# Button Control

A USB button pad plugged into the robot host lets an operator start, stop, and discard
demonstrations without touching a keyboard or the terminal. This matters when the
operator has both hands on the leader arms.

## How it works

The pad is read through the Linux input layer (`/dev/input/event*`) by the
`EvdevKeyboard` node, which runs on the machine the pad is plugged into. Because it
reads the kernel device directly, it needs no graphical session, no controlling
terminal, and no window focus. A script launched over SSH still sees presses on a pad
attached to the robot host.

Presses become `Command` values, which the control loop consumes. The command is
independent of which device raised it, so the same workflow can later be driven from a
different input source without changing the loop.

| Command | Meaning |
| --- | --- |
| `TOGGLE_EPISODE` | Start recording a demonstration, or stop and save the one in progress |
| `DISCARD_LAST` | Delete the most recently saved demonstration |
| `TOGGLE_INFERENCE` | Start or stop the policy inference loop |
| `GO_HOME` | Return the robot to its home pose, ready for another trial |

One button covers both halves of `TOGGLE_EPISODE`: pressing it while idle starts an
episode, pressing it again stops and saves. This is what lets a small pad run a full
collection session.

## One-time host setup

`/dev/input/event*` is owned by `root:input` with mode `0660`. A local desktop session
is granted access automatically, but an SSH session is not, so without this step the
device is invisible to a script started over SSH — and `evdev` reports *no devices*
rather than a permission error.

Install the udev rule from the hardware repository:

```bash
sudo cp rio-hw/scripts/setup/buttons/99-rio-buttons.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Then unplug and replug the pad. To grant an existing user access without a rule change,
add them to the `input` group instead — note the session must be fully reconnected
afterwards, since group membership is fixed at login:

```bash
sudo usermod -aG input "$USER"
```

Verify the pad is visible:

```bash
uv run python -m rio._scripts.list_input_devices
```

## Station configuration

Declare the pad as a node in the station config. Matching by USB id means the device
node number does not matter, so nothing breaks when devices re-enumerate after a reboot:

```python
buttons: str | None = "EvdevKeyboard"
buttons_module: str = "interfaces"
buttons_cfg: NodeCfg | None = field(
    default_factory=lambda: NodeCfg(
        vendor_id=0x3553,   # find with `lsusb`
        product_id=0xC011,
        grab=True,          # keep presses out of the operator's shell
        freq=100,
    )
)

button_bindings: dict[str, str] = field(
    default_factory=lambda: {
        "a": "TOGGLE_EPISODE",
        "c": "DISCARD_LAST",
    }
)
```

Bind by character (`"a"`) or by raw Linux input-event code (`30`) when a button sends
something without a character. `grab=True` takes exclusive access, so presses do not also
land in whatever shell has the terminal; leave it off for a keyboard the operator also
types on.

The buttons drive the recorder, so it must be enabled — several station configs ship
with `recorder = None` for testing, and with it unset a press is logged and ignored:

```python
recorder: str | None = "Recorder"
```

Set `start_recording=False` so the first episode begins on a button press rather than at
startup:

```python
recorder_cfg: RecorderCfg = field(
    default_factory=lambda: RecorderCfg(path=f"data/{TASK}/", start_recording=False)
)
```

Any pad works — the node is not specific to one model. Find a new pad's USB ids with
`lsusb`, and its keycodes with the device listing script above.

## Collecting data

```bash
STATION=BimanualYamStation uv run python -m examples.teleop_leader_follower
```

Teleoperation goes live immediately. The terminal prints a colour-coded banner on every
state change, so the current state is readable across the room:

| Banner | Meaning |
| --- | --- |
| `IDLE` (white) | Ready. Press the start button to begin an episode |
| `RECORDING` (red) | Recording. Press again to stop and save |
| `SAVING` (yellow) | Writing to disk. Buttons are ignored until it finishes |

`DISCARD_LAST` only applies between episodes; pressing it mid-episode is refused and
logged, so a mispress cannot destroy a demonstration being recorded. The discarded
trajectory's index is reused by the next episode, so numbering has no gaps.

## Policy inference

```bash
STATION=BimanualYamStation POLICY=MolmoAct2BimanualYamCfg uv run python -m examples.policy_inference
```

Inference starts **stopped** when buttons are configured, so the robot does not move until
someone presses start. The bound buttons then toggle inference and return the robot home
between trials.

Stopping and restarting inference re-reads the robot's actual pose and discards any
pending action chunk, so a chunk computed before the pause is never replayed against a
robot that has since moved.

### Going home

`GO_HOME` stops inference first, then drives the robot to `home_pose` along a cosine
ease-in-out ramp over `home_duration` seconds — accelerating from rest and arriving at
rest. It never commands the home pose directly: that would ask the arm to cross the whole
distance in a single control cycle.

```python
# Action layout: six joints then gripper, per arm. Zero joints, both grippers open.
home_pose: list[float] | None = field(
    default_factory=lambda: [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
)
home_duration: float = 5.0  # raise for a gentler return
```

Leave `home_pose` unset and it defaults to every joint at zero with the grippers open,
with the gripper positions discovered from the embodiment rather than hard-coded.

Inference stays stopped after homing; press the start button again for the next trial.

Homing is **not** available during leader-follower teleoperation. The leader arms run in
zero-gravity mode and ignore position commands, so homing the followers while the leaders
stayed put would snap the followers back to the leader pose on the next cycle.

## Running without buttons

Buttons are optional in both scripts. If the station declares none, or declares a pad
that is not currently plugged in, the script warns once and runs without it:

- **Data collection** prompts for Enter, then records the whole session as a single
  trajectory and saves it on Ctrl-C — the behaviour from before buttons existed.
- **Policy inference** prompts for Enter and then runs continuously, with no homing
  available.

A declared-but-absent pad costs a short startup delay while the node waits to become
ready; `timeout` in `buttons_cfg` bounds it. The node prints its own traceback in that
case, which is informational — the warning that follows says the script is continuing.

## Notes

A button pad routed through the control loop is **not** an emergency stop. Worst-case
latency is unbounded. Keep the physical e-stop as the safety device.
