<style>
  .rio-cta a {
    display: inline-block;
    padding: 0.65rem 1.6rem;
    border-radius: 6px;
    font-weight: 600;
    text-decoration: none;
    margin: 0.3rem;
  }
  .rio-cta .primary {
    background: var(--md-primary-fg-color);
    color: var(--md-primary-bg-color);
  }
  .rio-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.2rem;
    margin: 1rem auto;
    max-width: 860px;
    align-items: stretch;
  }
  .rio-grid .card {
    padding: 0.9rem;
    border-radius: 8px;
    border: 1px solid var(--md-default-fg-color--lightest);
    text-align: left;
  }
  .rio-grid .card p { font-size: 0.5rem; opacity: 0.85; margin-bottom: 0; line-height: 1.4; }
  .rio-grid .card p:first-child { font-size: 0.7rem; opacity: 1; margin-bottom: 0rem; margin-top: 0rem;}
</style>

<div class="rio-hero" markdown>
<p class="tagline">
  RIO is a real-time control library for cross-embodiment robot manipulation, supporting the full robot learning workflow.
</p>

<div class="rio-badges">
  <a href="https://github.com/astral-sh/uv">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" /></a>
  <a href="https://github.com/astral-sh/ruff">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
  <a href="https://github.com/astral-sh/ty">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json" /></a>
</div>

<div class="rio-cta">
  <a href="quickstart" class="primary">Get Started</a>
</div>

</div>

<div class="rio-grid" markdown>
<div class="card" markdown>
**:material-robot: Cross-Embodiment**

Support for XArm, UR5, Franka, SO100, Leap Hand, XHand — configure any robot with a single dataclass.
</div>

<div class="card" markdown>
**:material-gamepad-variant: Teleoperation**

Collect demonstrations with Spacemouse, gamepad, keyboard, or Gello leader arms. Record trajectories automatically.
</div>

<div class="card" markdown>
**:material-brain: Policy Deployment**

Export data to LeRobot/DROID format and fine-tune pi0 policies with the openpi training pipeline.
</div>

<div class="card" markdown>
**:material-play-circle: Closed-Loop Deployment**

Run fine-tuned VLA policies on real hardware at up to 250 Hz with action chunking and automatic re-planning.
</div>

</div>

## Architecture

RIO is split into two packages:

- [**RIO**](https://github.com/robot-i-o/rio) — utilities and abstractions for robot learning: data collection, format conversion, policy interfaces, and orchestration.
- [**RIO_HW**](https://github.com/robot-i-o/rio-hw) — backend code and hardware interfaces: real-time control loops, driver bindings, and communication middleware.

Everything is wired together through a **station config** which is a Python dataclass that declares every node in your setup (arms, grippers, cameras, recorders, policy servers). See [Station Configuration](workflow/station_cfg.md) for details.

---

## Installation
!!! note "Recommended System Setup"
    RIO should work on any *Linux* system.
    For replicating exact tested setup, see [Ubuntu RT Setup](setup-tutorials/ubuntu.md) for instructions.
    
```bash
# clone the repo
git clone git@github.com:robot-i-o/rio.git
cd rio

# install openpi (third-party policy server)
mkdir third_party && cd third_party
git clone git@github.com:Physical-Intelligence/openpi.git
cd ..

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv and install dependencies
uv venv --python 3.10
source .venv/bin/activate
uv sync --all-extras
```
