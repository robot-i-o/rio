import multiprocessing as mp
from dataclasses import dataclass

import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

import rio.envs.factory as F

LOG_PATH = "/env/camera/"


def camera_streaming_loop(args, cameras, visualizer):
    freq = args.freq
    dt = 1.0 / freq
    t_start = time.now()
    it = 0
    input("Press Enter to start...")
    try:
        while True:
            t_cycle_end = t_start + (it + 1) * dt

            for cam_name, cam in cameras.items():
                cam_state = cam.get_state()
                rgb = cam_state["color"]
                if rgb is not None and visualizer is not None:
                    path = f"{LOG_PATH}{cam_name}/rgb"
                    visualizer.log_image(path, rgb)
                else:
                    logger.warning(f"No RGB frame from camera {cam_name}")

                depth = cam_state.get("depth")
                if depth is not None and visualizer is not None:
                    path = f"{LOG_PATH}{cam_name}/depth"
                    visualizer.log_depth(path, depth, depth_units=cam_state.get("depth_units"))
            time.precise_wait(t_cycle_end)
            it += 1
    except KeyboardInterrupt:
        print("Shutting down camera streaming loop.")


def main(args):
    for cam in args.cameras.values():
        cam.cfg["enable_depth"] = args.depth
    cam_servers, cam_clients = F.make_cameras(args.mw, args.cameras)
    visualizer_server, visualizer_client = F.make_node(
        args.mw, "visualization", args.visualizer, {**F.asdict(args.visualizer_cfg), "max_queue_size": 100}, package="rio"
    )
    servers = [*list(cam_servers.values()), visualizer_server] if visualizer_server else list(cam_servers.values())
    with ServerManager(args.mw, servers):
        with F.init_clients(cam_clients) as cams, F.init_clients({"visualizer": visualizer_client}) as viz:
            camera_streaming_loop(args, cams, viz.get("visualizer", None))


if __name__ == "__main__":
    from examples import get_station_cfg

    StationCfg = get_station_cfg()

    @dataclass
    class Args(StationCfg):
        mw: str = "Thread"
        mp_method: str = "spawn"
        freq: int = 50
        depth: bool = True

        visualizer: str | None = "Rerun"

    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)
