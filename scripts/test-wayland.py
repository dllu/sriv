#!/usr/bin/env python3
"""Run the GPU surface regression test in an isolated, headless Sway session."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


def descendants(node):
    yield node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        yield from descendants(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sway", default=shutil.which("sway"))
    args = parser.parse_args()
    if not args.sway:
        parser.error("Sway is required; install it or pass --sway /path/to/sway")
    sway = Path(args.sway).resolve()
    swaymsg = sway.with_name("swaymsg")
    if not swaymsg.is_file():
        parser.error(f"swaymsg must be installed alongside {sway}")
    repo = Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "test", "--locked", "--no-run"], cwd=repo, check=True)

    with tempfile.TemporaryDirectory(prefix="sriv-wayland-") as directory:
        runtime = Path(directory)
        config = runtime / "sway.conf"
        config.write_text(
            "xwayland disable\n"
            "output HEADLESS-1 mode 960x720 position 0 0\n"
            "output HEADLESS-2 mode 1280x960 position 960 0 scale 1.5\n"
            "seat seat0 fallback true\n"
        )
        env = dict(os.environ)
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "SWAYSOCK"):
            env.pop(name, None)
        env.update(
            XDG_RUNTIME_DIR=directory,
            WLR_BACKENDS="headless",
            WLR_RENDERER="pixman",
            WLR_HEADLESS_OUTPUTS="2",
        )
        with (runtime / "sway.log").open("w+") as log:
            compositor = subprocess.Popen(
                [str(sway), "--unsupported-gpu", "-c", str(config)],
                env=env, stdout=log, stderr=log,
            )
            test = None
            try:
                deadline = time.monotonic() + 10
                while True:
                    sockets = list(runtime.glob("sway-ipc.*.sock"))
                    displays = [p for p in runtime.glob("wayland-*") if p.is_socket()]
                    if sockets and displays:
                        break
                    if compositor.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("headless Sway failed to start")
                    time.sleep(0.05)
                env.update(SWAYSOCK=str(sockets[0]), WAYLAND_DISPLAY=displays[0].name)

                def ipc(*command):
                    return json.loads(subprocess.check_output(
                        [str(swaymsg), "-s", env["SWAYSOCK"], *command], text=True,
                        timeout=5,
                    ))

                test = subprocess.Popen(
                    ["cargo", "test", "--locked", "wayland_surface_lifecycle",
                     "--", "--ignored", "--nocapture"], cwd=repo, env=env,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 45
                changes = 0
                while test.poll() is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError("Wayland test timed out")
                    windows = [n for n in descendants(ipc("-t", "get_tree"))
                               if n.get("name") == "sriv Wayland test"]
                    for window in windows:
                        assert window.get("shell") == "xdg_shell", window
                        output = 1 + changes % 2
                        width, height = 480 + changes % 7 * 40, 320 + changes % 5 * 32
                        results = ipc(
                            f'[con_id={window["id"]}] move container to output HEADLESS-{output}; '
                            f'[con_id={window["id"]}] floating enable; '
                            f'[con_id={window["id"]}] resize set {width} {height}'
                        )
                        if not all(r["success"] for r in results):
                            # The window can close between get_tree and these commands.
                            assert test.wait(timeout=3) == 0, results
                            break
                        changes += 1
                    time.sleep(0.05)
                if test.returncode:
                    raise RuntimeError(f"Wayland test exited with status {test.returncode}")
                assert changes > 0, "did not observe a native Wayland window"
                print(f"Passed native Wayland recovery and {changes} resize/output-scale changes")
            except BaseException:
                log.seek(0)
                print(log.read())
                raise
            finally:
                if test is not None and test.poll() is None:
                    os.killpg(test.pid, signal.SIGTERM)
                    try:
                        test.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(test.pid, signal.SIGKILL)
                        test.wait()
                compositor.terminate()
                try:
                    compositor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    compositor.kill()
                    compositor.wait()


if __name__ == "__main__":
    main()
