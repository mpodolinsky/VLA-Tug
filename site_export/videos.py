"""The rollout videos of the chosen episodes: checked for the browser (H.264, yuv420p) and for the length the page's clock relies on, and copied
(or re-encoded when they are not browser-ready) next to the page. Needs ffmpeg and ffprobe."""

import json
import shutil
import subprocess
from pathlib import Path

VIEWS = {"agent": "rollout.mp4", "wrist": "wrist.mp4"}  # the file names of a rollout folder of the released datasets


def probe(path: Path) -> dict:
    """Codec, pixel format, frame rate and the number of frames of the first video stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=codec_name,pix_fmt,r_frame_rate,nb_read_frames",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
    return {"codec": stream["codec_name"], "pix_fmt": stream["pix_fmt"], "fps": num / den, "frames": int(stream["nb_read_frames"])}


def browser_ready(info: dict) -> bool:
    return info["codec"] == "h264" and info["pix_fmt"] == "yuv420p"


def transcode(source: Path, target: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(target)], check=True)


def prepare(source: Path, key: str, out_dir: Path, rel_dir: str, expected_seconds: float) -> dict:
    """Copy the two videos of the episode ``key`` from ``source/<key>/`` to ``out_dir`` and return what the page needs: their paths (relative to the page),
    the duration and the frame rate. Raises when the length differs from ``expected_seconds`` (control steps / control rate) by more than a frame, because
    the page maps video time to policy steps with it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for view, name in VIEWS.items():
        src = source / key / name
        if not src.exists():
            raise FileNotFoundError(f"{src} not found; see the README for how to get the videos of the chosen episodes")
        info = probe(src)
        target = out_dir / f"{view}.mp4"
        if browser_ready(info):
            shutil.copyfile(src, target)
        else:
            transcode(src, target)
            info = probe(target)
        seconds = info["frames"] / info["fps"]
        if abs(seconds - expected_seconds) > 1 / info["fps"]:
            raise ValueError(f"{src}: {seconds:.2f} s long, the episode has {expected_seconds:.2f} s of control steps")
        result[view] = f"{rel_dir}/{view}.mp4"
        result.setdefault("duration", round(seconds, 3))
        result.setdefault("fps", info["fps"])
    return result
