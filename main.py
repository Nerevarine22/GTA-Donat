import argparse
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
import math
from pathlib import Path
import sys
from time import perf_counter

import cv2
import torch

import config
from renderer import OverlayRenderer
from tracker import PersonTracker, TrackedPerson


def _wait_for_playback(delay_ms: int, refresh: Callable[[], None]) -> bool:
    key = cv2.waitKey(delay_ms)
    if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        return False

    if key == ord(" "):
        while True:
            key = cv2.waitKey(30)
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return False
            refresh()
            if key in (ord(" "), ord("q"), ord("Q")):
                break
    return key not in (ord("q"), ord("Q"))


def _pick_track(people: list[TrackedPerson], x: int, y: int) -> int | None:
    selected = None
    smallest_area = math.inf
    for person in people:
        left, top, right, bottom = person.box
        if left <= x <= right and top <= y <= bottom:
            area = (right - left) * (bottom - top)
            if area < smallest_area:
                selected = person.track_id
                smallest_area = area
    return selected


def run_preview(input_video: Path, output_video: Path) -> None:
    capture = cv2.VideoCapture(str(input_video))
    writer = None
    window_created = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {input_video}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise RuntimeError("The input video does not report a valid frame rate.")

        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Could not decode the first frame of the input video.")
        height, width = frame.shape[:2]

        tracker = PersonTracker()
        renderer = OverlayRenderer(height)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*config.OUTPUT_FOURCC),
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video}")

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True
        scale = min(1.0, 1280 / width, 720 / height)
        cv2.resizeWindow(config.WINDOW_NAME, round(width * scale), round(height * scale))

        print(f"Device: CUDA ({torch.cuda.get_device_name(config.CUDA_DEVICE)})")
        print(f"Input: {width}x{height} at {source_fps:.2f} FPS.")
        print("Click a character to toggle their badge. Space: pause. Q: quit.")

        frame_period = 1.0 / source_fps
        smoothed_seconds: float | None = None
        total_seconds = 0.0
        frames_written = 0
        pending_write: Future[None] | None = None
        people: list[TrackedPerson] = []
        excluded_id: int | None = None
        displayed_exclusion: int | None = None
        fps: float | None = None

        def select_person(event: int, x: int, y: int, _flags: int, _data: object) -> None:
            nonlocal excluded_id
            if event == cv2.EVENT_LBUTTONDOWN:
                track_id = _pick_track(people, x, y)
                if track_id is not None:
                    excluded_id = None if excluded_id == track_id else track_id

        def render_frame() -> None:
            nonlocal displayed_exclusion, preview
            visible_people = [person for person in people if person.track_id != excluded_id]
            preview = frame.copy()
            renderer.draw(preview, visible_people, fps, "CUDA")
            cv2.imshow(config.WINDOW_NAME, preview)
            displayed_exclusion = excluded_id

        def refresh_paused_frame() -> None:
            if displayed_exclusion != excluded_id:
                render_frame()

        preview = frame
        cv2.setMouseCallback(config.WINDOW_NAME, select_person)

        # Keep one read and one write in flight. Inference and the GUI stay on the main thread.
        with ThreadPoolExecutor(max_workers=2) as video_io:
            frame_start = perf_counter()
            while ok:
                next_frame = video_io.submit(capture.read)
                people = tracker.update(frame)
                fps = 1.0 / smoothed_seconds if smoothed_seconds is not None else None
                render_frame()

                if pending_write is not None:
                    pending_write.result()
                pending_write = video_io.submit(writer.write, preview)

                elapsed = perf_counter() - frame_start
                frames_written += 1
                # Exclude first-frame model initialization and time spent waiting for playback.
                if frames_written > 1:
                    total_seconds += elapsed
                    if smoothed_seconds is None:
                        smoothed_seconds = elapsed
                    else:
                        smoothed_seconds += 0.15 * (elapsed - smoothed_seconds)

                delay_ms = max(1, round((frame_period - elapsed) * 1000))
                if not _wait_for_playback(delay_ms, refresh_paused_frame):
                    break
                frame_start = perf_counter()
                ok, frame = next_frame.result()

            if pending_write is not None:
                pending_write.result()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if window_created:
            cv2.destroyAllWindows()

    print(f"Saved {frames_written} frames to {output_video}")
    if total_seconds:
        print(f"Average processing FPS after startup: {(frames_written - 1) / total_seconds:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview NPC badges on a gameplay video.")
    parser.add_argument("--input", type=Path, default=config.INPUT_VIDEO, help="Input video path")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_VIDEO, help="Output MP4 path")
    args = parser.parse_args()
    if not args.input.is_file():
        print(f"Input video not found: {args.input}", file=sys.stderr)
        return 1
    if args.output.exists() and args.input.samefile(args.output):
        print("Input and output must be different files.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(
            "CUDA is unavailable. Install the CUDA-enabled PyTorch build and check the NVIDIA driver.",
            file=sys.stderr,
        )
        return 1

    try:
        run_preview(args.input, args.output)
    except (RuntimeError, OSError, cv2.error) as error:
        print(f"Preview failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
