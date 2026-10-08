"""
Step 1: Split a video into time-aligned frames (JPEG) and audio (WAV).

Usage:
  python extract_media.py <video_path> <out_dir> [--fps 2] [--audio_sr 16000]

Frames are saved as JPEG at the given FPS.
Audio is extracted as mono 16 kHz WAV.
An index CSV maps each frame index → wall-clock time so we can align
tokens across modalities later.
"""
import argparse, subprocess, os, csv
from pathlib import Path


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def probe_duration(video):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video
    ]).decode().strip()
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="frames per second to extract (default 2)")
    ap.add_argument("--audio_sr", type=int, default=16000)
    ap.add_argument("--size", type=int, default=128,
                    help="short-edge resize for frames (px)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    frames_dir = out / "frames"
    audio_path = out / "audio.wav"
    index_csv  = out / "frame_index.csv"
    frames_dir.mkdir(parents=True, exist_ok=True)

    dur = probe_duration(args.video)
    print(f"Video duration: {dur:.1f} s")
    n_expected = int(dur * args.fps)
    print(f"Extracting ~{n_expected} frames at {args.fps} fps, "
          f"resized to short edge = {args.size}px ...")

    # Frames
    run([
        "ffmpeg", "-y", "-i", args.video,
        "-vf", f"fps={args.fps},scale='if(gt(iw,ih),-2,{args.size})':'if(gt(iw,ih),{args.size},-2)'",
        "-q:v", "3",
        str(frames_dir / "frame_%06d.jpg"),
    ])

    # Audio (mono, downsampled)
    print(f"Extracting audio → {args.audio_sr} Hz mono WAV ...")
    run([
        "ffmpeg", "-y", "-i", args.video,
        "-vn", "-ac", "1", "-ar", str(args.audio_sr),
        "-acodec", "pcm_s16le", str(audio_path),
    ])

    # Build the frame-time index
    n_actual = len(list(frames_dir.glob("frame_*.jpg")))
    with open(index_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "time_sec"])
        for i in range(n_actual):
            w.writerow([i, i / args.fps])

    print(f"\n✅ Done. {n_actual} frames + {audio_path.stat().st_size/1e6:.1f} MB WAV")
    print(f"   frames: {frames_dir}")
    print(f"   audio : {audio_path}")
    print(f"   index : {index_csv}")


if __name__ == "__main__":
    main()
