#!/usr/bin/env python3
"""Slow deep voice on one stretch of a finished export (owner-approved move).

Usage: python3 -I tools/slow_voice.py <in.mp4> <out.mp4> <start_s> <end_s> [--tempo 0.625] [--pitch 0.62]

Picture and audio between start and end are slowed to `tempo` (video frames repeat, 30 fps kept);
the voice is pitched down by `pitch` (rubberband, formants shifted). The rest plays untouched.
Loudness is re-normalised to -14 LUFS and the audio is cut to the video length. Refuses to overwrite.
"""
import os, subprocess, sys
a = sys.argv[1:]
src, out, A, B = a[0], a[1], float(a[2]), float(a[3])
tempo = float(a[a.index("--tempo") + 1]) if "--tempo" in a else 0.625
pitch = float(a[a.index("--pitch") + 1]) if "--pitch" in a else 0.62
if os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}; bump the version")
A, B = round(A * 30) / 30, round(B * 30) / 30
tmp = out + ".tmp.mp4"
fc = (f"[0:v]split=3[v0][v1][v2];[0:a]asplit=3[a0][a1][a2];"
      f"[v0]trim=0:{A},setpts=PTS-STARTPTS[x0];[a0]atrim=0:{A},asetpts=PTS-STARTPTS[y0];"
      f"[v1]trim={A}:{B},setpts={1/tempo:.4f}*(PTS-STARTPTS),fps=30[x1];"
      f"[a1]atrim={A}:{B},asetpts=PTS-STARTPTS,rubberband=tempo={tempo}:pitch={pitch}:formant=shifted,volume=2.5dB[y1];"
      f"[v2]trim=start={B},setpts=PTS-STARTPTS[x2];[a2]atrim=start={B},asetpts=PTS-STARTPTS[y2];"
      f"[x0][y0][x1][y1][x2][y2]concat=n=3:v=1:a=1[v][a];[a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[an]")
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-filter_complex", fc, "-map", "[v]", "-map", "[an]",
                "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "256k", "-ar", "48000", tmp], check=True)
vd = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "csv=p=0", tmp],
                    capture_output=True, text=True).stdout.strip()
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", tmp, "-t", vd, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                "-movflags", "+faststart", out], check=True)
os.remove(tmp)
print(f"{out}: slowed {A:.3f}-{B:.3f}s at {tempo}x, pitch {pitch}; video {vd}s")
