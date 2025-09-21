#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path
import subprocess
import time
from typing import List, Optional

# -------- Progress bar setup --------
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def pwrite(msg: str):
    """Print without breaking the progress bar."""
    if _HAS_TQDM:
        tqdm.write(msg)
    else:
        print(msg)

# ------------------------------------

APPIMAGE = Path("/data2/weihanx/musicgpt/MuseScore-Studio-4.4.4.243461245-x86_64.AppImage")

# ---------- Helpers ----------

def load_txt(filename: Path) -> List[str]:
    """Load a TXT file as a list (stripped, non-empty)."""
    with open(filename, encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]

def _start_xvfb() -> Optional[subprocess.Popen]:
    """
    Start a private Xvfb if available (from conda-forge xorg-xvfb), return the process.
    If not available, return None (we'll rely on QT_QPA_PLATFORM=offscreen).
    """
    xvfb = shutil.which("Xvfb")
    if not xvfb:
        return None
    proc = subprocess.Popen([xvfb, ":99", "-screen", "0", "1920x1080x24"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # small wait so the display is ready
    time.sleep(0.2)
    return proc

def _ensure_executable(p: Path):
    try:
        p.chmod(p.stat().st_mode | 0o111)
    except Exception:
        pass  # best-effort

def run_musescore(input_path: Path, output_path: Path, use_xvfb: bool = True) -> bool:
    """
    Run MuseScore AppImage to export .mscz -> .musicxml.
    Returns True on success (non-empty output file), False otherwise.
    """
    _ensure_executable(APPIMAGE)
    input_str = str(input_path)
    output_str = str(output_path)

    # Build command. MuseScore 4 uses "-o" consistently for export:
    #   mscore -o out.musicxml in.mscz
    cmd = [str(APPIMAGE), "-o", output_str, input_str]

    # Headless-friendly environment
    env = os.environ.copy()
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("MUSESCORE_NO_AUDIO", "1")
    env.setdefault("JACK_NO_START_SERVER", "1")

    xvfb_proc = None
    if use_xvfb:
        xvfb_proc = _start_xvfb()
        if xvfb_proc is not None:
            env["DISPLAY"] = ":99"
        else:
            # Fall back to pure headless Qt if Xvfb is not available
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            env.setdefault("QT_OPENGL", "software")
    else:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("QT_OPENGL", "software")

    pwrite(f"Exporting {input_str} -> {output_str}")
    try:
        result = subprocess.run(
            cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        if result.returncode != 0:
            pwrite(f"MuseScore exited with error code: {result.returncode}")
            if result.stdout:
                pwrite("stdout: " + result.stdout.decode(errors="replace"))
            if result.stderr:
                pwrite("stderr: " + result.stderr.decode(errors="replace"))
            return False

        # Sanity check file created & non-empty
        if not output_path.exists() or output_path.stat().st_size == 0:
            pwrite(f"❌ Output file missing or empty: {output_str}")
            if result.stdout:
                pwrite("stdout: " + result.stdout.decode(errors="replace"))
            if result.stderr:
                pwrite("stderr: " + result.stderr.decode(errors="replace"))
            return False

        pwrite(f"✅ OK: {output_str} ({output_path.stat().st_size} bytes)")
        return True
    finally:
        if xvfb_proc is not None:
            try:
                xvfb_proc.terminate()
                xvfb_proc.wait(timeout=2)
            except Exception:
                xvfb_proc.kill()

def run_musescore_batch(video_name_path: Path, input_dir: Path, output_dir: Path):
    """
    video_name_path: TXT file with one base name per line, e.g. QmXX... (no extension)
    It will map to:
      input:  input_dir / vd[2] / vd[3] / f"{vd}.mscz"
      output: output_dir / vd[2] / vd[3] / f"{vd}.musicxml"
    (This matches your X/X sharding convention for hashes like 'QmXX...').
    """
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = load_txt(video_name_path)

    # Prefer Xvfb if available; otherwise use pure headless Qt
    use_xvfb = shutil.which("Xvfb") is not None

    saved = 0
    skipped_exists = 0
    missing_input = 0
    failed = 0

    iterable = names
    bar = None
    if _HAS_TQDM:
        # If not a TTY (e.g., slurm log), tqdm still works but may be verbose; tune update intervals.
        bar = tqdm(
            iterable,
            desc="Exporting MusicXML",
            unit="file",
            dynamic_ncols=True,
            mininterval=0.2,
            maxinterval=5.0,
            smoothing=0.1,
        )
        iterable = bar

    for vd in iterable:
        if len(vd) < 4:
            pwrite(f"Skipping malformed id (too short): {vd}")
            failed += 1
            if bar:
                bar.set_postfix(saved=saved, skip=skipped_exists, miss=missing_input, fail=failed, refresh=False)
            continue

        shard1, shard2 = vd[2], vd[3]  # 'X', 'X' for 'QmXX...'
        in_path = input_dir / shard1 / shard2 / f"{vd}.mscz"
        out_path = output_dir / shard1 / shard2 / f"{vd}.musicxml"

        # quick skip
        if out_path.exists() and out_path.stat().st_size > 0:
            pwrite(f"Already exists, skipping: {out_path}")
            skipped_exists += 1
            if bar:
                bar.set_postfix(saved=saved, skip=skipped_exists, miss=missing_input, fail=failed, refresh=False)
            continue

        if not in_path.exists():
            pwrite(f"❌ Input missing, skipping: {in_path}")
            missing_input += 1
            if bar:
                bar.set_postfix(saved=saved, skip=skipped_exists, miss=missing_input, fail=failed, refresh=False)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        ok = run_musescore(in_path, out_path, use_xvfb=use_xvfb)
        if not ok:
            # Retry once with opposite headless mode (often helps on finicky nodes)
            pwrite("Retrying with alternate headless mode...")
            ok2 = run_musescore(in_path, out_path, use_xvfb=not use_xvfb)
            if not ok2:
                pwrite(f"❌ Failed permanently: {vd}")
                failed += 1
            else:
                saved += 1
        else:
            saved += 1

        if bar:
            bar.set_postfix(saved=saved, skip=skipped_exists, miss=missing_input, fail=failed, refresh=False)

    if bar:
        bar.close()

    dt = time.time() - t_start
    total = len(names)
    pwrite("========== Summary ==========")
    pwrite(f"Total IDs:        {total}")
    pwrite(f"Saved:            {saved}")
    pwrite(f"Skipped (exists): {skipped_exists}")
    pwrite(f"Missing input:    {missing_input}")
    pwrite(f"Failed:           {failed}")
    pwrite(f"Wall time:        {dt/60:.2f} min ({(dt/max(1,total)):.2f} s/item)")

# ---------- Entrypoint ----------

if __name__ == "__main__":
    root_path = Path("/data2/weihanx/musicgpt/metascore_plus")
    video_name_path = Path("/data2/weihanx/musicgpt/musicxml_file_name/names_part_20of20.txt")

    input_dir = Path("/data3/weihanx/datasetforrelease/mscz")
    output_dir = Path("/data2/weihanx/musicgpt/musicxml")
    run_musescore_batch(video_name_path, input_dir, output_dir)
