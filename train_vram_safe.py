from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


def parse_image_sizes(raw: str) -> list[int]:
    sizes = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    if not sizes:
        raise ValueError("At least one image size is required.")
    return sizes


def build_trials(
    start_batch_size: int,
    start_accumulation_steps: int,
    min_batch_size: int,
    max_accumulation_steps: int,
) -> list[tuple[int, int]]:
    if start_batch_size < 1:
        raise ValueError("start_batch_size must be >= 1")
    if start_accumulation_steps < 1:
        raise ValueError("start_accumulation_steps must be >= 1")
    if min_batch_size < 1:
        raise ValueError("min_batch_size must be >= 1")
    if max_accumulation_steps < 1:
        raise ValueError("max_accumulation_steps must be >= 1")

    trials: list[tuple[int, int]] = []
    batch_size = start_batch_size
    accumulation_steps = start_accumulation_steps

    while True:
        current = (batch_size, accumulation_steps)
        if current not in trials:
            trials.append(current)

        if batch_size > min_batch_size:
            batch_size = max(min_batch_size, batch_size // 2)
            accumulation_steps = min(max_accumulation_steps, accumulation_steps * 2)
            continue

        if accumulation_steps >= max_accumulation_steps:
            break

        accumulation_steps = min(max_accumulation_steps, accumulation_steps * 2)

    return trials


def has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in args)


def remove_flag_and_value(args: list[str], flag: str) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for idx, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        if arg == flag:
            if idx + 1 < len(args):
                skip_next = True
            continue

        if arg.startswith(flag + "="):
            continue

        cleaned.append(arg)

    return cleaned


def get_flag_value(args: list[str], flag: str) -> str | None:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def resolve_run_dir(line: str, cwd: Path) -> Path | None:
    match = re.search(r"(?:Run directory:|Checkpoints saved to:)\s*(.+)", line)
    if not match:
        return None

    run_dir = Path(match.group(1).strip())
    if not run_dir.is_absolute():
        run_dir = cwd / run_dir
    return run_dir


def resolve_resume_checkpoint(run_dir: Path) -> Path | None:
    candidates = [run_dir / "last_model.pt", run_dir / "best_model.pt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_trial(
    python_executable: str,
    train_script: str,
    passthrough_args: list[str],
    batch_size: int,
    accumulation_steps: int,
    image_size: int,
    allocator_max_split_mb: int,
    resume_from: str | None,
) -> tuple[int, bool, Path | None, Path | None]:
    dynamic_args = remove_flag_and_value(passthrough_args, "--resume_from")
    if resume_from:
        dynamic_args.extend(["--resume_from", resume_from])

    command = [
        python_executable,
        "-u",
        train_script,
        *dynamic_args,
        "--batch_size",
        str(batch_size),
        "--accumulation_steps",
        str(accumulation_steps),
        "--image_size",
        str(image_size),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{allocator_max_split_mb}"

    print(
        f"\n[trial] image_size={image_size}, batch_size={batch_size}, "
        f"accumulation_steps={accumulation_steps}"
    )
    print("[trial] command:", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    seen_oom = False
    run_dir: Path | None = None
    resume_checkpoint: Path | None = None
    cwd = Path.cwd()

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lowered = line.lower()
        if any(pattern in lowered for pattern in OOM_PATTERNS):
            seen_oom = True

        parsed_run_dir = resolve_run_dir(line, cwd=cwd)
        if parsed_run_dir is not None:
            run_dir = parsed_run_dir
            resolved = resolve_resume_checkpoint(run_dir)
            if resolved is not None:
                resume_checkpoint = resolved

    return_code = process.wait()
    if run_dir is not None and resume_checkpoint is None:
        resume_checkpoint = resolve_resume_checkpoint(run_dir)

    return return_code, seen_oom, run_dir, resume_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch train.py with automatic VRAM-safe fallback settings."
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train_script", type=str, default="train.py")
    parser.add_argument("--start_batch_size", type=int, default=4)
    parser.add_argument("--start_accumulation_steps", type=int, default=4)
    parser.add_argument("--min_batch_size", type=int, default=1)
    parser.add_argument("--max_accumulation_steps", type=int, default=32)
    parser.add_argument("--image_sizes", type=str, default="224,192,160")
    parser.add_argument("--allocator_max_split_mb", type=int, default=128)
    parser.add_argument(
        "--auto_resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from the latest checkpoint between retries.",
    )
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    train_script_path = Path(args.train_script)
    if not train_script_path.exists():
        raise FileNotFoundError(f"Train script not found: {train_script_path}")

    passthrough_args = args.train_args[:]
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    if not has_flag(passthrough_args, "--use_amp"):
        passthrough_args.extend(["--use_amp", "true"])
    if not has_flag(passthrough_args, "--pin_memory"):
        passthrough_args.extend(["--pin_memory", "true"])

    image_sizes = parse_image_sizes(args.image_sizes)
    trial_pairs = build_trials(
        start_batch_size=args.start_batch_size,
        start_accumulation_steps=args.start_accumulation_steps,
        min_batch_size=args.min_batch_size,
        max_accumulation_steps=args.max_accumulation_steps,
    )

    print("[launcher] image sizes:", image_sizes)
    print("[launcher] (batch_size, accumulation_steps) trials:", trial_pairs)

    current_resume: str | None = get_flag_value(passthrough_args, "--resume_from")
    if current_resume:
        print(f"[launcher] initial resume checkpoint from args: {current_resume}")

    for image_size in image_sizes:
        for batch_size, accumulation_steps in trial_pairs:
            return_code, seen_oom, run_dir, resume_checkpoint = run_trial(
                python_executable=args.python,
                train_script=str(train_script_path),
                passthrough_args=passthrough_args,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                image_size=image_size,
                allocator_max_split_mb=args.allocator_max_split_mb,
                resume_from=current_resume,
            )

            if run_dir is not None:
                print(f"[launcher] detected run directory: {run_dir}")
            if resume_checkpoint is not None:
                print(f"[launcher] detected resume checkpoint: {resume_checkpoint}")

            if return_code == 0:
                print(
                    "\n[launcher] success with "
                    f"image_size={image_size}, batch_size={batch_size}, "
                    f"accumulation_steps={accumulation_steps}"
                )
                return

            if seen_oom:
                if args.auto_resume and resume_checkpoint is not None:
                    current_resume = str(resume_checkpoint)
                    print(f"[launcher] auto-resume enabled. Next retry will resume from: {current_resume}")
                print("[launcher] CUDA OOM detected, trying a safer setting...")
                continue

            raise RuntimeError(
                "Training failed with a non-OOM error. "
                "Please inspect logs above and fix that error first."
            )

    raise RuntimeError(
        "All VRAM-safe trials were exhausted. "
        "Try smaller image sizes, lower start batch size, or freeze the visual encoder."
    )


if __name__ == "__main__":
    main()
