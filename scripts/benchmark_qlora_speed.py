"""Two-stage, single-variable QLoRA throughput benchmark on a fixed 128-row set."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.qlora_common import AnswerOnlyCollator, CocoYesNoDataset, DATA_DIR, RESULTS_DIR, config, load_quantized_model, move_batch, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark checkpointing, then micro-batch size, without changing effective batch.")
    parser.add_argument("--phase", choices=("1", "2", "all"), default="all")
    parser.add_argument("--phase1-steps", type=int, default=5,
                        help="Short checkpointing/OOM probe per case (default: 5).")
    parser.add_argument("--phase1-warmup-steps", type=int, default=1)
    parser.add_argument("--phase2-steps", type=int, default=16,
                        help="Throughput measurement per micro-batch case (default: 16).")
    parser.add_argument("--phase2-warmup-steps", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fixed_128() -> list[dict[str, Any]]:
    rows = read_jsonl(DATA_DIR / "coco_train_e1.jsonl")
    yes = [row for row in rows if row["label"] == "yes"][:64]
    no = [row for row in rows if row["label"] == "no"][:64]
    selected = yes + no
    if len(selected) != 128: raise RuntimeError("The speed subset requires 64 yes + 64 no rows.")
    write_jsonl(DATA_DIR / "speed128.jsonl", selected)
    return selected


def run_case(name: str, micro_batch: int, accumulation: int, checkpointing: bool, rows: list[dict[str, Any]], steps: int, warmup: int) -> dict[str, Any]:
    """Manual loop exposes exact optimizer-step times and catches OOM per configuration."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    result: dict[str, Any] = {"name": name, "micro_batch": micro_batch, "gradient_accumulation": accumulation, "effective_batch": micro_batch * accumulation, "gradient_checkpointing": checkpointing, "oom": False}
    if result["effective_batch"] != 16: raise ValueError("Every benchmark configuration must keep effective batch=16.")
    model = processor = optimizer = None
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, processor, _ = load_quantized_model("e1", checkpointing)
        model.train(); optimizer = __import__("scripts.qlora_common", fromlist=["optimizer_groups"]).optimizer_groups(model, "e1")
        loader = cycle(DataLoader(CocoYesNoDataset(rows), batch_size=micro_batch, shuffle=False, collate_fn=AnswerOnlyCollator(processor)))
        device = model.get_input_embeddings().weight.device; timings: list[float] = []; measured_samples = 0
        progress = tqdm(range(steps), desc=f"测速 {name}", unit="optimizer-step", dynamic_ncols=True)
        for step in progress:
            torch.cuda.synchronize(); started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accumulation):
                batch = move_batch(next(loader), device); loss = model(**batch).loss / accumulation
                if not torch.isfinite(loss): raise RuntimeError("Non-finite loss during speed benchmark.")
                step_loss += float(loss.detach().cpu())
                loss.backward()
            optimizer.step(); torch.cuda.synchronize(); elapsed = time.perf_counter() - started
            progress.set_postfix(loss=f"{step_loss:.4f}", step_s=f"{elapsed:.2f}",
                                 peak_GB=f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}")
            if step >= warmup:
                timings.append(elapsed); measured_samples += 16
        result.update({"measured_optimizer_steps": len(timings), "mean_step_seconds": statistics.mean(timings), "p50_step_seconds": statistics.median(timings), "samples_per_second": measured_samples / sum(timings), "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3})
    except torch.OutOfMemoryError:
        result["oom"] = True
    except RuntimeError as error:
        if "out of memory" in str(error).lower(): result["oom"] = True
        else: raise
    finally:
        del optimizer, processor, model; gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass
    return result


def run_isolated_case(name: str, micro_batch: int, accumulation: int,
                      checkpointing: bool, steps: int, warmup: int) -> dict[str, Any]:
    """Run one configuration in a fresh process so CUDA baselines cannot leak."""
    result_path = RESULTS_DIR / f".qlora_speed_case_{name.lower()}.json"
    result_path.unlink(missing_ok=True)
    worker = PROJECT_DIR / "scripts" / "benchmark_qlora_case.py"
    command = [
        sys.executable, str(worker),
        "--name", name,
        "--micro-batch", str(micro_batch),
        "--accumulation", str(accumulation),
        "--checkpointing", "1" if checkpointing else "0",
        "--steps", str(steps),
        "--warmup-steps", str(warmup),
        "--output", str(result_path),
    ]
    print(f"\n启动独立子进程：{name}（micro={micro_batch}, accum={accumulation}, checkpointing={checkpointing}）", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    if not result_path.exists():
        raise RuntimeError(f"Isolated benchmark did not write its result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_path.unlink(missing_ok=True)
    return result


def fastest(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in results if not row["oom"]]
    if not candidates: raise RuntimeError("All speed configurations OOM; do not start smoke training.")
    return max(candidates, key=lambda row: float(row["samples_per_second"]))


def main() -> None:
    args = parse_args(); RESULTS_DIR.mkdir(exist_ok=True)
    phase1_output = RESULTS_DIR / "qlora_speed_phase1.json"
    final_output = RESULTS_DIR / "qlora_speed_benchmark.json"
    if args.phase in ("1", "all") and phase1_output.exists() and not args.force:
        raise FileExistsError(f"{phase1_output} exists; use --force only to intentionally repeat phase 1.")
    if args.phase in ("2", "all") and final_output.exists() and not args.force:
        raise FileExistsError(f"{final_output} exists; use --force only to intentionally repeat the final benchmark.")
    rows = fixed_128()

    if args.phase in ("1", "all"):
        # Probe C-off first: if it is stable, we learn the key OOM fact within
        # five steps instead of waiting for a long C-on reference run.
        phase1 = [
            run_isolated_case("C-off", 1, 16, False, args.phase1_steps, args.phase1_warmup_steps),
            run_isolated_case("C-on", 1, 16, True, args.phase1_steps, args.phase1_warmup_steps),
        ]
        phase1_payload = {
            "purpose": "Short checkpointing/OOM probe",
            "isolation": "one fresh Python/CUDA process per configuration",
            "rows": 128,
            "optimizer_steps_per_case": args.phase1_steps,
            "warmup_steps": args.phase1_warmup_steps,
            "phase1": phase1,
            "selected_checkpointing": fastest(phase1)["gradient_checkpointing"],
        }
        phase1_output.write_text(json.dumps(phase1_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.phase == "1":
            print(json.dumps(phase1_payload, ensure_ascii=False, indent=2))
            return
    else:
        if not phase1_output.exists():
            raise FileNotFoundError("Phase 1 result is missing. Run --phase 1 first.")
        phase1_payload = json.loads(phase1_output.read_text(encoding="utf-8"))
        phase1 = phase1_payload["phase1"]

    checkpointing = bool(phase1_payload["selected_checkpointing"])
    phase2 = [
        run_isolated_case(name, micro, accum, checkpointing,
                          args.phase2_steps, args.phase2_warmup_steps)
        for name, micro, accum in (("B1", 1, 16), ("B2", 2, 8))
    ]
    winner = fastest(phase2)
    phase1_source = (
        "fresh isolated phase 1 from this run"
        if args.phase == "all"
        else "existing qlora_speed_phase1.json (not rerun)"
    )
    payload = {
        "purpose": "Engineering-only benchmark; do not compare these values with E0 inference.",
        "phase1_source": phase1_source,
        "phase2_isolation": "one fresh Python/CUDA process per configuration",
        "rows": 128,
        "phase1_optimizer_steps_per_case": phase1_payload["optimizer_steps_per_case"],
        "phase1_warmup_steps": phase1_payload["warmup_steps"],
        "phase2_optimizer_steps_per_case": args.phase2_steps,
        "phase2_warmup_steps": args.phase2_warmup_steps,
        "phase1": phase1,
        "phase2": phase2,
        "skipped_configs": [{
            "name": "B4", "micro_batch": 4, "gradient_accumulation": 4,
            "reason": "Skipped by user because its memory pressure was too high; B1/B2 were sufficient to lock the engineering configuration.",
        }],
        "locked_config": {
            "per_device_train_batch_size": winner["micro_batch"],
            "gradient_accumulation_steps": winner["gradient_accumulation"],
            "gradient_checkpointing": winner["gradient_checkpointing"],
            "effective_batch": 16,
        },
    }
    final_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "qlora_locked_training_config.json").write_text(
        json.dumps(payload["locked_config"], indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
