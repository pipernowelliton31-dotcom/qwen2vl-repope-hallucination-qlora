"""Train E1/E2/E3/E4 or run the 128-row QLoRA smoke test; no evaluation leakage."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.qlora_common import AnswerOnlyCollator, CocoYesNoDataset, DATA_DIR, RESULTS_DIR, config, load_quantized_model, normalize_answer, optimizer_groups, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("smoke", "e1", "e2", "e3", "e4"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parent-architecture", choices=("e1", "e2"), help="Required for E3/E4 after selecting E1 versus E2 on dev.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); cfg = config()
    if args.experiment == "smoke": data_path, target, max_steps, merger = DATA_DIR / "speed128.jsonl", "e1", 8, False
    elif args.experiment in ("e1", "e2"): data_path, target, max_steps, merger = DATA_DIR / "coco_train_e1.jsonl", args.experiment, -1, args.experiment == "e2"
    elif args.experiment == "e3":
        if not args.parent_architecture: raise ValueError("E3 requires --parent-architecture selected from E1/E2 dev results.")
        data_path, target, max_steps, merger = DATA_DIR / "coco_train_e3.jsonl", args.parent_architecture, -1, args.parent_architecture == "e2"
    else:
        if not args.parent_architecture: raise ValueError("E4 requires --parent-architecture (the E3 architecture).")
        data_path, target, max_steps, merger = DATA_DIR / "coco_train_e3.jsonl", "e4", -1, args.parent_architecture == "e2"
    rows = read_jsonl(data_path); locked = json.loads((RESULTS_DIR / "qlora_locked_training_config.json").read_text(encoding="utf-8"))
    output = args.output_dir or RESULTS_DIR / "qlora_runs" / args.experiment
    if output.exists() and any(output.iterdir()) and not args.overwrite: raise FileExistsError(f"{output} exists; pass --overwrite only deliberately.")
    import torch
    from tqdm.auto import tqdm
    from transformers import Trainer, TrainerCallback, TrainingArguments

    model, processor, _ = load_quantized_model(target, bool(locked["gradient_checkpointing"]), cfg["max_visual_tokens"], include_merger=merger)
    collator = AnswerOnlyCollator(processor)
    # Static answer-only audit: inspect both labels and reject any prompt/image
    # content before spending a single optimizer step.
    probe_rows = [row for row in rows if row["label"] == "yes"][:3] + [row for row in rows if row["label"] == "no"][:2]
    supervised_examples = [collator.supervised_text(collator([row])) for row in probe_rows]
    for decoded in supervised_examples:
        if re.fullmatch(r"(yes|no)<\|im_end\|>\s*", decoded.lower()) is None:
            raise RuntimeError(f"Answer-only label check failed: {decoded!r}")
    experiment_name = args.experiment.upper()
    class TrainerWithGroups(Trainer):
        answer_correct = 0
        answer_total = 0
        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = optimizer_groups(self.model, target, merger)
            return self.optimizer
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Also retain a cheap, teacher-forced answer-token accuracy for the bar.

            The collator masks every user/image token, so the first supervised
            next-token position is the first token of ``yes`` or ``no``.  This is
            useful to diagnose optimisation while training, but is intentionally
            labelled *token* accuracy: model selection still uses the slower,
            greedy-generation Accuracy/F1/Recall/FPR on the leakage-free dev set.
            """
            outputs = model(**inputs)
            loss = outputs.loss
            labels = inputs["labels"]
            next_labels, next_logits = labels[:, 1:], outputs.logits[:, :-1, :]
            supervised = next_labels.ne(-100)
            usable = supervised.any(dim=1)
            if usable.any():
                first_position = supervised.to(torch.int64).argmax(dim=1)
                batch_index = torch.arange(labels.shape[0], device=labels.device)[usable]
                predicted = next_logits[batch_index, first_position[usable]].detach().argmax(dim=-1)
                expected = next_labels[batch_index, first_position[usable]]
                self.answer_correct += int((predicted == expected).sum().item())
                self.answer_total += int(expected.numel())
            return (loss, outputs) if return_outputs else loss
        def log(self, logs, start_time=None):
            # Aggregate every micro-batch since the previous log event. With
            # logging_steps=10 this is a stable 160-example training signal,
            # rather than the previous implementation's last micro-batch only.
            if self.answer_total:
                logs = dict(logs)
                logs["train_answer_accuracy"] = self.answer_correct / self.answer_total
                self.answer_correct = 0
                self.answer_total = 0
            return super().log(logs, start_time)
    class LiveLossCallback(TrainerCallback):
        """Show loss and aggregated teacher-forced binary training accuracy."""
        bar = None
        memory_allocated_gb = None
        def on_train_begin(self, args, state, control, **kwargs):
            self.memory_allocated_gb = []
            self.bar = tqdm(total=state.max_steps, desc=f"训练 {experiment_name}", unit="step", dynamic_ncols=True)
        def on_step_end(self, args, state, control, **kwargs):
            self.memory_allocated_gb.append(torch.cuda.memory_allocated() / 1024**3)
            if self.bar: self.bar.n = state.global_step; self.bar.refresh()
        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.bar and logs:
                postfix = {key: f"{float(value):.4f}" for key, value in logs.items() if key in {"loss", "grad_norm", "learning_rate"}}
                if "train_answer_accuracy" in logs:
                    postfix["train_acc"] = f"{float(logs['train_answer_accuracy']):.3f}"
                if self.memory_allocated_gb:
                    postfix["allocated_GB"] = f"{self.memory_allocated_gb[-1]:.2f}"
                if postfix: self.bar.set_postfix(postfix)
        def on_train_end(self, args, state, control, **kwargs):
            if self.bar: self.bar.n = state.global_step; self.bar.close()
    is_smoke = args.experiment == "smoke"
    training_args = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=int(locked["per_device_train_batch_size"]), gradient_accumulation_steps=int(locked["gradient_accumulation_steps"]),
        num_train_epochs=cfg["training"]["epochs"], max_steps=max_steps, learning_rate=cfg["lora"]["llm_lr"], warmup_ratio=cfg["training"]["warmup_ratio"], max_grad_norm=cfg["training"]["max_grad_norm"],
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), gradient_checkpointing=bool(locked["gradient_checkpointing"]),
        logging_strategy="steps", logging_steps=1 if is_smoke else cfg["training"]["logging_steps"],
        save_strategy="no" if is_smoke else "steps", save_steps=cfg["training"]["save_steps"],
        save_total_limit=5, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0 if is_smoke else 2, dataloader_pin_memory=True,
    )
    trainer = TrainerWithGroups(model=model, args=training_args, train_dataset=CocoYesNoDataset(rows), data_collator=collator)
    # Replace the default generic bar so loss and answer-token accuracy remain visible.
    from transformers.trainer_callback import ProgressCallback
    live_callback = LiveLossCallback()
    trainer.remove_callback(ProgressCallback); trainer.add_callback(live_callback)

    static_audit = None
    tracked_parameter = tracked_before = None
    if is_smoke:
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        all_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(parameter.numel() for _, parameter in trainable)
        print("\n[Smoke 静态检查] requires_grad=True 参数：", flush=True)
        for name, _ in trainable:
            print(name, flush=True)
        trainer.create_optimizer()
        optimizer_summary = [
            {
                "group": index,
                "tensor_count": len(group["params"]),
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
                "lr": float(group["lr"]),
            }
            for index, group in enumerate(trainer.optimizer.param_groups)
        ]
        tracked_name, tracked_parameter = next(
            ((name, parameter) for name, parameter in trainable if "lora_B" in name),
            trainable[0],
        )
        tracked_before = tracked_parameter.detach().float().cpu().clone()
        static_audit = {
            "supervised_examples": supervised_examples,
            "trainable_parameter_names": [name for name, _ in trainable],
            "trainable_parameter_count": trainable_parameter_count,
            "all_parameter_count": all_parameter_count,
            "trainable_ratio": trainable_parameter_count / all_parameter_count,
            "optimizer_groups": optimizer_summary,
            "tracked_lora_parameter": tracked_name,
        }
        print(json.dumps(static_audit, ensure_ascii=False, indent=2), flush=True)

    result = trainer.train()
    # Smoke uses save_strategy=no, so this is its only adapter save.
    trainer.save_model(); processor.save_pretrained(output)
    losses = [float(event["loss"]) for event in trainer.state.log_history if "loss" in event]
    gradient_norms = [float(event["grad_norm"]) for event in trainer.state.log_history if "grad_norm" in event]
    train_answer_accuracies = [
        float(event["train_answer_accuracy"])
        for event in trainer.state.log_history if "train_answer_accuracy" in event
    ]
    smoke_dynamic_check = smoke_reload_check = None
    if is_smoke:
        if trainer.state.global_step != 8:
            raise RuntimeError(f"Smoke must finish exactly 8 optimizer steps, got {trainer.state.global_step}.")
        if not losses or any(not math.isfinite(value) for value in losses):
            raise RuntimeError("Smoke loss is missing, NaN, or infinite.")
        if not gradient_norms or any(not math.isfinite(value) for value in gradient_norms) or max(gradient_norms) <= 0:
            raise RuntimeError(f"Smoke gradient norm is invalid or always zero: {gradient_norms}")
        parameter_change = float(
            (tracked_parameter.detach().float().cpu() - tracked_before).abs().max().item()
        )
        if parameter_change <= 0:
            raise RuntimeError("Tracked LoRA parameter did not change after optimizer steps.")
        memory = list(live_callback.memory_allocated_gb or [])
        # Variable-resolution images can move allocation up/down. Flag only a
        # sustained monotonic rise larger than 0.5 GiB after the first step.
        steady_memory = memory[1:]
        monotonic_growth = len(steady_memory) >= 3 and all(
            later > earlier + 0.01 for earlier, later in zip(steady_memory, steady_memory[1:])
        )
        memory_growth = steady_memory[-1] - steady_memory[0] if len(steady_memory) >= 2 else 0.0
        leak_suspected = monotonic_growth and memory_growth > 0.5
        if leak_suspected:
            raise RuntimeError(f"Smoke detected sustained GPU-memory growth: {memory}")
        smoke_dynamic_check = {
            "optimizer_steps": trainer.state.global_step,
            "samples_processed": trainer.state.global_step * int(locked["effective_batch"]),
            "logged_losses": losses,
            "all_losses_finite": True,
            "gradient_norms": gradient_norms,
            "nonzero_gradient_observed": max(gradient_norms) > 0,
            "train_answer_accuracies": train_answer_accuracies,
            "tracked_lora_max_abs_change": parameter_change,
            "optimizer_updated_lora": True,
            "memory_allocated_gb_after_each_step": memory,
            "memory_growth_gb_after_first_step": memory_growth,
            "memory_leak_suspected": leak_suspected,
        }

        adapter_config = output / "adapter_config.json"
        adapter_weights = output / "adapter_model.safetensors"
        if not adapter_config.exists() or not adapter_weights.exists():
            raise RuntimeError(
                f"Smoke adapter save incomplete: config={adapter_config.exists()}, "
                f"weights={adapter_weights.exists()}"
            )

        from PIL import Image
        def fixed_inference(inference_model, inference_processor) -> dict[str, str]:
            inference_model.eval()
            with Image.open(rows[0]["image_path"]) as source:
                image = source.convert("RGB")
            prompt = inference_processor.apply_chat_template(
                AnswerOnlyCollator._conversation(rows[0]["question"]),
                tokenize=False, add_generation_prompt=True,
            )
            inputs = inference_processor(text=[prompt], images=[image], return_tensors="pt").to(
                inference_model.get_input_embeddings().weight.device
            )
            with torch.inference_mode():
                ids = inference_model.generate(**inputs, max_new_tokens=4, do_sample=False)
            raw = inference_processor.batch_decode(
                ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            return {"raw": raw, "parsed": normalize_answer(raw)}

        before_reload_prediction = fixed_inference(model, processor)
        if before_reload_prediction["parsed"] not in {"yes", "no"}:
            raise RuntimeError(f"Pre-reload smoke answer is not parseable: {before_reload_prediction}")
        del trainer, model
        gc.collect(); torch.cuda.empty_cache()
        from peft import PeftModel
        base, reload_processor, _ = load_quantized_model(
            "e1", bool(locked["gradient_checkpointing"]), cfg["max_visual_tokens"],
            attach_lora=False,
        )
        reloaded = PeftModel.from_pretrained(base, output)
        after_reload_prediction = fixed_inference(reloaded, reload_processor)
        if after_reload_prediction["parsed"] not in {"yes", "no"}:
            raise RuntimeError(f"Reloaded smoke answer is not parseable: {after_reload_prediction}")
        if before_reload_prediction != after_reload_prediction:
            raise RuntimeError(
                "Greedy prediction changed after adapter reload: "
                f"before={before_reload_prediction}, after={after_reload_prediction}"
            )
        smoke_reload_check = {
            "adapter_config_exists": True,
            "adapter_safetensors_exists": True,
            "before_reload_prediction": before_reload_prediction,
            "after_reload_prediction": after_reload_prediction,
            "greedy_prediction_identical": True,
            "reload_and_inference_succeeded": True,
        }
        (output / "smoke_reload_ok.txt").write_text(
            json.dumps(smoke_reload_check, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "experiment": args.experiment,
        "adapter_target": target,
        "parent_architecture": args.parent_architecture,
        "merger_lora": merger,
        "data": str(data_path),
        "supervised_examples": supervised_examples,
        "locked_training_config": locked,
        "train_metrics": result.metrics,
        "train_answer_accuracy_history": train_answer_accuracies,
        "smoke_static_audit": static_audit,
        "smoke_dynamic_check": smoke_dynamic_check,
        "smoke_reload_check": smoke_reload_check,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
