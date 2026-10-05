"""Supervised fine-tuning with LoRA / QLoRA.

Built on plain `transformers` + `peft` (not a trainer wrapper) so every step is
visible: chat-template tokenization, assistant-only loss masking, gradient accumulation,
checkpointing, and eval loss. Heavy imports are lazy: `--dry-run` validates data and
produces a GPU memory plan without torch installed.

Output layout (one run = one immutable directory):
    .forge/artifacts/runs/<run_id>/
        manifest.json       config, dataset hash, memory plan, metrics, loss curve
        adapter/            LoRA weights, loadable by vLLM via --lora-modules
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field

from forge.config import settings
from forge.training.dataset import prepare

# Approximate shapes for memory planning when the HF config isn't downloaded.
KNOWN_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": dict(params=0.49e9, hidden=896, inter=4864, kv=128, layers=24, vocab=151936),
    "Qwen/Qwen2.5-1.5B-Instruct": dict(params=1.54e9, hidden=1536, inter=8960, kv=256, layers=28, vocab=151936),
    "Qwen/Qwen2.5-3B-Instruct": dict(params=3.09e9, hidden=2048, inter=11008, kv=256, layers=36, vocab=151936),
    "meta-llama/Llama-3.2-1B-Instruct": dict(params=1.24e9, hidden=2048, inter=8192, kv=512, layers=16,
                                             vocab=128256),
    "meta-llama/Llama-3.2-3B-Instruct": dict(params=3.21e9, hidden=3072, inter=8192, kv=1024, layers=28,
                                             vocab=128256),
    "HuggingFaceTB/SmolLM2-135M-Instruct": dict(params=0.135e9, hidden=576, inter=1536, kv=192, layers=30,
                                                vocab=49152),
}


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = "data/sample/train.jsonl"
    output_name: str = "qwen-farm"
    qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"])
    max_seq_len: int = 1024
    epochs: float = 2.0
    lr: float = 2e-4
    micro_batch_size: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.05
    gradient_checkpointing: bool = True
    seed: int = 42
    eval_every: int = 50
    save_every: int = 200


def memory_plan(cfg: TrainConfig) -> dict:
    """Back-of-envelope GPU memory budget. Worth understanding before renting a GPU.

    weights      : params x bytes/param (bf16 = 2, NF4 ~ 0.55 incl. quant constants)
    lora + optim : trainable params x (2 weight + 2 grad + 8 Adam m/v + 4 fp32 master) bytes
    activations  : ~ micro_batch x seq x hidden x layers x k bytes
                   (k~34 without checkpointing, ~2 with: only layer inputs are kept)
    logits       : micro_batch x seq x vocab x 4 bytes (fp32 upcast for the loss), often the surprise
    """
    shape = KNOWN_MODELS.get(cfg.base_model, KNOWN_MODELS["Qwen/Qwen2.5-1.5B-Instruct"])
    gb = 1024**3
    weights = shape["params"] * (0.55 if cfg.qlora else 2) / gb
    h, i, kv = shape["hidden"], shape["inter"], shape["kv"]
    # LoRA on W (in x out) adds A (in x r) + B (r x out) = r * (in + out) params
    dims = {"q_proj": (h, h), "o_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv),
            "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}
    per_layer = sum(cfg.lora_r * sum(dims.get(t, (h, h))) for t in cfg.target_modules)
    trainable = per_layer * shape["layers"]
    optim = trainable * 16 / gb
    k = 2 if cfg.gradient_checkpointing else 34
    acts = cfg.micro_batch_size * cfg.max_seq_len * shape["hidden"] * shape["layers"] * k / gb
    logits = cfg.micro_batch_size * cfg.max_seq_len * shape["vocab"] * 4 / gb
    overhead = 1.0  # CUDA context, allocator fragmentation
    total = weights + optim + acts + logits + overhead
    return {
        "base_params_b": round(shape["params"] / 1e9, 2), "trainable_params_m": round(trainable / 1e6, 2),
        "trainable_pct": round(100 * trainable / shape["params"], 3),
        "weights_gb": round(weights, 2), "lora_and_optimizer_gb": round(optim, 2),
        "activations_gb": round(acts, 2), "logits_gb": round(logits, 2), "overhead_gb": overhead,
        "total_gb": round(total, 2), "effective_batch": cfg.micro_batch_size * cfg.grad_accum,
        "fits": {g: total < cap * 0.92 for g, cap in (("T4-16GB", 16), ("L4-24GB", 24), ("A100-40GB", 40),
                                                       ("A100-80GB", 80))},
    }


def train(cfg: TrainConfig, dry_run: bool = False) -> dict:
    ds = prepare(cfg.dataset, seed=cfg.seed)
    run_id = f"{cfg.output_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = settings().artifacts / "runs" / run_id
    manifest = {"run_id": run_id, "config": asdict(cfg), "dataset": {"path": cfg.dataset, "sha256": ds.sha256,
                **ds.stats}, "memory_plan": memory_plan(cfg), "status": "planned", "started_at": time.time()}
    if dry_run:
        return manifest

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs: dict = {"dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    if cfg.qlora:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)
    if cfg.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in cfg.target_modules if t in present] or ["c_attn"]
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    if not cfg.qlora:
        model.to(device)

    def render(msgs: list[dict], gen_prompt: bool = False) -> str:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt) if msgs else ""

    def encode(example: dict) -> dict:
        """Tokenize with the chat template; loss only on assistant content (labels = -100 elsewhere).

        Each turn's text is the diff between the rendered conversation before and after it,
        so template boilerplate (system preamble, role headers) is always masked.
        """
        msgs, ids, labels, so_far = example["messages"], [], [], ""

        def add(text: str, train_on: bool) -> None:
            seg = tok(text, add_special_tokens=False)["input_ids"]
            ids.extend(seg)
            labels.extend(seg if train_on else [-100] * len(seg))

        for i, msg in enumerate(msgs):
            if msg["role"] == "assistant":
                header = render(msgs[:i], gen_prompt=True)
                add(header[len(so_far):], False)
                so_far = header
            full = render(msgs[: i + 1])
            if not full.startswith(so_far):
                raise ValueError(f"chat template for {cfg.base_model} is not prefix-stable; cannot mask turns")
            add(full[len(so_far):], msg["role"] == "assistant")
            so_far = full
        return {"input_ids": ids[: cfg.max_seq_len], "labels": labels[: cfg.max_seq_len]}

    train_enc = [encode(e) for e in ds.train]
    val_enc = [encode(e) for e in ds.val]

    def batches(data, shuffle: bool):
        idx = list(range(len(data)))
        if shuffle:
            torch.manual_seed(cfg.seed + len(history))
            idx = torch.randperm(len(idx)).tolist()
        for s in range(0, len(idx), cfg.micro_batch_size):
            chunk = [data[j] for j in idx[s: s + cfg.micro_batch_size]]
            n = max(len(c["input_ids"]) for c in chunk)
            pad = tok.pad_token_id
            yield {
                "input_ids": torch.tensor([c["input_ids"] + [pad] * (n - len(c["input_ids"])) for c in chunk]),
                "attention_mask": torch.tensor([[1] * len(c["input_ids"]) + [0] * (n - len(c["input_ids"]))
                                                for c in chunk]),
                "labels": torch.tensor([c["labels"] + [-100] * (n - len(c["labels"])) for c in chunk]),
            }

    @torch.no_grad()
    def eval_loss() -> float:
        model.eval()
        losses = [model(**{k: v.to(model.device) for k, v in b.items()}).loss.item() for b in batches(val_enc, False)]
        model.train()
        return sum(losses) / len(losses)

    history: list[dict] = []
    steps_per_epoch = math.ceil(len(train_enc) / (cfg.micro_batch_size * cfg.grad_accum))
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * cfg.warmup_ratio), total_steps)
    model.train()
    step, micro, t0 = 0, 0, time.time()
    best_val = float("inf")
    while step < total_steps:
        for b in batches(train_enc, True):
            out = model(**{k: v.to(model.device) for k, v in b.items()})
            (out.loss / cfg.grad_accum).backward()
            micro += 1
            if micro % cfg.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": round(out.loss.item(), 4), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            if step % cfg.eval_every == 0 or step == total_steps:
                rec["val_loss"] = round(eval_loss(), 4)
                best_val = min(best_val, rec["val_loss"])
            if step % cfg.save_every == 0:
                model.save_pretrained(run_dir / f"checkpoint-{step}")
            history.append(rec)
            print(json.dumps(rec), flush=True)
            if step >= total_steps:
                break

    model.save_pretrained(run_dir / "adapter")
    tok.save_pretrained(run_dir / "adapter")
    manifest.update(status="completed", finished_at=time.time(), steps=step, loss_curve=history,
                    metrics={"final_train_loss": history[-1]["loss"], "best_val_loss": best_val,
                             "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                             if device == "cuda" else None},
                    adapter_uri=str(run_dir / "adapter"))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
