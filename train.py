import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

import torch

from model import GPT, GPTConfig
from tokenizer import JokeTokenizer
import platform

CPU_DEFAULTS = {
    "bpe_vocab_size": 2000,
    "bpe_min_pair_freq": 2,
    "bpe_train_chars": 3_000_000,
    "block_size": 128,
    "batch_size": 32,
    "max_iters": 5000,
    "eval_interval": 100,
    "eval_iters": 100,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "dropout": 0.2,
}

GPU_DEFAULTS = {
    "bpe_vocab_size": 2000,
    "bpe_min_pair_freq": 2,
    "bpe_train_chars": 3_000_000,
    "block_size": 128,
    "batch_size": 64,
    "max_iters": 10000,
    "eval_interval": 500,
    "eval_iters": 100,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.2,
}

def main():
    args = parse_train_args()

    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_info = get_runtime_info(args.device)
    log(f"Runtime infos: {runtime_info}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    text = data_path.read_text(encoding="utf-8")

    print("Training BPE tokenizer with tokenizers library...")

    if args.bpe_train_chars > 0:
        tokenizer_train_path = out_dir / "_tokenizer_train_sample.txt"
        tokenizer_train_text = text[: args.bpe_train_chars]
        tokenizer_train_path.write_text(tokenizer_train_text, encoding="utf-8")
    else:
        tokenizer_train_path = data_path

    tokenizer = JokeTokenizer.train_from_file(
        tokenizer_train_path,
        vocab_size=args.bpe_vocab_size,
        min_frequency=args.bpe_min_pair_freq,
        show_progress=True,
    )

    tokenizer.save(out_dir / "tokenizer.json")

    if tokenizer_train_path.name == "_tokenizer_train_sample.txt":
        tokenizer_train_path.unlink(missing_ok=True)

    print("Encoding dataset with BPE tokenizer...")

    log(f"Saved tokenizer at: {out_dir}\\tokenizer.json")
    log("Encoding dataset with BPE tokenizer...")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    if len(ids) < args.block_size * 20:
        raise ValueError(
            f"Tokenized dataset too small: {len(ids)} tokens"
        )

    shutil.copyfile(data_path, out_dir / "data_clean.txt")
    log(f"Saved data at: {out_dir}\data_clean.txt")    
    split_index = int(0.9 * len(ids))

    train_data = ids[:split_index]
    val_data = ids[split_index:]

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    model = GPT(config).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_step = None
    log(f"Device: {args.device}")
    log(f"BPE vocab size: {tokenizer.vocab_size}")
    log(f"Total tokens: {len(ids)}")
    log(f"Train tokens: {len(train_data)}")
    log(f"Val tokens: {len(val_data)}")
    log(f"Block size: {args.block_size}")

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=args.device,
                eval_iters=args.eval_iters,
            )

            log(
                f"step {step}/{args.max_iters}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_step = step
                save_checkpoint(
                    out_dir=out_dir,
                    model=model,
                    config=config,
                    args=args,
                    best_val_loss=best_val_loss,
                )
                #log(f"Saved checkpoint model+config to: {out_dir}")

        xb, yb = get_batch(
            data=train_data,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=args.device,
        )

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

    log("Training done.")
    save_config_to_file(args, data_path, out_dir, runtime_info, text, tokenizer, ids, train_data, val_data, config, best_val_loss, best_step)    
    log(f"Saved data, tokenizer, config, model to: {out_dir}")

def get_batch(data, batch_size, block_size, device):
    max_start = len(data) - block_size - 1

    if max_start <= 0:
        raise ValueError("Dataset is too small for this block_size")

    ix = torch.randint(max_start, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    device,
    eval_iters,
):
    model.eval()

    result = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses[i] = loss.item()

        result[split] = losses.mean().item()

    model.train()

    return result


def save_checkpoint(
    out_dir: Path,
    model: GPT,
    config: GPTConfig,
    args,
    best_val_loss: float,
):
    torch.save(model.state_dict(), out_dir / "model.pt")

    payload = {
        "model": asdict(config),
        "training": vars(args),
        "best_val_loss": best_val_loss,
    }

    (out_dir / "config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

def save_config_to_file(args, data_path, out_dir, runtime_info, text, tokenizer, ids, train_data, val_data, config, best_val_loss, best_step):
    dataset_stats = {
        "raw_chars": len(text),
        "total_tokens": len(ids),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "train_ratio": 0.9,
    }
    final_config = build_config_payload(
        args=args,
        model_config=config,
        tokenizer=tokenizer,
        data_path=data_path,
        out_dir=out_dir,
        dataset_stats=dataset_stats,
        runtime_info=runtime_info,
        best_val_loss=best_val_loss,
        best_step=best_step,
        status="finished",
    )

    save_config(out_dir, final_config)

def build_config_payload(
    args,
    model_config: GPTConfig,
    tokenizer: JokeTokenizer,
    data_path: Path,
    out_dir: Path,
    dataset_stats: dict,
    runtime_info: dict,
    best_val_loss: float | None = None,
    best_step: int | None = None,
    status: str = "initialized",
) -> dict:
    return {
        "status": status,
        "paths": {
            "data": str(data_path),
            "out_dir": str(out_dir),
            "model_file": str(out_dir / "model.pt"),
            "tokenizer_file": str(out_dir / "tokenizer.json"),
            "config_file": str(out_dir / "config.json"),
            "copied_clean_data_file": str(out_dir / "data_clean.txt"),
        },
        "model": asdict(model_config),
        "tokenizer": {
            "type": "huggingface_tokenizers_byte_level_bpe",
            "vocab_size": tokenizer.vocab_size,
            "requested_vocab_size": args.bpe_vocab_size,
            "min_pair_freq": args.bpe_min_pair_freq,
            "train_chars": args.bpe_train_chars,
            "special_tokens": tokenizer.SPECIAL_TOKENS,
        },
        "training": {
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
            "device": args.device,
        },
        "dataset": dataset_stats,
        "checkpoint": {
            "best_val_loss": best_val_loss,
            "best_step": best_step,
        },
        "runtime": runtime_info,
    }

def save_config(out_dir: Path, payload: dict):
    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def get_runtime_info(device: str) -> dict:
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()

    return info

def normalize_device(device: str) -> str:
    device = device.lower()

    if device == "auto":
        return get_device()

    if device == "gpu":
        return "cuda"

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    return device


def apply_device_defaults(args):
    defaults = GPU_DEFAULTS if args.device == "cuda" else CPU_DEFAULTS

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data_clean.txt")
    parser.add_argument("--out_dir", default="models/jokes_bpe")

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "gpu"],
        help="auto = cuda if available else cpu; gpu = cuda",
    )

    parser.add_argument("--bpe_vocab_size", type=int, default=None)
    parser.add_argument("--bpe_min_pair_freq", type=int, default=None)
    parser.add_argument("--bpe_train_chars", type=int, default=None)

    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--seed", type=int, default=1337)

    return parser

def parse_train_args():
    parser = build_arg_parser()
    args = parser.parse_args()

    args.device = normalize_device(args.device)
    args = apply_device_defaults(args)

    return args

def log(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")

def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    return device

if __name__ == "__main__":
    main()