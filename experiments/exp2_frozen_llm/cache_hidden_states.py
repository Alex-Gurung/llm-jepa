from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def extract_pair(record: dict) -> tuple[str, str]:
    if "text" in record and "code" in record:
        return str(record["text"]), str(record["code"])
    if "messages" in record:
        messages = record["messages"]
        users = [m["content"] for m in messages if m.get("role") == "user"]
        assistants = [m["content"] for m in messages if m.get("role") == "assistant"]
        if users and assistants:
            return str(users[-1]), str(assistants[0])
    raise ValueError("record must contain text/code or chat messages with user and assistant turns")


@torch.no_grad()
def pool_batch(model, tokenizer, strings: list[str], *, max_length: int, pooling: str, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        strings,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    outputs = model(**tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    mask = tokens["attention_mask"]
    if pooling == "last":
        idx = mask.sum(dim=1).clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=device), idx].float().cpu()
    if pooling == "mean":
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1).div(denom).float().cpu()
    raise ValueError(f"unknown pooling: {pooling}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen LLM pooled hidden states for NL-code pairs.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with text/code or chat messages.")
    parser.add_argument("--output", type=Path, default=Path("results/exp2_frozen_llm/cache.pt"))
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = []
    with args.input.open() as f:
        for line in f:
            if line.strip():
                pairs.append(extract_pair(json.loads(line)))
            if args.limit is not None and len(pairs) >= args.limit:
                break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    text_states = []
    code_states = []
    texts = [p[0] for p in pairs]
    codes = [p[1] for p in pairs]
    for start in range(0, len(pairs), args.batch_size):
        end = start + args.batch_size
        text_states.append(pool_batch(model, tokenizer, texts[start:end], max_length=args.max_length, pooling=args.pooling, device=device))
        code_states.append(pool_batch(model, tokenizer, codes[start:end], max_length=args.max_length, pooling=args.pooling, device=device))
        print(f"cached {min(end, len(pairs))}/{len(pairs)}")

    payload = {
        "text": texts,
        "code": codes,
        "anchor_text": torch.cat(text_states),
        "anchor_code": torch.cat(code_states),
        "config": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
