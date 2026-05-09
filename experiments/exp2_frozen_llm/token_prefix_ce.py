from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sphere_jepa.spherify import perturb_and_respherify, spherify


class PrefixProjector(nn.Module):
    def __init__(self, anchor_dim: int, hidden_size: int, prefix_tokens: int) -> None:
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(anchor_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, prefix_tokens * hidden_size),
        )

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        out = self.net(anchor)
        return out.view(anchor.shape[0], self.prefix_tokens, self.hidden_size)


def labels_from_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = ignore_index
    return labels


@torch.no_grad()
def evaluate_no_prefix(model, code_tokens: dict[str, torch.Tensor], batch_size: int, device: torch.device) -> float:
    losses = []
    n_items = code_tokens["input_ids"].shape[0]
    for start in range(0, n_items, batch_size):
        sl = slice(start, start + batch_size)
        input_ids = code_tokens["input_ids"][sl].to(device)
        attention_mask = code_tokens["attention_mask"][sl].to(device)
        labels = labels_from_input_ids(input_ids, attention_mask)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(float(outputs.loss.detach()) * input_ids.shape[0])
    return sum(losses) / n_items


def prefix_loss(model, projector, anchors, code_tokens, sl, *, sigma_max: float, device: torch.device) -> torch.Tensor:
    input_ids = code_tokens["input_ids"][sl].to(device)
    attention_mask = code_tokens["attention_mask"][sl].to(device)
    labels = labels_from_input_ids(input_ids, attention_mask)
    token_embeds = model.get_input_embeddings()(input_ids)
    prefix, _ = perturb_and_respherify(spherify(anchors[sl].to(device)), sigma_max=sigma_max)
    prefix_embeds = projector(prefix)
    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
    prefix_mask = torch.ones(input_ids.shape[0], prefix_embeds.shape[1], dtype=attention_mask.dtype, device=device)
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    prefix_labels = torch.full((input_ids.shape[0], prefix_embeds.shape[1]), -100, dtype=labels.dtype, device=device)
    full_labels = torch.cat([prefix_labels, labels], dim=1)
    return model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels).loss


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 token-anchor prefix CE experiment.")
    parser.add_argument("--cache", type=Path, default=Path("results/exp2_frozen_llm/cache.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/exp2_frozen_llm/e1_prefix.json"))
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-code-length", type=int, default=256)
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = torch.load(args.cache, map_location="cpu")
    anchors = cache["anchor_text"].float()
    codes = cache["code"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    code_tokens = tokenizer(
        codes,
        padding=True,
        truncation=True,
        max_length=args.max_code_length,
        return_tensors="pt",
    )
    no_prefix_ce = evaluate_no_prefix(model, code_tokens, args.batch_size, device)

    projector = PrefixProjector(anchors.shape[-1], model.config.hidden_size, args.prefix_tokens).to(device)
    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    dataset = TensorDataset(torch.arange(len(anchors)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        total = 0.0
        for (idx,) in loader:
            loss = prefix_loss(model, projector, anchors, code_tokens, idx, sigma_max=args.sigma_max, device=device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        print(f"epoch={epoch + 1} prefix_ce={total / len(anchors):.4f}")

    with torch.no_grad():
        total = 0.0
        for start in range(0, len(anchors), args.batch_size):
            idx = torch.arange(start, min(start + args.batch_size, len(anchors)))
            total += float(prefix_loss(model, projector, anchors, code_tokens, idx, sigma_max=0.0, device=device).detach()) * len(idx)
    prefix_ce = total / len(anchors)
    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "E1_no_prefix_ce": no_prefix_ce,
        "E1_prefix_ce": prefix_ce,
        "ce_improvement": no_prefix_ce - prefix_ce,
        "bits_saved": (no_prefix_ce - prefix_ce) / math.log(2.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
