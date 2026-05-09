from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sphere_jepa.heads import AnchorPreprocessor, MLP, make_ema_copy, update_ema
from sphere_jepa.losses import (
    clean_jepa_alignment,
    cross_view_cap_anchor_loss,
    info_nce_loss,
    noisy_cap_anchor_loss,
    sigreg_loss,
    vicreg_loss,
)
from sphere_jepa.metrics import (
    anchor_geometry_diagnostics,
    cosine_pair_stats,
    paired_vs_unpaired_cosine,
    rankme,
    retrieval_at_k,
    uniformity_loss,
)
from sphere_jepa.spherify import perturb_and_respherify, spherify


VARIANTS = [
    "A",
    "B",
    "C",
    "C_detach",
    "C_ema",
    "D_own_0",
    "D_own_1",
    "D_cross_0",
    "D_cross_1",
    "D_cross_sym_0",
    "D_cross_sym_1",
    "F",
    "G",
    "H",
]


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def split_indices(n: int, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * train_fraction)
    return torch.tensor(idx[:cut]), torch.tensor(idx[cut:])


def fit_preprocessors(anchor_text: torch.Tensor, anchor_code: torch.Tensor, mode: str) -> tuple[AnchorPreprocessor, AnchorPreprocessor]:
    text_proc = AnchorPreprocessor(mode).fit(anchor_text)
    code_proc = AnchorPreprocessor(mode).fit(anchor_code)
    return text_proc, code_proc


def train_variant(args: argparse.Namespace, variant: str, train_data: TensorDataset, eval_tensors: tuple[torch.Tensor, ...]) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    input_dim = train_data.tensors[0].shape[-1]
    anchor_dim = train_data.tensors[2].shape[-1]
    g_text = MLP(input_dim, args.emb_dim, hidden_dim=args.hidden_dim, num_layers=3).to(device)
    g_code = MLP(input_dim, args.emb_dim, hidden_dim=args.hidden_dim, num_layers=3).to(device)
    predictor = MLP(args.emb_dim, args.emb_dim, hidden_dim=args.hidden_dim).to(device)
    cotrain_cap = MLP(args.emb_dim, args.emb_dim, hidden_dim=args.hidden_dim).to(device)
    cap_text = MLP(args.emb_dim, anchor_dim, hidden_dim=args.hidden_dim).to(device)
    cap_code = MLP(args.emb_dim, anchor_dim, hidden_dim=args.hidden_dim).to(device)
    cap_cross_t2c = MLP(args.emb_dim, anchor_dim, hidden_dim=args.hidden_dim).to(device)
    cap_cross_c2t = MLP(args.emb_dim, anchor_dim, hidden_dim=args.hidden_dim).to(device)
    ema_g_code = make_ema_copy(g_code).to(device) if variant == "C_ema" else None

    modules = [g_text, g_code]
    if variant not in {"F"}:
        modules.append(predictor)
    if variant.startswith("C"):
        modules.append(cotrain_cap)
    if variant.startswith("D_own"):
        modules.extend([cap_text, cap_code])
    if variant.startswith("D_cross"):
        modules.append(cap_cross_t2c)
        if variant.startswith("D_cross_sym"):
            modules.append(cap_cross_c2t)
    params = [param for module in modules for param in module.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=len(train_data) >= args.batch_size)

    for epoch in range(args.epochs):
        for text_x, code_x, anchor_text, anchor_code in loader:
            text_x = text_x.to(device)
            code_x = code_x.to(device)
            anchor_text = anchor_text.to(device)
            anchor_code = anchor_code.to(device)
            z_text = g_text(text_x)
            z_code = g_code(code_x)
            if variant == "A":
                loss = F.mse_loss(predictor(z_text), z_code.detach())
            elif variant == "B":
                loss = clean_jepa_alignment(z_text, z_code, predictor, metric="cosine")
            elif variant == "C":
                target = spherify(z_code)
                noisy, _ = perturb_and_respherify(spherify(z_text), sigma_max=args.sigma_max)
                loss = clean_jepa_alignment(z_text, z_code, predictor, metric="cosine")
                loss = loss + F.mse_loss(cotrain_cap(noisy), target)
            elif variant == "C_detach":
                target = spherify(z_code).detach()
                noisy, _ = perturb_and_respherify(spherify(z_text), sigma_max=args.sigma_max)
                loss = clean_jepa_alignment(z_text, z_code, predictor, metric="cosine")
                loss = loss + F.mse_loss(cotrain_cap(noisy), target)
            elif variant == "C_ema":
                assert ema_g_code is not None
                with torch.no_grad():
                    target = spherify(ema_g_code(code_x))
                noisy, _ = perturb_and_respherify(spherify(z_text), sigma_max=args.sigma_max)
                loss = clean_jepa_alignment(z_text, target, predictor, metric="cosine", detach_target=True)
                loss = loss + F.mse_loss(cotrain_cap(noisy), target)
            elif variant in {"D_own_0", "D_own_1"}:
                sigma = 0.0 if variant.endswith("_0") else args.sigma_max
                align = clean_jepa_alignment(z_text, z_code, predictor, metric="cosine")
                cap_loss = noisy_cap_anchor_loss(z_text, anchor_text, cap_text, sigma_max=sigma)
                cap_loss = cap_loss + noisy_cap_anchor_loss(z_code, anchor_code, cap_code, sigma_max=sigma)
                loss = args.lambda_jepa * align + args.lambda_cap * cap_loss
            elif variant in {"D_cross_0", "D_cross_1"}:
                sigma = 0.0 if variant.endswith("_0") else args.sigma_max
                align = clean_jepa_alignment(z_text, z_code, predictor, metric="cosine")
                cap_loss = cross_view_cap_anchor_loss(z_text, anchor_code, cap_cross_t2c, sigma_max=sigma)
                loss = args.lambda_jepa * align + args.lambda_cap * cap_loss
            elif variant in {"D_cross_sym_0", "D_cross_sym_1"}:
                sigma = 0.0 if variant.endswith("_0") else args.sigma_max
                align = clean_jepa_alignment(z_text, z_code, predictor, metric="cosine")
                cap_loss = cross_view_cap_anchor_loss(z_text, anchor_code, cap_cross_t2c, sigma_max=sigma)
                cap_loss = cap_loss + cross_view_cap_anchor_loss(z_code, anchor_text, cap_cross_c2t, sigma_max=sigma)
                loss = args.lambda_jepa * align + args.lambda_cap * 0.5 * cap_loss
            elif variant == "F":
                loss = info_nce_loss(z_text, z_code)
            elif variant == "G":
                loss = vicreg_loss(z_text, z_code)
            elif variant == "H":
                align = clean_jepa_alignment(z_text, z_code, predictor, metric="cosine")
                loss = align + args.sigreg_weight * (sigreg_loss(z_text) + sigreg_loss(z_code))
            else:
                raise ValueError(f"unknown variant: {variant}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema_g_code is not None:
                update_ema(ema_g_code, g_code, args.ema_momentum)
        print(f"variant={variant} epoch={epoch + 1} loss={float(loss.detach()):.4f}")

    text_eval, code_eval, anchor_text_eval, anchor_code_eval = [t.to(device) for t in eval_tensors]
    with torch.no_grad():
        z_text = g_text(text_eval)
        z_code = g_code(code_eval)
        pred_code = predictor(spherify(z_text))
        records = {
            "variant": variant,
            "text_rankme": rankme(z_text.cpu()),
            "code_rankme": rankme(z_code.cpu()),
            "text_uniformity": uniformity_loss(z_text.cpu()),
            "code_uniformity": uniformity_loss(z_code.cpu()),
            "text_cosine": cosine_pair_stats(z_text.cpu()),
            "code_cosine": cosine_pair_stats(z_code.cpu()),
            "paired_vs_unpaired": paired_vs_unpaired_cosine(z_text.cpu(), z_code.cpu()),
            "retrieval_embedding": retrieval_at_k(z_text.cpu(), z_code.cpu(), ks=(1, 10)),
            "retrieval_predicted_code_embedding": retrieval_at_k(pred_code.cpu(), z_code.cpu(), ks=(1, 10)),
        }
        if variant.startswith("D_cross"):
            projected = cap_cross_t2c(spherify(z_text))
            records["retrieval_text_cap_to_code_anchor"] = retrieval_at_k(projected.cpu(), anchor_code_eval.cpu(), ks=(1, 10))
        if variant.startswith("D_own"):
            text_cap = cap_text(spherify(z_text))
            code_cap = cap_code(spherify(z_code))
            records["retrieval_text_cap_to_text_anchor"] = retrieval_at_k(text_cap.cpu(), anchor_text_eval.cpu(), ks=(1, 10))
            records["retrieval_code_cap_to_code_anchor"] = retrieval_at_k(code_cap.cpu(), anchor_code_eval.cpu(), ks=(1, 10))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train projection heads over cached frozen LLM states.")
    parser.add_argument("--cache", type=Path, default=Path("results/exp2_frozen_llm/cache.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp2_frozen_llm"))
    parser.add_argument("--anchor-preprocess", choices=["raw", "norm", "white", "sphere"], default="sphere")
    parser.add_argument("--variants", nargs="*", default=["D_own_0", "D_own_1", "C", "F", "G", "H"])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--lambda-jepa", type=float, default=1.0)
    parser.add_argument("--lambda-cap", type=float, default=1.0)
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    payload = torch.load(args.cache, map_location="cpu")
    raw_text = payload["anchor_text"].float()
    raw_code = payload["anchor_code"].float()
    train_idx, eval_idx = split_indices(len(raw_text), args.train_fraction, args.seed)
    text_proc, code_proc = fit_preprocessors(raw_text[train_idx], raw_code[train_idx], args.anchor_preprocess)
    anchor_text = text_proc.transform(raw_text)
    anchor_code = code_proc.transform(raw_code)
    train_data = TensorDataset(raw_text[train_idx], raw_code[train_idx], anchor_text[train_idx], anchor_code[train_idx])
    eval_tensors = (raw_text[eval_idx], raw_code[eval_idx], anchor_text[eval_idx], anchor_code[eval_idx])

    diagnostics = {
        "raw_text": anchor_geometry_diagnostics(raw_text),
        "raw_code": anchor_geometry_diagnostics(raw_code),
        f"{args.anchor_preprocess}_text": anchor_geometry_diagnostics(anchor_text),
        f"{args.anchor_preprocess}_code": anchor_geometry_diagnostics(anchor_code),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [train_variant(args, variant, train_data, eval_tensors) for variant in args.variants]
    out_payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cache_config": jsonable(payload.get("config", {})),
        "anchor_geometry": diagnostics,
        "records": records,
        "headline_contrasts": {
            "noise": "D_own_1 vs D_own_0.",
            "anchor": "D_own_1 vs C/C_detach/C_ema.",
            "preprocessing": "Run once per anchor_preprocess mode and compare D_own_sphere to D_own_raw.",
        },
    }
    out = args.output_dir / f"summary_{args.anchor_preprocess}.json"
    out.write_text(json.dumps(out_payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
