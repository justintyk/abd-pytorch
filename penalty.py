"""
Apply a smooth penalty to outlier coordinates to guide activations back within boundary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

# -----------------------------------------------------------------------------

def apply_penalty(
    x: Tensor,
    mu: float,
    alpha: float,
    beta: float,
    k: float,
) -> Tensor:
    """
    Penalty function: x' = alpha * tanh(beta * (x - mu)) + mu.

    Applied per element along the hidden dim, but only to the top-k fraction of
    coordinates per token row (ranked by |x - mu|).    
    """
    if k <= 0.0:
        return x
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be non-negative")
    if k > 1.0:
        raise ValueError("k must lie in (0, 1]")

    delta = x - mu
    penalized = alpha * torch.tanh(beta * delta) + mu

    hidden_dim = x.shape[-1]
    n_pen = max(1, int(round(k * hidden_dim)))
    if n_pen >= hidden_dim:
        return penalized

    abs_delta = delta.abs()
    _, top_idx = torch.topk(abs_delta, k=n_pen, dim=-1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, top_idx, True)
    return torch.where(mask, penalized, x)

# -----------------------------------------------------------------------------

def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer decoder layer ModuleList on common HF models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Vicuna, Qwen2, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2, GPT-Neo
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX
    raise AttributeError(
        "Could not locate decoder layers on model. "
        "Expected model.model.layers, model.transformer.h, or model.gpt_neox.layers."
    )


def _layer_output_hidden(output) -> Tensor:
    """Decoder layers return either a Tensor or a tuple (hidden, ...)."""
    if isinstance(output, Tensor):
        return output
    return output[0]


@torch.no_grad()
def compute_layer_means(
    model: nn.Module,
    tokenizer,
    prompts: Iterable[str],
    device: str | torch.device = "cpu",
    max_length: int = 512,
) -> dict[int, tuple[float, float]]:
    """Compute (mu_D^l, sigma_D^l) for every decoder layer.

    Paper protocol (Appendix E.3): ~400 non-overlapping AdvBench prompts.
    mu and sigma are scalars pooled across all hidden dims and all samples,
    using the last-token activation at each layer.
    """
    layers = _get_decoder_layers(model)
    L = len(layers)
    sums = [0.0] * L
    sum_sq = [0.0] * L
    counts = [0] * L
    captured: list[Tensor | None] = [None] * L

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            hidden = _layer_output_hidden(output)
            captured[idx] = hidden[:, -1, :].detach().to(torch.float32).cpu()
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in range(L)]
    try:
        model.eval()
        for prompt in prompts:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            model(**enc)
            for i in range(L):
                v = captured[i]
                if v is None:
                    continue
                arr = v.numpy().reshape(-1)
                sums[i] += float(arr.sum())
                sum_sq[i] += float((arr * arr).sum())
                counts[i] += arr.size
                captured[i] = None
    finally:
        for h in handles:
            h.remove()

    out: dict[int, tuple[float, float]] = {}
    for i in range(L):
        if counts[i] == 0:
            raise RuntimeError(f"No activations captured for layer {i}")
        mu = sums[i] / counts[i]
        var = max(0.0, sum_sq[i] / counts[i] - mu * mu)
        out[i] = (mu, float(np.sqrt(var)))
    return out


def js_divergence_normal(
    values: np.ndarray,
    mu: float,
    sigma: float,
    n_bins: int = 200,
) -> float:
    """JSD between the empirical distribution of `values` and N(mu, sigma).

    Sanity check; paper reports max JSD = 0.0839 across layers (Appendix E.1).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    lo = float(min(values.min(), mu - 6 * sigma))
    hi = float(max(values.max(), mu + 6 * sigma))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    emp, _ = np.histogram(values, bins=edges, density=False)
    p = emp.astype(np.float64)
    p = p / max(p.sum(), 1.0)

    z = (centers - mu) / sigma
    q = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
    q = q * width
    q = q / max(q.sum(), 1e-12)

    m = 0.5 * (p + q)
    eps = 1e-12

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

# -----------------------------------------------------------------------------

@dataclass
class LayerParams:
    alpha: float
    beta: float
    k: float

class ABDHooker:
    """
    Registers forward hooks on each decoder layer    
    """

    def __init__(self, model: nn.Module, layer_means: dict[int, float | tuple[float, float]]):
        self.model = model
        self.layers = _get_decoder_layers(model)
        self.L = len(self.layers)
        self.means: dict[int, float] = {}
        for l, v in layer_means.items():
            self.means[l] = float(v[0]) if isinstance(v, tuple) else float(v)
        self.theta: dict[int, LayerParams] = {}
        self.mask: dict[int, int] = {l: 0 for l in range(self.L)}
        self._handles: list = []

    def set_params(self, theta: dict[int, dict], mask: dict[int, int]) -> None:
        self.theta = {
            int(l): LayerParams(float(p["alpha"]), float(p["beta"]), float(p["k"]))
            for l, p in theta.items()
        }
        self.mask = {int(l): int(m) for l, m in mask.items()}

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            if self.mask.get(idx, 0) != 1:
                return output
            params = self.theta.get(idx)
            if params is None or params.k <= 0.0:
                return output
            mu = self.means[idx]
            hidden = _layer_output_hidden(output)
            new_hidden = apply_penalty(hidden, mu, params.alpha, params.beta, params.k)
            if isinstance(output, Tensor):
                return new_hidden
            return (new_hidden, *output[1:])
        return hook

    def attach(self) -> "ABDHooker":
        if self._handles:
            return self
        for i in range(self.L):
            self._handles.append(self.layers[i].register_forward_hook(self._make_hook(i)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "ABDHooker":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()