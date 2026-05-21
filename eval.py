"""
Produce DSR table
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import torch
from torch import nn

from optimize import compute_dsr

# -----------------------------------------------------------------------------

@torch.no_grad()
def build_dsr_table(
    model: nn.Module,
    tokenizer,
    hooks,                  # penalty.ABDHooker
    best,                   # optimize.TrialResult
    attack_dict: Mapping[str, Sequence[str]],
    device: str | torch.device = "cuda",
    max_new_tokens: int = 128,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for name, prompts in attack_dict.items():
        # --- No Defense ---
        hooks.detach()
        if verbose:
            print(f"[{name}] No Defense ... ", end="", flush=True)
        no_def = compute_dsr(
            model, tokenizer, prompts, device=device, max_new_tokens=max_new_tokens
        )
        if verbose:
            print(f"DSR={no_def:.3f}")

        # --- ABD ---
        hooks.set_params(best.theta, best.mask)
        hooks.attach()
        if verbose:
            print(f"[{name}] ABD        ... ", end="", flush=True)
        with_abd = compute_dsr(
            model, tokenizer, prompts, device=device, max_new_tokens=max_new_tokens
        )
        if verbose:
            print(f"DSR={with_abd:.3f}")
        hooks.detach()

        rows[name] = {"No Defense": no_def, "ABD (Ours)": with_abd}

    df = pd.DataFrame.from_dict(rows, orient="index") * 100.0
    df.index.name = "Attack"
    return df.round(2)