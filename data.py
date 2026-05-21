"""
Dataloader for AdvBench
"""

from __future__ import annotations

import random
from typing import Sequence

# -----------------------------------------------------------------------------

ADVBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

# -----------------------------------------------------------------------------

def load_advbench() -> list[str]:
    """
    Return the ~520 AdvBench harmful behaviors as plain strings.

    Try HF dataset first, fall back to the GCG repo CSV.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("walledai/AdvBench", split="train")
        return [str(p) for p in ds["prompt"]]
    except Exception:
        import csv
        import io
        import urllib.request

        with urllib.request.urlopen(ADVBENCH_CSV_URL) as f:
            text = f.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return [row["goal"] for row in reader]

def split(
    prompts: Sequence[str],
    sizes: tuple[int, ...] = (200, 50, 50),
    seed: int = 0,
) -> list[list[str]]:    
    if sum(sizes) > len(prompts):
        raise ValueError(
            f"Requested {sum(sizes)} prompts but only {len(prompts)} available."
        )
    rng = random.Random(seed)
    idx = list(range(len(prompts)))
    rng.shuffle(idx)
    out: list[list[str]] = []
    start = 0
    for n in sizes:
        out.append([prompts[i] for i in idx[start : start + n]])
        start += n
    return out


def format_chat(prompts: Sequence[str], tokenizer) -> list[str]:    
    out: list[str] = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        out.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return out