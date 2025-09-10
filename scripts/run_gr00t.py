#!/usr/bin/env python
"""
Minimal GR00T N1.5 runner for Jetson (PyTorch/Transformers path).

Usage:
  HF_TOKEN=hf_... python scripts/run_gr00t.py --model nvidia/GR00T-N1.5-3B --prompt "Describe how to grasp a mug."

Notes:
- Ensure Jetson-specific torch is installed. Use float16 to reduce memory.
- For 32GB devices, enable swap and consider reducing max_new_tokens.
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GR00T N1.5 with Transformers")
    p.add_argument("--model", default="nvidia/GR00T-N1.5-3B", help="Model repo id")
    p.add_argument("--prompt", required=True, help="Input prompt")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"], help="Computation dtype")
    p.add_argument("--device", default="cuda", help="Device, e.g., cuda or cpu")
    return p.parse_args()


def get_dtype(flag: str):
    return torch.float16 if flag == "fp16" else torch.float32


def main() -> None:
    args = parse_args()
    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    print(f"Loading model {args.model} with dtype={dtype} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    print(tok.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)


