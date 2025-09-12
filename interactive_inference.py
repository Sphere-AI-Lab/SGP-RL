#!/usr/bin/env python3
"""
sample_vllm.py — Load a (local or Hugging Face) LLM with vLLM and chat interactively.

Usage:
  pip install vllm torch  # torch should match your CUDA; see PyTorch site for wheels
  python sample_vllm.py --model Qwen/Qwen2.5-7B-Instruct
  python sample_vllm.py --model /path/to/your/checkpoint --system "You are helpful."

Multi-GPU example:
  python sample_vllm.py --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 2

Notes:
- Uses tokenizer.chat_template when available; otherwise falls back to a simple prompt format.
- Keeps the conversation history within this session; use --no-history to only send the latest user turn.
- For very long chats, raise --max-model-len to avoid truncation (if your GPU memory allows).
- If your model repo requires custom code, pass --trust-remote-code (defaults to True).
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict

from vllm import LLM, SamplingParams


# -----------------------------
# Utilities
# -----------------------------

def maybe_apply_chat_template(tokenizer, system_prompt: Optional[str], turns: List[Dict[str, str]]) -> str:
    """
    Build a text prompt using the tokenizer's chat_template if present.
    'turns' is a list of {"role": "user"/"assistant", "content": str}.
    """
    has_chat_template = getattr(tokenizer, "chat_template", None) not in (None, "", False)
    if has_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(turns)
        # Ensure the template ends expecting the assistant
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback: plain preamble
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for m in turns:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)


@dataclass
class GenConfig:
    temperature: float = 0.7


def build_sampling_params(cfg: GenConfig) -> SamplingParams:
    return SamplingParams(
        n=1,
        best_of=1,
        stop=["</answer>"],
        max_tokens = 3000,
        include_stop_str_in_output=True,
        temperature=cfg.temperature,
    )


# -----------------------------
# Main
# -----------------------------

def load_llm(
    model_id_or_path: str,
    trust_remote_code: bool,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    dtype: Optional[str],
):
    """
    Construct a vLLM LLM instance.
    - dtype can be: "auto", "float16", "bfloat16", "float32" (None acts like "auto")
    """
    llm = LLM(
        model=model_id_or_path,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype if dtype != "auto" else None,
        enforce_eager=False,  # generally faster with CUDA graphs when False
    )
    tok = llm.get_tokenizer()
    return llm, tok


def interactive_loop(llm: LLM, tokenizer, system_prompt: Optional[str], cfg: GenConfig, keep_history: bool):
    turns: List[Dict[str, str]] = []
    sampler = build_sampling_params(cfg)

    print("\n[ready] Chat is live. Press Ctrl+C to quit.")
    if system_prompt:
        print(f"[info] System prompt: {system_prompt!r}", file=sys.stderr)

    while True:
        try:
            user_msg = input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[bye]")
            break
        if not user_msg:
            continue

        if keep_history:
            turns.append({"role": "user", "content": user_msg})
            prompt_text = maybe_apply_chat_template(tokenizer, system_prompt, turns)
        else:
            prompt_text = maybe_apply_chat_template(tokenizer, system_prompt, [{"role": "user", "content": user_msg}])

        # Generate (non-streaming). For streaming, prefer the OpenAI server entrypoint.
        outputs = llm.generate([prompt_text], sampler)
        out_text = outputs[0].outputs[0].text

        print("Assistant > " + out_text, flush=True)

        if keep_history:
            turns.append({"role": "assistant", "content": out_text})


def main():
    ap = argparse.ArgumentParser(description="Interactive chat using vLLM.")
    ap.add_argument("--model",default = "/home/chenyamei/codes/understand-r1-zero/oat-output/reward_siglipsmall_0502T17:43:27/saved_models/step_00780", help="HF repo id or local path to the model checkpoint.")
    ap.add_argument("--system", default="A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.", help="Optional system prompt.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs to shard across.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Fraction of GPU memory to use (0-1).")
    ap.add_argument("--max-model-len", type=int, default=10000, help="Set to override model context length (caution: VRAM).")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--trust-remote-code", action="store_true", help="Enable if your model repo requires it.")
    ap.add_argument("--no-history", action="store_true", help="If set, only send current user turn each time.")
    args = ap.parse_args()

    llm, tokenizer = load_llm(
        args.model,
        trust_remote_code=args.trust_remote_code or True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    cfg = GenConfig(
        temperature=args.temperature,
    )

    interactive_loop(llm, tokenizer, args.system, cfg, keep_history=not args.no_history)


if __name__ == "__main__":
    main()
