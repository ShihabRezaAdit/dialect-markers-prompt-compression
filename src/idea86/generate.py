"""Answer generation with vLLM (internal plan Section 12). Runs in env-gen.

One GPU per job; one model per job. Prompts are deduplicated by their text, so the
Arm B prompt with the SAE question and no compression is served from the Arm A S-U
generation, as the plan states.

Run as a GPU job: ./run_pilot.sh generate submits one job per model.
"""
from __future__ import annotations

import hashlib
import os
import time

from .config import model_path
from .score import extract_answer

SQUAD_TEMPLATE = (
    "Answer the question using a short span copied from the passage. Reply with the answer only.\n\n"
    "Passage: {PASSAGE}\n\n"
    "Question: {QUESTION}"
)


def squad_prompt(passage: str, question: str) -> str:
    return SQUAD_TEMPLATE.replace("{PASSAGE}", passage).replace("{QUESTION}", question)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _batch_invariant_on() -> str:
    """Check that the pinned vLLM knows VLLM_BATCH_INVARIANT and that it is on; stop otherwise
    (decisions log: batch-invariant generation)."""
    import vllm
    import vllm.envs as envs

    try:
        on = bool(envs.VLLM_BATCH_INVARIANT)
    except AttributeError:
        raise SystemExit(f"vLLM {vllm.__version__} has no VLLM_BATCH_INVARIANT; see the decisions-log fallback")
    if not on:
        raise SystemExit("VLLM_BATCH_INVARIANT is not active")
    return vllm.__version__


def run(prompts: list[dict], model_id: str, cfg, batch_size: int = 512) -> list[dict]:
    g = cfg["generate"]
    if g.get("batch_invariant"):
        # Outputs must not depend on how prompts are grouped into batches (G1, flips).
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm import LLM, SamplingParams  # env-gen only

    vllm_version = _batch_invariant_on() if g.get("batch_invariant") else None
    m = cfg["generate"]["models"][model_id]
    path = model_path(m["repo"], m["revision"])
    llm = LLM(model=path, dtype=g["dtype"], max_model_len=g["max_model_len"],
              tensor_parallel_size=g["tensor_parallel_size"])
    sp = SamplingParams(temperature=g["temperature"], max_tokens=g["max_tokens"], seed=g["seed"])
    tok = llm.get_tokenizer()

    unique: dict[str, str] = {}
    for p in prompts:
        unique.setdefault(prompt_hash(p["prompt"]), p["prompt"])
    keys = list(unique)
    results: dict[str, dict] = {}
    t0 = time.perf_counter()
    for i in range(0, len(keys), batch_size):
        chunk = keys[i:i + batch_size]
        msgs = [[{"role": "user", "content": unique[k]}] for k in chunk]
        outs = llm.chat(msgs, sp, use_tqdm=False)
        for k, o in zip(chunk, outs):
            c = o.outputs[0]
            results[k] = {"output": c.text, "finish_reason": c.finish_reason,
                          "n_output_tokens": len(c.token_ids)}
    secs = time.perf_counter() - t0
    # The rendered chat text of one prompt, to record any default system text a template adds.
    template_probe = tok.apply_chat_template([{"role": "user", "content": "PROBE"}], tokenize=False,
                                             add_generation_prompt=True)
    rows = []
    for p in prompts:
        k = prompt_hash(p["prompt"])
        r = results[k]
        rows.append({**{x: p[x] for x in p if x != "prompt"}, "model": model_id, "prompt_hash": k,
                     "output": r["output"], "answer": extract_answer(r["output"]),
                     "finish_reason": r["finish_reason"], "length_stop": r["finish_reason"] == "length",
                     "n_output_tokens": r["n_output_tokens"],
                     "n_prompt_tokens": len(tok(p["prompt"], add_special_tokens=False)["input_ids"])})
    rows.append({"_meta": True, "model": model_id, "repo": m["repo"], "revision": m["revision"],
                 "unique_prompts": len(keys), "seconds": secs, "seconds_per_prompt": secs / max(1, len(keys)),
                 "batch_invariant": bool(g.get("batch_invariant")), "vllm_version": vllm_version,
                 "chat_template_probe": template_probe})
    return rows
