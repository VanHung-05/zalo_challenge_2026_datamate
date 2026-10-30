"""LLM Wrapper supporting backends: llama.cpp GGUF (local dev), Ollama (dev on Mac) and vLLM (prod on GPU).
"""

import os
import re
from typing import Any, Dict, List, Optional
import requests

# Qwen3.x may emit a reasoning block before the answer; only the answer is parsed downstream
THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


class LLMWrapper:
    """Unified LLM inference interface.
    - Dev: Ollama + qwen2.5:7b / qwen3:8b over local REST API.
    - Dev: llama-cpp-python + local GGUF file (e.g. model/Qwen3.5-4B-Q4_K_M.gguf), no server needed.
    - Prod: vLLM + Qwen2.5-14B / Qwen3-14B with Tensor Parallelism on 2x RTX 5090.
    """

    def __init__(self, config: Dict[str, Any]):
        llm_cfg = config.get("llm", {})
        self.backend = llm_cfg.get("backend", "ollama")
        self.model = llm_cfg.get("model", "qwen2.5:7b")
        self.seed = llm_cfg.get("seed", 42)
        self.default_temp = llm_cfg.get("temperature", 0.0)
        self.default_max_tokens = llm_cfg.get("max_tokens", 4096)

        if self.backend == "llama_cpp":
            try:
                from llama_cpp import Llama, llama_supports_gpu_offload
            except ImportError as e:
                # Keep the original error: a failed CUDA build and a missing package look identical otherwise
                raise ImportError(f"llama-cpp-python could not be imported ({e}); install it (see inference.ipynb cell 2b)") from e
            model_path = llm_cfg.get("model_path", "")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"GGUF model not found: {model_path} (run scripts/download_model.py)")
            self.n_ctx = llm_cfg.get("n_ctx", 32768)
            n_gpu_layers = llm_cfg.get("n_gpu_layers", -1)
            gpu_build = llama_supports_gpu_offload()
            # A CPU-only build silently ignores n_gpu_layers, so say which device is actually used
            if n_gpu_layers != 0 and not gpu_build:
                print("[LLM Warning] llama-cpp-python was built without GPU support; running on CPU")
            print(f"[LLM] llama.cpp device: {'GPU' if gpu_build and n_gpu_layers != 0 else 'CPU'}")
            self.engine = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,  # -1: offload all layers if built with GPU support
                n_threads=llm_cfg.get("n_threads"),
                seed=self.seed,
                verbose=llm_cfg.get("verbose", False),
            )
        elif self.backend == "ollama":
            self.base_url = llm_cfg.get("ollama_url", "http://localhost:11434")
            # Ollama's default context (2k-4k tokens) silently truncates long prompts (rulebook + schema)
            self.num_ctx = llm_cfg.get("num_ctx", 32768)
            self.request_timeout = llm_cfg.get("request_timeout", 300)
            self.engine = None
        elif self.backend == "vllm":
            try:
                from vllm import LLM
                model_path = llm_cfg.get("model_path", self.model)
                tp_size = llm_cfg.get("tp_size", 2)
                self.engine = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    seed=self.seed,
                    trust_remote_code=True
                )
            except ImportError:
                raise ImportError("vLLM is not installed. Use 'ollama' backend for local development.")
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        stop: Optional[List[str]] = None
    ) -> List[str]:
        temp = temperature if temperature is not None else self.default_temp
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        stop_tokens = stop if stop is not None else ["```\n\n", "User:", "<|im_end|>"]

        if self.backend == "llama_cpp":
            responses = []
            for _ in range(n):
                try:
                    n_prompt = len(self.engine.tokenize(prompt.encode("utf-8"), add_bos=False))
                    if n_prompt + max_tok > self.n_ctx:
                        print(f"[LLM Warning] prompt has {n_prompt} tokens + max_tokens {max_tok} > n_ctx={self.n_ctx}")
                    # Chat API applies the chat template embedded in the GGUF file
                    out = self.engine.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=max_tok,
                        stop=stop_tokens,
                        seed=self.seed,
                    )
                    text = out["choices"][0]["message"].get("content") or ""
                    responses.append(THINK_BLOCK_RE.sub("", text).strip())
                except Exception as e:
                    print(f"[LLM Error] llama.cpp generation failed: {e}")
                    responses.append("")
            return responses

        if self.backend == "ollama":
            responses = []
            for _ in range(n):
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tok,
                        "num_ctx": self.num_ctx,
                        "seed": self.seed,
                        "stop": stop_tokens
                    },
                    "stream": False
                }
                try:
                    resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.request_timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("prompt_eval_count", 0) >= self.num_ctx:
                        print(f"[LLM Warning] prompt filled num_ctx={self.num_ctx}; it was likely truncated")
                    responses.append(data.get("response", ""))
                except Exception as e:
                    print(f"[LLM Error] Ollama request failed: {e}")
                    responses.append("")
            return responses

        elif self.backend == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=temp,
                max_tokens=max_tok,
                n=n,
                stop=stop_tokens,
                seed=self.seed
            )
            outputs = self.engine.generate([prompt], params)
            return [o.text for o in outputs[0].outputs]

        return []
