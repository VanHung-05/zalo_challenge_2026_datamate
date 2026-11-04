"""LLM Wrapper: Ollama (dev on Mac), vLLM (prod on GPU), llama_cpp (GGUF in-process, e.g. Kaggle).
"""

import re
from typing import Any, Dict, List, Optional
import requests


class LLMWrapper:
    """Unified LLM inference interface.
    - Dev: Ollama + qwen2.5:7b / qwen3:8b over local REST API.
    - Prod: vLLM + Qwen2.5-14B / Qwen3-14B with Tensor Parallelism on 2x RTX 5090.
    """

    def __init__(self, config: Dict[str, Any]):
        llm_cfg = config.get("llm", {})
        self.backend = llm_cfg.get("backend", "ollama")
        self.model = llm_cfg.get("model", "qwen2.5:7b")
        self.seed = llm_cfg.get("seed", 42)
        self.default_temp = llm_cfg.get("temperature", 0.0)
        self.default_max_tokens = llm_cfg.get("max_tokens", 4096)

        if self.backend == "ollama":
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
        elif self.backend == "llama_cpp":
            # GGUF models in-process (llama-cpp-python), e.g. on Kaggle. The chat template stored in the
            # GGUF is applied by create_chat_completion.
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError("llama-cpp-python is not installed (pip install llama-cpp-python with CUDA).")
            self.n_ctx = int(llm_cfg.get("n_ctx", 32768))
            self.enable_thinking = bool(llm_cfg.get("enable_thinking", False))
            self.engine = Llama(
                model_path=llm_cfg.get("model_path", self.model),
                n_ctx=self.n_ctx,
                n_gpu_layers=int(llm_cfg.get("n_gpu_layers", -1)),
                n_batch=int(llm_cfg.get("n_batch", 512)),
                flash_attn=bool(llm_cfg.get("flash_attn", False)),
                seed=self.seed,
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def _llama_cpp_generate(self, prompt: str, temp: float, max_tok: int, n: int, stop: List[str], seed: int) -> List[str]:
        # "/no_think" is Qwen3's switch; for other models (Gemma...) it would only be noise in the prompt
        is_qwen3 = "qwen3" in f"{self.model}".lower()
        content = prompt + "\n/no_think" if (is_qwen3 and not self.enable_thinking) else prompt
        try:
            n_prompt = len(self.engine.tokenize(content.encode("utf-8")))
            if n_prompt + max_tok > self.n_ctx:
                print(f"[LLM Warning] prompt {n_prompt} + max_tokens {max_tok} tokens > n_ctx {self.n_ctx}; it will be cut")
        except Exception:
            pass
        out = []
        for i in range(n):
            try:
                r = self.engine.create_chat_completion(
                    messages=[{"role": "user", "content": content}], temperature=temp, max_tokens=max_tok,
                    stop=stop, seed=seed + i)
                text = r["choices"][0]["message"].get("content") or ""
            except Exception as e:
                print(f"[LLM Error] llama_cpp generation failed: {e}")
                text = ""
            # Reasoning models may still emit a <think> block: only the answer after it is parsed
            out.append(re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip())
        return out

    def _chat(self, prompt: str) -> str:
        """Instruct models expect their chat template; a raw prompt degrades them (vLLM .generate()
        does not apply it, Ollama /api/generate does)."""
        try:
            tok = self.engine.get_tokenizer()
            if getattr(tok, "chat_template", None):
                return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[LLM Warning] chat template not applied: {e}")
        return prompt

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None
    ) -> List[str]:
        """seed: per-request seed (self-consistency samples use base seed + i, so a rerun draws the
        same samples; Rule 8 reproducibility)."""
        temp = temperature if temperature is not None else self.default_temp
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        stop_tokens = stop if stop is not None else ["```\n\n", "User:", "<|im_end|>"]
        req_seed = self.seed if seed is None else seed

        if self.backend == "ollama":
            responses = []
            for i in range(n):
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tok,
                        "num_ctx": self.num_ctx,
                        "seed": req_seed + i,
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
                seed=req_seed
            )
            outputs = self.engine.generate([self._chat(prompt)], params)
            return [o.text for o in outputs[0].outputs]

        elif self.backend == "llama_cpp":
            return self._llama_cpp_generate(prompt, temp, max_tok, n, stop_tokens, req_seed)

        return []
