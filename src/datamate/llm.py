"""LLM Wrapper supporting dual backends: Ollama (dev on Mac) and vLLM (prod on GPU).
"""

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
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

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

        return []
