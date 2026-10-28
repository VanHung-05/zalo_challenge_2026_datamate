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

        if self.backend == "ollama":
            responses = []
            for _ in range(n):
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tok,
                        "seed": self.seed,
                        "stop": stop_tokens
                    },
                    "stream": False
                }
                try:
                    resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
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
