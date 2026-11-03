"""LLM Wrapper supporting dual backends:
- Ollama: development on Mac
- llama.cpp: Kaggle / single GPU GGUF
- vLLM: production on GPU
"""

from typing import Any, Dict, List, Optional
import requests


class LLMWrapper:
    """Unified LLM inference interface.

    Dev:
        Ollama + qwen2.5:7b / qwen3:8b over local REST API.

    Kaggle:
        llama.cpp + GGUF on a single GPU.

    Production:
        vLLM + Qwen2.5-14B / Qwen3-14B with Tensor Parallelism.
    """

    def __init__(self, config: Dict[str, Any]):
        llm_cfg = config.get("llm", {})

        self.backend = llm_cfg.get("backend", "ollama")
        self.model = llm_cfg.get("model", "qwen2.5:7b")
        self.seed = int(llm_cfg.get("seed", 42))
        self.default_temp = float(llm_cfg.get("temperature", 0.0))
        self.default_max_tokens = int(llm_cfg.get("max_tokens", 4096))

        self.engine = None

        # =====================================================
        # 1. OLLAMA BACKEND
        # =====================================================
        if self.backend == "ollama":
            self.base_url = llm_cfg.get("ollama_url", "http://localhost:11434")
            self.num_ctx = int(llm_cfg.get("num_ctx", 32768))
            self.request_timeout = int(llm_cfg.get("request_timeout", 300))

        # =====================================================
        # 2. LLAMA.CPP BACKEND
        # =====================================================
        elif self.backend == "llama_cpp":
            try:
                import ctypes
                import glob
                import os

                cuda_patterns = [
                    "/usr/local/lib/python*/dist-packages/nvidia/*/lib/libcudart.so.12*",
                    "/usr/local/lib/python*/site-packages/nvidia/*/lib/libcudart.so.12*",
                    "/usr/local/lib/python*/dist-packages/nvidia/*/lib/libcublasLt.so.12*",
                    "/usr/local/lib/python*/site-packages/nvidia/*/lib/libcublasLt.so.12*",
                    "/usr/local/lib/python*/dist-packages/nvidia/*/lib/libcublas.so.12*",
                    "/usr/local/lib/python*/site-packages/nvidia/*/lib/libcublas.so.12*",
                    "/usr/local/cuda*/lib64/libcudart.so.12*",
                    "/usr/local/cuda*/lib64/libcublasLt.so.12*",
                    "/usr/local/cuda*/lib64/libcublas.so.12*",
                ]

                preloaded = set()
                for pattern in cuda_patterns:
                    for lib_path in sorted(glob.glob(pattern)):
                        real_path = os.path.realpath(lib_path)
                        if real_path in preloaded:
                            continue
                        try:
                            ctypes.CDLL(real_path, mode=ctypes.RTLD_GLOBAL)
                            preloaded.add(real_path)
                        except OSError as load_error:
                            print(
                                f"[LLM Warning] Could not preload {real_path}: {load_error}",
                                flush=True
                            )

                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is not installed. "
                    "On Kaggle, install a CUDA-enabled wheel first."
                )

            import os

            model_path = llm_cfg.get("model_path")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"GGUF model not found: {model_path}. "
                    "Run the model download cell first."
                )

            self.enable_thinking = bool(llm_cfg.get("enable_thinking", False))

            kw = {
                "model_path": model_path,
                "n_ctx": int(llm_cfg.get("n_ctx", 32768)),
                "n_gpu_layers": int(llm_cfg.get("n_gpu_layers", -1)),
                "n_batch": int(llm_cfg.get("n_batch", 1024)),
                "seed": self.seed,
                "verbose": bool(llm_cfg.get("verbose", False)),
            }
            if "flash_attn" in llm_cfg:
                kw["flash_attn"] = bool(llm_cfg["flash_attn"])

            self.engine = Llama(**kw)

        # =====================================================
        # 3. vLLM BACKEND
        # =====================================================
        elif self.backend == "vllm":
            try:
                from vllm import LLM

                model_path = llm_cfg.get("model_path", self.model)
                tp_size = int(llm_cfg.get("tp_size", 2))

                self.engine = LLM(
                    model=model_path,
                    tensor_parallel_size=tp_size,
                    seed=self.seed,
                    trust_remote_code=True,
                )
            except ImportError:
                raise ImportError(
                    "vLLM is not installed. "
                    "Use 'ollama' for Mac dev or 'llama_cpp' for Kaggle."
                )

        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    # =========================================================
    # CHAT TEMPLATE HELPER
    # =========================================================
    def _chat(self, prompt: str) -> str:
        """Wrap prompt in the target model's chat template."""
        if self.backend == "llama_cpp":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        if self.backend == "vllm":
            try:
                tok = self.engine.get_tokenizer()
                if getattr(tok, "chat_template", None):
                    return tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            except Exception as e:
                print(f"[LLM Warning] vLLM chat template not applied: {e}")

        return prompt

    # =========================================================
    # GENERATE
    # =========================================================
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate one or multiple responses.
        Each sample uses base_seed + i for reproducibility.
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        temp = temperature if temperature is not None else self.default_temp
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        stop_tokens = (
            stop
            if stop is not None
            else ["```\n\n", "\n```\n", "<|im_end|>", "<|im_start|>", "\n\n\n\n"]
        )

        req_seed = self.seed if seed is None else int(seed)

        # 1. OLLAMA
        if self.backend == "ollama":
            responses = []
            for i in range(n):
                payload = {
                    "model": self.model,
                    "prompt": self._chat(prompt),
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tok,
                        "num_ctx": self.num_ctx,
                        "seed": req_seed + i,
                        "stop": stop_tokens,
                    },
                    "stream": False,
                }
                try:
                    resp = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=self.request_timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    responses.append(data.get("response", ""))
                except Exception as e:
                    print(f"[LLM Error] Ollama request failed: {e}", flush=True)
                    responses.append("")
            return responses

        # 2. LLAMA.CPP
        elif self.backend == "llama_cpp":
            text_prompt = self._chat(prompt)
            responses = []
            for i in range(n):
                try:
                    output = self.engine.create_completion(
                        prompt=text_prompt,
                        max_tokens=max_tok,
                        temperature=temp,
                        top_p=0.95 if temp > 0 else 1.0,
                        stop=stop_tokens,
                        seed=req_seed + i,
                    )
                    responses.append(output["choices"][0]["text"])
                except Exception as e:
                    print(f"[LLM Error] llama.cpp generation failed: {e}", flush=True)
                    responses.append("")
            return responses

        # 3. vLLM
        elif self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(
                temperature=temp,
                max_tokens=max_tok,
                n=n,
                stop=stop_tokens,
                seed=req_seed,
            )
            text_prompt = self._chat(prompt)
            try:
                outputs = self.engine.generate([text_prompt], params)
                return [item.text for item in outputs[0].outputs]
            except Exception as e:
                print(f"[LLM Error] vLLM generation failed: {e}", flush=True)
                return [""] * n

        return []
