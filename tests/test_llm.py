import sys
import types

from src.datamate.llm import LLMWrapper


class FakeLlama:
    """Stands in for llama_cpp.Llama (not installed on the dev machine)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def tokenize(self, b):
        return list(range(len(b) // 4))

    def create_chat_completion(self, messages, temperature, max_tokens, stop, seed):
        self.calls.append({"messages": messages, "temperature": temperature, "seed": seed, "stop": stop})
        return {"choices": [{"message": {"content": "<think>hmm</think>TYPE: number\n```sql\nSELECT 1\n```"}}]}


def make(monkeypatch, model):
    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    cfg = {"llm": {"backend": "llama_cpp", "model": model, "model_path": "/m.gguf", "n_ctx": 8192,
                   "n_gpu_layers": -1, "flash_attn": True, "seed": 42, "temperature": 0.0, "max_tokens": 256}}
    return LLMWrapper(cfg)


def test_llama_cpp_backend(monkeypatch):
    llm = make(monkeypatch, "gemma-4-12b-it-Q4_K_M.gguf")
    assert llm.engine.kwargs["n_ctx"] == 8192 and llm.engine.kwargs["model_path"] == "/m.gguf"
    out = llm.generate("hỏi", n=2, seed=7)
    assert out == ["TYPE: number\n```sql\nSELECT 1\n```"] * 2          # think block stripped
    assert [c["seed"] for c in llm.engine.calls] == [7, 8]           # per-sample seeds (Rule 8)
    assert llm.engine.calls[0]["messages"][0]["content"] == "hỏi"    # no Qwen3 switch for Gemma


def test_qwen3_no_think_switch(monkeypatch):
    llm = make(monkeypatch, "Qwen3.5-4B-GGUF")
    llm.generate("hỏi")
    assert llm.engine.calls[0]["messages"][0]["content"].endswith("/no_think")
