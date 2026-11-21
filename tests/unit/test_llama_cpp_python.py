"""Guards against llama-cpp-python upgrades breaking llacie.

llacie relies on a small slice of llama-cpp-python's API (see llacie/inference/llama_cpp_python.py
and AbstractVllmOrLcpStrategy). These tests pin down that contract:

- The offline tests check the API surface and that every strategy's JSON schema still compiles to
  a llama.cpp grammar. They need no model and run in well under a second.
- The `requires_llm` test loads a ~3 MB random-weight GGUF from HuggingFace and runs real
  schema-constrained generation through llacie's wrapper. The output is gibberish by design; what
  is asserted is that loading, grammar compilation and constrained decoding all still work.
"""
import inspect
import json
from glob import glob
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

llama_cpp = pytest.importorskip("llama_cpp")
from llama_cpp import Llama, LlamaGrammar  # noqa: E402

from llacie.inference.llama_cpp_python import LlamaCppPython  # noqa: E402
from llacie.strategies.abstract_vllm_or_lcp import AbstractVllmOrLcpStrategy  # noqa: E402

pytestmark = pytest.mark.unit

MIN_LLAMA_CPP_VERSION = (0, 3, 21)  # first release with universal py3 wheels; see pyproject.toml
FEATURE_STRATEGIES_DIR = Path(__file__).parents[2] / "llacie" / "strategies" / "feature"

TINY_GGUF_REPO = "aladar/tiny-random-LlamaForCausalLM-GGUF"
TINY_GGUF_FILE = "tiny-random-LlamaForCausalLM.gguf"


def _llama_cpp_strategies():
    """All concrete feature strategies that can run on llama-cpp-python.

    Discovers modules by globbing, like StrategyFinder does, because some strategy directories
    (e.g. feature/antibiotics) have no __init__.py and would be skipped by pkgutil."""
    strategies = []
    for py_file in sorted(glob(str(FEATURE_STRATEGIES_DIR / "*" / "*.py"))):
        if Path(py_file).name.startswith("__"):
            continue
        rel = Path(py_file).relative_to(FEATURE_STRATEGIES_DIR).with_suffix("")
        mod = import_module("llacie.strategies.feature." + ".".join(rel.parts))
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, AbstractVllmOrLcpStrategy) and obj.__module__ == mod.__name__
                    and not name.startswith("Abstract")):
                strategies.append(obj)
    return strategies


LLAMA_CPP_STRATEGIES = _llama_cpp_strategies()


def _translated_sampling_params(strategy_cls):
    """Runs a strategy's vllm-style SAMPLING_PARAMS through llacie's own translation step."""
    fake_self = SimpleNamespace(backend="llama-cpp-python")
    return AbstractVllmOrLcpStrategy._translate_sampling_params(
        fake_self, dict(strategy_cls.SAMPLING_PARAMS))


def _schema_ids(cls):
    return cls.name


def test_strategies_were_discovered():
    names = {s.name for s in LLAMA_CPP_STRATEGIES}
    assert "feature.presenting_sx.llama3_8b" in names


def test_llama_cpp_version_floor():
    version = tuple(int(p) for p in llama_cpp.__version__.split(".")[:3])
    assert version >= MIN_LLAMA_CPP_VERSION, llama_cpp.__version__


def test_llama_constructors_accept_llacie_engine_args():
    """LlamaCppPython passes these to Llama() / Llama.from_pretrained()."""
    from_pretrained = inspect.signature(Llama.from_pretrained).parameters
    assert {"repo_id", "filename"} <= set(from_pretrained)

    init = inspect.signature(Llama.__init__).parameters
    assert {"model_path", "n_ctx", "verbose", "seed"} <= set(init)


@pytest.mark.parametrize("strategy_cls", LLAMA_CPP_STRATEGIES, ids=_schema_ids)
def test_create_chat_completion_accepts_strategy_params(strategy_cls):
    """Every sampling param a strategy sends must still be a create_chat_completion argument.

    Catches renamed/removed kwargs (cf. vllm's `repetition_penalty` -> llama-cpp's
    `repeat_penalty`, which llacie translates)."""
    accepted = set(inspect.signature(Llama.create_chat_completion).parameters)
    sent = set(_translated_sampling_params(strategy_cls)) | {"response_format", "max_tokens"}
    assert sent <= accepted, f"not accepted by llama-cpp: {sorted(sent - accepted)}"


@pytest.mark.parametrize("strategy_cls", LLAMA_CPP_STRATEGIES, ids=_schema_ids)
def test_strategy_json_schema_compiles_to_grammar(strategy_cls):
    """create_chat_completion(response_format={"schema": ...}) converts the schema with
    LlamaGrammar.from_json_schema, which raises on schema features it doesn't support."""
    schema = strategy_cls.JSON_SCHEMA
    json.loads(schema)  # must be valid JSON in the first place
    assert LlamaGrammar.from_json_schema(schema, verbose=False) is not None


def test_default_wrapper_json_schema_compiles_to_grammar():
    assert LlamaGrammar.from_json_schema(LlamaCppPython.JSON_SCHEMA, verbose=False) is not None


@pytest.fixture(scope="module")
def tiny_llm():
    """llacie's LlamaCppPython wrapper around a tiny random-weight model, loaded the same way
    strategies load models from HuggingFace (via the `filename` engine arg)."""
    schema = json.dumps({
        "type": "array", "minItems": 1, "maxItems": 3,
        "items": {"type": "string", "minLength": 1, "maxLength": 8},
    })
    try:
        return LlamaCppPython(
            {}, TINY_GGUF_REPO, "List the antibiotics.", "Note: {input}", schema,
            filename=TINY_GGUF_FILE, n_ctx=512, verbose=False, seed=0,
        )
    except OSError as e:  # e.g. no network to download the model
        pytest.skip(f"Could not fetch {TINY_GGUF_REPO}: {e}")


@pytest.mark.requires_llm
def test_tiny_model_output_obeys_json_schema(tiny_llm):
    """Greedy, schema-constrained generation must produce an array matching the schema."""
    prompt = tiny_llm.create_prompt(input="Started on vancomycin and cefepime.")
    result = tiny_llm._model.create_chat_completion(
        prompt,
        response_format={"type": "json_object", "schema": json.loads(tiny_llm.json_schema)},
        temperature=0, max_tokens=128,
    )
    content = result["choices"][0]["message"]["content"]
    # strict=False: a random-weight model can emit raw control characters inside strings
    output = json.loads(content, strict=False)
    assert isinstance(output, list) and 1 <= len(output) <= 3, content
    assert all(isinstance(s, str) and 1 <= len(s) <= 8 for s in output), content


@pytest.mark.requires_llm
def test_tiny_model_through_llacie_wrapper(tiny_llm):
    """LlamaCppPython.__call__ returns one parsed output (or None) per prompt, using the exact
    sampling params a real strategy would send."""
    params = _translated_sampling_params(LLAMA_CPP_STRATEGIES[0])
    prompts = [tiny_llm.create_prompt(input=f"Note {i}: given ceftriaxone.") for i in range(3)]
    outputs = tiny_llm(prompts, max_tokens=128, **params)
    assert len(outputs) == len(prompts)
    assert all(o is None or (isinstance(o, list) and all(isinstance(s, str) for s in o))
               for o in outputs)
    assert any(o is not None for o in outputs), "no prompt produced parseable JSON"
