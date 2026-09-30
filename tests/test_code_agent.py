"""Unit tests for CodeAgent multi-file code generation and manifest builder."""
from __future__ import annotations

import pytest
from app.agents.models import EvidenceLink
from app.guardrails.models import GuardrailResult, AIParadigm, Requirement, EvidenceState, GuardrailTerminalState
from app.agents.code_agent import CodeAgent, CodeGenResult, GeneratedFile


def test_detect_architecture_family():
    """Basic single-family keyword detection."""
    attention_specs = {"key_parameters": {"value": "d_model=512, nhead=8", "status": "SUPPORTED"}}
    conv_specs = {"key_parameters": {"value": "conv2d, kernel_size=3, resnet", "status": "SUPPORTED"}}
    recurrent_specs = {"key_parameters": {"value": "LSTM with 2 layers", "status": "SUPPORTED"}}
    gan_specs = {"key_parameters": {"value": "Generator and Discriminator with Wasserstein loss", "status": "SUPPORTED"}}
    graph_specs = {"key_parameters": {"value": "Graph Convolutional Network (GCN) with adjacency matrix", "status": "SUPPORTED"}}
    generic_specs = {"key_parameters": {"value": "Custom MLP architecture", "status": "SUPPORTED"}}

    assert CodeAgent._detect_architecture_family(attention_specs) == "attention"
    assert CodeAgent._detect_architecture_family(conv_specs) == "convolutional"
    assert CodeAgent._detect_architecture_family(recurrent_specs) == "recurrent"
    assert CodeAgent._detect_architecture_family(gan_specs) == "gan"
    assert CodeAgent._detect_architecture_family(graph_specs) == "graph"
    assert CodeAgent._detect_architecture_family(generic_specs) == "generic_deep_learning"


def test_detect_architecture_family_graph_attention():
    """Graph Attention Networks (GAT) — graph-heavy text with one incidental
    'attention' keyword.  Must return 'graph', not 'attention'."""
    gat_specs = {
        "core_method_description": {
            "value": "Graph Attention Network using masked self-attention over graph neighborhoods",
            "status": "SUPPORTED",
        },
        "key_parameters": {
            "value": "8 attention heads, node embedding dim=64, graph adjacency matrix, message passing layers",
            "status": "SUPPORTED",
        },
    }
    assert CodeAgent._detect_architecture_family(gat_specs) == "graph"


def test_detect_architecture_family_gan_with_conv():
    """GAN-heavy text with one incidental 'conv' mention (e.g. 'convolutional
    discriminator').  Must return 'gan', not 'convolutional'."""
    dcgan_specs = {
        "core_method_description": {
            "value": "Deep Convolutional GAN with convolutional discriminator and generator using adversarial training",
            "status": "SUPPORTED",
        },
        "key_parameters": {
            "value": "generator latent dim=100, discriminator, adversarial loss",
            "status": "SUPPORTED",
        },
    }
    assert CodeAgent._detect_architecture_family(dcgan_specs) == "gan"


def test_detect_architecture_family_pure_transformer():
    """Pure vanilla-Transformer spec dict (no graph/gan/conv/rnn keywords).
    Must still return 'attention'."""
    transformer_specs = {
        "core_method_description": {
            "value": "Transformer with multi-head self-attention and encoder-decoder architecture",
            "status": "SUPPORTED",
        },
        "key_parameters": {
            "value": "d_model=512, d_ff=2048, num_heads=8, qkv projections",
            "status": "SUPPORTED",
        },
    }
    assert CodeAgent._detect_architecture_family(transformer_specs) == "attention"


def test_build_file_manifest():
    attention_manifest = CodeAgent._build_file_manifest("attention", {})
    filenames = [f["filename"] for f in attention_manifest]
    assert filenames == ["attention.py", "layers.py", "blocks.py", "model.py", "train.py", "main.py"]

    conv_manifest = CodeAgent._build_file_manifest("convolutional", {})
    filenames_conv = [f["filename"] for f in conv_manifest]
    assert filenames_conv == ["layers.py", "model.py", "train.py", "main.py"]


def test_looks_truncated():
    # Complete Python code without __main__
    code_no_main = "import torch\nclass Foo:\n    pass\n"
    # Should not look truncated if require_main=False
    assert CodeAgent._looks_truncated(code_no_main, require_main=False) is False
    # Should look truncated if require_main=True
    assert CodeAgent._looks_truncated(code_no_main, require_main=True) is True

    # Unbalanced parenthesis
    code_unbalanced = "def foo():\n    bar("
    assert CodeAgent._looks_truncated(code_unbalanced, require_main=False) is True


def test_generate_implementation_multifile_success():
    mock_responses = {
        "attention.py": "import torch\nimport torch.nn as nn\n# SPEC: key_parameters = d_model=512 (source: page 4)\nclass MultiHeadAttention(nn.Module):\n    pass\n",
        "layers.py": "import torch.nn as nn\nclass PositionwiseFeedForward(nn.Module):\n    pass\n",
        "blocks.py": "import torch.nn as nn\nclass EncoderBlock(nn.Module):\n    pass\n",
        "model.py": "import torch.nn as nn\nclass Transformer(nn.Module):\n    pass\n",
        "train.py": "def train():\n    pass\n",
        "main.py": "if __name__ == '__main__':\n    print('Running transformer model')\n",
    }

    def mock_llm_fn(prompt: str) -> str:
        for fname, resp in mock_responses.items():
            if f"write ONLY `{fname}`" in prompt or f"`{fname}`" in prompt:
                return f"```python\n{resp}\n```"
        return "```python\n# dummy output\n```"

    agent = CodeAgent(llm_fn=mock_llm_fn)
    gr = GuardrailResult(
        terminal_state=GuardrailTerminalState.PASS,
        attempt_count=1,
        code_generation_allowed=True,
        detected_paradigm=AIParadigm.DEEP_LEARNING,
        task_description="Implement Transformer model",
        requirements={
            "key_parameters": Requirement(
                name="key_parameters",
                value="d_model=512, nhead=8 (multi-head attention)",
                state=EvidenceState.SUPPORTED,
                evidence=EvidenceLink(source="paper.pdf", page=4, chunk_id="c1", evidence_text="d_model=512"),
            )
        },
    )

    result = agent.generate_implementation_multifile(gr)
    assert result.success is True
    assert len(result.files) == 6
    file_map = {f.filename: f.code for f in result.files}
    assert "attention.py" in file_map
    assert "main.py" in file_map
    assert "MultiHeadAttention" in file_map["attention.py"]
    assert "key_parameters" in result.spec_citations_found[0]

    # Verify spec coverage was computed
    assert isinstance(result.missing_specs, list)
    assert isinstance(result.fully_grounded, bool)


def test_auto_fix_syntax_missing_indented_block():
    """Verify auto_fix_syntax inserts pass for missing indented blocks after if/def/class/for."""
    broken_code = (
        "import torch\n"
        "def train_epoch():\n"
        "    if epoch % 5 == 0:\n"
        "    print('Done')\n"
    )
    fixed = CodeAgent.auto_fix_syntax(broken_code)
    # Should compile cleanly without SyntaxError
    compile(fixed, "<test>", "exec")
    assert "pass" in fixed


def test_auto_fix_syntax_eof_dangling_colon():
    """Verify auto_fix_syntax handles dangling colon at end of file."""
    broken_code = "if __name__ == '__main__':"
    fixed = CodeAgent.auto_fix_syntax(broken_code)
    compile(fixed, "<test>", "exec")
    assert "pass" in fixed


def test_generate_implementation_multifile_recovers_from_syntax_error():
    """Verify multifile generator automatically fixes syntax error in train.py."""
    mock_responses = {
        "attention.py": "class MHA:\n    pass\n",
        "layers.py": "class Layer:\n    pass\n",
        "blocks.py": "class Block:\n    pass\n",
        "model.py": "class Model:\n    pass\n",
        # train.py contains missing indented block after 'if' statement
        "train.py": "def train():\n    if True:\n    return\n",
        "main.py": "if __name__ == '__main__':\n    print('OK')\n",
    }

    def mock_llm_fn(prompt: str) -> str:
        for fname, resp in mock_responses.items():
            if f"`{fname}`" in prompt:
                return f"```python\n{resp}\n```"
        return "```python\n# fallback\n```"

    agent = CodeAgent(llm_fn=mock_llm_fn)
    gr = GuardrailResult(
        terminal_state=GuardrailTerminalState.PASS,
        attempt_count=1,
        code_generation_allowed=True,
        detected_paradigm=AIParadigm.DEEP_LEARNING,
        task_description="Test recovery",
    )

    result = agent.generate_implementation_multifile(gr)
    assert result.success is True
    file_map = {f.filename: f.code for f in result.files}
    assert "train.py" in file_map
    # Verify train.py now compiles cleanly
    compile(file_map["train.py"], "<train.py>", "exec")


def test_heal_cross_file_imports_missing_symbol():
    """Verify heal_cross_file_imports injects a fallback stub into layers.py when model.py imports a missing symbol."""
    from app.agents.code_agent import GeneratedFile

    files = [
        GeneratedFile(filename="layers.py", code="import torch\nimport torch.nn as nn\nclass PositionalEncoding(nn.Module):\n    pass\n"),
        GeneratedFile(filename="model.py", code="import torch\nimport torch.nn as nn\nfrom layers import TokenEmbedding, PositionalEncoding\nclass TransformerModel(nn.Module):\n    pass\n"),
    ]

    healed = CodeAgent.heal_cross_file_imports(files)
    layers_code = next(f.code for f in healed if f.filename == "layers.py")
    assert "class TokenEmbedding(nn.Module):" in layers_code


