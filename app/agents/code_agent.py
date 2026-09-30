from __future__ import annotations

import os
import ast
import re
import logging
from typing import Any, Dict, Callable, Optional

from pydantic import BaseModel
from app.guardrails.models import GuardrailResult, AIParadigm

import difflib

logger = logging.getLogger(__name__)


class CodeGenerationError(RuntimeError):
    """Raised when code generation is attempted despite guardrail restriction."""
    pass


class GeneratedFile(BaseModel):
    """Represents a single generated file in a multi-file project."""
    filename: str
    code: str
    spec_citations_found: list[str] = []
    assumptions_found: list[str] = []
    deferred: list[str] = []


class CodeGenResult(BaseModel):
    """Structured result of a code generation attempt."""
    files: list[GeneratedFile] = []
    generated_code: str = ""          # kept for backward compat: concatenation of all files
    spec_citations_found: list[str] = []
    assumptions_found: list[str] = []
    deferred_specs: list[str] = []
    missing_specs: list[str] = []          # GROUNDING SIGNAL
    fully_grounded: bool = True            # GROUNDING SIGNAL
    raw_model_response: str = ""
    success: bool = True
    error_reason: str | None = None


class CodeAgent:
    """Code generation agent that respects the Evidence Guardrail."""

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None) -> None:
        """
        Args:
            llm_fn: A callable that takes a prompt string and returns the raw
                LLM response string. If not provided, `call_llm` (the module-level
                stub) is used. Inject a real implementation here in production.
        """
        self._llm_fn = llm_fn or call_llm

    # ---------------------------------------------------------------------
    # Helper: ensure generation is permitted
    # ---------------------------------------------------------------------
    @staticmethod
    def _assert_allowed(guardrail_result: GuardrailResult) -> None:
        if not getattr(guardrail_result, "code_generation_allowed", False):
            raise CodeGenerationError(
                "Code generation is not allowed by the Evidence Guardrail. "
                "Inspect the GuardrailResult before attempting generation."
            )

    # ---------------------------------------------------------------------
    # Specification formatting
    # ---------------------------------------------------------------------
    @staticmethod
    def format_specs_for_prompt(grounded_specs: Dict[str, Dict[str, Any]]) -> str:
        """Transform ``grounded_specs`` into a bullet-list string.

        Only entries where ``value`` is non-empty and ``status`` equals
        ``SUPPORTED`` are rendered. Each line follows the pattern::

            - <spec_name>: <value> (source: page <citation>)
        """
        lines: list[str] = []
        for name, spec in grounded_specs.items():
            value = (spec.get("value") or "").strip()
            status = (spec.get("status") or "").upper()
            citation = spec.get("citation")
            if not value or status != "SUPPORTED":
                continue
            if citation is not None:
                line = f"- {name}: {value} (source: page {citation})"
            else:
                line = f"- {name}: {value}"
            lines.append(line)
        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Prompt construction
    # ---------------------------------------------------------------------
    @staticmethod
    def _build_component_checklist(grounded_specs: Dict[str, Dict[str, Any]]) -> str:
        """Derive a mandatory implementation checklist from the grounded specs
        so the model must explicitly account for every verified spec, rather
        than silently dropping ones that don't fit neatly into a code comment.
        """
        items = []
        kp = (grounded_specs.get("key_parameters") or {}).get("value", "")

        # pull out likely layer-count mentions
        layer_match = re.search(r"(?:N|num_layers|layers)\s*=\s*(\d+)", kp, re.IGNORECASE)
        num_layers_hint = layer_match.group(1) if layer_match else None

        items.append("Core architecture module(s) implementing core_method_description")
        items.append(
            f"Full stack of {num_layers_hint or '[N, from key_parameters]'} layers "
            "(use nn.ModuleList — do NOT implement only a single block)"
            if "core_equations_or_formal_rules" in grounded_specs
            or "key_parameters" in grounded_specs
            else "Full architecture stack per key_parameters"
        )

        if "training_or_optimization_procedure" in grounded_specs:
            items.append(
                "Training/optimization procedure: "
                f"{grounded_specs['training_or_optimization_procedure'].get('value')} "
                "— implement the optimizer setup and/or the described schedule as a "
                "function or class, even if not exercised in the __main__ demo."
            )

        if "dataset_or_example_input" in grounded_specs:
            items.append(
                "Dataset/input handling: "
                f"{grounded_specs['dataset_or_example_input'].get('value')} "
                "— reflect this in variable names, comments, or a data-loading stub; "
                "do not omit it silently."
            )

        for i, item in enumerate(items, 1):
            items[i - 1] = f"{i}. {item}"

        return "\n".join(items)

    def build_prompt(self, guardrail_result: GuardrailResult) -> str:
        """Create the full LLM prompt: paradigm context + formatting rules +
        grounded specifications + task description."""
        self._assert_allowed(guardrail_result)

        paradigm = guardrail_result.detected_paradigm

        if paradigm == AIParadigm.DEEP_LEARNING:
            paradigm_context = (
                "Context: Deep Learning paradigm.\n"
                "You MUST write a complete, high-quality, production-grade PyTorch implementation (nn.Module subclasses).\n"
                "CRITICAL ARCHITECTURAL CONTRACTS FOR DEEP LEARNING / TRANSFORMER MODELS:\n"
                "1. MULTI-HEAD ATTENTION:\n"
                "   - If the paper uses Attention, implement true Multi-Head Attention with Query (Q), Key (K), Value (V) linear projections.\n"
                "   - Scaled Dot-Product Attention: compute scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k).\n"
                "   - Support optional `attention_mask` (causal or padding mask) applied to scores before softmax.\n"
                "   - Apply Softmax over key sequence length (`dim=-1`), multiply by V, reshape/concat heads, and apply final linear projection.\n"
                "2. RESIDUAL CONNECTIONS & LAYERNORM:\n"
                "   - Every sublayer (Attention and FFN) MUST use actual residual connections: `x = x + sublayer(norm(x))` or `x = norm(x + sublayer(x))`.\n"
                "   - Do NOT use dummy identity layers (`nn.Identity()`) or out-of-bounds indexing.\n"
                "   - Include LayerNorm or BatchNorm for each sublayer as specified in the paper.\n"
                "3. FEED-FORWARD NETWORKS & EMBEDDINGS:\n"
                "   - Position-wise FFN must have 2 linear transformations with specified activation (e.g. ReLU/GELU).\n"
                "   - If embeddings are scaled (e.g. math.sqrt(d_model)), include the scaling factor.\n"
                "   - Positional Encodings: Implement Sinusoidal positional encoding tensors or learned positional embeddings as specified.\n"
                "4. FULL STRUCTURE & ENTRYPOINT:\n"
                "   - If the paper features an Encoder and Decoder, implement both `Encoder` and `Decoder` stacks with Cross-Attention.\n"
                "   - Include final linear projection head (`nn.Linear(d_model, vocab_size_or_output_dim)`).\n"
                "   - Include a top-level `if __name__ == '__main__':` block unindented at the root level that instantiates the model with sample dimensions, passes dummy tensors, and prints output shapes."
            )
        elif paradigm == AIParadigm.SYMBOLIC_PROBABILISTIC:
            paradigm_context = (
                "Context: Symbolic AI / Probabilistic Graphical Models paradigm.\n"
                "Implement the described method as plain Python class structures "
                "(e.g. logic proposition managers, probability interval "
                "propagators, constraint solvers). Do NOT force this into a "
                "PyTorch nn.Module or neural network shape."
            )
        elif paradigm == AIParadigm.CLASSICAL_ML:
            paradigm_context = (
                "Context: Classical ML paradigm.\n"
                "Implement the described method using scikit-learn compatible "
                "class structures, or custom logic if the method is not a "
                "standard scikit-learn estimator."
            )
        elif paradigm == AIParadigm.CONTROL_INDUSTRIAL:
            paradigm_context = (
                "Context: Control / Industrial paradigm.\n"
                "Implement the described method as state machines, automation "
                "logic, or control-theory system code."
            )
        else:
            paradigm_context = (
                f"Context: {paradigm} paradigm.\n"
                "Implement the described method according to the specifications "
                "below, in the most natural code shape for this kind of method."
            )

        specs_block = self.format_specs_for_prompt(guardrail_result.grounded_specs)
        checklist = self._build_component_checklist(guardrail_result.grounded_specs)

        prompt = f"""You are an expert Machine Learning Software Engineer implementing a paper as Python code.
Your code must be mathematically accurate, architecturally complete, and fully functional. Follow these rules:

1. ARCHITECTURAL FAITHFULNESS:
   - Implement the complete, full-scale architecture faithfully.
   - Do NOT use placeholder stubs, dummy identity loops, or missing operations.

2. SPECIFICATION CITATIONS:
   - For every specification implemented, add an inline comment directly above:
     # SPEC: <spec_name> = <value> (source: <page/citation>)

3. ASSUMPTIONS MARKING:
   - If any parameter/activation/dimension is missing from the extracted specs, choose a standard value and mark it clearly:
     # ASSUMED (not in paper): <what you chose> - <one-line justification>

4. EXPOSE HYPERPARAMETERS:
   - All hyperparameters (e.g. d_model, num_heads, num_layers, lr) must be named arguments in `__init__` with default values.

5. OUTPUT FORMAT:
   - Output ONLY a single Python code block - no conversational prose outside code comments.

6. RUNNABLE ENTRYPOINT:
   - End with a top-level `if __name__ == '__main__':` block that initializes the model, runs a dummy input tensor, and verifies output shapes.

7. MANDATORY COMPONENT CHECKLIST:
   You MUST address every item below. For each item, either implement it fully,
   or if you deliberately choose not to implement it (e.g. it's out of scope
   for a minimal demo), you MUST add a comment immediately before where it
   would go:
     # DEFERRED: <component name> - <one-line reason not implemented>
   Do NOT simply omit a checklist item without a DEFERRED comment. Omission
   without a DEFERRED marker will be treated as a defect.

   CHECKLIST:
{checklist}

{paradigm_context}

VERIFIED SPECIFICATIONS FROM THE PAPER:
{specs_block if specs_block else "(no additional grounded specifications provided)"}

TASK: {guardrail_result.task_description}

Produce the full Python code now:
"""
        return prompt

    # ---------------------------------------------------------------------
    # Code / annotation extraction & sanitization
    # ---------------------------------------------------------------------
    @staticmethod
    def extract_code_block(raw_response: str) -> str:
        """Strip markdown code fences from the raw LLM response."""
        fence_match = re.search(
            r"```(?:python)?\s*\n(.*?)```", raw_response, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            code = fence_match.group(1).strip()
        else:
            import_match = re.search(r"^\s*(import|from)\s", raw_response, re.MULTILINE)
            if import_match:
                code = raw_response[import_match.start():].strip()
            else:
                code = raw_response.strip()

        return CodeAgent.sanitize_code(code)

    @staticmethod
    def auto_fix_syntax(code: str) -> str:
        """Attempt deterministic AST and syntax repairs for common LLM generation flaws,
        such as missing indented blocks after control statements (if/elif/else/def/class/try/for/while).
        """
        if not code or not code.strip():
            return code

        current_code = code
        for _ in range(5):
            try:
                compile(current_code, "<check>", "exec")
                return current_code
            except SyntaxError as exc:
                lines = current_code.splitlines()
                if not lines:
                    break
                err_line = exc.lineno if exc.lineno is not None else len(lines)
                err_msg = str(exc).lower()

                fixed = False

                # Handle "expected an indented block"
                if "expected an indented block" in err_msg or "indented block" in err_msg:
                    header_idx = max(0, min(err_line - 1, len(lines) - 1))
                    
                    # Search upwards for nearest statement line ending with ':'
                    search_idx = header_idx
                    while search_idx >= 0:
                        line_str = lines[search_idx].strip()
                        if line_str and not line_str.startswith("#"):
                            if line_str.endswith(":"):
                                header_idx = search_idx
                                break
                        search_idx -= 1

                    header_line = lines[header_idx]
                    header_indent = len(header_line) - len(header_line.lstrip())
                    pass_indent = " " * (header_indent + 4)
                    
                    insert_pos = header_idx + 1
                    while insert_pos < len(lines) and lines[insert_pos].strip().startswith("#"):
                        insert_pos += 1
                    
                    lines.insert(insert_pos, f"{pass_indent}pass")
                    current_code = "\n".join(lines)
                    fixed = True

                # Handle unexpected EOF or unclosed block at end of file
                elif ("unexpected eof" in err_msg or "was never closed" in err_msg or err_line >= len(lines)) and len(lines) > 0:
                    last_line = lines[-1].strip()
                    if last_line.endswith(":"):
                        indent = len(lines[-1]) - len(lines[-1].lstrip())
                        lines.append(" " * (indent + 4) + "pass")
                        current_code = "\n".join(lines)
                        fixed = True
                    elif last_line.endswith(("=", "(", "[", "{", ",", "+", "-", "*", "/", "\\")):
                        lines.pop()
                        current_code = "\n".join(lines)
                        fixed = True

        # Check AST for truncated/undefined class references (e.g. 'Sc' instead of 'ScaledDotProductAttention')
        try:
            tree = ast.parse(current_code)
            defined_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            if defined_classes:
                import builtins
                undefined_replacements = {}
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        name_id = node.id
                        if (
                            name_id not in defined_classes
                            and name_id not in ("self", "cls")
                            and not hasattr(builtins, name_id)
                        ):
                            matches = [c for c in defined_classes if c.startswith(name_id) and len(c) > len(name_id)]
                            if matches:
                                undefined_replacements[name_id] = f"{matches[0]}()"

                for wrong_name, right_name in undefined_replacements.items():
                    current_code = re.sub(rf"\b{wrong_name}\b", right_name, current_code)
        except Exception:
            pass

        return current_code

    @staticmethod
    def sanitize_code(code: str) -> str:
        """Fix common structural LLM quirks like indented def __main__ inside class bodies."""
        lines = code.splitlines()
        fixed_lines = []
        inside_indented_main = False
        
        for line in lines:
            if re.match(r"^\s+def\s+__main__\s*\([^)]*\)\s*:", line):
                fixed_lines.append("\nif __name__ == '__main__':")
                inside_indented_main = True
                continue
            if inside_indented_main:
                if line.startswith("        "):
                    fixed_lines.append(line[4:])
                elif line.startswith("    "):
                    fixed_lines.append(line[4:])
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        sanitized = "\n".join(fixed_lines)
        return CodeAgent.auto_fix_syntax(sanitized)

    @staticmethod
    def extract_annotations(code: str) -> tuple[list[str], list[str], list[str]]:
        """Extract `# SPEC:`, `# ASSUMED:`, and `# DEFERRED:` annotations."""
        spec_citations: list[str] = []
        assumptions: list[str] = []
        deferred: list[str] = []

        for line in code.splitlines():
            if m := re.search(r"#\s*SPEC:?\s*(.+)", line, re.IGNORECASE):
                spec_citations.append(m.group(1).strip())
                continue
            if m := re.search(r"#\s*ASSUMED[^:]*:\s*(.+)", line, re.IGNORECASE):
                assumptions.append(m.group(1).strip())
                continue
            if m := re.search(r"#\s*DEFERRED:?\s*(.+)", line, re.IGNORECASE):
                deferred.append(m.group(1).strip())

        return spec_citations, assumptions, deferred

    # ---------------------------------------------------------------------
    # Spec coverage verification
    # ---------------------------------------------------------------------
    @staticmethod
    def check_spec_coverage(
        grounded_specs: Dict[str, Dict[str, Any]],
        spec_citations_found: list[str],
        deferred_found: list[str] = [],
    ) -> tuple[list[str], list[str]]:
        """Return (missing_specs, deferred_specs).

        A spec counts as "covered" if it appears in spec_citations_found OR in
        deferred_found. If it appears in deferred_found, it's categorized as
        a disclosed gap (deferred_specs). Otherwise if absent from both, it's
        categorized as a silent gap (missing_specs).
        """
        citations_blob = " ".join(spec_citations_found).lower()
        deferred_blob = " ".join(deferred_found).lower()

        missing: list[str] = []
        deferred_specs: list[str] = []

        for name, spec in grounded_specs.items():
            status = (spec.get("status") or "").upper()
            if status != "SUPPORTED":
                continue  # only enforce coverage for specs the guardrail actually verified

            value = str(spec.get("value") or "")
            name_tokens = re.split(r"[_\s]+", name.lower())
            value_tokens = [
                t.strip(".,()").lower()
                for t in re.split(r"[\s,=]+", value)
                if len(t.strip(".,()")) > 2
            ]

            candidate_tokens = name_tokens + value_tokens

            is_cited = any(tok in citations_blob for tok in candidate_tokens if tok)
            if not is_cited and citations_blob.strip():
                close = difflib.get_close_matches(
                    name.lower(), citations_blob.split(), n=1, cutoff=0.75
                )
                is_cited = bool(close)

            is_deferred = any(tok in deferred_blob for tok in candidate_tokens if tok)
            if not is_deferred and deferred_blob.strip():
                close = difflib.get_close_matches(
                    name.lower(), deferred_blob.split(), n=1, cutoff=0.75
                )
                is_deferred = bool(close)

            if is_cited:
                pass
            elif is_deferred:
                deferred_specs.append(name)
            else:
                missing.append(name)

        return missing, deferred_specs

    # ---------------------------------------------------------------------
    # Architecture Manifest & Multi-File Generation
    # ---------------------------------------------------------------------
    @staticmethod
    def _detect_architecture_family(grounded_specs: Dict[str, Dict[str, Any]]) -> str:
        """Score every architecture family by keyword frequency in the
        concatenated grounded_specs text and return the highest-scoring one.

        This avoids first-match-wins bugs where e.g. "Graph Attention Network"
        would match "attention" before "graph".
        """
        text = ""
        for k, s in grounded_specs.items():
            val = str(s.get("value", "") if isinstance(s, dict) else getattr(s, "value", s)).lower()
            text += f" {k.lower()} {val}"

        families: Dict[str, list[str]] = {
            "graph":         ["gnn", "graph", "adjacency", "message passing", "node embedding"],
            "gan":           ["gan", "generator", "discriminator", "adversarial"],
            "recurrent":     ["lstm", "gru", "rnn", "recurrent", "hidden state"],
            "convolutional": ["conv", "convolutional", "resnet", "cnn", "kernel_size", "stride", "pooling"],
            "attention":     ["attention", "transformer", "multi-head", "d_model", "qkv", "encoder-decoder", "self-attention"],
        }

        scores: Dict[str, int] = {}
        for family, keywords in families.items():
            scores[family] = sum(text.count(kw) for kw in keywords)

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            return "generic_deep_learning"
        return best

    @staticmethod
    def _build_file_manifest(architecture_family: str, grounded_specs: Dict[str, Dict[str, Any]]) -> list[dict]:
        """Return an ordered list of {filename, responsibility} describing
        what each generated file must contain. Later files can reference
        classes defined in earlier files."""
        if architecture_family == "attention":
            manifest = [
                {"filename": "attention.py",
                 "responsibility": "ScaledDotProductAttention and MultiHeadAttention classes only. "
                                    "Must correctly implement Q/K/V projections, scaling by sqrt(d_k), "
                                    "optional masking, and head reshape/concat preserving batch_size."},
                {"filename": "layers.py",
                 "responsibility": "PositionwiseFeedForward, positional encoding (sinusoidal or learned "
                                    "per spec), and embedding scaling by sqrt(d_model). Import from attention.py "
                                    "if needed."},
                {"filename": "blocks.py",
                 "responsibility": "EncoderBlock (and DecoderBlock with cross-attention if the paper "
                                    "specifies a decoder) using the residual/norm convention specified. "
                                    "Import MultiHeadAttention from attention.py and PositionwiseFeedForward "
                                    "from layers.py."},
                {"filename": "model.py",
                 "responsibility": "Full model: stacks num_layers of EncoderBlock (and DecoderBlock if "
                                    "applicable) into Encoder/Decoder, embedding layer, output projection "
                                    "head to vocab size. Import from blocks.py and layers.py."},
                {"filename": "train.py",
                 "responsibility": "Training/optimization procedure per the verified spec (optimizer, "
                                    "learning rate schedule, dataset loading stub) even if not exercised "
                                    "in a runnable demo."},
                {"filename": "main.py",
                 "responsibility": "if __name__ == '__main__' entrypoint: instantiate the model from "
                                    "model.py, run a dummy forward pass, print output shapes."},
            ]
        elif architecture_family == "convolutional":
            manifest = [
                {"filename": "layers.py", "responsibility": "Conv blocks, pooling, normalization per spec."},
                {"filename": "model.py", "responsibility": "Full network assembled from layers.py per key_parameters."},
                {"filename": "train.py", "responsibility": "Training/optimization procedure per spec."},
                {"filename": "main.py", "responsibility": "Entrypoint: instantiate, dummy forward pass, print shapes."},
            ]
        elif architecture_family == "recurrent":
            manifest = [
                {"filename": "layers.py", "responsibility": "Recurrent cells/layers (LSTM/GRU/RNN) and attention mechanism if present."},
                {"filename": "model.py", "responsibility": "Full sequence model assembled from layers.py."},
                {"filename": "train.py", "responsibility": "Training/optimization procedure per spec."},
                {"filename": "main.py", "responsibility": "Entrypoint: instantiate, dummy forward pass, print shapes."},
            ]
        elif architecture_family == "gan":
            manifest = [
                {"filename": "models.py", "responsibility": "Generator and Discriminator networks per spec."},
                {"filename": "train.py", "responsibility": "GAN training procedure, adversarial loss setup, optimizer steps."},
                {"filename": "main.py", "responsibility": "Entrypoint: instantiate Generator and Discriminator, run dummy forward pass, print shapes."},
            ]
        elif architecture_family == "graph":
            manifest = [
                {"filename": "layers.py", "responsibility": "Graph convolution/attention layers and message passing modules."},
                {"filename": "model.py", "responsibility": "Full GNN architecture assembled from layers.py."},
                {"filename": "train.py", "responsibility": "Training/optimization procedure per spec."},
                {"filename": "main.py", "responsibility": "Entrypoint: instantiate model with dummy graph input, print shapes."},
            ]
        else:
            manifest = [
                {"filename": "model.py", "responsibility": "Full architecture per core_method_description and core_equations_or_formal_rules."},
                {"filename": "train.py", "responsibility": "Training/optimization procedure per spec."},
                {"filename": "main.py", "responsibility": "Entrypoint: instantiate, dummy forward pass, print shapes."},
            ]
        return manifest

    # ---------------------------------------------------------------------
    # Truncation detector & continuation loop
    # ---------------------------------------------------------------------
    @staticmethod
    def _looks_truncated(code: str, require_main: bool = False) -> bool:
        """Heuristic check for a response cut off mid-generation."""
        if not code.strip():
            return True
            
        try:
            compile(code, "<check>", "exec")
            if require_main and "__main__" not in code:
                return True
            return False
        except SyntaxError as e:
            err_str = str(e).lower()
            if "unexpected eof" in err_str or "was never closed" in err_str:
                return True
                
            stripped = code.rstrip()
            last_line = stripped.splitlines()[-1].rstrip()
            if last_line.endswith((",", "(", "=", "+", "-", "*", "/", "\\", ":")):
                return True
                
            opens = sum(stripped.count(c) for c in "([{")
            closes = sum(stripped.count(c) for c in ")]}")
            if opens != closes:
                return True
                
            return False

    def _generate_with_continuation(self, prompt: str, max_continuations: int = 3, require_main: bool = False) -> str:
        """Call the LLM, and if the response looks truncated, ask it to
        continue from exactly where it stopped rather than restarting."""
        full_response = self._llm_fn(prompt)
        accumulated_code = self.extract_code_block(full_response)

        attempts = 0
        while self._looks_truncated(accumulated_code, require_main=require_main) and attempts < max_continuations:
            continuation_prompt = (
                f"{prompt}\n\n"
                "--- PARTIAL RESPONSE ALREADY GENERATED (DO NOT REPEAT THIS) ---\n"
                f"{accumulated_code}\n"
                "--- END PARTIAL RESPONSE ---\n\n"
                "Your previous response was cut off. Continue writing the Python "
                "code EXACTLY from where it stopped. Do not repeat any code "
                "already shown above. Do not add markdown fences or commentary. "
                "Output ONLY the remaining code, continuing seamlessly."
            )
            continuation = self._llm_fn(continuation_prompt)
            continuation_code = self.extract_code_block(continuation)
            accumulated_code = self.sanitize_code(accumulated_code + "\n" + continuation_code)
            attempts += 1

        return accumulated_code

    # ---------------------------------------------------------------------
    # Multi-File Generation Engine
    # ---------------------------------------------------------------------
    @staticmethod
    def _summarize_context(context_so_far: str, max_chars: int = 2000) -> str:
        """Reduce prior-file context to class/def signatures + imports only.

        Keeps the model aware of what classes/functions exist in earlier files
        (so it can write correct imports) without blowing up the prompt with
        full implementation bodies. Hard-caps at max_chars to protect small
        local models.
        """
        if not context_so_far.strip():
            return "(none yet — this is the first file)"

        summary_lines: list[str] = []
        for line in context_so_far.splitlines():
            stripped = line.rstrip()
            # Keep file-section headers, imports, class/def signatures, decorators
            if (
                stripped.startswith("# ---")          # section header
                or stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("class ")
                or stripped.startswith("def ")
                or stripped.startswith("    def ")     # method signatures
                or stripped.startswith("    class ")
                or stripped.startswith("@")
            ):
                # For def/class lines keep only the signature (strip the body-start colon onward
                # for single-line defs that happen to have a body)
                summary_lines.append(stripped)

        summary = "\n".join(summary_lines)
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "\n# ... (truncated for brevity)"
        return summary or "(no signatures found in prior files)"

    def generate_implementation_multifile(self, guardrail_result: GuardrailResult) -> CodeGenResult:
        """Generate a modular, multi-file PyTorch implementation based on a file manifest."""
        self._assert_allowed(guardrail_result)

        architecture_family = self._detect_architecture_family(guardrail_result.grounded_specs)
        manifest = self._build_file_manifest(architecture_family, guardrail_result.grounded_specs)
        specs_block = self.format_specs_for_prompt(guardrail_result.grounded_specs)

        generated_files: list[GeneratedFile] = []
        context_so_far = ""  # accumulated prior files, for import consistency

        for entry in manifest:
            is_main = (entry["filename"] == "main.py")
            file_prompt = f"""You are implementing ONE FILE of a multi-file PyTorch project reproducing a paper.

VERIFIED SPECIFICATIONS:
{specs_block if specs_block else "(none)"}

FILES ALREADY WRITTEN (for reference — import from these, do not redefine their classes):
{self._summarize_context(context_so_far) if context_so_far else "(none yet — this is the first file)"}

YOUR TASK: write ONLY `{entry['filename']}`.
RESPONSIBILITY: {entry['responsibility']}

RULES:
- Output ONLY the code for this one file, no other files, no commentary outside comments.
- For every specification implemented, add: # SPEC: <name> = <value> (source: <citation>)
- For anything not in the paper, add: # ASSUMED (not in paper): <choice> - <justification>
- If you deliberately skip something the spec requires (e.g. it belongs in a different file), add: # DEFERRED: <what> - <where it belongs instead>
- Assume all imports from other listed files are correct and available.

Produce the code for {entry['filename']} now:
"""
            # -------------------------------------------
            # Generate file - single pass
            # -------------------------------------------
            try:
                code = self._generate_with_continuation(file_prompt, require_main=is_main)
            except Exception as exc:
                logger.exception("LLM call failed during code generation for %s.", entry['filename'])
                return CodeGenResult(
                    files=generated_files,
                    generated_code=context_so_far,
                    success=False,
                    error_reason=f"LLM call failed on {entry['filename']}: {exc}",
                )

            spec_c, assumed, deferred = self.extract_annotations(code)
            generated_files.append(GeneratedFile(
                filename=entry["filename"],
                code=code,
                spec_citations_found=spec_c,
                assumptions_found=assumed,
                deferred=deferred,
            ))
            context_so_far += f"\n# --- {entry['filename']} ---\n{code}\n"

        all_spec_citations = [c for f in generated_files for c in f.spec_citations_found]
        all_assumptions = [a for f in generated_files for a in f.assumptions_found]
        all_deferred_annotations = [d for f in generated_files for d in f.deferred]

        missing, deferred_specs = self.check_spec_coverage(
            guardrail_result.grounded_specs, all_spec_citations, all_deferred_annotations
        )
        all_deferred = list(dict.fromkeys(deferred_specs + all_deferred_annotations))

        # compile-check each file independently with automatic syntax repair & self-healing
        for f in generated_files:
            repaired_code = self.auto_fix_syntax(f.code)
            try:
                compile(repaired_code, f"<{f.filename}>", "exec")
                f.code = repaired_code
            except SyntaxError as exc:
                healed = False
                if self._llm_fn:
                    repair_prompt = (
                        f"The generated Python code for `{f.filename}` has a syntax error:\n"
                        f"{exc}\n\n"
                        f"CODE:\n```python\n{f.code}\n```\n\n"
                        f"Output ONLY the corrected code for `{f.filename}` with proper syntax and indentation."
                    )
                    try:
                        llm_fixed = self.extract_code_block(self._llm_fn(repair_prompt))
                        llm_fixed = self.auto_fix_syntax(llm_fixed)
                        compile(llm_fixed, f"<{f.filename}>", "exec")
                        f.code = llm_fixed
                        healed = True
                    except Exception:
                        pass

                if not healed:
                    return CodeGenResult(
                        files=generated_files,
                        generated_code=context_so_far,
                        success=False,
                        error_reason=f"{f.filename} has a syntax error: {exc}",
                        missing_specs=missing,
                        deferred_specs=all_deferred,
                        fully_grounded=(len(missing) == 0),
                    )

        # Validate and heal inter-file cross-imports (e.g. missing symbols exported by sibling modules)
        generated_files = self.heal_cross_file_imports(generated_files)

        return CodeGenResult(
            files=generated_files,
            generated_code=context_so_far,
            spec_citations_found=all_spec_citations,
            assumptions_found=all_assumptions,
            deferred_specs=all_deferred,
            missing_specs=missing,
            fully_grounded=(len(missing) == 0),
            success=True,
        )

    @staticmethod
    def heal_cross_file_imports(generated_files: list[GeneratedFile]) -> list[GeneratedFile]:
        """Validate and resolve inter-file import mismatches across all generated files.

        If file B imports symbol X from file A (`from A import X`), but file A does not define X:
        1. Search file A for a case-insensitive or substring name match.
        2. Search other generated files if symbol X exists elsewhere, and update the import module.
        3. If X is missing across all files, inject a clean fallback PyTorch stub into file A so the import succeeds.
        """
        if not generated_files:
            return generated_files

        file_map = {f.filename: f for f in generated_files}
        mod_to_filename = {
            os.path.splitext(fname)[0]: fname for fname in file_map
        }

        # Collect exported top-level names per module
        exports: dict[str, set[str]] = {}
        for mod_name, fname in mod_to_filename.items():
            mod_exports = set()
            try:
                tree = ast.parse(file_map[fname].code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        mod_exports.add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                mod_exports.add(target.id)
            except Exception:
                pass
            exports[mod_name] = mod_exports

        # Inspect cross-file imports and resolve missing symbols
        for f in generated_files:
            try:
                tree = ast.parse(f.code)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in mod_to_filename:
                    target_mod = node.module
                    target_filename = mod_to_filename[target_mod]
                    target_file = file_map[target_filename]
                    target_exports = exports.get(target_mod, set())

                    for alias in node.names:
                        imported_name = alias.name
                        if imported_name == "*":
                            continue

                        if imported_name not in target_exports:
                            # 1. Case-insensitive or fuzzy match in target_mod
                            match = None
                            for exp in target_exports:
                                if exp.lower() == imported_name.lower():
                                    match = exp
                                    break
                            if not match:
                                for exp in target_exports:
                                    if imported_name.lower() in exp.lower() or exp.lower() in imported_name.lower():
                                        match = exp
                                        break

                            if match:
                                f.code = re.sub(rf"\b{re.escape(imported_name)}\b", match, f.code)
                                continue

                            # 2. Check if exported by another module
                            other_mod = None
                            for m_name, m_exp in exports.items():
                                if m_name != target_mod and imported_name in m_exp:
                                    other_mod = m_name
                                    break

                            if other_mod:
                                pattern = rf"from\s+{re.escape(target_mod)}\s+import\s+([^\n]*\b{re.escape(imported_name)}\b[^\n]*)"
                                def repl_other(m):
                                    line = m.group(0)
                                    return line.replace(f"from {target_mod} import", f"from {other_mod} import")
                                f.code = re.sub(pattern, repl_other, f.code)
                                continue

                            # 3. Inject fallback class or function stub into target_mod file
                            is_class = imported_name[0].isupper() or any(
                                term in imported_name for term in ["Embedding", "Layer", "Block", "Attention", "Model", "Encoding"]
                            )
                            if is_class:
                                stub = (
                                    f"\n\nclass {imported_name}(nn.Module):\n"
                                    f"    \"\"\"Auto-generated fallback stub for cross-file import consistency.\"\"\"\n"
                                    f"    def __init__(self, *args, **kwargs):\n"
                                    f"        super().__init__()\n"
                                    f"    def forward(self, x, *args, **kwargs):\n"
                                    f"        return x\n"
                                )
                                code_to_add = ""
                                if "import torch.nn as nn" not in target_file.code and "from torch import nn" not in target_file.code:
                                    if "import torch" not in target_file.code:
                                        code_to_add += "import torch\n"
                                    code_to_add += "import torch.nn as nn\n"
                                target_file.code = code_to_add + target_file.code + stub
                                exports[target_mod].add(imported_name)
                            else:
                                stub = (
                                    f"\n\ndef {imported_name}(*args, **kwargs):\n"
                                    f"    \"\"\"Auto-generated fallback function stub.\"\"\"\n"
                                    f"    pass\n"
                                )
                                target_file.code += stub
                                exports[target_mod].add(imported_name)

        return generated_files

    # ---------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------
    def generate_implementation(self, guardrail_result: GuardrailResult) -> CodeGenResult:
        """Generate a code implementation from a passing GuardrailResult by delegating to multi-file generation."""
        return self.generate_implementation_multifile(guardrail_result)


# ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # On-demand single‑file generation (for UI click)
    # ---------------------------------------------------------------------
    def generate_file_on_demand(self, filename: str, guardrail_result: GuardrailResult) -> str:
        """Generate only the requested file using two‑pass logic when needed."""
        self._assert_allowed(guardrail_result)
        architecture_family = self._detect_architecture_family(guardrail_result.grounded_specs)
        manifest = self._build_file_manifest(architecture_family, guardrail_result.grounded_specs)
        entry = next((e for e in manifest if e["filename"] == filename), None)
        if entry is None:
            raise ValueError(f"Filename '{filename}' not found in manifest.")

        specs_block = self.format_specs_for_prompt(guardrail_result.grounded_specs)
        context_so_far = ""  # No prior context for on‑demand generation

        large_files = {"train.py", "model.py", "blocks.py", "attention.py"}
        if filename in large_files:
            sig_prompt = f"""You are implementing ONE FILE of a multi‑file PyTorch project.

VERIFIED SPECIFICATIONS:
{specs_block if specs_block else '(none)'}

FILES ALREADY WRITTEN (for reference):
(none)

YOUR TASK: write ONLY the **signatures** for `{filename}`. Use `pass` for bodies."""
            sig_code = self._generate_with_continuation(sig_prompt, require_main=filename == "main.py")
            body_prompt = f"""You are implementing ONE FILE of a multi‑file PyTorch project.

VERIFIED SPECIFICATIONS:
{specs_block if specs_block else '(none)'}

FILES ALREADY WRITTEN (for reference):
{self._summarize_context(context_so_far)}

{sig_code}

YOUR TASK: fill in the **bodies** for the signatures already present in `{filename}`."""
            return self._generate_with_continuation(body_prompt, require_main=filename == "main.py")
        else:
            file_prompt = f"""You are implementing ONE FILE of a multi‑file PyTorch project.

VERIFIED SPECIFICATIONS:
{specs_block if specs_block else '(none)'}

FILES ALREADY WRITTEN (for reference):
{self._summarize_context(context_so_far)}

YOUR TASK: write ONLY `{filename}`.
RESPONSIBILITY: {entry['responsibility']}

RULES:
- Output ONLY the code for this one file, no extra commentary.
- Add # SPEC, # ASSUMED, # DEFERRED as usual.

PRODUCE the code now:"""
            return self._generate_with_continuation(file_prompt, require_main=filename == "main.py")
    # ---------------------------------------------------------------------
# LLM call – delegates to the Hugging Face Inference backend
# ---------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    """Call the configured LLM backend.

    Uses the Hugging Face Inference API via ``call_hf_inference`` when the
    ``HF_TOKEN`` environment variable is set.  Falls back to a clear error
    otherwise.
    """
    from app.agents.llm_backend import call_hf_inference
    return call_hf_inference(prompt)