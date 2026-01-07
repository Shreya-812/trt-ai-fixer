# TRT-Fixer

A TensorRT inference analysis and rewrite tool for fixing unsupported operators
and improving determinism and latency in TensorRT workflows.

---

## Overview

TRT-Fixer is a developer-focused tool that analyzes ONNX models before TensorRT
compilation, detects graph patterns that are known to fail or perform
suboptimally, and applies targeted graph rewrites or fixes.

While modern TensorRT versions (via the Myelin compiler) can compile many
previously unsupported operators, they often prioritize general correctness and
broad compatibility over aggressive latency optimization or explainability.
TRT-Fixer intervenes at the graph level to give developers control, visibility,
and predictability.

---

## Why this tool exists

In real-world TensorRT inference pipelines, developers frequently encounter:

- Unsupported or partially supported operators (e.g. Einsum, GELU, LayerNorm)
- Silent performance regressions caused by conservative compiler decisions
- Opaque TensorRT build errors that are difficult to debug
- Manual and repetitive graph surgery

TRT-Fixer addresses these issues by:

- Detecting problematic subgraphs before TensorRT execution
- Explaining why a pattern may fail or perform poorly
- Applying deterministic, rule-based graph rewrites
- Falling back safely when no fix is applicable

---

## Key capabilities

- ONNX graph scanning and shape inference
- TensorRT risk detection for known operator patterns
- Rewrite rules for:
  - Einsum
  - GELU
  - LayerNorm
- Dry-run analysis mode (no model modification)
- Explainable output for failures and applied fixes
- CLI and Python package support

---

## Installation

```bash
pip install trt-fixer
```
Note:
TensorRT itself is not installed via pip. Ensure TensorRT is available in your
system environment when building engines.

## CLI usage

Scan a model:
```bash
trt-fix model.onnx
```

Dry run (analysis only):
```bash
trt-fix model.onnx --dry-run
```

Explain detected issues:
```bash
trt-fix model.onnx --explain
```

Apply rewrites and export a fixed model:
```bash
trt-fix model.onnx --out model_fixed.onnx
```

## Example output
Detected pattern: Einsum → GELU

TensorRT risk: high (unsupported contraction and FP16 instability)

Applied rewrite: einsum_fusion_v1

Result: TensorRT-compatible graph


## Design philosophy

- Fail early and explain clearly
- Prefer graph-level fixes over runtime hacks
- Deterministic rewrites over heuristic guessing
- Never silently modify user models
- Generated artifacts are not source

## What this tool is NOT

- Not a TensorRT replacement
- Not a generic ONNX optimizer
- Not a benchmarking framework

TRT-Fixer is a diagnostic and corrective layer for TensorRT-bound inference
workflows.

## Roadmap

- Backend abstraction (TensorRT, ONNX Runtime)
- Latency-aware rewrite selection
- AI-assisted decision engine
- Edge and embedded inference support
- Expanded operator coverage

## License

MIT

## Status

Early-stage but functional. APIs and rewrite rules may evolve as the project
matures.


---

### After pasting, run:

```bash
git add README.md
git commit -m "Add project README"
git push
```
