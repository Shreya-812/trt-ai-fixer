# Roadmap

This roadmap reflects realistic, engineering-focused growth.

---

## Phase 1 — Foundation (Completed)

- ONNX graph scanner
- TensorRT compatibility detection
- Rewrite feasibility analysis
- Human-readable explanations
- Ranked fix suggestions
- CLI interface
- Test coverage

---

## Phase 2 — Preventive Execution

- Apply graph rewrites directly to ONNX
- Generate rewritten ONNX artifacts
- Output parity validation
- Precision override injection

---

## Phase 3 — Plugin Assistance

- Auto-generate TensorRT plugin skeletons
- CMake + build scripts
- Python registration helpers
- Reference CPU implementations for validation

---

## Phase 4 — Runtime Integration

- TensorRT dry-run engine builds
- Capture real parser errors
- Correlate runtime failures with graph nodes

---

## Phase 5 — AI Assistance (Optional)

- Explain failures using LLM reasoning
- Suggest optimal rewrite strategies
- Assist in CUDA kernel drafting (human-reviewed)

---

## Phase 6 — Ecosystem Expansion

- TensorRT 9+ support
- TorchScript input
- TVM / OpenVINO backends
- Visualization UI

---

## Guiding Principle

Automation should **reduce developer effort**, not hide complexity.
Every step must remain explainable and reversible.
