# trt-ai-fixer — Architecture

## Overview

`trt-ai-fixer` is a compiler-style analysis and remediation tool designed to
identify, explain, and mitigate TensorRT incompatibilities in deep learning models.

Instead of failing late during engine build, the tool performs **static graph analysis**
and provides **preventive fixes** or **actionable guidance**.

---

## High-level Pipeline


Each stage is isolated, deterministic, and testable.

---

## Stage 1: Scan

**Purpose:** Understand the model graph without making judgments.

- Parses ONNX models
- Performs best-effort shape inference
- Builds a Model Intelligence Graph (MIG)

**Output:** Structured intermediate representation (IR)

---

## Stage 2: Detect

**Purpose:** Determine TensorRT supportability per node.

Classification buckets:
- `native_supported`
- `rewrite_supported`
- `plugin_required`
- `blocked`
- `unknown`

Detection is:
- Version-aware
- Rule-based
- Explainable

---

## Stage 3: Prevent

**Purpose:** Prevent TensorRT failures where possible.

Strategies:
- Graph-level rewrites (e.g. Einsum → MatMul)
- Precision-level fixes (FP16 → FP32)
- Shape stabilization

This stage produces **action plans**, not silent mutations.

---

## Stage 4: Explain

**Purpose:** Translate low-level TensorRT failures into human-readable reasons.

- Parses TensorRT error messages
- Correlates errors with graph structure
- Explains *why* a node fails, not just *that* it fails

---

## Stage 5: Suggest

**Purpose:** Guide the developer toward the best fix.

Suggestions are:
- Ranked by feasibility and impact
- Explicit about trade-offs
- Honest about engineering cost

---

## Design Principles

- Deterministic over black-box AI
- Explainability over automation
- Safe defaults over aggressive rewrites
- Extensible architecture

---

## Non-Goals

- Replacing TensorRT
- Automatically generating CUDA kernels without review
- Runtime benchmarking (handled externally)

---

## Extensibility

The architecture allows future support for:
- TensorRT 9+
- TorchScript
- Other runtimes (TVM, OpenVINO)
- Auto-generated plugins
