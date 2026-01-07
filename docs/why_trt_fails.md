# Why TensorRT Fails (And Why Errors Are Hard to Debug)

TensorRT failures are rarely caused by a single issue.
They usually emerge from **graph semantics**, **shape constraints**, or **unsupported patterns**.

This document explains the most common causes.

---

## 1. Unsupported Operators

TensorRT does not support all ONNX operators natively.

Examples:
- Einsum
- LayerNormalization
- Control-flow ops (Loop, Scan)

These ops often appear in models exported from PyTorch or JAX.

---

## 2. Einsum Is Specially Problematic

Einsum represents **arbitrary tensor contractions**.

Why TensorRT struggles:
- No fixed kernel selection
- Dynamic reduction axes
- Multiple equivalent contraction paths

Some Einsum patterns *can* be rewritten safely.
Others require plugins.

---

## 3. Dynamic Shapes

TensorRT prefers:
- Static shapes
- Or bounded dynamic profiles

Failures occur when:
- Shapes are fully symbolic
- Reductions depend on runtime dimensions

---

## 4. Precision Sensitivity

Some ops behave poorly in FP16 / INT8:
- LayerNorm
- ReduceMean
- Variance-based ops

TensorRT may reject these or produce unstable results.

---

## 5. Plugin Gaps

TensorRT expects:
- Explicit plugin registration
- Correct format support
- Proper serialization

Missing any of these causes opaque errors.

---

## Why Errors Are Opaque

TensorRT error messages are:
- Parser-level
- Not graph-aware
- Lacking semantic context

Example:

What it *should* say:
> This Einsum pattern cannot be lowered to a fixed contraction and requires a plugin.

---

## How trt-ai-fixer Helps

- Detects failure points *before* engine build
- Explains the structural reason for failure
- Suggests ranked fixes with trade-offs

This turns TensorRT debugging from trial-and-error into analysis.
