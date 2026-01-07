# Demo: Fixing an Einsum TensorRT Failure

This demo shows how `trt-ai-fixer` analyzes a model
that fails TensorRT engine build due to an Einsum operator.

---

## Step 1: Analyze the model

```bash
trt-fix analyze models/einsum_fail.onnx --out outputs/einsum_analysis.json
```
## Step 2: Understand the failure

The tool detects:

Einsum is not natively supported

The contraction pattern is rewriteable

## Step 3: Review suggestions

Recommended fix:

Rewrite Einsum into a batched MatMul

Alternative:

TensorRT plugin

## Result

The developer now understands:

Why TensorRT failed

What can be fixed automatically

What requires manual intervention


This makes the repo **self-explanatory**.

---

# 🧠 Why `examples/` matters
- Shows **real usage**
- Proves **end-to-end flow**
- Makes reviewers trust the tool
- Enables quick demos without setup

This is the difference between:
> “Interesting idea”  
and  
> “Oh wow, this actually works.”

---

# What you should do now

1. Create the folders:
```bash
mkdir -p examples/{models,outputs}
