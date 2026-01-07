import argparse
import sys

from trt_fixer.scanner.onnx_scanner import scan_onnx_model
from trt_fixer.detector.support_db import TensorRTSupportDB
from trt_fixer.detector.classify_node import classify_node
from trt_fixer.explain.failure_reasoner import explain_failure
from trt_fixer.suggest.fix_ranker import rank_fixes

from trt_fixer.prevent.rewrite_plan import build_rewrite_plan
from trt_fixer.prevent.rewrite_engine import apply_rewrites


# --------------------------------------------------
# ANALYZE
# --------------------------------------------------
def analyze_model(model_path: str, trt_version: str):
    graph = scan_onnx_model(model_path)
    support_db = TensorRTSupportDB(trt_version)

    print(f"[INFO] Scanning model: {model_path}")

    for node in graph.nodes:
        result = classify_node(node, support_db)
        status = result["status"]

        print(f"[INFO] [{node.name}] {status}")

        if status != "native_supported":
            explanation = explain_failure(node, trt_version)
            suggestions = rank_fixes(node)

            print("  Explanation:", explanation["explanation"])
            print("  Suggestions:")
            for s in suggestions:
                print(f"   - ({s['rank']}) {s['type']}: {s['summary']}")


# --------------------------------------------------
# APPLY
# --------------------------------------------------
def apply_fixes(model_path: str):
    print(f"[INFO] Scanning model: {model_path}")

    graph = scan_onnx_model(model_path)
    plan = build_rewrite_plan(graph)

    if not plan:
        print("[INFO] No rewrites applicable. Model left unchanged.")
        return

    print("[INFO] Rewrite plan:")
    for step in plan:
        print(f" - {step['op']} @ {step['node_name']} ({step['strategy']})")

    out_path = apply_rewrites(model_path, plan)

    print(f"[SUCCESS] Rewritten model saved to:")
    print(f"  {out_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="trt-fix",
        description="AI-assisted TensorRT compatibility fixer",
    )

    sub = parser.add_subparsers(dest="command")

    # analyze
    analyze = sub.add_parser("analyze", help="Analyze TensorRT compatibility")
    analyze.add_argument("model", help="Path to ONNX model")
    analyze.add_argument(
        "--trt-version",
        default="TensorRT-8.6",
        help="TensorRT version",
    )

    # apply
    apply = sub.add_parser("apply", help="Apply safe ONNX rewrites")
    apply.add_argument("model", help="Path to ONNX model")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_model(args.model, args.trt_version)

    elif args.command == "apply":
        apply_fixes(args.model)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
