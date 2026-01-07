from typing import List, Dict

from trt_fixer.detector.gelu_patterns import is_erf_gelu_pattern
from trt_fixer.scanner.graph_repr import IRGraph, IRNode


def build_rewrite_plan(graph: IRGraph) -> List[Dict]:
    """
    Inspect IRGraph and produce a rewrite plan.
    """

    plan: List[Dict] = []

    for node in graph.nodes:
        # -------- GELU --------
        if is_erf_gelu_pattern(graph, node):
            plan.append({
                "op": "GELU",
                "node_name": node.name,
                "strategy": "exact_to_approx",
            })

        # -------- EINSUM --------
        elif node.op_type == "Einsum":
            plan.append({
                "op": "EINSUM",
                "node_name": node.name,
                "strategy": "decompose_matmul",
            })

        # -------- LayerNorm --------
        # Explicitly NOT added here (plugin-only)

    return plan
