from typing import List, Dict, Optional


class IRNode:
    def __init__(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Dict,
        dynamic: bool = False,
        shape: Optional[Dict] = None,
    ):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes
        self.dynamic = dynamic
        self.shape = shape


class IRGraph:
    def __init__(self, nodes: List[IRNode], model_info: Dict):
        self.nodes = nodes
        self.model_info = model_info

        # consumer map for pattern analysis (read-only)
        self._consumers: Dict[str, List[IRNode]] = {}
        for n in nodes:
            for t in n.inputs:
                self._consumers.setdefault(t, []).append(n)

        # 🔑 CRITICAL: attach back-reference
        for n in self.nodes:
            n.graph = self

    def consumers_of(self, tensor: str) -> List[IRNode]:
        return self._consumers.get(tensor, [])
