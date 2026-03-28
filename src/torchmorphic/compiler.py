import torch.fx as fx
from torchmorphic.extractor import extract_pytorch_graph
from discopy.monoidal import Id
from torchmorphic.registry import TRANSLATION_REGISTRY
from torch import nn

def to_diagram(model: nn.Module):
	traced = fx.symbolic_trace(model)
	nodes = extract_pytorch_graph(traced.graph)
	diagram = None

	for node in nodes:
		# 1. Look up the categorical generator using the generic target string
		if node.target in TRANSLATION_REGISTRY:
			BoxClass = TRANSLATION_REGISTRY[node.target]
			generator = BoxClass(node.name)
		elif node.op_class == "sample":
			# Fallback for generic inputs if not explicitly registered
			generator = TRANSLATION_REGISTRY["placeholder"](node.name)
		else:
			raise ValueError(f"Target '{node.target}' not found in registry.")

		# 2. Compose the diagram
		if diagram is None:
			# First node (usually a sample) starts the diagram
			diagram = generator
		elif node.op_class in ["sample", "init"]:
			# New inputs or parameters are placed in parallel (Tensor Product)
			diagram = diagram @ generator
		elif node.op_class == "transform":
			# Computations are executed sequentially
			# Note: For strict routing with multiple wires, Swap/Id logic goes here.
			diagram = diagram >> generator

	return diagram