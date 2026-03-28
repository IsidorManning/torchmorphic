import torch.fx as fx
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class MorphicNode:
    """A framework-agnostic representation of a computational step."""
    name: str           # Unique identifier (e.g., 'x', 'linear_1')
    op_class: str       # The category of operation: 'sample', 'init', or 'transform'
    target: str         # The specific operation name (e.g., 'attention', 'relu')
    inputs: list[str]   # Names of the nodes feeding into this one 


def extract_pytorch_graph(graph: fx.Graph) -> list[MorphicNode]:
	generic_nodes = []

	for node in graph.nodes:
		inputs = [n.name for n in node.all_input_nodes]

		if node.op == "placeholder":
			generic_nodes.append(MorphicNode(node.name, "sample", "input", inputs))
			
		elif node.op == "get_attr":
			generic_nodes.append(MorphicNode(node.name, "init", "weight", inputs))
			
		elif node.op in ["call_module", "call_method"]:
			# node.target holds the name of the operation (e.g., 'attention')
			target_name = str(node.target).split('.')[-1] 
			generic_nodes.append(MorphicNode(node.name, "transform", target_name, inputs))
			
		elif node.op == "call_function":
			target_name = str(node.target)

			if "getitem" in target_name:
				# Option A: Label it as a structural projection
				generic_nodes.append(MorphicNode(node.name, "transform", "projection", inputs))
			else:
				# Option B: Treat it as a standard function
				generic_nodes.append(MorphicNode(node.name, "transform", target_name, inputs))
            
	return generic_nodes