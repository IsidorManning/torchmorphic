from dataclasses import dataclass

import torch.fx as fx


@dataclass
class MorphicNode:
    """A framework-agnostic representation of a computational step."""

    name: str  # Unique identifier (e.g., 'x', 'linear_1')
    op_class: str  # The category of operation: 'sample', 'init', or 'transform'
    target: str  # The specific operation name (e.g., 'attention', 'relu')
    inputs: list[str]  # Names of the nodes feeding into this one
    fan_out: int  # NEW: How many future nodes use this?


def extract_pytorch_graph(graph: fx.Graph) -> list[MorphicNode]:
    generic_nodes = []

    # 1. Accurately count how many times each tensor is used
    usage_counts = {node.name: 0 for node in graph.nodes}
    for node in graph.nodes:

        def count_uses(arg):
            if isinstance(arg, fx.Node):
                usage_counts[arg.name] += 1
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    count_uses(a)

        count_uses(node.args)
        count_uses(node.kwargs)

    # 2. Build the intermediate representation
    for node in graph.nodes:
        inputs = []

        def get_inputs(arg):
            if isinstance(arg, fx.Node):
                inputs.append(arg.name)
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    get_inputs(a)

        get_inputs(node.args)
        get_inputs(node.kwargs)

        fan_out = usage_counts[node.name]

        if node.op == "placeholder":
            generic_nodes.append(
                MorphicNode(node.name, "sample", "input", inputs, fan_out)
            )
        elif node.op == "get_attr":
            generic_nodes.append(
                MorphicNode(node.name, "init", "weight", inputs, fan_out)
            )
        elif node.op in ["call_module", "call_method"]:
            target_name = str(node.target).split(".")[-1]
            generic_nodes.append(
                MorphicNode(node.name, "transform", target_name, inputs, fan_out)
            )
        elif node.op == "call_function":
            target_name = str(node.target)
            if "getitem" in target_name:
                generic_nodes.append(
                    MorphicNode(node.name, "transform", "projection", inputs, fan_out)
                )
            elif "add" in target_name:
                generic_nodes.append(
                    MorphicNode(node.name, "transform", "add", inputs, fan_out)
                )
            else:
                generic_nodes.append(
                    MorphicNode(node.name, "transform", target_name, inputs, fan_out)
                )

    return generic_nodes
