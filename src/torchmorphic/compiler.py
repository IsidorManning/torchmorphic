import torch.fx as fx
from discopy.symmetric import Id, Swap
from torch import nn

from torchmorphic.extractor import extract_pytorch_graph
from torchmorphic.registry import TRANSLATION_REGISTRY


def route_wire_to_right(diagram, active_wires: list[str], wire_index: int):
    """
    Bubbles a wire to the far right of the diagram and updates the active_wires list.
    """
    if wire_index >= len(diagram.cod) - 1:
        return diagram, active_wires

    current_idx = wire_index
    while current_idx < len(diagram.cod) - 1:
        # 1. Build the mathematical Swap layer
        left_type = diagram.cod[:current_idx]
        wire_type = diagram.cod[current_idx : current_idx + 1]
        next_type = diagram.cod[current_idx + 1 : current_idx + 2]
        right_type = diagram.cod[current_idx + 2 :]

        swap_layer = Swap(wire_type, next_type)
        if left_type:
            swap_layer = Id(left_type) @ swap_layer
        if right_type:
            swap_layer = swap_layer @ Id(right_type)

        # 2. Apply to the diagram
        diagram = diagram >> swap_layer

        # 3. Mirror the swap in our tracking list
        active_wires[current_idx], active_wires[current_idx + 1] = (
            active_wires[current_idx + 1],
            active_wires[current_idx],
        )

        current_idx += 1

    return diagram, active_wires


def to_diagram(model: nn.Module):
    traced = fx.symbolic_trace(model)
    nodes = extract_pytorch_graph(traced.graph)

    diagram = None
    active_wires = []  # Tracks the names of the wires hanging at the bottom

    for node in nodes:
        # --- 1. Fetch Operation ---
        if node.target in TRANSLATION_REGISTRY:
            box = TRANSLATION_REGISTRY[node.target](node.name)
        elif node.op_class == "sample":
            box = TRANSLATION_REGISTRY["placeholder"](node.name)
        else:
            raise ValueError(f"Target '{node.target}' not found in registry.")

        # --- 2. Initial Setup ---
        if diagram is None:
            diagram = box
            active_wires.append(node.name)
        elif node.op_class in ["sample", "init"]:
            diagram = diagram @ box
            active_wires.append(node.name)

        # --- 3. The Transformation & Routing ---
        elif node.op_class == "transform":
            # A. Parameter Injection
            needs_param = any(t.name == "P" for t in box.dom.inside)
            if needs_param:
                param_name = f"θ_{node.name}"
                diagram = diagram @ TRANSLATION_REGISTRY["init"](param_name)
                active_wires.append(param_name)
                # We artificially add the parameter to the required inputs
                node.inputs.append(param_name)

            # B. Route required inputs to the right
            # Resolve multi-wire inputs (tuples) from PyTorch into individual Proc wires
            resolved_inputs = []
            for input_name in node.inputs:
                if input_name in active_wires:
                    resolved_inputs.append(input_name)
                elif f"{input_name}_0" in active_wires:
                    # It's a tuple! Collect all wires belonging to this output
                    i = 0
                    while f"{input_name}_{i}" in active_wires:
                        resolved_inputs.append(f"{input_name}_{i}")
                        i += 1
                else:
                    raise ValueError(
                        f"'{input_name}' not found in active wires: {active_wires}"
                    )

            # Route the resolved inputs in order so they line up perfectly for the box
            for input_name in resolved_inputs:
                # Find where the wire currently is
                wire_idx = active_wires.index(input_name)
                # Bubble it to the far right
                diagram, active_wires = route_wire_to_right(
                    diagram, active_wires, wire_idx
                )

            # C. Apply the operation
            left_type = diagram.cod[: -len(box.dom)]
            spatial_layer = Id(left_type) @ box if left_type else box
            diagram = diagram >> spatial_layer

            # D. Update active wires (remove consumed inputs, add the new output)
            active_wires = active_wires[: -len(box.dom)]
            # NEW: Add a name for EVERY wire in the output (codomain)
            # If a box outputs T @ T, we add 'node.name_0', 'node.name_1'
            if len(box.cod) > 1:
                for i in range(len(box.cod)):
                    active_wires.append(f"{node.name}_{i}")
            else:
                active_wires.append(node.name)

        # --- 4. Immediate Fan-Out (Copy) ---
        if node.fan_out > 1:
            copy_box = TRANSLATION_REGISTRY["copy"](node.fan_out)
            left_type = diagram.cod[:-1]
            spatial_layer = Id(left_type) @ copy_box if left_type else copy_box
            diagram = diagram >> spatial_layer

            # Update tracking list: replace the single wire with 'k' copies
            base_name = active_wires.pop()
            active_wires.extend([base_name] * node.fan_out)

    return diagram
