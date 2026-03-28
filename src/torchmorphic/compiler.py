import torch.fx as fx
import torch
from discopy.monoidal import Id
from torchmorphic.registry import TRANSLATION_REGISTRY, Sample, Attention, register_translation

# --- Core Compiler ---

def create_string_diagram(model: torch.nn.Module):
    """
    Transpiles a PyTorch model into a DisCoPy string diagram.
    """
    from torchmorphic.extractor import extract_graph
    graph = extract_graph(model)
    
    diagram = None
    
    # MVP assumption: Nodes are processed in topological order.
    # We build the diagram layer by layer using the tensor product (@) for parallel 
    # inputs, and sequential composition (>>) for operations.
    for node in graph.nodes:
        if node.op == "placeholder":
            generator = TRANSLATION_REGISTRY["placeholder"](node)
            # Tensor product (@) stacks inputs in parallel
            diagram = generator if diagram is None else diagram @ generator
            
        elif node.op == "call_module":
            # In FX, node.target is the string name of the module attribute
            if node.target in TRANSLATION_REGISTRY:
                generator = TRANSLATION_REGISTRY[node.target](node)
                # Sequential composition (>>) feeds the accumulated inputs into the module
                diagram = diagram >> generator
            else:
                raise NotImplementedError(f"Module '{node.target}' not registered.")
                
        elif node.op == "output":
            # For the MVP, we assume the previous node contains the final state
            pass 

    return diagram