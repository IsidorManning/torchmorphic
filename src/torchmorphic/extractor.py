import torch.nn as nn
import torch.fx as fx

def extract_graph(model: nn.Module) -> fx.Graph:
    """
    Symbolically traces a PyTorch module and returns its intermediate 
    representation graph.
    """
    # symbolic_trace treats sub-modules (like MultiheadAttention) as single nodes,
    # which is ideal for a high-level categorical representation.
    traced_module = fx.symbolic_trace(model)
    return traced_module.graph