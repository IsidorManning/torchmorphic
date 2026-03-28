from typing import Any, Callable, Dict
from discopy.monoidal import Ty, Box

# --- Object Types (T) ---
C = Ty('C') # Context / Abstract Source
T = Ty('T') # Tensor Workspace
P = Ty('P') # Trainable Parameter

# --- Registration ---
TRANSLATION_REGISTRY: Dict[str, Callable] = {}

def register_translation(target_name: str):
    """Decorator to map PyTorch node targets directly to DisCoPy Box classes."""
    def decorator(cls: Callable):
        TRANSLATION_REGISTRY[target_name] = cls
        return cls
    return decorator

# --- Generators (G) ---

@register_translation("placeholder")
class Sample(Box):
    """Lifts abstract context into a tensor."""
    def __init__(self, node_or_name: Any):
        # Extract the string name if a torch.fx.Node is passed
        name = node_or_name.name if hasattr(node_or_name, 'name') else str(node_or_name)
        super().__init__(name, dom=C, cod=T)  # type: ignore

@register_translation("linear")
class Linear(Box):
    def __init__(self, name: str):
        # This box only cares about the CATEGORICAL signature.
        # It doesn't know if the data came from Torch or JAX.
        super().__init__(name, dom=T @ P, cod=T) # type: ignore

@register_translation("attention")
class Attention(Box):
    """Standard Multi-Head Attention operation."""
    def __init__(self, node_or_name: Any):
        name = node_or_name.name if hasattr(node_or_name, 'name') else str(node_or_name)
        super().__init__(name, dom=T @ T @ T, cod=T)  # type: ignore

# --- Execution ---
if __name__ == "__main__":
    print("--- Current Registry Status ---")
    for target, cls in TRANSLATION_REGISTRY.items():
        print(f"Target: '{target}' -> Class: {cls.__name__}")