from typing import Callable, Any
# Assume DisCoPy box imports go here

# A dictionary mapping PyTorch node targets to your translation functions
TRANSLATION_REGISTRY: dict[str, Callable] = {}

def register_translation(target_name: str):
    """Decorator to easily add new node translations."""
    def decorator(func: Callable):
        TRANSLATION_REGISTRY[target_name] = func
        return func
    return decorator

# Example: Registering a standard linear layer
@register_translation("torch.nn.modules.linear.Linear")
def translate_linear(node) -> Any:
    # Logic to return a DisCoPy Box representing linear transformation
    pass

# Example: Registering a tensor addition
@register_translation("torch.add")
def translate_add(node) -> Any:
    # Logic to return a DisCoPy Box for residual connections
    pass