import operator

import torch.nn as nn

from torchmorphic.compiler import to_diagram


class VerifiableResidual(nn.Module):
    """
    A strict mathematical representation of x -> x + Linear(x).
    Demonstrates spatial routing, Fan-Out (Copy), and Fan-In (Add).
    """

    def __init__(self):
        super().__init__()
        # We name this 'linear' to perfectly match your registry target
        self.linear = nn.Linear(128, 128)

    def forward(self, x):
        # 1. The transpiler detects this as a Fan-Out (Copy)
        residual = x

        # 2. The transpiler routes this using Id @ Linear
        out = self.linear(x)

        # 3. The transpiler detects this as a Fan-In (Add)
        return operator.add(residual, out)


if __name__ == "__main__":
    model = VerifiableResidual()
    diagram = to_diagram(model)

    print("\n--- Verifiable Residual Block ---")
    print(f"Domain (Inputs): {diagram.dom}")
    print(f"Codomain (Outputs): {diagram.cod}")

    diagram.draw(figsize=(8, 8), path="./residual_diagram.png")
    print("Saved to ./residual_diagram.png")
