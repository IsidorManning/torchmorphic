import operator

import torch.nn as nn

from torchmorphic.compiler import to_diagram


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # ONE shared parameter space
        self.shared_layer = nn.Linear(128, 128)

    def forward(self, x1, x2):
        # Two independent data streams using the same layer
        out1 = self.shared_layer(x1)
        out2 = self.shared_layer(x2)

        # Merge them (e.g., to calculate a distance metric)
        return operator.add(out1, out2)


if __name__ == "__main__":
    model = SiameseNetwork()

    diagram = to_diagram(model)

    # Output properties for verification
    print(f"Domain (Inputs): {diagram.dom}")
    print(f"Codomain (Outputs): {diagram.cod}")

    # Draws the string diagram
    diagram.draw(figsize=(6, 6), path="./siamese.png")
