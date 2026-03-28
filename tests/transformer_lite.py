import operator

import torch.nn as nn

from torchmorphic.compiler import to_diagram


class TransformerLite(nn.Module):
    """
    A formal string diagram of a Transformer Encoder block.
    Demonstrates sequential composition of spatial layers.
    """

    def __init__(self):
        super().__init__()
        # Naming matches the registry targets exactly
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.linear = nn.Linear(256, 256)

    def forward(self, x):
        # --- Block 1: Self-Attention ---
        res1 = x
        attn_out = self.attention(x, x, x)[0]
        x = operator.add(res1, attn_out)

        # --- Block 2: Feed-Forward ---
        res2 = x
        ff_out = self.linear(x)
        out = operator.add(res2, ff_out)

        return out


if __name__ == "__main__":
    model = TransformerLite()
    diagram = to_diagram(model)

    print("\n--- Transformer Lite Block ---")
    print(f"Domain (Inputs): {diagram.dom}")
    print(f"Codomain (Outputs): {diagram.cod}")

    diagram.draw(figsize=(10, 14), path="./transformer_diagram_2.png")
