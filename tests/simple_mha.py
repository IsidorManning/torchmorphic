import torch.nn as nn
from torchmorphic.compiler import create_string_diagram

class SimpleMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

    def forward(self, query, key, value):
        # PyTorch MHA returns a tuple (attn_output, attn_output_weights).
        # We index [0] to keep the graph simple for the MVP.
        return self.attention(query, key, value)[0]

if __name__ == "__main__":
    model = SimpleMHA()
    
    diagram = create_string_diagram(model)
    
    # Output properties for verification
    print(f"Domain (Inputs): {diagram.dom}")
    print(f"Codomain (Outputs): {diagram.cod}")
    
    # Draws the string diagram
    diagram.draw(figsize=(6, 6), path="./mha_diagram.png")