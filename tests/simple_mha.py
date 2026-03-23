import torch.nn as nn
# from torchmorphic import X

class SimpleMHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

    def forward(self, query, key, value):
        return self.attention(query, key, value)
    

model = SimpleMHA()

# diagram = create_string_diagram(model) 
# diagram.draw()