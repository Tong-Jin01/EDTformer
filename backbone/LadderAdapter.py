import torch.nn as nn

class SideAdapter(nn.Module):
    def __init__(self, D_features, D_hidden_features=4, act_layer=nn.GELU, skip_connect=True, alpha=0.5):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.alpha = alpha

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + self.alpha * xs
        else:
            x = xs
        return x
