import torch
import torch.nn as nn
import math

# def derivative_of(input, dt=1, radian=False):
#     device = input.device
#     x = input.clone().detach().cpu()
#     x = x.numpy()

#     dx = np.full_like(x, np.nan)
#     dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)
#     dx = torch.from_numpy(dx).float().to(device)
#     return dx

def derivative_of(input, dt=1, radian=False):
    '''
    input: (T,B,N)
    output: (T,B,N)
    '''
    device = input.device
    x = input.clone().detach()
    # x = x.numpy()
    # x = input
    dx = torch.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt) # (T-2, 2)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    # dx = torch.from_numpy(dx).float().to(device)
    return dx.float().to(device)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # (max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx)) # (B, 1, dim_ctx) -> (B,1, dim_out)
        bias = self._hyper_bias(ctx) # (B, 1, dim_ctx) -> (B,1, dim_out)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias  # element-wise multiplication,(B, 12, dim_out) * (B, 1, dim_out) + (B, 1, dim_out) -> (B, 12, dim_out) 
        return ret
    
    def batch_generate(self, ctx, x):
        # ctx: (B, n, 1, F+3)
        # x: (B, n, T, 2)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret