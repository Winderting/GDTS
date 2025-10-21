import torch
import torch.nn.functional as F
import numpy as np
from src.models.common import PositionalEncoding, ConcatSquashLinear
import math

# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    import math
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class VarianceSchedule(torch.nn.Module):
    # control the beta increase from t=1 to t=T, check diffusion principle for more details.
    # num_steps: diffusion step, 100 in this paper
    # beta_1: initial beta
    # beta_T: final beta
    # consine_s: 
    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding, betas size is (num_steps+1,) = (101,) 100 is the diffusion iterative step

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        # select (b,) tensor from the array np.arange(...) 
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0.0):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class TransformerConcatLinear(torch.nn.Module):

    def __init__(self, context_dim, tf_layer):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(2,2*context_dim,context_dim+3)
        self.layer = torch.nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)


    def forward(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        x = self.concat1(ctx_emb,x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)


        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)
    
# class TransformerConcatLinear(torch.nn.Module):
#     def __init__(self, context_dim, tf_layer):
#         super().__init__()
#         self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
#         self.concat1 = ConcatSquashLinear(2,2*context_dim,context_dim+3)
#         self.layer = torch.nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=4*context_dim) # single Transformer Encoder layer, input [T,B,context_dim]
#         self.transformer_encoder = torch.nn.TransformerEncoder(self.layer, num_layers=tf_layer) # a stack of N encoder layers, N=3 self-attention layers

#         self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
#         self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
#         self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)

#         self.context_dim = context_dim


#     def forward(self, x, beta, context):
#         '''
#         x: refinement, (B, Tp, 2)
#         context: (B, 1, context_dim = 4 * To )'''
#         batch_size = x.size(0)
#         beta = beta.view(batch_size, 1, 1)          # (B, 1, 1) time 
#         context = context.view(batch_size, 1, -1)   # (B, 1, context_dim) the encoding from encoder
#         time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3), calculate the time embedding, 'k' in the figure of paper
#         # print(time_emb.device, context.device, x.device)
#         ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, context_dim+3), concatenate it with the feature of the observed trajectory (the encoding from encoder)
#         x = self.concat1(ctx_emb,x) # upsample and sum up the output as the fused features ctx_emb:(B,1,context_dim+3), x:(B,12,2) -> x:(B,12,2*context_dim)
#         final_emb = x.permute(1,0,2) # (12,B,2*context_dim)
#         final_emb = self.pos_emb(final_emb)  # position embedding of the fused features (12,B,2*context_dim)
#         trans = self.transformer_encoder(final_emb).permute(1,0,2) # Transformer network to learn the spatial-temporal clues
#         trans = self.concat3(ctx_emb, trans) # ctx_emb:(B,12,context_dim+3) trans:(B,12,context_dim*2) -> trans: (B,12,context_dim)
#         trans = self.concat4(ctx_emb, trans) # ctx_emb:(B,12,context_dim+3) trans:(B,12,context_dim) -> trans: (B,12,context_dim//2)
#         return self.linear(ctx_emb, trans) # ctx_emb:(B,12,context_dim+3) trans:(B,12,context_dim//2) -> (B,12,2)
