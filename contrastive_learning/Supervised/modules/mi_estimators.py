import numpy as np
import torch


# Lower bound
def InfoNCE(mu, z):
    mu = mu.unsqueeze(0)
    z = z.unsqueeze(1)
    score = -((z-mu)**2).sum(-1)/20.
    lower_bound = -score.logsumexp(dim=1).mean()
    return lower_bound

# def InfoNCE(mu, z):
#     AA = (mu*mu).sum(1).unsqueeze(0)
#     BB = (z*z).sum(1, keepdims=True)
#     AB = torch.mm(z, mu.transpose(0, 1))
#     score = -(AA-2.*AB+BB)/20.
#     lower_bound = -score.logsumexp(dim=1).mean()
#     return lower_bound