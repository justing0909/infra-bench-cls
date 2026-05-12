from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -9e15)

    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    numerator = torch.exp(positives)
    denominator = torch.exp(sim).sum(dim=1)
    loss = -torch.log(numerator / denominator)
    return loss.mean()
