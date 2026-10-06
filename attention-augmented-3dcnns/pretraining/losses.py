import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Compute NT-Xent (InfoNCE) loss for a batch of positive pairs (z1, z2).
    Args:
        z1, z2: [B, D] normalized embeddings
        temperature: float
    Returns:
        Scalar loss
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]
    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels, labels])
    mask = ~torch.eye(2*N, dtype=torch.bool, device=z1.device)
    sim = sim.masked_select(mask).view(2*N, -1)
    positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
    logits = sim
    targets = torch.arange(N, device=z1.device)
    targets = torch.cat([targets, targets])
    loss = F.cross_entropy(logits, targets)
    return loss 
