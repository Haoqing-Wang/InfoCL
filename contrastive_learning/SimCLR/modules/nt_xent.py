import torch
import torch.nn as nn

def gen_mask(k, feat_dim):
    mask = None
    for i in range(k):
        tmp_mask = torch.triu(torch.randint(0, 2, (feat_dim, feat_dim)), 1)
        tmp_mask = tmp_mask + torch.triu(1-tmp_mask,1).t()
        tmp_mask = tmp_mask.view(tmp_mask.shape[0], tmp_mask.shape[1],1)
        mask = tmp_mask if mask is None else torch.cat([mask,tmp_mask],2)
    return mask

def entropy(prob):
    # assume m x m x k input
    return -torch.sum(prob*torch.log(prob),1)

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, mask):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = mask
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.batch_size * 2, 1)
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).long().cuda()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        
        return loss
