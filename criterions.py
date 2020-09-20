import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothKLDivLoss(nn.Module):
    def __init__(self, epsilon=0.1, nclass=1000, device="cpu"):
        super(LabelSmoothKLDivLoss, self).__init__()
        self.epsilon = epsilon
        self.nclass = nclass
        self.device = device
        self.pb = epsilon / (nclass-1)
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def __call__(self, x, y):
        N = x.size(0)
        with torch.no_grad():
            target = torch.zeros(N, self.nclass, device=self.device) + self.pb
            for i, cls in enumerate(y):
                target[i, cls] = 1.0 - self.epsilon
        log_prob = F.log_softmax(x, dim=1)
        loss = self.kl_div(log_prob, target)
        return loss / N
