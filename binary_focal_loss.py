import torch
from torch import nn
import torch.nn.functional as F


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='sum'):
        super(BinaryFocalWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        probs = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - probs) ** self.gamma * BCE_loss
        # print("F_loss:", F_loss.shape)

        if self.reduction == "sum":
            # print("F_loss:", F_loss.shape)
            # print("F_loss mean:", torch.mean(F_loss))
            # print("F_loss sum:", torch.sum(F_loss))
            # exit(0)
            return torch.sum(F_loss)
        else:
            return torch.mean(F_loss)

if __name__ == "__main__":
    print(__doc__)