
import torch
from informal.utils import split_into_groups


def KLDist(p, q, eps=1e-8):
    log_p, log_q = torch.log(p + eps), torch.log(q + eps)
    return torch.sum(p * (log_p - log_q))

class MeanLoss(torch.nn.Module):
    def __init__(self, base_loss):
        super(MeanLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, pred, gt, domain):
        _, group_indices, _ = split_into_groups(domain)
        total_loss, total_cnt = 0, 0

        losses_per_group = []
        group_loss_list = []
        for i_group in group_indices:
            group_loss = self.base_loss(pred[i_group], gt[i_group])
            total_loss += group_loss
            total_cnt += 1
            losses_per_group.append(group_loss.mean())
            group_loss_list.append(group_loss)
        return total_loss / total_cnt, losses_per_group, group_loss_list