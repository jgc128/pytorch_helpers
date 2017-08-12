import torch
import torch.nn
import torch.nn.functional as F


class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, masks):
        loss_bce = self.bce_loss(outputs, masks)

        return loss_bce
