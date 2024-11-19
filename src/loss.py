from torch import nn
from LibMTL.loss import AbsLoss

class BCELoss(AbsLoss):
    r"""The cross-entropy loss function.
    """
    def __init__(self, factor=1):
        super(BCELoss, self).__init__()
        
        self.loss_fn = nn.BCELoss()
        self.factor = factor
        
    def compute_loss(self, pred, gt):
        r"""
        """
        try:
            loss = self.loss_fn(pred, gt) * self.factor
        except Exception as e:
            print(f'pred: {pred}')
            print(f'pred shape: {pred.shape}')
            print(f'ground truth: {gt}')
            print(f'ground truth shape: {gt.shape}')
            print(f'Max gt: {gt.max()}')
            print(e.with_traceback())
        return loss