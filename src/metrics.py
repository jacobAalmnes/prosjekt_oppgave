import LibMTL as mtl
import numpy as np
import torch
import torch.nn as nn
# from torchmetrics.classification import MultilabelAccuracy
from torcheval.metrics import MultilabelAccuracy

class BCEMetric(mtl.metrics.AbsMetric):
    r"""Calculate the Binary Cross-Entropy (BCE) loss.
    """
    def __init__(self):
        super(BCEMetric, self).__init__()
        self.criterion = nn.BCELoss()
        
    def update_fun(self, pred, gt):
        r"""
        Args:
            pred (torch.Tensor): The predicted tensor with shape (batch_size, num_classes).
            gt (torch.Tensor): The ground-truth tensor with shape (batch_size, num_classes).
        """
        print()
        
        loss = self.criterion(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size(0))
        
    def score_fun(self):
        r"""
        Returns:
            list: A list containing the average BCE loss across all iterations.
        """
        avg_loss = sum(self.record) / len(self.record)
        return [avg_loss]

class MAEMetric(mtl.metrics.AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """
    def __init__(self):
        super(MAEMetric, self).__init__()
        
    def update_fun(self, pred, gt):
        r"""
        Args:
            pred (torch.Tensor): The predicted tensor with shape (batch_size, num_classes).
            gt (torch.Tensor): The ground-truth tensor with shape (batch_size, num_classes).
        """
        abs_err = torch.abs(pred - gt).mean()
        self.record.append(abs_err.item())
        self.bs.append(pred.size(0))
        
    def score_fun(self):
        r"""
        Returns:
            list: A list containing the average MAE across all iterations.
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records * batch_size).sum() / (sum(batch_size))]


class MultiLabelAccuracy(mtl.metrics.AbsMetric):
    def __init__(self, num_labels, threshold=0.5):
        super(MultiLabelAccuracy, self).__init__()
        self.num_labels = num_labels
        self.threshold = threshold
        self.metric = MultilabelAccuracy(num_labels=num_labels, threshold=threshold)
    
    def update_fun(self, pred, gt):
        pred = torch.sigmoid(pred)  # Apply sigmoid to ensure values are between 0 and 1
        accuracy = self.metric(pred, gt)
        self.record.append(accuracy.item())
        self.bs.append(gt.size(0))
    
    def score_fun(self):
        return [sum(self.record) / len(self.record)]
    
    def reinit(self):
        super().reinit()
        self.metric()

class BinaryMultilabelAccuracy(mtl.metrics.AbsMetric):
    def __init__(self, criteria, threshold):
        super(BinaryMultilabelAccuracy, self).__init__()
        self.metric = MultilabelAccuracy(criteria=criteria, threshold=threshold)
        self.threshold = threshold
    
    def update_fun(self, pred, gt):
        # pred = torch.sigmoid(pred)  # Apply sigmoid to ensure values are between 0 and 1
        pred = (pred >= self.threshold).float()
        self.metric.update(pred, gt)
        accuracy = self.metric.compute()
        self.record.append(accuracy.item())
        self.bs.append(gt.size(0))
    
    def score_fun(self):
        return [sum(self.record) / len(self.record)]
    
    def reinit(self):
        super().reinit()
        self.metric.reset()
