import os
import torch
import numpy as np
import adamp
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict
from contextlib import nullcontext
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.optimizer import Optimizer
from datetime import datetime 
from tqdm import tqdm, trange
from src.utils import get_paths
import LibMTL as mtl
from LibMTL._record import _PerformanceMeter
from LibMTL.weighting.Aligned_MTL import Aligned_MTL
from LibMTL.utils import count_parameters
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from src.weighting import Fixed_Aligned_MTL
import torchmetrics

optim_dict = {
        'sgd': torch.optim.SGD,
        'sgdp': adamp.SGDP,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'adamp': adamp.AdamP,
        'adagrad': torch.optim.Adagrad,
        'rmsprop': torch.optim.RMSprop,
    }
    
scheduler_dict = {
        'exp': torch.optim.lr_scheduler.ExponentialLR,
        'step': torch.optim.lr_scheduler.StepLR,
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
        'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
    }

def handle_epoch(epoch_type:str, dl:DataLoader, model:nn.Module, loss_fn:Callable, metric_fn:torchmetrics.Metric, optimizer:Optimizer=None, scheduler=None, device=None, threshold=None) -> Tuple[Optional[float], float]:
    valid_epoch_types = ('train', 'test', 'val', 'validation')
    if epoch_type.lower() not in valid_epoch_types:
        raise TypeError(f'Argument "epoch_type" must be one of {valid_epoch_types}')
    model.train() if epoch_type == 'train' else model.eval()

    running_loss = 0
    total_samples = 0

    metric_fn.reset()
    context_manager = torch.no_grad() if epoch_type != 'train' else nullcontext()
    with context_manager:
        for data_x, data_y in dl:
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            if epoch_type == 'train':
                optimizer.zero_grad()

            predictions = model(data_x)

            loss = loss_fn(predictions, data_y) if epoch_type != 'test' else None

            if epoch_type == 'train':
                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()
            
            if threshold is not None:
                predictions = (predictions >= threshold).int()
        
            metric_fn.update(predictions, data_y)
            if epoch_type != 'test':
                running_loss += loss.item() * data_x.size(0)
            total_samples += data_y.numel()

    avg_loss = running_loss / total_samples if epoch_type != 'test' else None
    avg_metric = metric_fn.compute().item()
    return avg_loss, avg_metric


def train(
    model: nn.Module,
    dataloaders: dict,
    optimizer: Optimizer,
    loss_fn: Callable,
    metric_fn: torchmetrics.Metric,
    n_epochs: int,
    checkpoint_name: str = 'default',
    patience = 5,
    higher_is_better: bool = True,
    device = None,
    logger = None,
    scheduler = None,
    threshold= None
):

    if logger:
        for k, v in dataloaders.items(): 
          print(f'Dataloader {k}: {len(v.dataset)}') 
        logger.info(f'Model: {model}')

    start = datetime.now()
    paths = get_paths()
    model_path = paths['training']['checkpoints'] / Path(f'{checkpoint_name}.pth')

    best_epoch = -1
    best_metric = float('-inf') if higher_is_better else float('inf')

    train_loop = trange(n_epochs, desc='Training', leave=True)
    for epoch in train_loop:

        avg_loss, avg_metric = handle_epoch('train', dataloaders['train'], model, loss_fn=loss_fn, metric_fn=metric_fn, optimizer=optimizer, scheduler=scheduler, device=device, threshold=threshold)

        avg_vloss, avg_vmetric = handle_epoch('val', dataloaders['val'], model, loss_fn=loss_fn, metric_fn=metric_fn, device=device, threshold=threshold)

        train_loop.set_description(f'EPOCH {epoch}: Loss train: {avg_loss:.3f}, val: {avg_vloss:.3f} | Metric train: {avg_metric:.3f}, val: {avg_vmetric:.3f}')

        if (higher_is_better and avg_vmetric > best_metric) or (not higher_is_better and avg_vmetric < best_metric):
            best_metric = avg_vmetric
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
        elif epoch - best_epoch > patience:
            tqdm.write(f'Early stopped after {patience} epochs of no progress. Best validation metric {best_metric:.3f}')
            break
    
    avg_tloss, avg_tmetric = handle_epoch('test', dataloaders['test'], model, loss_fn=loss_fn, metric_fn=metric_fn, device=device, threshold=threshold)

    end = datetime.now()
    total_time = end - start

    tqdm.write(f'Training finished after {total_time}. Test metric {avg_tmetric:.3f}')

    results = {
        'best_val_metric': best_metric,
        'best_epoch': best_epoch,
        'avg_tloss': avg_tloss,
        'avg_tmetric': avg_tmetric,
        'total_time': total_time
    }

    print(results)

    return results


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class MTLTrainer(mtl.Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                    rep_grad, multi_input, optim_param, scheduler_param, 
                    save_path=None, load_path=None, device=torch.device('cuda:0'), **kwargs):
        nn.Module.__init__(self)
        
        print(encoder_class)
        self.device = device # added
        mtl.utils.set_device(device.type)
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

    def _prepare_optimizer(self, optim_param, scheduler_param):
        alg, optim_arg = get_optimizer(optim_param)
        self.optimizer = alg(self.model.parameters(), **optim_arg)
        self.scheduler = get_scheduler(self.optimizer, scheduler_param)

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label

    def train(self, train_dataloaders, test_dataloaders, epochs, 
                val_dataloaders=None, return_weight=False, patience=None):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        if patience:
            early_stopping = EarlyStopping(patience=patience)
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            with trange(train_batch, desc="Training", unit="batch") as pprange:
                for batch_index in pprange:
                    allocated = round(torch.cuda.memory_allocated(0)/1024**3,1)
                    cached = round(torch.cuda.memory_reserved(0)/1024**3,1)
                    pprange.set_description(f"Epoch {epoch}, Allocated {allocated}GB, Cached {cached}GB")
                    if not self.multi_input:
                        train_inputs, train_gts = self._process_data(train_loader)
                        train_preds = self.model(train_inputs)
                        train_preds = self.process_preds(train_preds)
                        train_losses = self._compute_loss(train_preds, train_gts)
                        self.meter.update(train_preds, train_gts)
                    else:
                        train_losses = torch.zeros(self.task_num).to(self.device)
                        for tn, task in enumerate(self.task_name):
                            train_input, train_gt = self._process_data(train_loader[task])
                            train_pred = self.model(train_input, task)
                            train_pred = train_pred[task]
                            train_pred = self.process_preds(train_pred, task)
                            train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                            self.meter.update(train_pred, train_gt, task)

                    self.optimizer.zero_grad(set_to_none=False)
                    w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                    if w is not None:
                        self.batch_weight[:, epoch, batch_index] = w
                    self.optimizer.step()
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
                if patience:
                    early_stopping(val_improvement)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight
        else:
            results = {
                # 'results': self.meter.results,
                'best_result': self.meter.best_result,
                # 'losses': self.meter.losses,
                # 'metrics': self.meter.metrics,
                'total_time': self.meter.end_time - self.meter.beg_time
            }

            return results
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):

        weighting = weighting_method.__dict__[weighting]
        # FIX deprecated function
        if weighting == Aligned_MTL:
            weighting = Fixed_Aligned_MTL
        architecture = architecture_method.__dict__[architecture]
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)
        
def get_optimizer(optim_param:dict):
    optim = optim_param['optim']
    try: alg = optim_dict[optim]
    except KeyError: raise TypeError(f'Optimizer not recognized. Must be one of: {optim_dict.keys()}')
    exclude = ['optim'] if 'adam' in optim else ['optim', 'betas']
    params = {k: v for k, v in optim_param.items() if k not in exclude}
    return alg, params

def get_scheduler(optimizer, scheduler_param:dict):
    scheduler = scheduler_dict[scheduler_param['scheduler']]
    if scheduler_param is not None:
        scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
        return scheduler(optimizer, **scheduler_arg)
    else: 
        return None