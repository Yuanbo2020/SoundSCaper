import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model_path, decrease=True, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.delta = delta
        self.path = model_path
        self.trace_func = trace_func
        self.decrease = decrease

    def __call__(self, metrics, model):
        """
        :param metrics:
        :param model:
        :param decrease: True for losses, False for accuracy, auc.
        :return:
        """

        if self.decrease:
            score = -metrics
        else:
            score = metrics

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metrics, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metrics, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.decrease:
            '''Saves model when validation loss decrease.'''
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
        else:
            '''Saves model when validation accuracy increase.'''
            if self.verbose:
                self.trace_func(
                    f'Validation acc increased ({self.val_acc_max:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_acc_max = val_loss









