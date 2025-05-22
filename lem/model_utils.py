import torch
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts, SequentialLR

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough P
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # Compute the positional encodings once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        x = x + self.pe[:, :x.size(1)]
        return x

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, outputs: torch.Tensor, targets) -> torch.Tensor:
        if isinstance(targets, tuple) or isinstance(targets, list):
            targets = targets[0]
        return nn.functional.mse_loss(outputs, targets)

class EarlyStopping:
    def __init__(self, patience=50, delta=0.0001, warmup=10, average_last=10):
        """
        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            warmup (int): Number of epochs to wait before starting to monitor for early stopping.
            average_last (int): Number of epochs to average for reporting.
        """
        self.patience = patience
        self.delta = delta
        self.warmup = warmup
        
        self.total_epochs = 0
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        self.best_acc = None
        self.best_epoch = None
        self.running_loss = None
        self._average_acc_list = []
        self.average_acc = None
        self.average_last = average_last

    def copy_model(self, model):
        device = next(model.parameters()).device
        model.to('cpu')
        self.best_model = copy.deepcopy(model)
        model.to(device)
        
    def test_early_stop(self, val_loss, model=None, acc=None):
        self.total_epochs += 1
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = acc
            self.best_epoch = self.total_epochs
            self.running_loss = val_loss
        self.best_loss = min(self.best_loss, val_loss)
        if acc is not None:
            self.best_acc = max(self.best_acc, acc)
            self._average_acc_list.append(acc)
            if len(self._average_acc_list) > self.average_last:
                self._average_acc_list.pop(0)
            self.average_acc = sum(self._average_acc_list) / len(self._average_acc_list)
        if self.total_epochs < self.warmup:
            self.early_stop = False
        else:
            # Check if the loss did not improve by at least delta
            if val_loss > self.running_loss - self.delta:
                # Did not improve
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                # Improved!
                self.running_loss = val_loss
                if model is not None:
                    self.copy_model(model)
                self.best_epoch = self.total_epochs
                self.counter = 0
                self.early_stop = False
        return self.early_stop

def compute_rank_accuracy(y_pred, y_possible, y_true, individual=False):
    """
    Computes rank accuracy for a batch of network outputs.
    
    Args:
        y_pred (torch.Tensor): Predicted vectors, shape (batch_size, feature_dim)
        y_possible (torch.Tensor): Possible output vectors, shape (num_classes, feature_dim)
        y_true (torch.Tensor): Ground truth indices, shape (batch_size,)
        
    Returns:
        ranks (torch.Tensor): Ranks of the correct class for each sample
        accuracy_at_1 (float): Top-1 accuracy
        accuracy_at_5 (float): Top-5 accuracy
    """
    # Compute cosine similarity (or dot product) of y_pred and y_possible:
    scores = torch.matmul(y_pred, y_possible.T)  # (batch_size, num_classes)
    # Get the ranking of each correct answer
    ranks = torch.argsort(scores, dim=1, descending=True)  # Sorted indices (batch_size, num_classes)
    # Find where the true class appears in sorted rankings
    correct_ranks = torch.nonzero(ranks == y_true.unsqueeze(1), as_tuple=True)[1]
    # Compute accuracy@1 and accuracy@5
    accuracy_at_1 = (correct_ranks == 0).float().mean().item()
    accuracy_at_5 = (correct_ranks <= 4).float().mean().item()
    rank_acc = 1 - (correct_ranks / y_possible.shape[0]).mean().item()
    if individual:
        return [1 - x for x in (correct_ranks / y_possible.shape[0]).tolist()]
    else:
        return rank_acc, accuracy_at_1, accuracy_at_5

def get_cosine_schedule_with_warmup_restarts(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    T_0: int,
    T_mult: int = 1,
    eta_min: float = 0.0,
    last_epoch: int = -1
):
    """
    Returns a scheduler that does:
      1) Linear warmup for `warmup_steps` steps
      2) Cosine annealing with restarts thereafter (T_0, T_mult, etc.)
    
    Arguments:
      optimizer:       The optimizer we are scheduling.
      warmup_steps:    How many steps to warm up for.
      T_0:             Number of steps for the first restart in CosineAnnealingWarmRestarts.
      T_mult:          A factor increases T_{i} after a restart. Defaults to 1 (no change).
      eta_min:         Minimum learning rate for the cosine schedule.
      last_epoch:      The index of the last epoch. Can be used to resume training.
    """

    # 1) Define a warmup schedule that linearly ramps the learning rate
    def warmup_lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step+1) / float(max(1, warmup_steps))
        else:
            return 1.0  # After warmup_steps, keep LR multiplier at 1.0 (until we switch)

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda, last_epoch=last_epoch)

    # 2) Define the cosine schedule with warm restarts
    #    - T_0 = number of steps for the first cycle
    #    - T_mult = factor by which the period grows after each restart
    #    - eta_min = minimum learning rate
    cos_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        last_epoch=last_epoch
    )

    # 3) Chain them using SequentialLR
    #    - We'll switch to cos_scheduler after `warmup_steps` are finished
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cos_scheduler],
        milestones=[warmup_steps]  # Step index at which we move to the next scheduler
    )

    return scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)