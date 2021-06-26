import torch
import torch.nn as nn
import copy
from utils import train_epoch, predict_epoch

class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(16, 120, 5, stride = 1, padding = 0)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.pool(y)
        y = self.conv3(y)
        y = torch.flatten(y, start_dim=1)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        return y

class DeterministicWrapper(torch.nn.Module):
    def __init__(self, model, hyperparams):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.optimizer = hyperparams['optimizer'](self.model.parameters(), **hyperparams['optimizer_params'])
    
    def forward(self, x):
        return self.model(x)

    def train_epoch(self, data_loader, loss_fn):
        train_epoch(self.model, self.optimizer, data_loader, loss_fn)

    def predict_epoch(self, data_loader, return_logits=False):
        return predict_epoch(self, data_loader, return_logits )

class EnsembleWrapper(torch.nn.Module):
    def __init__(self, model, hyperparams):
        super().__init__()
        self.ensemble_size = hyperparams['ensemble_size']
        self.models = []
        self.optimizers = []
        for _ in range(self.ensemble_size):
            self.models.append( copy.deepcopy(model) )
        for i in range(self.ensemble_size):
            self.optimizers.append(hyperparams['optimizer'](self.models[i].parameters(), **hyperparams['optimizer_params']))

    def forward(self, x): # DO NOT USE FOR TRAINING. ONLY FOR PREDICTING
        preds = []
        for model in self.models:
            preds.append( model(x.clone()) )
        return torch.mean(torch.stack(preds), dim=0)

    def train_epoch(self, data_loader, loss_fn):
        for i in range(self.ensemble_size):
            train_epoch(self.models[i], self.optimizers[i], data_loader, loss_fn)

    def predict_epoch(self, data_loader, return_logits=False):
        return predict_epoch(self, data_loader, return_logits )