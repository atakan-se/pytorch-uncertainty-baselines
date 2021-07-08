import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import copy
from utils import train_epoch, predict_epoch

def init_weights(m):
    if type(m) == nn.Linear:
        #nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

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

class MobilenetV2_CIFAR(torch.nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        model = mobilenet_v2(pretrained=False)
        features = model.features
        features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        features[2].conv[1][0] = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.features = nn.Sequential(features)
        self.global_avg = nn.AvgPool2d(kernel_size=(4,4))
        self.classifier = nn.Linear(in_features=1280, out_features=classes, bias=True)
    def forward(self, x):
        y = self.features(x)
        y = self.global_avg(y)
        y = torch.squeeze(y)
        y = self.classifier(y)
        return y

class DeterministicWrapper(torch.nn.Module):
    def __init__(self, model, hyperparams):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.apply(init_weights)
        self.optimizer = hyperparams['optimizer'](self.model.parameters(), **hyperparams['optimizer_params'])
    
    def forward(self, x):
        return self.model(x)

    def train_epoch(self, data_loader, loss_fn):
        train_epoch(self.model, self.optimizer, data_loader, loss_fn)

    def predict_epoch(self, data_loader, return_logits=False):
        return predict_epoch(self, data_loader, return_logits )

    def multiply_lr(self, factor):
        for group in self.optimizer.param_groups:
            group['lr'] *= factor

    def save_weights(self):
        torch.save(self.model.state_dict(), './saved_models/model_weights.pt')

class EnsembleWrapper(torch.nn.Module):
    def __init__(self, model, hyperparams):
        super().__init__()
        self.ensemble_size = hyperparams['ensemble_size']
        self.models = []
        self.optimizers = []
        for i in range(self.ensemble_size):
            self.models.append( copy.deepcopy(model) )
            self.models[i].apply(init_weights)
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

    def multiply_lr(self, factor):
        for optim in self.optimizers:
            for group in self.optimizer.param_groups:
                group['lr'] *= factor

    def save_weights(self):
        for i, model in enumerate(self.models):
                torch.save(model.state_dict(), './saved_models/model_weights' + str(i) + '.pt')