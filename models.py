import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import copy
from utils import train_epoch, predict_epoch

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(m.weight.data , nonlinearity='relu')

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
        self.device = hyperparams['device']

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, data_loader, loss_fn):
        train_epoch(self.model, self.optimizer, data_loader, loss_fn, self.device)

    def predict_epoch(self, data_loader, return_logits=False):
        return predict_epoch(self, data_loader, self.device, return_logits )

    def multiply_lr(self, factor):
        for group in self.optimizer.param_groups:
            group['lr'] *= factor

    def save_weights(self):
        torch.save(self.model.state_dict(), './saved_models/deterministic.pt')

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
        self.device = hyperparams['device']

    def forward(self, x): # DO NOT USE FOR TRAINING. ONLY FOR PREDICTING
        assert self.training, "Do not use 'forward' for training directly."
        preds = []
        for model in self.models:
            preds.append( model(x.clone()) )
        return torch.mean(torch.stack(preds), dim=0)

    def train_epoch(self, data_loader, loss_fn):
        for i in range(self.ensemble_size):
            train_epoch(self.models[i], self.optimizers[i], data_loader, loss_fn, self.device)

    def predict_epoch(self, data_loader, return_logits=False):
        return predict_epoch(self, data_loader, self.device, return_logits )

    def multiply_lr(self, factor):
        for optim in self.optimizers:
            for group in self.optimizer.param_groups:
                group['lr'] *= factor

    def save_weights(self):
        for i, model in enumerate(self.models):
                torch.save(model.state_dict(), './saved_models/ensemble' + str(i) + '.pt')

class DropoutWrapper(torch.nn.Module):
    def __init__(self, model, hyperparams):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.apply(init_weights)
        self.optimizer = hyperparams['optimizer'](self.model.parameters(), **hyperparams['optimizer_params'])
        # Adding Dropout after every feature layer:
        total_blocks = len(self.model.features[0])
        self.dropouts = nn.ModuleList([ nn.Dropout(hyperparams['dropout_rate']) for _ in range(total_blocks)] )
        self.device = hyperparams['device']

    def forward(self, x):
        y = self.model.features[0][0](x)
        y = self.dropouts[0](y)
        for i in range(1, len(self.model.features[0]) ):
            y = self.model.features[0][i](y)
            y = self.dropouts[i](y)
        y = self.model.global_avg(y)
        y = torch.squeeze(y)
        y = self.model.classifier(y)
        return y
        
    def train_epoch(self, data_loader, loss_fn):
        train_epoch(self.model, self.optimizer, data_loader, loss_fn, self.device)

    def predict_epoch(self, data_loader, return_logits=False):
        return predict_epoch(self, data_loader, self.device, return_logits )

    def multiply_lr(self, factor):
        for group in self.optimizer.param_groups:
            group['lr'] *= factor

    def save_weights(self):
        torch.save(self.model.state_dict(), './saved_models/dropout.pt')


class Conv2d_BatchEnsemble_Wrapper(torch.nn.Module): # Wrapper for a single Conv2d layer
    def __init__(self, conv_layer, ensemble_size=4):
        super().__init__()
        self.conv =  copy.deepcopy(conv_layer)
        self.ensemble_size = ensemble_size
        self.alpha = nn.Parameter( torch.ones((ensemble_size, self.conv.in_channels)) )
        self.gamma = nn.Parameter( torch.ones((ensemble_size, self.conv.out_channels)) )

    def forward(self, x):
        sample_per_model = x.shape[0] / self.ensemble_size # Batch size / model count
        alpha = torch.repeat_interleave(self.alpha, sample_per_model, dim=0).unsqueeze(-1).unsqueeze(-1)
        gamma = torch.repeat_interleave(self.gamma, sample_per_model, dim=0).unsqueeze(-1).unsqueeze(-1)
        y = self.conv(x*alpha)*gamma
        return y            

class BatchEnsembleWrapper(torch.nn.Module): # wrapper for entire module
    def __init__(self, model, hyperparams):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.apply(init_weights)

        def conv_to_batchconv(feature):
        # Replaces Conv2d layers with BatchEnsemble Conv2d
            for n, module in feature.named_children():
                if len(list(module.children())) > 0:
                    conv_to_batchconv(module)
            if isinstance(module, nn.Conv2d):
                setattr(feature, n, Conv2d_BatchEnsemble_Wrapper(module))

        conv_to_batchconv(self.model.features)
        self.ensemble_size = hyperparams['ensemble_size']
        self.optimizer = hyperparams['optimizer'](self.model.parameters(), **hyperparams['optimizer_params'])
        self.device = hyperparams['device']

    def forward(self, x):
        return self.model(x)

    def train_epoch(self, data_loader, loss_fn):
        train_epoch(self.model, self.optimizer, data_loader, loss_fn, self.device)

    def predict_epoch(self, data_loader, return_logits=False):
        # TO DO: Change to predict function like the rest of the wrappers
        epoch_size = len(data_loader.dataset)
        self.model.eval()
        if return_logits:
            logit_lst = []
            label_lst = []
        correct_preds = 0
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs = inputs.repeat(self.ensemble_size, 1, 1, 1).to(self.device) 
                labels = labels.squeeze().to(self.device) 
                logits = self.model(inputs)
                logits = torch.chunk(logits, self.ensemble_size)
                logits = torch.mean(torch.stack(logits), dim=0)
                if return_logits:
                    logit_lst.append(logits)
                    label_lst.append(labels)
                predicted = torch.argmax(logits.data, dim=1)
                correct_preds += (predicted == labels).sum().item()
        print("Accuracy: ", correct_preds / epoch_size)
        if return_logits: return torch.cat(logit_lst,dim=0), torch.cat(label_lst,dim=0)

    def multiply_lr(self, factor):
        for group in self.optimizer.param_groups:
            group['lr'] *= factor

    def save_weights(self):
        torch.save(self.model.state_dict(), './saved_models/batchensemble.pt')