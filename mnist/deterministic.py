import torch
import torchvision
import numpy as np
import os
import sys
import random
import argparse

sys.path.append('..')
parser = argparse.ArgumentParser()
parser.add_argument("-gpu_id", type=str, default='1')
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-weight_decay", type=float, default=0)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-epoch", type=int, default=25)
parser.add_argument("-ensemble_size", type=int, default=None)
parser.add_argument("-validation_frequency", type=int, default=5)
parser.add_argument("-fashion", action='store_true')
parser.add_argument("-cores", type=int, default=1) # NUM WORKERS FOR DATA LOADER
args = parser.parse_args()

from models import LeNet5, DeterministicWrapper, EnsembleWrapper
from datasets import MNIST, FashionMNIST
from utils import ECELoss

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
assert torch.cuda.is_available(), "Only CUDA is supported."
torch.cuda.manual_seed_all(0)
device = torch.device("cuda:" + args.gpu_id)

hyperparams={'optimizer':torch.optim.Adam, 
                'optimizer_params':{'lr':args.lr, 'weight_decay':args.weight_decay}, 
            'batch_size':args.batch_size,
            'epoch':args.epoch,
            'ensemble_size':args.ensemble_size,
            "device":device,
            "cores":args.cores,
            "val_freq": args.validation_frequency}

model = LeNet5().to(device)

if hyperparams['ensemble_size']:
    model = EnsembleWrapper(model, hyperparams)
else:
    model = DeterministicWrapper(model, hyperparams)

if args.fashion:
    dataset = FashionMNIST
else:
    dataset = MNIST


loss_func = torch.nn.CrossEntropyLoss().to(device)

transform = torchvision.transforms.Compose([       
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

target_transform = torchvision.transforms.Compose([       
        ])

train_data = dataset(transform=transform, target_transform=target_transform)
test_data =  dataset(train=False, transform=transform, target_transform=target_transform)
train_loader = train_data.get_loader(batch_size=hyperparams['batch_size'], num_workers=hyperparams['cores'])
test_loader = test_data.get_loader(batch_size=hyperparams['batch_size'], num_workers=hyperparams['cores'])

best_accuracy = 0
for epoch in range(hyperparams['epoch']):
    model.train_epoch(train_loader, loss_func)
    print("Epoch", epoch, "done.")
    if not epoch % hyperparams['val_freq']:
        print("--Validation--")
        accuracy = model.predict_epoch(test_loader)
        print("Accuracy: ", accuracy)
        if accuracy > best_accuracy:
            model.save_weights()
            best_accuracy = accuracy

model.load_weights() # load best weights

print("-Training Accuracy-")
accuracy = model.predict_epoch(train_loader)
print("Accuracy: ", accuracy)

print("-Validation Accuracy-")
logits, labels = model.predict_epoch(test_loader, return_logits=True)
accuracy = (torch.argmax(logits.data, dim=1) == labels).sum().item() / logits.shape[0]
print("Accuracy: ", accuracy)

ECE_loss_func = ECELoss()
ECE_loss = ECE_loss_func(logits, labels).item()
print("Expected Calibration Error: ", ECE_loss)