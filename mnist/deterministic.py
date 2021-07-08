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
parser.add_argument("-fashion", action='store_true')
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
            'ensemble_size':args.ensemble_size}

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
        transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.Lambda(lambda x: x.to(device))
        ])

target_transform = torchvision.transforms.Compose([       
        torchvision.transforms.Lambda(lambda x: torch.tensor([x]).to(device))
        ])

train_data = dataset(transform=transform, target_transform=target_transform)
test_data =  dataset(train=False, transform=transform, target_transform=target_transform)
train_loader = train_data.get_loader(batch_size=hyperparams['batch_size'])
test_loader = test_data.get_loader(batch_size=hyperparams['batch_size'])

for epoch in range(hyperparams['epoch']):
  model.train_epoch(train_loader, loss_func)
  print("Epoch", epoch, "done.")
  if epoch and not epoch%5:
        print("--Validation--")
        model.predict_epoch(test_loader)

print("-Training Accuracy-")
model.predict_epoch(train_loader)
print("-Validation Accuracy-")
logits, labels = model.predict_epoch(test_loader, return_logits=True)

ECE_loss_func = ECELoss()
ECE_loss = ECE_loss_func(logits, labels).item()
print("Expected Calibration Error: ", ECE_loss)
