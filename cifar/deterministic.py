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
parser.add_argument("-lr", type=float, default=0.1)
parser.add_argument("-weight_decay", type=float, default=4e-5)
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-epoch", type=int, default=350)
parser.add_argument("-validation_frequency", type=int, default=5)
parser.add_argument("-ensemble_size", type=int, default=None)
parser.add_argument("-cifar100", action='store_true')
parser.add_argument("-cores", type=int, default=1) # NUM WORKERS FOR DATA LOADER
args = parser.parse_args()

from models import MobilenetV2_CIFAR, DeterministicWrapper, EnsembleWrapper
from datasets import CIFAR10, CIFAR100
from utils import ECELoss

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
assert torch.cuda.is_available(), "Only CUDA is supported."
torch.cuda.manual_seed_all(0)
device = torch.device("cuda:" + args.gpu_id)

hyperparams={'optimizer':torch.optim.SGD, 
                'optimizer_params':{'lr':args.lr, 'weight_decay':args.weight_decay, 'momentum':0.9}, 
            'batch_size':args.batch_size,
            'epoch':args.epoch,
            'ensemble_size':args.ensemble_size,
            "device":device,
            "cores":args.cores,
            "val_freq": args.validation_frequency}



if args.cifar100:
    dataset = CIFAR100
    model = MobilenetV2_CIFAR(classes=100).to(device)
else:
    dataset = CIFAR10
    model = MobilenetV2_CIFAR(classes=10).to(device)
    print("Training with CIFAR10")

if hyperparams['ensemble_size']:
    model = EnsembleWrapper(model, hyperparams)
else:
    model = DeterministicWrapper(model, hyperparams)


loss_func = torch.nn.CrossEntropyLoss().to(device)

train_transform = torchvision.transforms.Compose([       
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

test_transform = torchvision.transforms.Compose([       
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

target_transform = torchvision.transforms.Compose([
        ])

train_data = dataset(transform=train_transform, target_transform=target_transform)
test_data =  dataset(train=False, transform=test_transform, target_transform=target_transform)
train_loader = train_data.get_loader(batch_size=hyperparams['batch_size'], num_workers=hyperparams['cores'], shuffle=True)
test_loader = test_data.get_loader(batch_size=hyperparams['batch_size'], num_workers=hyperparams['cores'])

best_accuracy = 0
schedules = [150,250]
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
    if epoch in schedules:
        model.multiply_lr(0.1)
        
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