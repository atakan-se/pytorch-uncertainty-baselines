All scores achieved using default parameters. (No command line arguments passed)
### MNIST 
                    
Model  | Train Acc. | NLL | Test Acc. |  ECE
------------- | ------------- | -------------| ------------- |------------- |
Deterministic | 0.98905 |-| 0.9828  | 9.4e-7
Ensemble(n=10)| 0.9943  |-| 0.9873  | 7.5e-7

### Fashion MNIST 
                    
Model  | Train Acc. | NLL | Test Acc. |  ECE
------------- | ------------- | -------------| ------------- |------------- |
Deterministic | 0.91697 |-| 0.8934 | 2.7e-6
Ensemble(n=10)| 0.92757 |-| 0.8969 | 2.8e-6

### CIFAR10
                    
Model  | Train Acc. | NLL | Test Acc. |  ECE
------------- | ------------- | -------------| ------------- |------------- |
Deterministic | 0.99996 |0.315| 0.9417 | 0.03967
Dropout (0.1) | 0.99994 |0.30774| 0.9423 | 0.040057
BatchEnsemble (size=4) | 0.99996 |0.31389| 0.9418 | 0.0401
MIMO(in=1,out=4) | 0.99998 |0.40129| 0.9422 | 0.205784

### CIFAR100
                    
Model  | Train Acc. | NLL | Test Acc. |  ECE
------------- | ------------- | -------------| ------------- |------------- |
Deterministic | 0.99984 |1.3001| 0.7399 | 0.14592
Dropout (0.1) | 0.99982 |1.24013| 0.7477 | 0.13941
BatchEnsemble (size=4) |0.9998 |1.27273| 0.743 | 0.14199
MIMO(in=1,out=4) | 0.9868 |1.8614| 0.6798 | 0.36682

