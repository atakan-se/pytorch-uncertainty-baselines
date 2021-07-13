### MNIST 
                    
Model  | Train Acc. | Test Acc. |  ECE
------------- | ------------- | ------------- |------------- |
Deterministic | 0.98905 | 0.9828  | 9.4e-7
Ensemble(n=10)| 0.9943  | 0.9873  | 7.5e-7

### Fashion MNIST 
                    
Model  | Train Acc. | Test Acc. |  ECE
------------- | ------------- | ------------- |------------- |
Deterministic | 0.91697 | 0.8934 | 2.7e-6
Ensemble(n=10)| 0.92757 | 0.8969 | 2.8e-6

### CIFAR10
                    
Model  | Train Acc. | Test Acc. |  ECE
------------- | ------------- | ------------- |------------- |
Deterministic | 0.99974 | 0.9344 | 0.04796
Dropout (0.1) | 0.9998 | 0.9355 | 0.046998
BatchEnsemble (size=4) | 0.99982 | 0.9366 | 0.045764
