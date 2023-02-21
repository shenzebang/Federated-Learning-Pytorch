# Federated-Learning-Pytorch
A repo on federated learning for research purpose. The implementation is based on pytorch. 
To support acceleration with multiple GPUs, we use ray.

Currently supporting multiclass image classification task.

Inlcuding baselines:
1. fed-avg
2. *scaffold
3. *fedprox
4. *mime
5. *feddyn
...

Including models:
1. MLP
2. LeNet-5
3. Resnet
4. *VGG
5. *Alexnet
...

Including datasets:
1. cifar10
2. mnist
3. *emnist
4. *cifar100
5. *imagenet

Structure of the code:
1. load configuration
2. prepare the local datasets
3. prepare logger, local objective function
4. run FL
5. save model

To adapt the current template to your algorithm, simply implement the following five functions:
1. server_init
2. client_init
3. clients_step
4. server_step
5. clients_update
