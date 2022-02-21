# Federated-Learning-Pytorch



This repo contains code accompaning the following papers (and more to come), 
1. Federated Functional Gradient Boosting (Shen et al., AISTATS 2022).
2. An Agnostic Approach to Federated Learning with Class Imbalance (Shen et al., ICLR 2022).

It includes code for running the multiclass image classification experiments in the Federated Learning paradigm.
A few different settings are considered, including standard Federated Learning, Functional Federated Learning, and Constrained Federated Learning.
The implementation is based on pytorch. To support acceleration with multiple GPUs, we use ray.

To run experiments in the standard setting of Federated Learning, please use run_FL.py. 
```
python run_FL.py
```
To run experiments in the setting of Functional Federated Learning, please use run_FFL.py.
```
python run_FFL.py
```
To run experiments in the setting of Constrained Federated Learning, please use run_PD_FL.py
```
python run_PD_FL.py
```
Another goal of this repo is to provide a template of the optimization methods for Federated Learning.

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
