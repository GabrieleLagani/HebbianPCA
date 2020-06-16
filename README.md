Pytorch implementation of Hebbian learning algorithms to train
deep convolutional neural networks. This work, which focuses on 
Hebbian PCA algorithms, is a continuation of 
https://github.com/GabrieleLagani/HebbianLearningThesis  
A neural network model is trained on CIFAR10 both using Hebbian PCA 
algorithms and SGD in order to compare the results.

In order to launch a training session, type:  
`PYTHONPATH=<project root> python <project root>/train.py --config <config family>/<config name>`  
Where `<config family>` is either `gdes` or `hebb`, depending whether 
you want to run gradient descent or hebbian training, and 
`<config name>` is the name of one of the training configurations in 
the `config.py` file.  
Example:  
`PYTHONPATH=<project root> python <project root>/train.py --config gdes/config_base`  
To evaluate the network on the CIFAR10 test set, type:  
`PYTHONPATH=<project root> python <project root>/evaluate.py --config <config family>/<config name>`


Author: Gabriele Lagani - gabriele.lagani@gmail.com

