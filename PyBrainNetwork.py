import numpy as np
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import csv

np.random.seed(19230834)

inputNumber=1
outputNumber=1
ds = SupervisedDataSet(inputNumber, outputNumber)
net = buildNetwork(2,5,5,1)

trainer = BackpropTrainer(net, ds)
print(trainer.trainUntilConvergence())