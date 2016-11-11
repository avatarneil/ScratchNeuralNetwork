import numpy as np
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

np.random.seed(19230834)
x = np.linspace(0,1,10)
z = np.linspace(0,1,10)
i=0
while i<len(x):
    z[i]=x[i]**2
    i+=1
i=0
ds = SupervisedDataSet(1, 1)
net = buildNetwork(1, 3, 1, bias=True)
while i<len(x):
    ds.addSample((x[i]),(z[i],))
    i+=1
i=0

trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence()

print(net.activate([3]))