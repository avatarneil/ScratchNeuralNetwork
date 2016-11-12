import numpy as np
import matplotlib.pyplot as plt
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

np.random.seed(19230834)
ds = SupervisedDataSet(2, 1)
net = buildNetwork(2, 1000, 1, bias='True')
#while i<len(x):
   # ds.addSample((x[i]),(z[i],))
  #  i+=1
#i=0
ds.addSample((0,0),(0,))
ds.addSample((0,1),(1,))
ds.addSample((1,0),(1,))
ds.addSample((1,1),(0,))
trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence()

print(net.activate([0,1]))