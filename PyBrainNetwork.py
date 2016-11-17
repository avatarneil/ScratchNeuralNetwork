import numpy as np
import matplotlib.pyplot as plt
from cybrain.tools.shortcuts import buildNetwork
from cybrain.datasets import SupervisedDataSet
from cybrain.supervised.trainers import BackpropTrainer
import csv

np.random.seed(19230834)

inputNumber=1
outputNumber=1

fileName='demodataset.csv'
array = np.loadtxt(fileName, delimiter=',', skiprows=1)

# assume last field in csv is single target variable
# and all other fields are input variables
number_of_columns = array.shape[1]
#dataset = SupervisedDataSet(number_of_columns - 1, 1)

#print array[0]
#print array[:,:-1]
#print array[:,-1]

ds = SupervisedDataSet(number_of_columns - 1, outputNumber)
ds.setField('input',array[:,:-1])
ds.setField('target',array[:,-1:])

net = buildNetwork(inputNumber,100,100,100,outputNumber)

trainer = BackpropTrainer(net, ds)
for i in range(1,1000):
    trainer.train()
    print(i)
print(net.activate([(np.pi)/6]))