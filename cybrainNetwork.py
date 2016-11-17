import sys
sys.path.append("..")

import cybrain as cb
import numpy as np
from time import time

#TRUTH TABLE (DATA)
#X =     [[0.0,0.0]];     Y = [[-1.0]]
#X.append([1.0,0.0]); Y.append([1.0])
#X.append([0.0,1.0]); Y.append([1.0])
#X.append([1.0,1.0]); Y.append([-1.0])

def column(matrix, i):
    return [row[i] for row in matrix]
    
fileName='demodataset.csv'
arraything = np.loadtxt(fileName, delimiter=',', skiprows=1)

xlist = column(arraything,0)
ylist = column(arraything,1)
xlist = [float(v) for v in xlist]
print(type(xlist[1]))
ylist = [float(v) for v in ylist]
xlist2 = []
ylist2 = []
for i in range(len(xlist)):
    xlist2.append([xlist[i]])
    ylist2.append([ylist[i]])
xlist2, ylist2 = np.array(xlist2),np.array(ylist2)
#CONVERT DATA TO NUMPY ARRAY
#X, Y = np.array(X), np.array(Y)

#CREATE NETWORK
nnet = cb.Network()

#CREATE LAYERS
Lin = cb.LinearLayer(1)
Lhidden = cb.TanhLayer(10)
Lout = cb.TanhLayer(1)
bias = cb.BiasUnit()

#ADD LAYERS TO NETWORK
nnet.inputLayers = [Lin]
nnet.hiddenLayers = [Lhidden]
nnet.outputLayers = [Lout]
nnet.autoInputLayers = [bias]

#CONNECT LAYERS
Lin.fullConnectTo(Lhidden)
Lhidden.fullConnectTo(Lout)
bias.fullConnectTo(Lhidden)
bias.fullConnectTo(Lout)

#CREATE BATCH TRAINER
rate = 0.1
nnet.setup()
batch = cb.FullBatchTrainer(nnet, xlist2, ylist2, rate)
#print(type(xlist2))
#TRAIN
t1 = time()
batch.epochs(200)
print "Time CyBrain {}".format(time()-t1)


#PRINT RESULTS
for i in range(len(xlist2)):
    print "{} ==> {}".format(xlist2[i], np.array(nnet.activate(xlist2[i:i+1,:])))