import numpy as np
import matplotlib.pyplot as plt
np.random.seed(19230834)
x = np.linspace(0,1,100)
z= np.linspace(0,1,100)
i=0
while i<len(x):
    z[i]=2*x[i]
    i+=1
i=0
step = .01
wumbo1 = np.random.rand(1)
wumbo2 = np.random.rand(1)
w1=wumbo1[0]
w2=wumbo2[0]
#cs = np.array([])

def sigmoid(a):
    return (1/(1+np.exp(-a)))

def calcVar(a):
    p1 = (w1*a)
    y = sigmoid(p1)
    p2 = (w2*y)
    return sigmoid(p2)

def training():
    global i
    global w1
    global w2
    while (i<len(z)):
        v = calcVar(x[i])
        y = sigmoid(w1*x[i])
        c=(z[i]-v)
        dcw1 =2*(z[i]-v)*v*(1-v)*w2*y*(1-y)*x[i]
        dcw2 =2*(z[i]-v)*v*(1-v)*y
        w1 =w1-(dcw1*c)
        w2 =w2-(dcw2*c)
        #plt.plot(i,c)
        print(c,w1,w2)
        i+=1
    i=0
training()