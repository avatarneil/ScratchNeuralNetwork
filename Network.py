import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1,50,50)
z= np.linspace(1,50,50)
while i<=len(x):
    z[i]=x[i]**2
    
step = .01
i=0
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
    while (i<=len(z)):
        v = calcVar(x[i])
        y = sigmoid(w1*x[i])
        c=(z[i]-v)**2
        dcw1 = 2*(z[i]-v)*v*(1-v)*w2*y*(1-y)*x[i]
        dcw2 = 2*(z[i]-v)*v*(1-v)*y
        w1 =w1-(dcw1*step)
        w2 =w2-(dcw2*step)
        print(c)
        #plt.plot(i,c)
        i+=1

training()
