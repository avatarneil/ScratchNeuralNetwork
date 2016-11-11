import numpy as np
x = [1,2,3,4,5,6]
z = [1,4,9,16,25,36]
step = .01

def sigmoid(a):
    return (1/(1+np.exp(-a)))

w1 = np.random(0,1)
w2 = np.random(0,1)

def calcVar(a):
    p1 = (w1*a)
    y = sigmoid(p1)
    p2 = (w2*y)
    return sigmoid(p2)

def training():
    while i<len(z):
        v = calcVar(x[i])
        c=(z[i]-v)^2
        dcw1 = 
        dcw2 = 2*(z[i]-v)*v*(1-v)*
        w1 = w1-(dcw1*step)
        w2 = w2-(dcw2*step)
        i++
        