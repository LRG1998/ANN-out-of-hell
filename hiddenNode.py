import numpy as np
class HiddenNode():
    def __init__(self, learnInput):
        self.output = 0
        self.learningrate = learnInput
        self.inputs = []
        self.weights = []
        self.bias = np.random.uniform(-1,1)
        self.delta = 0

    def genweights(self):
        for i in range(len(self.inputs)):
            self.weights.append(np.random.uniform(0,1))

    def addup(self):
        self.x = 0
        for v in range(len(self.inputs)):
            self.x += self.inputs[v]* self.weights[v]
        self.x += self.bias
        return self.x

    def activation(self,x):
        #sigmoid activation function. 
        self.output = (1/(1+np.exp(-x)))
        return self.output
        
    def upDelta(self):
        self.delta = (self.activation(self.addup()) * (1 - self.activation(self.addup())))
        return self.delta

    def update(self):
        for x in range(len(self.inputs)):
            self.weights[x] += self.inputs[x] * self.delta * self.learningrate
            self.bias += self.learningrate*self.delta * 1
        print(self.weights)
        print(self.bias)
        
        
