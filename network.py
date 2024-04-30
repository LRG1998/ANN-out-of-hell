from Node import Node
from hiddenNode import HiddenNode
import numpy as np


hiddenCount = 6
learningrate = 0.001
#Confidence you want out of 1
confidence = .99
inputNodes = [Node(2), Node(2)]
expected = inputNodes[0].value * inputNodes[1].value
output = 0
hiddenLayer = []

for i in range(hiddenCount):
    hiddenLayer.append(HiddenNode(learningrate))
    for t in range(len(inputNodes)):
        hiddenLayer[i].inputs.append(inputNodes[t].value)
    hiddenLayer[i].genweights()


#This will be important later.
expected = inputNodes[0].value * inputNodes[1].value

for i in range(hiddenCount):
    hiddenLayer[i].activation(hiddenLayer[i].addup())
    output += hiddenLayer[i].output
loss = output - expected
rmse = np.sqrt(np.mean(loss)**2)




for i in range(hiddenCount):
    hiddenLayer[i].update()
#print(output)



