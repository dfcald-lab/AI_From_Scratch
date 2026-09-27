from src.neuron import Neuron
from src.layer import Layer


layer = Layer(
    neurons=[
        Neuron(weights=[1.0, 2.0], bias=0.0),
        Neuron(weights=[3.0, 1.0], bias=1.0),
    ]
)

inputs = [2, 1]

raw_outputs, activated_outputs = layer.forward(inputs)

print("Inputs:", inputs)
print("Raw outputs:", raw_outputs)
print("Activated outputs:", activated_outputs)
