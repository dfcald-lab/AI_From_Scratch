from src.neuron import Neuron


neuron = Neuron(
    weights=[0.5, 0.5],
    bias=0.0
)

inputs = [2, 1]

raw_output, activated_output = neuron.forward(inputs)

print("Weights:", neuron.weights)
print("Bias:", neuron.bias)
print("Inputs:", inputs)
print("Raw output:", raw_output)
print("ReLU output:", activated_output)
print("ReLU derivative:", neuron.relu_derivative(raw_output))
