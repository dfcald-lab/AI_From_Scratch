from src.layer import Layer


layer = Layer(
    number_of_inputs=2,
    number_of_neurons=3,
    seed=0
)

inputs = [2, 1]

raw_outputs, activated_outputs = layer.forward(inputs)

print("NUMBER OF NEURONS:", len(layer.neurons))
print()

for number, neuron in enumerate(layer.neurons, start=1):
    print("NEURON", number)
    print("Weights:", neuron.weights)
    print("Bias:", neuron.bias)
    print()

print("Inputs:", inputs)
print("Raw outputs:", raw_outputs)
print("Activated outputs:", activated_outputs)
