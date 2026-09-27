from src.neuron import Neuron


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs):
        raw_outputs = []
        activated_outputs = []

        for neuron in self.neurons:
            raw_output, activated_output = neuron.forward(inputs)

            raw_outputs.append(raw_output)
            activated_outputs.append(activated_output)

        return raw_outputs, activated_outputs
