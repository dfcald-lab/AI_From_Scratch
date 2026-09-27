import random

from src.neuron import Neuron


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons, seed=0):
        random_generator = random.Random(seed)

        self.neurons = []

        for _ in range(number_of_neurons):
            weights = [
                random_generator.uniform(-1, 1)
                for _ in range(number_of_inputs)
            ]

            bias = 0.0

            self.neurons.append(
                Neuron(
                    weights=weights,
                    bias=bias
                )
            )

    def forward(self, inputs):
        raw_outputs = []
        activated_outputs = []

        for neuron in self.neurons:
            raw_output, activated_output = neuron.forward(inputs)

            raw_outputs.append(raw_output)
            activated_outputs.append(activated_output)

        return raw_outputs, activated_outputs
