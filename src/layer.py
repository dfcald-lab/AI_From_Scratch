import random

from src.neuron import Neuron


class Layer:
    def __init__(
        self,
        neurons=None,
        number_of_inputs=None,
        number_of_neurons=None,
        seed=0
    ):
        if neurons is not None:
            self.neurons = neurons
            return

        if number_of_inputs is None or number_of_neurons is None:
            raise ValueError(
                "Provide either neurons or both "
                "number_of_inputs and number_of_neurons."
            )

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

    def backward(self, output_gradients):
        input_gradients = [0.0] * len(self.neurons[0].weights)

        for neuron, gradient in zip(self.neurons, output_gradients):
            neuron_input_gradients = neuron.backward(gradient)

            for index, value in enumerate(neuron_input_gradients):
                input_gradients[index] += value

        return input_gradients

    def update(self, learning_rate):
        for neuron in self.neurons:
            neuron.update(learning_rate)
