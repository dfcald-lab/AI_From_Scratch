from src.layer import Layer


class Network:
    def __init__(
        self,
        layers=None,
        number_of_inputs=None,
        layer_sizes=None,
        activations=None,
        seed=0
    ):
        if layers is not None:
            self.layers = layers
            return

        if number_of_inputs is None or layer_sizes is None:
            raise ValueError(
                "Provide either layers or both "
                "number_of_inputs and layer_sizes."
            )

        if activations is None:
            activations = ["relu"] * len(layer_sizes)

        if len(activations) != len(layer_sizes):
            raise ValueError(
                "activations and layer_sizes must have the same length."
            )

        self.layers = []

        inputs_for_layer = number_of_inputs

        for index, (number_of_neurons, activation) in enumerate(
            zip(layer_sizes, activations)
        ):
            layer = Layer(
                number_of_inputs=inputs_for_layer,
                number_of_neurons=number_of_neurons,
                seed=seed + index,
                activation=activation
            )

            self.layers.append(layer)
            inputs_for_layer = number_of_neurons

    def forward(self, inputs):
        current = list(inputs)

        for layer in self.layers:
            _, current = layer.forward(current)

        return current

    def backward(self, gradients):
        current_gradients = list(gradients)

        for layer in reversed(self.layers):
            current_gradients = layer.backward(current_gradients)

        return current_gradients

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
