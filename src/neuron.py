class Neuron:
    def __init__(self, weights, bias, activation="relu"):
        self.weights = weights
        self.bias = bias
        self.activation = activation

        self.inputs = None
        self.raw_output = None

        self.weight_gradients = [0.0] * len(weights)
        self.bias_gradient = 0.0

    def activate(self, value):
        if self.activation == "relu":
            return max(0.0, value)

        if self.activation == "linear":
            return value

        raise ValueError(f"Unknown activation: {self.activation}")

    def activation_derivative(self, value):
        if self.activation == "relu":
            return 1.0 if value > 0 else 0.0

        if self.activation == "linear":
            return 1.0

        raise ValueError(f"Unknown activation: {self.activation}")

    def relu(self, value):
        return max(0.0, value)

    def relu_derivative(self, value):
        return 1.0 if value > 0 else 0.0

    def forward(self, inputs):
        self.inputs = list(inputs)

        total = self.bias

        for weight, input_value in zip(self.weights, self.inputs):
            total += weight * input_value

        self.raw_output = total
        activated = self.activate(total)

        return total, activated

    def backward(self, gradient):
        activation_gradient = self.activation_derivative(self.raw_output)
        dz = gradient * activation_gradient

        self.weight_gradients = [
            dz * input_value
            for input_value in self.inputs
        ]

        self.bias_gradient = dz

        input_gradients = [
            dz * weight
            for weight in self.weights
        ]

        return input_gradients

    def update(self, learning_rate):
        self.weights = [
            weight - learning_rate * gradient
            for weight, gradient in zip(
                self.weights,
                self.weight_gradients
            )
        ]

        self.bias -= learning_rate * self.bias_gradient
