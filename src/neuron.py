class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        self.inputs = None
        self.raw_output = None

        self.weight_gradients = [0.0] * len(weights)
        self.bias_gradient = 0.0

    def relu(self, value):
        return max(0.0, value)

    def relu_derivative(self, value):
        if value > 0:
            return 1.0
        return 0.0

    def forward(self, inputs):
        self.inputs = inputs

        total = self.bias

        for weight, input_value in zip(self.weights, inputs):
            total += weight * input_value

        self.raw_output = total
        activated = self.relu(total)

        return total, activated

    def backward(self, gradient):
        relu_gradient = self.relu_derivative(self.raw_output)
        dz = gradient * relu_gradient

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
