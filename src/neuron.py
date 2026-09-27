class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def relu(self, value):
        return max(0, value)

    def relu_derivative(self, value):
        if value > 0:
            return 1
        return 0

    def forward(self, inputs):
        total = self.bias

        for weight, input_value in zip(self.weights, inputs):
            total += weight * input_value

        activated = self.relu(total)

        return total, activated
