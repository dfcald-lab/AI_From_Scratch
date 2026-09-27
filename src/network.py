class Network:
    def __init__(self, layers):
        self.layers = layers

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
