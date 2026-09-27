from src.layer import Layer
from src.neuron import Neuron


hidden_layer = Layer(
    neurons=[
        Neuron(weights=[0.8, 0.9], bias=0.1),
        Neuron(weights=[1.1, 0.8], bias=-0.9),
    ]
)

output_layer = Layer(
    neurons=[
        Neuron(
            weights=[0.8, -1.7],
            bias=0.1,
            activation="linear"
        )
    ]
)


training_data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


learning_rate = 0.05
epochs = 1000


for epoch in range(epochs):
    total_loss = 0.0

    for inputs, target in training_data:
        # -------------------------
        # Forward pass
        # -------------------------
        _, hidden_outputs = hidden_layer.forward(inputs)
        _, outputs = output_layer.forward(hidden_outputs)

        prediction = outputs[0]
        expected = target[0]

        # -------------------------
        # Loss
        # -------------------------
        error = prediction - expected
        loss = 0.5 * error ** 2

        total_loss += loss

        # -------------------------
        # Backward pass
        # -------------------------
        output_gradient = [error]

        hidden_gradients = output_layer.backward(
            output_gradient
        )

        hidden_layer.backward(
            hidden_gradients
        )

        # -------------------------
        # Update
        # -------------------------
        output_layer.update(learning_rate)
        hidden_layer.update(learning_rate)

    if epoch in [0, 1, 9, 99, 499, 999]:
        print(
            f"epoch={epoch + 1:4d} "
            f"loss={total_loss:.12f}"
        )


print()
print("FINAL XOR RESULTS")

for inputs, target in training_data:
    _, hidden_outputs = hidden_layer.forward(inputs)
    _, outputs = output_layer.forward(hidden_outputs)

    print(
        "Input:", inputs,
        "Expected:", target[0],
        "Predicted:", outputs[0]
    )
