from src.network import Network


network = Network(
    number_of_inputs=2,
    layer_sizes=[2, 1],
    activations=["relu", "linear"],
    seed=1
)


training_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


learning_rate = 0.05
epochs = 1000


for epoch in range(epochs):
    total_loss = 0.0

    for inputs, target in training_data:
        outputs = network.forward(inputs)
        prediction = outputs[0]

        error = prediction - target
        loss = 0.5 * error ** 2

        total_loss += loss

        network.backward([error])
        network.update(learning_rate)

    if epoch in [0, 1, 9, 99, 499, 999]:
        print(
            f"epoch={epoch + 1:4d} "
            f"loss={total_loss:.12f}"
        )


print()
print("NETWORK STRUCTURE")

for index, layer in enumerate(network.layers, start=1):
    print(
        f"Layer {index}: "
        f"{len(layer.neurons)} neurons"
    )

print()
print("FINAL XOR RESULTS")

for inputs, target in training_data:
    prediction = network.forward(inputs)[0]

    print(
        "Input:", inputs,
        "Expected:", target,
        "Predicted:", prediction
    )
