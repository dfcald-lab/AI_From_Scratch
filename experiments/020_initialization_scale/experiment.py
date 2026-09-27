from src.network import Network


training_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


learning_rate = 0.05
epochs = 1000
seed = 1
scales = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


for scale in scales:
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed
    )

    for layer in network.layers:
        for neuron in layer.neurons:
            neuron.weights = [
                weight * scale
                for weight in neuron.weights
            ]

    for _ in range(epochs):
        for inputs, target in training_data:
            prediction = network.forward(inputs)[0]

            error = prediction - target
            network.backward([error])
            network.update(learning_rate)

    total_loss = 0.0

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        error = prediction - target
        total_loss += 0.5 * error ** 2

    successful = total_loss < 1e-8

    print(
        f"scale={scale:>4} "
        f"loss={total_loss:.12f} "
        f"success={successful}"
    )
