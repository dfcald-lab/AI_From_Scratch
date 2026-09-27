from src.network import Network


training_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


learning_rate = 0.05
epochs = 1000
seeds = range(10)


for seed in seeds:
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed
    )

    for _ in range(epochs):
        for inputs, target in training_data:
            prediction = network.forward(inputs)[0]

            error = prediction - target
            network.backward([error])
            network.update(learning_rate)

    total_loss = 0.0
    predictions = []

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        error = prediction - target
        total_loss += 0.5 * error ** 2
        predictions.append(prediction)

    successful = total_loss < 1e-8

    print(
        f"seed={seed:2d} "
        f"loss={total_loss:.12f} "
        f"success={successful}"
    )
