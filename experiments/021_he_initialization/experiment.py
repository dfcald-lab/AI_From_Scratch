import math
import random

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


def apply_he_initialization(network, seed):
    random_generator = random.Random(seed)

    for layer in network.layers:
        if layer.neurons[0].activation != "relu":
            continue
        fan_in = len(layer.neurons[0].weights)
        standard_deviation = math.sqrt(2.0 / fan_in)

        for neuron in layer.neurons:
            neuron.weights = [
                random_generator.gauss(0.0, standard_deviation)
                for _ in neuron.weights
            ]
            neuron.bias = 0.0


def train(network):
    for _ in range(epochs):
        for inputs, target in training_data:
            prediction = network.forward(inputs)[0]

            error = prediction - target
            network.backward([error])
            network.update(learning_rate)


def evaluate(network):
    total_loss = 0.0

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        error = prediction - target
        total_loss += 0.5 * error ** 2

    return total_loss


for initialization in ["uniform", "he"]:
    successes = 0

    print(f"\n{initialization.upper()} INITIALIZATION")

    for seed in seeds:
        network = Network(
            number_of_inputs=2,
            layer_sizes=[2, 1],
            activations=["relu", "linear"],
            seed=seed
        )

        if initialization == "he":
            apply_he_initialization(network, seed)

        train(network)

        total_loss = evaluate(network)
        successful = total_loss < 1e-8

        if successful:
            successes += 1

        print(
            f"seed={seed:2d} "
            f"loss={total_loss:.12f} "
            f"success={successful}"
        )

    print(f"successful_runs={successes}/{len(seeds)}")
