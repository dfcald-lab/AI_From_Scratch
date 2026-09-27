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
seeds = range(10)
checkpoints = [0, 1, 10, 100, 500, 1000]


def apply_he_initialization(network, seed):
    rng = random.Random(seed)

    for layer in network.layers:
        if layer.neurons[0].activation != "relu":
            continue

        fan_in = len(layer.neurons[0].weights)
        std = math.sqrt(2.0 / fan_in)

        for neuron in layer.neurons:
            neuron.weights = [
                rng.gauss(0.0, std)
                for _ in neuron.weights
            ]
            neuron.bias = 0.0


def train_one_epoch(network):
    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]
        network.backward([prediction - target])
        network.update(learning_rate)


def inspect(network):
    hidden = network.layers[0].neurons
    output = network.layers[1].neurons[0]

    rows = []

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        h0 = hidden[0].activate(hidden[0].raw_output)
        h1 = hidden[1].activate(hidden[1].raw_output)

        rows.append(
            f"  input={inputs} target={target:.0f} "
            f"h=({h0:.6f},{h1:.6f}) "
            f"pred={prediction:.6f}"
        )

    print(
        f"  output_weights={output.weights} "
        f"output_bias={output.bias:.6f}"
    )

    for row in rows:
        print(row)


print("\n" + "=" * 60)
print("HE INITIALIZATION — HIDDEN REPRESENTATION")

for seed in seeds:
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    print(f"\nseed={seed}")

    for checkpoint in checkpoints:
        while getattr(network, "_epochs_completed", 0) < checkpoint:
            train_one_epoch(network)
            network._epochs_completed = (
                getattr(network, "_epochs_completed", 0) + 1
            )

        print(f"\nepoch={checkpoint}")
        inspect(network)
