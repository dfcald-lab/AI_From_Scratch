import math
import random
from statistics import mean

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


def evaluate(network):
    loss = 0.0

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]
        error = prediction - target
        loss += 0.5 * error ** 2

    return loss


def measure_neuron(network, neuron):
    activations = []
    gradients = []
    zero_gradients = 0
    total_gradients = 0
    pattern = []

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        activation = neuron.activate(neuron.raw_output)
        activations.append(activation)
        pattern.append("1" if activation > 0.0 else "0")

        network.backward([prediction - target])

        for gradient in neuron.weight_gradients:
            gradients.append(abs(gradient))
            total_gradients += 1
            if gradient == 0.0:
                zero_gradients += 1

        gradients.append(abs(neuron.bias_gradient))
        total_gradients += 1
        if neuron.bias_gradient == 0.0:
            zero_gradients += 1

    weight_norm = math.sqrt(sum(weight ** 2 for weight in neuron.weights))

    return {
        "active": sum(value > 0.0 for value in activations) / len(activations),
        "mean_act": mean(activations),
        "mean_grad": mean(gradients),
        "zero_grad": zero_gradients / total_gradients,
        "pattern": "".join(pattern),
        "w_norm": weight_norm,
    }


print("\n" + "=" * 60)
print("HE INITIALIZATION — INDIVIDUAL NEURON LIFETIME")

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

        print(f"epoch={checkpoint:4d} loss={evaluate(network):.12f}")

        for index, neuron in enumerate(network.layers[0].neurons):
            state = measure_neuron(network, neuron)

            print(
                f"  neuron={index} "
                f"active={state['active']:.3f} "
                f"mean_act={state['mean_act']:.6f} "
                f"mean_grad={state['mean_grad']:.6f} "
                f"zero_grad={state['zero_grad']:.3f} "
                f"pattern={state['pattern']} "
                f"w_norm={state['w_norm']:.6f}"
            )
