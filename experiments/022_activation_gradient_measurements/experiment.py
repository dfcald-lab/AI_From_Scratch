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


def measure_initial_state(network):
    activations = []
    gradients = []
    zero_gradients = 0
    total_gradients = 0

    hidden_layer = network.layers[0]

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        for neuron in hidden_layer.neurons:
            activations.append(neuron.activate(neuron.raw_output))

        error = prediction - target
        network.backward([error])

        for neuron in hidden_layer.neurons:
            for gradient in neuron.weight_gradients:
                gradients.append(abs(gradient))

                if gradient == 0.0:
                    zero_gradients += 1

                total_gradients += 1

            gradients.append(abs(neuron.bias_gradient))

            if neuron.bias_gradient == 0.0:
                zero_gradients += 1

            total_gradients += 1

    active_values = [value for value in activations if value > 0.0]

    return {
        "active_fraction": len(active_values) / len(activations),
        "mean_activation": mean(activations),
        "max_activation": max(activations),
        "mean_abs_gradient": mean(gradients),
        "max_abs_gradient": max(gradients),
        "zero_gradient_fraction": zero_gradients / total_gradients,
    }


for initialization in ["uniform", "he"]:
    print(f"\n{initialization.upper()} INITIALIZATION")

    results = []

    for seed in seeds:
        network = Network(
            number_of_inputs=2,
            layer_sizes=[2, 1],
            activations=["relu", "linear"],
            seed=seed
        )

        if initialization == "he":
            apply_he_initialization(network, seed)

        measurements = measure_initial_state(network)

        train(network)

        total_loss = evaluate(network)
        successful = total_loss < 1e-8

        results.append(measurements)

        print(
            f"seed={seed:2d} "
            f"active={measurements['active_fraction']:.3f} "
            f"mean_act={measurements['mean_activation']:.6f} "
            f"max_act={measurements['max_activation']:.6f} "
            f"mean_grad={measurements['mean_abs_gradient']:.6f} "
            f"max_grad={measurements['max_abs_gradient']:.6f} "
            f"zero_grad={measurements['zero_gradient_fraction']:.3f} "
            f"loss={total_loss:.12f} "
            f"success={successful}"
        )

    print("\nAVERAGES")

    for key in results[0]:
        print(f"{key}={mean(result[key] for result in results):.6f}")
