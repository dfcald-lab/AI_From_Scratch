import math
import random

from src.network import Network
from src.neuron import Neuron


training_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

learning_rate = 0.10
epochs = 4000
injection_epoch = 100

seeds = range(10)
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]

extra_width = 2
success_threshold = 1e-6


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


def make_width2_network(seed):
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    return network


def batch_gradients(network):
    accumulated = []

    for layer in network.layers:
        accumulated.append(
            [
                {
                    "weights": [0.0 for _ in neuron.weights],
                    "bias": 0.0,
                }
                for neuron in layer.neurons
            ]
        )

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        network.backward([prediction - target])

        for layer_index, layer in enumerate(network.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                for weight_index, gradient in enumerate(
                    neuron.weight_gradients
                ):
                    accumulated[layer_index][neuron_index]["weights"][
                        weight_index
                    ] += gradient

                accumulated[layer_index][neuron_index]["bias"] += (
                    neuron.bias_gradient
                )

    return accumulated


def batch_epoch(network):
    gradients = batch_gradients(network)
    batch_size = len(training_data)

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.weight_gradients = [
                gradient / batch_size
                for gradient in gradients[layer_index][neuron_index]["weights"]
            ]

            neuron.bias_gradient = (
                gradients[layer_index][neuron_index]["bias"]
                / batch_size
            )

    network.update(learning_rate)


def loss(network):
    total = 0.0

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]
        error = prediction - target
        total += 0.5 * error ** 2

    return total


def standard_new_weights(seed, initialization_offset):
    rng = random.Random(
        seed + 10000 + initialization_offset
    )

    fan_in = 2
    std = math.sqrt(2.0 / fan_in)

    first = [
        rng.gauss(0.0, std)
        for _ in range(fan_in)
    ]

    second = [
        rng.gauss(0.0, std)
        for _ in range(fan_in)
    ]

    return first, second


def append_new_neurons(
    network,
    weights,
    biases,
):
    hidden_layer = network.layers[0]
    output_neuron = network.layers[1].neurons[0]

    for weight_vector, bias in zip(weights, biases):
        hidden_layer.neurons.append(
            Neuron(
                weights=list(weight_vector),
                bias=bias,
                activation="relu",
            )
        )

    output_neuron.weights.extend(
        [0.0] * extra_width
    )

    output_neuron.weight_gradients.extend(
        [0.0] * extra_width
    )


def initialization_scale(seed, initialization_offset):
    first, second = standard_new_weights(
        seed,
        initialization_offset,
    )

    first_norm = math.sqrt(
        sum(value * value for value in first)
    )

    second_norm = math.sqrt(
        sum(value * value for value in second)
    )

    return (
        first_norm + second_norm
    ) / 2.0


def construct_weights(
    condition,
    seed,
    initialization_offset,
):
    target_norm = initialization_scale(
        seed,
        initialization_offset,
    )

    a = target_norm / math.sqrt(2.0)

    if condition == "standard":
        first, second = standard_new_weights(
            seed,
            initialization_offset,
        )

        return (
            [first, second],
            [0.0, 0.0],
        )

    if condition == "target_selective":
        return (
            [
                [a, -a],
                [-a, a],
            ],
            [0.0, 0.0],
        )

    if condition == "asymmetric_selective":
        return (
            [
                [a, -a],
                [-0.75 * a, 0.75 * a],
            ],
            [0.0, 0.0],
        )

    if condition == "hinge_decomposition":
        return (
            [
                [a, a],
                [-a, -a],
            ],
            [0.0, -a],
        )

    if condition == "axis_pair":
        return (
            [
                [target_norm, 0.0],
                [0.0, target_norm],
            ],
            [0.0, 0.0],
        )

    if condition == "redundant":
        return (
            [
                [a, -a],
                [a, -a],
            ],
            [0.0, 0.0],
        )

    raise ValueError(
        f"Unknown condition: {condition}"
    )


def new_activation_patterns(network):
    hidden_neurons = network.layers[0].neurons[-extra_width:]

    patterns = []

    for neuron in hidden_neurons:
        values = []

        for inputs, _ in training_data:
            network.forward(inputs)
            values.append(
                neuron.activate(neuron.raw_output)
            )

        patterns.append(
            "".join(
                "1" if value > 1e-12 else "0"
                for value in values
            )
        )

    return patterns


def run(
    condition,
    seed,
    initialization_offset,
):
    network = make_width2_network(seed)

    for _ in range(injection_epoch):
        batch_epoch(network)

    weights, biases = construct_weights(
        condition,
        seed,
        initialization_offset,
    )

    append_new_neurons(
        network,
        weights,
        biases,
    )

    patterns_at_injection = new_activation_patterns(
        network
    )

    for _ in range(
        epochs - injection_epoch
    ):
        batch_epoch(network)

    final_loss = loss(network)
    success = final_loss < success_threshold

    return (
        patterns_at_injection,
        final_loss,
        success,
    )


conditions = [
    "standard",
    "target_selective",
    "asymmetric_selective",
    "hinge_decomposition",
    "axis_pair",
    "redundant",
]


print("\n" + "=" * 70)
print("EXPERIMENT 060 — COMPLEMENTARITY GENERALIZATION")
print("Width 2 -> add 2 hidden neurons at epoch 100")
print("Learning rate = 0.10")
print("Seeds = 0-9")
print("Initialization offsets = [0, 1, 2, 3, 4, 5, 10, 100]")
print("New output weights start at zero.")
print("=" * 70)

all_results = {}

for condition in conditions:
    successes = []
    losses = []
    pattern_counts = {}

    print("\n" + "-" * 70)
    print(condition)

    for offset in initialization_offsets:
        offset_successes = 0

        for seed in seeds:
            (
                patterns,
                final_loss,
                success,
            ) = run(
                condition,
                seed,
                offset,
            )

            losses.append(final_loss)

            canonical_pattern = "|".join(
                sorted(patterns)
            )

            if canonical_pattern not in pattern_counts:
                pattern_counts[
                    canonical_pattern
                ] = {
                    "success": 0,
                    "failure": 0,
                }

            if success:
                pattern_counts[
                    canonical_pattern
                ]["success"] += 1

                successes.append(
                    (seed, offset)
                )

                offset_successes += 1
            else:
                pattern_counts[
                    canonical_pattern
                ]["failure"] += 1

        print(
            f"offset={offset:3d} "
            f"successful={offset_successes}/10"
        )

    mean_loss = sum(losses) / len(losses)

    all_results[condition] = {
        "successes": len(successes),
        "mean_loss": mean_loss,
        "patterns": pattern_counts,
    }

    print(
        f"\nsummary {condition}: "
        f"successful={len(successes)}/80 "
        f"mean_loss={mean_loss:.12f}"
    )

    print("activation patterns at injection:")

    for pattern, counts in sorted(
        pattern_counts.items()
    ):
        print(
            f"  {pattern} "
            f"success={counts['success']} "
            f"fail={counts['failure']}"
        )


print("\n" + "=" * 70)
print("EXPERIMENT 060 SUMMARY")
print("=" * 70)

for condition in conditions:
    result = all_results[condition]

    print(
        f"{condition:24s} "
        f"successful={result['successes']:2d}/80 "
        f"mean_loss={result['mean_loss']:.12f}"
    )
