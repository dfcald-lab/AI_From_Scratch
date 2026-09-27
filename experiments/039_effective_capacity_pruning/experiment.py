import math
import random

from src.network import Network


training_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

learning_rate = 0.1
checkpoints = [0, 1, 2, 5, 10, 20, 50, 100, 4000]
epochs = 4000
seeds = range(10)


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


def batch_epoch(network, learning_rate):
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


def cross(a, b, c):
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def on_segment(a, b, p, eps=1e-12):
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        and abs(cross(a, b, p)) <= eps
    )


def segments_intersect(a, b, c, d):
    c1 = cross(a, b, c)
    c2 = cross(a, b, d)
    c3 = cross(c, d, a)
    c4 = cross(c, d, b)

    eps = 1e-12

    if (
        ((c1 > eps and c2 < -eps) or (c1 < -eps and c2 > eps))
        and
        ((c3 > eps and c4 < -eps) or (c3 < -eps and c4 > eps))
    ):
        return True

    return (
        on_segment(a, b, c)
        or on_segment(a, b, d)
        or on_segment(c, d, a)
        or on_segment(c, d, b)
    )


def point_segment_distance(p, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_squared = dx * dx + dy * dy

    if length_squared == 0.0:
        return math.dist(p, a)

    t = (
        (p[0] - a[0]) * dx
        + (p[1] - a[1]) * dy
    ) / length_squared

    t = max(0.0, min(1.0, t))

    closest = (
        a[0] + t * dx,
        a[1] + t * dy,
    )

    return math.dist(p, closest)


def segment_distance(a, b, c, d):
    if segments_intersect(a, b, c, d):
        return 0.0

    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def separation_gap(network):
    points = []

    for inputs, _ in training_data:
        network.forward(inputs)
        hidden = network.layers[0].neurons

        points.append(
            (
                hidden[0].activate(hidden[0].raw_output),
                hidden[1].activate(hidden[1].raw_output),
            )
        )

    return segment_distance(
        points[1],
        points[2],
        points[0],
        points[3],
    )


def predictions(network):
    return [
        network.forward(inputs)[0]
        for inputs, _ in training_data
    ]


from itertools import combinations


hidden_width = 6


def hidden_activity(network):
    activity = []

    for neuron in network.layers[0].neurons:
        values = []

        for inputs, _ in training_data:
            network.forward(inputs)
            values.append(
                neuron.activate(neuron.raw_output)
            )

        activity.append(values)

    return activity


def prune_mask_loss(network, mask):
    output_neuron = network.layers[1].neurons[0]
    original_weights = list(output_neuron.weights)

    output_neuron.weights = [
        weight if mask[index] else 0.0
        for index, weight in enumerate(original_weights)
    ]

    value = loss(network)

    output_neuron.weights = original_weights

    return value


print("\n" + "=" * 60)
print("HE INITIALIZATION — EFFECTIVE CAPACITY BY PRUNING")
print("Width 6, full-batch training, learning rate 0.1.")
print("Every hidden-neuron subset is tested after training.")

successful = 0
minimum_sizes = []

for seed in seeds:
    network = Network(
        number_of_inputs=2,
        layer_sizes=[hidden_width, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    for _ in range(epochs):
        batch_epoch(network, learning_rate)

    final_loss = loss(network)
    success = final_loss < 1e-6

    if success:
        successful += 1

    activity = hidden_activity(network)

    active_counts = [
        sum(value > 1e-12 for value in values)
        for values in activity
    ]

    print(
        f"\nseed={seed} "
        f"result={'SUCCESS' if success else 'FAIL'} "
        f"loss={final_loss:.12f}"
    )

    print(
        "  active_examples_per_neuron="
        + " ".join(str(value) for value in active_counts)
    )

    if not success:
        continue

    best_size = hidden_width
    best_masks = []

    for size in range(hidden_width + 1):
        found = []

        for indices in combinations(range(hidden_width), size):
            mask = [
                index in indices
                for index in range(hidden_width)
            ]

            pruned_loss = prune_mask_loss(network, mask)

            if pruned_loss < 1e-6:
                found.append((mask, pruned_loss))

        if found:
            best_size = size
            best_masks = found
            break

    minimum_sizes.append(best_size)

    print(
        f"  minimum_neurons_needed={best_size}"
        f"  successful_masks={len(best_masks)}"
    )

    if best_masks:
        mask, pruned_loss = best_masks[0]

        print(
            "  example_mask="
            + "".join("1" if value else "0" for value in mask)
            + f" loss={pruned_loss:.12f}"
        )

print("\n" + "-" * 60)
print(f"WIDTH-6 SUCCESS: {successful}/10")

if minimum_sizes:
    print(
        "MINIMUM NEURONS AMONG SUCCESSFUL RUNS:",
        minimum_sizes,
    )

    print(
        "MEAN MINIMUM NEURONS:",
        f"{sum(minimum_sizes) / len(minimum_sizes):.2f}",
    )
