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


hidden_widths = [1, 2, 3, 4, 6]

print("\n" + "=" * 60)
print("HE INITIALIZATION — HIDDEN CAPACITY SWEEP")
print("Full-batch training, learning rate 0.1.")
print("Hidden widths:", hidden_widths)

all_results = {}

for hidden_width in hidden_widths:
    successful = 0
    total_loss = 0.0
    total_gap = 0.0

    print(f"\n{'-' * 60}")
    print(f"HIDDEN WIDTH: {hidden_width}")

    seed_results = []

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

        if hidden_width == 2:
            final_gap = separation_gap(network)
            total_gap += final_gap
        else:
            final_gap = None

        successful += int(success)
        total_loss += final_loss

        seed_results.append(
            (seed, final_loss, final_gap, success)
        )

        if final_gap is None:
            print(
                f"seed={seed} "
                f"loss={final_loss:.12f} "
                f"successful={'YES' if success else 'NO'}"
            )
        else:
            print(
                f"seed={seed} "
                f"loss={final_loss:.12f} "
                f"gap={final_gap:.9f} "
                f"successful={'YES' if success else 'NO'}"
            )

    all_results[hidden_width] = seed_results

    print(
        f"\nsummary width={hidden_width}: "
        f"successful={successful}/10 "
        f"mean_loss={total_loss / 10:.12f}"
    )

    if hidden_width == 2:
        print(f"mean_gap={total_gap / 10:.9f}")

print("\n" + "=" * 60)
print("CAPACITY SUMMARY")

for hidden_width in hidden_widths:
    seed_results = all_results[hidden_width]

    successful_seeds = [
        seed for seed, _, _, success in seed_results
        if success
    ]

    mean_loss = (
        sum(loss_value for _, loss_value, _, _ in seed_results)
        / len(seed_results)
    )

    print(
        f"width={hidden_width} "
        f"successful={len(successful_seeds)}/10 "
        f"seeds={successful_seeds} "
        f"mean_loss={mean_loss:.12f}"
    )
