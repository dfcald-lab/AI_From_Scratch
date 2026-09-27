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

orders = {
    "fixed": [0, 1, 2, 3],
    "reverse": [3, 2, 1, 0],
    "oddeven": [1, 3, 0, 2],
    "evenodd": [0, 2, 1, 3],
}


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


def train_epoch(network, order):
    for index in order:
        inputs, target = training_data[index]
        prediction = network.forward(inputs)[0]
        network.backward([prediction - target])
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
    values = []

    for inputs, _ in training_data:
        values.append(network.forward(inputs)[0])

    return values


print("\n" + "=" * 60)
print("HE INITIALIZATION — TRAINING ORDER SENSITIVITY")

summary = {}

for order_name, order in orders.items():
    print(f"\n{'-' * 60}")
    print(f"ORDER: {order_name} {order}")

    successful = 0
    final_losses = []
    final_gaps = []

    for seed in seeds:
        network = Network(
            number_of_inputs=2,
            layer_sizes=[2, 1],
            activations=["relu", "linear"],
            seed=seed,
        )

        apply_he_initialization(network, seed)

        for _ in range(epochs):
            train_epoch(network, order)

        final_loss = loss(network)
        final_gap = separation_gap(network)

        if final_loss < 1e-6 and final_gap > 1e-9:
            successful += 1

        final_losses.append(final_loss)
        final_gaps.append(final_gap)

        print(
            f"seed={seed} "
            f"loss={final_loss:.12f} "
            f"gap={final_gap:.9f} "
            f"separable={'YES' if final_gap > 1e-9 else 'NO'}"
        )

        print(
            "  predictions="
            + " ".join(f"{value:.6f}" for value in predictions(network))
        )

    summary[order_name] = {
        "successful": successful,
        "mean_loss": sum(final_losses) / len(final_losses),
        "mean_gap": sum(final_gaps) / len(final_gaps),
    }

print("\n" + "=" * 60)
print("SUMMARY")

for order_name, result in summary.items():
    print(
        f"{order_name:8s} "
        f"successful={result['successful']}/10 "
        f"mean_loss={result['mean_loss']:.12f} "
        f"mean_gap={result['mean_gap']:.9f}"
    )
