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


def hidden_points(network):
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

    return points


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


def representation_geometry(network):
    points = hidden_points(network)

    negative_a = points[0]  # [0,0]
    positive_a = points[1]  # [0,1]
    positive_b = points[2]  # [1,0]
    negative_b = points[3]  # [1,1]

    gap = segment_distance(
        positive_a,
        positive_b,
        negative_a,
        negative_b,
    )

    return points, gap, gap > 1e-9


def evaluate(network):
    loss = 0.0

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]
        error = prediction - target
        loss += 0.5 * error ** 2

    return loss


print("\n" + "=" * 60)
print("HE INITIALIZATION — REPRESENTATION GEOMETRY")
print("Positive class: [0,1], [1,0]")
print("Negative class: [0,0], [1,1]")
print("gap = distance between the two class segments")

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

        points, gap, separable = representation_geometry(network)

        print(
            f"epoch={checkpoint:4d} "
            f"loss={evaluate(network):.12f} "
            f"gap={gap:.9f} "
            f"separable={'YES' if separable else 'NO'}"
        )

        for row, point in zip(training_data, points):
            print(
                f"  input={row[0]} "
                f"hidden=({point[0]:.6f},{point[1]:.6f})"
            )
