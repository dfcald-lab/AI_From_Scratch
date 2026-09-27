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

# Seeds with informative transitions from Experiment 027.
seed_transitions = {
    0: 27,
    2: 131,
    4: 146,
    6: 1,
    8: 80,
}

window = 2


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


def geometry(network):
    points = hidden_points(network)

    positive_a = points[1]
    positive_b = points[2]
    negative_a = points[0]
    negative_b = points[3]

    gap = segment_distance(
        positive_a,
        positive_b,
        negative_a,
        negative_b,
    )

    return points, gap, gap > 1e-9


def loss(network):
    total = 0.0

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]
        error = prediction - target
        total += 0.5 * error ** 2

    return total


def capture_gradients(network, inputs, target):
    prediction = network.forward(inputs)[0]
    network.backward([prediction - target])

    gradients = []

    for neuron in network.layers[0].neurons:
        gradients.append(
            {
                "weights": list(neuron.weight_gradients),
                "bias": neuron.bias_gradient,
                "norm": math.sqrt(
                    sum(g ** 2 for g in neuron.weight_gradients)
                    + neuron.bias_gradient ** 2
                ),
            }
        )

    return prediction, gradients


def describe_update(
    network,
    inputs,
    target,
    epoch,
    sample_index,
):
    before_points, before_gap, before_separable = geometry(network)
    before_loss = loss(network)

    prediction, gradients = capture_gradients(
        network,
        inputs,
        target,
    )

    output = network.layers[1].neurons[0]

    print(
        f"\n  sample={sample_index} "
        f"input={inputs} "
        f"target={target:.0f}"
    )
    print(
        f"    before loss={before_loss:.12f} "
        f"gap={before_gap:.9f} "
        f"separable={'YES' if before_separable else 'NO'} "
        f"prediction={prediction:.9f}"
    )

    for index, gradient in enumerate(gradients):
        print(
            f"    neuron={index} "
            f"grad_weights={gradient['weights']} "
            f"grad_bias={gradient['bias']:.9f} "
            f"grad_norm={gradient['norm']:.9f}"
        )

    print(
        f"    output_grad_weights={output.weight_gradients} "
        f"output_grad_bias={output.bias_gradient:.9f}"
    )

    network.update(learning_rate)

    after_points, after_gap, after_separable = geometry(network)
    after_loss = loss(network)

    print(
        f"    after  loss={after_loss:.12f} "
        f"gap={after_gap:.9f} "
        f"delta_gap={after_gap - before_gap:+.9f} "
        f"separable={'YES' if after_separable else 'NO'}"
    )

    for index, (before, after) in enumerate(
        zip(before_points, after_points)
    ):
        print(
            f"    hidden_point input={training_data[index][0]} "
            f"before=({before[0]:.6f},{before[1]:.6f}) "
            f"after=({after[0]:.6f},{after[1]:.6f})"
        )


print("\n" + "=" * 60)
print("HE INITIALIZATION — GRADIENT GEOMETRY")

for seed, transition in seed_transitions.items():
    print(f"\nseed={seed} transition_epoch={transition}")

    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    start_epoch = max(0, transition - window)
    end_epoch = transition + window

    for epoch in range(1, end_epoch + 1):
        if start_epoch <= epoch <= end_epoch:
            print(f"\n epoch={epoch}")

            for sample_index, (inputs, target) in enumerate(training_data):
                describe_update(
                    network,
                    inputs,
                    target,
                    epoch,
                    sample_index,
                )
        else:
            train_one_epoch(network)

        if start_epoch <= epoch <= end_epoch:
            # The four sample updates were already performed above.
            continue
