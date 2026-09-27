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
seeds = [0, 2, 4, 6, 8]
window = 5


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


def gradient_summary(network):
    hidden = network.layers[0].neurons
    summaries = []

    for neuron in hidden:
        gradients = []
        zeros = 0

        for inputs, target in training_data:
            prediction = network.forward(inputs)[0]
            network.backward([prediction - target])

            for gradient in neuron.weight_gradients:
                gradients.append(abs(gradient))
                if gradient == 0.0:
                    zeros += 1

            gradients.append(abs(neuron.bias_gradient))
            if neuron.bias_gradient == 0.0:
                zeros += 1

        summaries.append(
            (
                mean(gradients),
                zeros / len(gradients),
            )
        )

    return summaries


def capture(network, epoch):
    points, gap, separable = geometry(network)
    gradients = gradient_summary(network)
    output = network.layers[1].neurons[0]

    return {
        "epoch": epoch,
        "loss": loss(network),
        "gap": gap,
        "separable": separable,
        "points": points,
        "output_weights": list(output.weights),
        "output_bias": output.bias,
        "gradients": gradients,
    }


def find_transition(seed):
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )
    apply_he_initialization(network, seed)

    _, _, previous = geometry(network)

    for epoch in range(1, epochs + 1):
        train_one_epoch(network)

        _, _, current = geometry(network)

        if current != previous:
            return epoch

        previous = current

    return None


print("\n" + "=" * 60)
print("HE INITIALIZATION — TRANSITION MICROSCOPE")

for seed in seeds:
    transition = find_transition(seed)

    if transition is None:
        print(f"\nseed={seed} no transition found")
        continue

    print(f"\nseed={seed} transition_epoch={transition}")

    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )
    apply_he_initialization(network, seed)

    start = max(0, transition - window)
    end = min(epochs, transition + window)

    snapshots = {}

    for epoch in range(0, end + 1):
        if start <= epoch <= end:
            snapshots[epoch] = capture(network, epoch)

        if epoch < end:
            train_one_epoch(network)

    for epoch in range(start, end + 1):
        state = snapshots[epoch]

        print(
            f"\nepoch={epoch:4d} "
            f"{'*** TRANSITION ***' if epoch == transition else ''}"
        )
        print(
            f"  loss={state['loss']:.12f} "
            f"gap={state['gap']:.9f} "
            f"separable={'YES' if state['separable'] else 'NO'}"
        )
        print(
            f"  output_weights={state['output_weights']} "
            f"output_bias={state['output_bias']:.9f}"
        )

        for index, point in enumerate(state["points"]):
            print(
                f"  input={training_data[index][0]} "
                f"hidden=({point[0]:.6f},{point[1]:.6f})"
            )

        for index, (mean_gradient, zero_fraction) in enumerate(
            state["gradients"]
        ):
            print(
                f"  neuron={index} "
                f"mean_grad={mean_gradient:.9f} "
                f"zero_grad={zero_fraction:.3f}"
            )
