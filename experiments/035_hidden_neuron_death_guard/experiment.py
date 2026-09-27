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


def hidden_neuron_dead(network, neuron):
    for inputs, _ in training_data:
        network.forward(inputs)

        if neuron.activate(neuron.raw_output) > 1e-12:
            return False

    return True


def run_baseline(seed):
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    first_dead_epoch = None

    for epoch in range(epochs):
        batch_epoch(network, learning_rate)

        dead_neurons = [
            hidden_neuron_dead(network, neuron)
            for neuron in network.layers[0].neurons
        ]

        if any(dead_neurons) and first_dead_epoch is None:
            first_dead_epoch = epoch + 1

    return network, first_dead_epoch


def run_guarded(seed):
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    interventions = 0

    for _ in range(epochs):
        previous_state = [
            (
                list(neuron.weights),
                neuron.bias,
            )
            for neuron in network.layers[0].neurons
        ]

        batch_epoch(network, learning_rate)

        for neuron_index, neuron in enumerate(network.layers[0].neurons):
            if hidden_neuron_dead(network, neuron):
                neuron.weights = list(previous_state[neuron_index][0])
                neuron.bias = previous_state[neuron_index][1]
                interventions += 1

    return network, interventions


print("\n" + "=" * 60)
print("HE INITIALIZATION — HIDDEN NEURON DEATH GUARD")
print("Full-batch training, learning rate 0.1.")
print("Baseline vs. intervention preventing complete hidden-neuron death.")

baseline_success = 0
guarded_success = 0

for seed in seeds:
    baseline_network, first_dead_epoch = run_baseline(seed)
    guarded_network, interventions = run_guarded(seed)

    baseline_loss = loss(baseline_network)
    baseline_gap = separation_gap(baseline_network)

    guarded_loss = loss(guarded_network)
    guarded_gap = separation_gap(guarded_network)

    baseline_ok = (
        baseline_loss < 1e-6
        and baseline_gap > 1e-9
    )

    guarded_ok = (
        guarded_loss < 1e-6
        and guarded_gap > 1e-9
    )

    baseline_success += int(baseline_ok)
    guarded_success += int(guarded_ok)

    print(
        f"\nseed={seed}"
        f" baseline={'SUCCESS' if baseline_ok else 'FAIL'}"
        f" first_dead_epoch={first_dead_epoch}"
        f" loss={baseline_loss:.9f}"
        f" gap={baseline_gap:.9f}"
    )

    print(
        f"       guarded={'SUCCESS' if guarded_ok else 'FAIL'}"
        f" interventions={interventions}"
        f" loss={guarded_loss:.9f}"
        f" gap={guarded_gap:.9f}"
    )

print("\n" + "-" * 60)
print(f"BASELINE SUCCESS: {baseline_success}/10")
print(f"GUARDED SUCCESS:  {guarded_success}/10")
