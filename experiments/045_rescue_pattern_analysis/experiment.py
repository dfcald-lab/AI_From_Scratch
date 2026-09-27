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


from src.neuron import Neuron


switch_epochs = [0, 5, 20, 50, 100]


def add_capacity(network, seed, extra_width=4):
    hidden_layer = network.layers[0]
    output_neuron = network.layers[1].neurons[0]

    rng = random.Random(seed + 10000)

    fan_in = len(hidden_layer.neurons[0].weights)
    std = math.sqrt(2.0 / fan_in)

    for _ in range(extra_width):
        hidden_layer.neurons.append(
            Neuron(
                weights=[
                    rng.gauss(0.0, std)
                    for _ in range(fan_in)
                ],
                bias=0.0,
                activation="relu",
            )
        )

    output_neuron.weights.extend([0.0] * extra_width)
    output_neuron.weight_gradients.extend([0.0] * extra_width)


def make_width2_network(seed):
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)
    return network


def run_condition(seed, switch_epoch):
    network = make_width2_network(seed)

    if switch_epoch == 0:
        add_capacity(network, seed)

    for epoch in range(epochs):
        if switch_epoch != 0 and epoch == switch_epoch:
            add_capacity(network, seed)

        batch_epoch(network, learning_rate)

    final_loss = loss(network)

    success = final_loss < 1e-6

    return final_loss, success


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]


def add_capacity_with_offset(
    network,
    seed,
    extra_width,
    initialization_offset,
):
    hidden_layer = network.layers[0]
    output_neuron = network.layers[1].neurons[0]

    rng = random.Random(
        seed + 10000 + initialization_offset
    )

    fan_in = len(hidden_layer.neurons[0].weights)
    std = math.sqrt(2.0 / fan_in)

    for _ in range(extra_width):
        hidden_layer.neurons.append(
            Neuron(
                weights=[
                    rng.gauss(0.0, std)
                    for _ in range(fan_in)
                ],
                bias=0.0,
                activation="relu",
            )
        )

    output_neuron.weights.extend(
        [0.0] * extra_width
    )

    output_neuron.weight_gradients.extend(
        [0.0] * extra_width
    )


def run(seed, initialization_offset):
    network = make_width2_network(seed)

    for epoch in range(epochs):
        if epoch == injection_epoch:
            add_capacity_with_offset(
                network,
                seed,
                extra_width,
                initialization_offset,
            )

        batch_epoch(
            network,
            learning_rate,
        )

    final_loss = loss(network)
    final_gap = None

    if len(network.layers[0].neurons) == 2:
        final_gap = separation_gap(network)

    success = final_loss < 1e-6

    return network, final_loss, success


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]


def new_neuron_activation_patterns(network):
    patterns = []

    hidden_neurons = network.layers[0].neurons[-extra_width:]

    for neuron in hidden_neurons:
        values = []

        for inputs, _ in training_data:
            network.forward(inputs)
            values.append(
                neuron.activate(neuron.raw_output)
            )

        pattern = "".join(
            "1" if value > 1e-12 else "0"
            for value in values
        )

        patterns.append(pattern)

    return patterns


def run(seed, initialization_offset):
    network = make_width2_network(seed)

    for epoch in range(epochs):
        if epoch == injection_epoch:
            add_capacity_with_offset(
                network,
                seed,
                extra_width,
                initialization_offset,
            )

            injection_patterns = (
                new_neuron_activation_patterns(network)
            )

        batch_epoch(
            network,
            learning_rate,
        )

    final_loss = loss(network)
    success = final_loss < 1e-6

    return (
        injection_patterns,
        final_loss,
        success,
    )


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]


def run(seed, initialization_offset):
    network = make_width2_network(seed)

    for epoch in range(epochs):
        if epoch == injection_epoch:
            add_capacity_with_offset(
                network,
                seed,
                extra_width,
                initialization_offset,
            )

            injection_patterns = (
                new_neuron_activation_patterns(network)
            )

        batch_epoch(
            network,
            learning_rate,
        )

    final_loss = loss(network)
    success = final_loss < 1e-6

    # Neuron order should not matter, so canonicalize the pair.
    canonical_pattern = "|".join(
        sorted(injection_patterns)
    )

    return (
        canonical_pattern,
        final_loss,
        success,
    )


print("\n" + "=" * 60)
print("HE INITIALIZATION — RESCUE ACTIVATION PATTERN ANALYSIS")
print("Start with width 2; add 2 neurons at epoch 100.")
print("Only the new-neuron initialization changes.")
print("Neuron order is canonicalized before aggregation.")
print("Pattern order: [0,0], [0,1], [1,0], [1,1].")

results = []

for offset in initialization_offsets:
    print("\n" + "-" * 60)
    print(f"INITIALIZATION OFFSET: {offset}")

    for seed in seeds:
        pattern, final_loss, success = run(
            seed,
            offset,
        )

        results.append({
            "seed": seed,
            "offset": offset,
            "pattern": pattern,
            "loss": final_loss,
            "success": success,
        })

        print(
            f"seed={seed} "
            f"pattern={pattern} "
            f"result={'SUCCESS' if success else 'FAIL'} "
            f"loss={final_loss:.9f}"
        )

print("\n" + "=" * 60)
print("EXACT PATTERN SUMMARY")

patterns = sorted(
    {
        result["pattern"]
        for result in results
    }
)

for pattern in patterns:
    matching = [
        result
        for result in results
        if result["pattern"] == pattern
    ]

    successful = sum(
        result["success"]
        for result in matching
    )

    failed = len(matching) - successful

    print(
        f"pattern={pattern} "
        f"cases={len(matching)} "
        f"success={successful} "
        f"fail={failed} "
        f"success_rate={successful / len(matching):.3f}"
    )

print("\n" + "=" * 60)
print("FAILED PATTERNS")

for result in results:
    if not result["success"]:
        print(
            f"seed={result['seed']} "
            f"offset={result['offset']} "
            f"pattern={result['pattern']} "
            f"loss={result['loss']:.9f}"
        )
