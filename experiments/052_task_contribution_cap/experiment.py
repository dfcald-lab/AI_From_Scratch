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


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]
relative_checkpoints = [0, 1, 2, 3, 5, 10, 20]


def new_neuron_metrics(network):
    hidden_neurons = network.layers[0].neurons[-extra_width:]
    output_neuron = network.layers[1].neurons[0]

    active_counts = []

    for neuron in hidden_neurons:
        count = 0

        for inputs, _ in training_data:
            network.forward(inputs)

            if neuron.activate(neuron.raw_output) > 1e-12:
                count += 1

        active_counts.append(count)

    accumulated = batch_gradients(network)
    hidden_index = len(network.layers[0].neurons) - extra_width

    gradients = []

    for neuron_index in range(
        hidden_index,
        hidden_index + extra_width,
    ):
        gradients.extend(
            accumulated[0][neuron_index]["weights"]
        )

        gradients.append(
            accumulated[0][neuron_index]["bias"]
        )

    mean_abs_gradient = (
        sum(abs(value) for value in gradients)
        / len(gradients)
    )

    zero_gradient_fraction = (
        sum(abs(value) < 1e-12 for value in gradients)
        / len(gradients)
    )

    output_weights = output_neuron.weights[-extra_width:]

    mean_abs_output_weight = (
        sum(abs(value) for value in output_weights)
        / len(output_weights)
    )

    return (
        active_counts,
        mean_abs_output_weight,
        mean_abs_gradient,
        zero_gradient_fraction,
    )


def run(seed, initialization_offset):
    network = make_width2_network(seed)

    for _ in range(injection_epoch):
        batch_epoch(network, learning_rate)

    add_capacity_with_offset(
        network,
        seed,
        extra_width,
        initialization_offset,
    )

    trajectory = {}

    trajectory[0] = (
        loss(network),
        new_neuron_metrics(network),
    )

    for relative_epoch in range(1, max(relative_checkpoints) + 1):
        batch_epoch(network, learning_rate)

        if relative_epoch in relative_checkpoints:
            trajectory[relative_epoch] = (
                loss(network),
                new_neuron_metrics(network),
            )

    remaining_updates = (
        epochs - injection_epoch - max(relative_checkpoints)
    )

    for _ in range(remaining_updates):
        batch_epoch(network, learning_rate)

    final_loss = loss(network)
    success = final_loss < 1e-6

    return trajectory, final_loss, success


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]
relative_checkpoints = [0, 1, 5, 20, 100, 500, 3900]


def learned_contribution_metrics(network):
    hidden_neurons = network.layers[0].neurons[-extra_width:]
    output_neuron = network.layers[1].neurons[0]
    output_weights = output_neuron.weights[-extra_width:]

    metrics = []

    for neuron, output_weight in zip(
        hidden_neurons,
        output_weights,
    ):
        activations = []

        for inputs, _ in training_data:
            network.forward(inputs)
            activations.append(
                neuron.activate(neuron.raw_output)
            )

        contributions = [
            output_weight * activation
            for activation in activations
        ]

        active_count = sum(
            value > 1e-12
            for value in activations
        )

        mean_activation = (
            sum(activations)
            / len(activations)
        )

        mean_abs_contribution = (
            sum(abs(value) for value in contributions)
            / len(contributions)
        )

        contribution_norm = math.sqrt(
            sum(value * value for value in contributions)
        )

        metrics.append({
            "output_weight": output_weight,
            "active_count": active_count,
            "mean_activation": mean_activation,
            "mean_abs_contribution": mean_abs_contribution,
            "contribution_norm": contribution_norm,
        })

    total_contribution_norm = math.sqrt(
        sum(
            metric["contribution_norm"]
            ** 2
            for metric in metrics
        )
    )

    return metrics, total_contribution_norm


def run(seed, initialization_offset):
    network = make_width2_network(seed)

    for _ in range(injection_epoch):
        batch_epoch(network, learning_rate)

    add_capacity_with_offset(
        network,
        seed,
        extra_width,
        initialization_offset,
    )

    trajectory = {}

    trajectory[0] = (
        loss(network),
        learned_contribution_metrics(network),
    )

    for relative_epoch in range(
        1,
        max(relative_checkpoints) + 1,
    ):
        batch_epoch(
            network,
            learning_rate,
        )

        if relative_epoch in relative_checkpoints:
            trajectory[relative_epoch] = (
                loss(network),
                learned_contribution_metrics(network),
            )

    final_loss = loss(network)
    success = final_loss < 1e-6

    return trajectory, final_loss, success


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]

# Maximum absolute output weight allowed for each new neuron.
# None means no cap.
output_weight_caps = [0.02, 0.05, 0.10, 0.20, None]


def clamp_new_output_weights(network, cap):
    if cap is None:
        return

    output_neuron = network.layers[1].neurons[0]

    for index in range(
        len(output_neuron.weights) - extra_width,
        len(output_neuron.weights),
    ):
        weight = output_neuron.weights[index]

        if weight > cap:
            output_neuron.weights[index] = cap
        elif weight < -cap:
            output_neuron.weights[index] = -cap


def run(seed, initialization_offset, cap):
    network = make_width2_network(seed)

    for _ in range(injection_epoch):
        batch_epoch(
            network,
            learning_rate,
        )

    add_capacity_with_offset(
        network,
        seed,
        extra_width,
        initialization_offset,
    )

    for _ in range(epochs - injection_epoch):
        batch_epoch(
            network,
            learning_rate,
        )

        clamp_new_output_weights(
            network,
            cap,
        )

    final_loss = loss(network)
    success = final_loss < 1e-6

    return final_loss, success


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]

contribution_caps = [0.10, 0.25, 0.50, 1.00, None]


def neuron_contribution_norm(network, neuron_index):
    hidden_neuron = network.layers[0].neurons[neuron_index]
    output_neuron = network.layers[1].neurons[0]

    output_weight = output_neuron.weights[neuron_index]

    contributions = []

    for inputs, _ in training_data:
        network.forward(inputs)

        activation = hidden_neuron.activate(
            hidden_neuron.raw_output
        )

        contributions.append(
            output_weight * activation
        )

    return math.sqrt(
        sum(value * value for value in contributions)
    )


def clamp_new_contribution_norms(network, cap):
    if cap is None:
        return

    first_new_index = (
        len(network.layers[0].neurons)
        - extra_width
    )

    output_neuron = network.layers[1].neurons[0]

    for neuron_index in range(
        first_new_index,
        first_new_index + extra_width,
    ):
        contribution_norm = neuron_contribution_norm(
            network,
            neuron_index,
        )

        if contribution_norm <= cap:
            continue

        if contribution_norm == 0.0:
            continue

        scale = cap / contribution_norm

        output_neuron.weights[neuron_index] *= scale


def run(seed, initialization_offset, cap):
    network = make_width2_network(seed)

    for _ in range(injection_epoch):
        batch_epoch(
            network,
            learning_rate,
        )

    add_capacity_with_offset(
        network,
        seed,
        extra_width,
        initialization_offset,
    )

    for _ in range(epochs - injection_epoch):
        batch_epoch(
            network,
            learning_rate,
        )

        clamp_new_contribution_norms(
            network,
            cap,
        )

    final_loss = loss(network)
    success = final_loss < 1e-6

    return final_loss, success


print("\n" + "=" * 60)
print("HE INITIALIZATION — TASK CONTRIBUTION CAP")
print("Width 2 + 2 neurons injected at epoch 100.")
print("Caps apply to each new neuron's contribution norm")
print("across the four XOR examples.")
print("Learning rate 0.1, full-batch training, 4000 updates.")
print("Contribution caps:", contribution_caps)

results = {}

for cap in contribution_caps:
    successful_cases = []
    losses = []

    label = (
        "uncapped"
        if cap is None
        else f"cap_{cap}"
    )

    print("\n" + "-" * 60)
    print(label)

    for offset in initialization_offsets:
        offset_success = 0

        for seed in seeds:
            final_loss, success = run(
                seed,
                offset,
                cap,
            )

            losses.append(final_loss)

            if success:
                successful_cases.append(
                    (seed, offset)
                )
                offset_success += 1

        print(
            f"offset={offset} "
            f"successful={offset_success}/10"
        )

    mean_loss = sum(losses) / len(losses)

    results[cap] = (
        successful_cases,
        mean_loss,
    )

    print(
        f"\nsummary {label}: "
        f"successful={len(successful_cases)}/80 "
        f"mean_loss={mean_loss:.12f}"
    )

print("\n" + "=" * 60)
print("TASK CONTRIBUTION CAP SUMMARY")

for cap in contribution_caps:
    successful_cases, mean_loss = results[cap]

    label = (
        "uncapped"
        if cap is None
        else f"{cap:.2f}"
    )

    print(
        f"cap={label:8s} "
        f"successful={len(successful_cases)}/80 "
        f"mean_loss={mean_loss:.12f}"
    )
