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

boost_factors = [1.0, 2.0, 4.0, 8.0]
boost_steps = 20


def boosted_batch_epoch(
    network,
    learning_rate,
    boost_factor,
):
    output_neuron = network.layers[1].neurons[0]

    old_output_weights = list(
        output_neuron.weights
    )

    batch_epoch(
        network,
        learning_rate,
    )

    first_new_index = (
        len(output_neuron.weights)
        - extra_width
    )

    for index in range(
        first_new_index,
        len(output_neuron.weights),
    ):
        old_weight = old_output_weights[index]
        normal_weight = output_neuron.weights[index]

        delta = normal_weight - old_weight

        output_neuron.weights[index] = (
            old_weight
            + boost_factor * delta
        )


def run(seed, initialization_offset, boost_factor):
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

    for step in range(
        epochs - injection_epoch
    ):
        if step < boost_steps:
            boosted_batch_epoch(
                network,
                learning_rate,
                boost_factor,
            )
        else:
            batch_epoch(
                network,
                learning_rate,
            )

    final_loss = loss(network)
    success = final_loss < 1e-6

    return final_loss, success


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]

hidden_boost_factors = [0.0, 1.0, 2.0, 4.0]
boost_steps = 20


def hidden_boost_batch_epoch(
    network,
    learning_rate,
    boost_factor,
):
    first_new_index = (
        len(network.layers[0].neurons)
        - extra_width
    )

    old_weights = [
        list(network.layers[0].neurons[index].weights)
        for index in range(
            first_new_index,
            first_new_index + extra_width,
        )
    ]

    old_biases = [
        network.layers[0].neurons[index].bias
        for index in range(
            first_new_index,
            first_new_index + extra_width,
        )
    ]

    batch_epoch(
        network,
        learning_rate,
    )

    for local_index, neuron_index in enumerate(
        range(
            first_new_index,
            first_new_index + extra_width,
        )
    ):
        neuron = network.layers[0].neurons[neuron_index]

        normal_weights = list(neuron.weights)

        for weight_index in range(len(normal_weights)):
            delta = (
                normal_weights[weight_index]
                - old_weights[local_index][weight_index]
            )

            neuron.weights[weight_index] = (
                old_weights[local_index][weight_index]
                + boost_factor * delta
            )

        normal_bias_delta = (
            neuron.bias
            - old_biases[local_index]
        )

        neuron.bias = (
            old_biases[local_index]
            + boost_factor * normal_bias_delta
        )


def run(seed, initialization_offset, boost_factor):
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

    for step in range(
        epochs - injection_epoch
    ):
        if step < boost_steps:
            hidden_boost_batch_epoch(
                network,
                learning_rate,
                boost_factor,
            )
        else:
            batch_epoch(
                network,
                learning_rate,
            )

    final_loss = loss(network)
    success = final_loss < 1e-6

    return final_loss, success


injection_epoch = 100
extra_width = 2
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]
relative_checkpoints = [0, 1, 5, 20, 100, 500, 3900]


def cosine_similarity(a, b):
    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(y * y for y in b)
    )

    if norm_a == 0.0 or norm_b == 0.0:
        return None

    return dot / (norm_a * norm_b)


def hidden_direction_metrics(
    network,
    initial_weights,
    initial_biases,
):
    metrics = []

    first_new_index = (
        len(network.layers[0].neurons)
        - extra_width
    )

    for local_index, neuron_index in enumerate(
        range(
            first_new_index,
            first_new_index + extra_width,
        )
    ):
        neuron = network.layers[0].neurons[
            neuron_index
        ]

        similarity = cosine_similarity(
            initial_weights[local_index],
            neuron.weights,
        )

        displacement = math.sqrt(
            sum(
                (
                    current - initial
                ) ** 2
                for current, initial in zip(
                    neuron.weights,
                    initial_weights[local_index],
                )
            )
        )

        bias_change = abs(
            neuron.bias
            - initial_biases[local_index]
        )

        metrics.append({
            "cosine": similarity,
            "displacement": displacement,
            "bias_change": bias_change,
        })

    valid_cosines = [
        metric["cosine"]
        for metric in metrics
        if metric["cosine"] is not None
    ]

    mean_cosine = (
        sum(valid_cosines)
        / len(valid_cosines)
        if valid_cosines
        else 0.0
    )

    mean_displacement = (
        sum(
            metric["displacement"]
            for metric in metrics
        )
        / len(metrics)
    )

    mean_bias_change = (
        sum(
            metric["bias_change"]
            for metric in metrics
        )
        / len(metrics)
    )

    return (
        mean_cosine,
        mean_displacement,
        mean_bias_change,
    )


def run(seed, initialization_offset):
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

    first_new_index = (
        len(network.layers[0].neurons)
        - extra_width
    )

    initial_weights = [
        list(
            network.layers[0].neurons[
                first_new_index + index
            ].weights
        )
        for index in range(extra_width)
    ]

    initial_biases = [
        network.layers[0].neurons[
            first_new_index + index
        ].bias
        for index in range(extra_width)
    ]

    trajectory = {}

    trajectory[0] = (
        loss(network),
        hidden_direction_metrics(
            network,
            initial_weights,
            initial_biases,
        ),
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
                hidden_direction_metrics(
                    network,
                    initial_weights,
                    initial_biases,
                ),
            )

    remaining_updates = (
        epochs
        - injection_epoch
        - max(relative_checkpoints)
    )

    for _ in range(remaining_updates):
        batch_epoch(
            network,
            learning_rate,
        )

    final_loss = loss(network)
    success = final_loss < 1e-6

    return trajectory, final_loss, success


print("\n" + "=" * 60)
print("HE INITIALIZATION — HIDDEN FEATURE DIRECTION")
print("Width 2 + 2 neurons injected at epoch 100.")
print("Tracking movement of new hidden-weight vectors.")
print("Offsets:", initialization_offsets)

results = []

for offset in initialization_offsets:
    for seed in seeds:
        (
            trajectory,
            final_loss,
            success,
        ) = run(
            seed,
            offset,
        )

        results.append({
            "seed": seed,
            "offset": offset,
            "trajectory": trajectory,
            "final_loss": final_loss,
            "success": success,
        })

successful = [
    result
    for result in results
    if result["success"]
]

failed = [
    result
    for result in results
    if not result["success"]
]

print("\n" + "-" * 60)
print(
    f"successful cases={len(successful)} "
    f"failed cases={len(failed)}"
)

print("\nGROUP MEANS")

for checkpoint in relative_checkpoints:
    for label, group in [
        ("SUCCESS", successful),
        ("FAIL", failed),
    ]:
        if not group:
            continue

        mean_cosine = sum(
            result["trajectory"][checkpoint][1][0]
            for result in group
        ) / len(group)

        mean_displacement = sum(
            result["trajectory"][checkpoint][1][1]
            for result in group
        ) / len(group)

        mean_bias_change = sum(
            result["trajectory"][checkpoint][1][2]
            for result in group
        ) / len(group)

        mean_loss = sum(
            result["trajectory"][checkpoint][0]
            for result in group
        ) / len(group)

        print(
            f"step={checkpoint:4d} "
            f"group={label:7s} "
            f"loss={mean_loss:.9f} "
            f"cosine={mean_cosine:.6f} "
            f"displacement={mean_displacement:.6f} "
            f"bias_change={mean_bias_change:.6f}"
        )

print("\nFAILED CASES")

for result in failed:
    checkpoint = 500

    (
        checkpoint_loss,
        (
            mean_cosine,
            mean_displacement,
            mean_bias_change,
        ),
    ) = result["trajectory"][checkpoint]

    print(
        f"seed={result['seed']} "
        f"offset={result['offset']} "
        f"loss_at_500={checkpoint_loss:.9f} "
        f"cosine={mean_cosine:.6f} "
        f"displacement={mean_displacement:.6f} "
        f"bias_change={mean_bias_change:.6f}"
    )
