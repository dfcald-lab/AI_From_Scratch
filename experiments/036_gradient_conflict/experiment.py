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


def example_hidden_gradient(network, inputs, target):
    prediction = network.forward(inputs)[0]
    network.backward([prediction - target])

    gradient = []

    for neuron in network.layers[0].neurons:
        gradient.extend(neuron.weight_gradients)
        gradient.append(neuron.bias_gradient)

    return gradient


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return None

    return dot / (norm_a * norm_b)


def gradient_conflict_metrics(network):
    gradients = [
        example_hidden_gradient(network, inputs, target)
        for inputs, target in training_data
    ]

    similarities = []

    for i in range(len(gradients)):
        for j in range(i + 1, len(gradients)):
            similarity = cosine_similarity(
                gradients[i],
                gradients[j],
            )

            if similarity is not None:
                similarities.append(similarity)

    mean_cosine = (
        sum(similarities) / len(similarities)
        if similarities
        else 0.0
    )

    negative_fraction = (
        sum(value < 0.0 for value in similarities)
        / len(similarities)
        if similarities
        else 0.0
    )

    minimum_cosine = (
        min(similarities)
        if similarities
        else 0.0
    )

    norms = [
        math.sqrt(sum(value * value for value in gradient))
        for gradient in gradients
    ]

    mean_gradient_norm = sum(norms) / len(norms)

    return (
        mean_cosine,
        negative_fraction,
        minimum_cosine,
        mean_gradient_norm,
        len(similarities),
    )


print("\n" + "=" * 60)
print("HE INITIALIZATION — GRADIENT CONFLICT")
print("Full-batch training, learning rate 0.1.")
print("Pairwise cosine similarity of per-example hidden gradients.")
print("Negative cosine = opposing hidden update directions.")

results = {}

for seed in seeds:
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    trajectory = []

    for epoch in range(epochs + 1):
        if epoch in checkpoints:
            current_loss = loss(network)
            current_gap = separation_gap(network)

            (
                mean_cosine,
                negative_fraction,
                minimum_cosine,
                mean_gradient_norm,
                valid_pairs,
            ) = gradient_conflict_metrics(network)

            trajectory.append({
                "epoch": epoch,
                "loss": current_loss,
                "gap": current_gap,
                "mean_cosine": mean_cosine,
                "negative_fraction": negative_fraction,
                "minimum_cosine": minimum_cosine,
                "mean_gradient_norm": mean_gradient_norm,
                "valid_pairs": valid_pairs,
            })

        if epoch == epochs:
            break

        batch_epoch(network, learning_rate)

    final = trajectory[-1]

    successful = (
        final["loss"] < 1e-6
        and final["gap"] > 1e-9
    )

    results[seed] = (successful, trajectory)

for seed in seeds:
    successful, trajectory = results[seed]

    print(
        f"\nseed={seed} "
        f"result={'SUCCESS' if successful else 'FAIL'}"
    )

    for point in trajectory:
        print(
            f"  epoch={point['epoch']:4d} "
            f"loss={point['loss']:.9f} "
            f"gap={point['gap']:.9f} "
            f"mean_cos={point['mean_cosine']:.6f} "
            f"neg_frac={point['negative_fraction']:.3f} "
            f"min_cos={point['minimum_cosine']:.6f} "
            f"mean_grad={point['mean_gradient_norm']:.6f} "
            f"pairs={point['valid_pairs']}"
        )

successful_seeds = [
    seed for seed, (successful, _) in results.items()
    if successful
]

failed_seeds = [
    seed for seed, (successful, _) in results.items()
    if not successful
]

print("\n" + "-" * 60)
print("SUCCESSFUL SEEDS:", successful_seeds)
print("FAILED SEEDS:    ", failed_seeds)
print(f"SUMMARY: successful={len(successful_seeds)}/10")
