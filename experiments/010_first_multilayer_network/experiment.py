import random


def relu(value):
    return max(0, value)


def relu_derivative(value):
    if value > 0:
        return 1
    return 0


training_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]


random_generator = random.Random(0)

weight_11 = random_generator.uniform(-1, 1)
weight_12 = random_generator.uniform(-1, 1)
bias_1 = 0.5

weight_21 = random_generator.uniform(-1, 1)
weight_22 = random_generator.uniform(-1, 1)
bias_2 = 0.5

output_weight_1 = random_generator.uniform(-1, 1)
output_weight_2 = random_generator.uniform(-1, 1)
output_bias = 0.1

learning_rate = 0.05
epochs = 1000


for epoch in range(epochs):
    total_loss = 0

    for x1, x2, correct_answer in training_data:

        # Forward pass
        z1 = weight_11 * x1 + weight_12 * x2 + bias_1
        a1 = relu(z1)

        z2 = weight_21 * x1 + weight_22 * x2 + bias_2
        a2 = relu(z2)

        prediction = (
            output_weight_1 * a1
            + output_weight_2 * a2
            + output_bias
        )

        error = prediction - correct_answer
        loss = error ** 2

        total_loss += loss

        # Backward pass
        output_gradient = 2 * error

        output_weight_1_gradient = output_gradient * a1
        output_weight_2_gradient = output_gradient * a2
        output_bias_gradient = output_gradient

        a1_gradient = output_gradient * output_weight_1
        a2_gradient = output_gradient * output_weight_2

        z1_gradient = a1_gradient * relu_derivative(z1)
        z2_gradient = a2_gradient * relu_derivative(z2)

        weight_11_gradient = z1_gradient * x1
        weight_12_gradient = z1_gradient * x2
        bias_1_gradient = z1_gradient

        weight_21_gradient = z2_gradient * x1
        weight_22_gradient = z2_gradient * x2
        bias_2_gradient = z2_gradient

        # Update parameters
        weight_11 -= learning_rate * weight_11_gradient
        weight_12 -= learning_rate * weight_12_gradient
        bias_1 -= learning_rate * bias_1_gradient

        weight_21 -= learning_rate * weight_21_gradient
        weight_22 -= learning_rate * weight_22_gradient
        bias_2 -= learning_rate * bias_2_gradient

        output_weight_1 -= learning_rate * output_weight_1_gradient
        output_weight_2 -= learning_rate * output_weight_2_gradient
        output_bias -= learning_rate * output_bias_gradient

    if epoch in [0, 1, 9, 99, 199, 999]:
        print(
            "Epoch:", epoch + 1,
            "Total Loss:", total_loss
        )


print()
print("FINAL PARAMETERS")

print("Hidden Neuron 1:")
print("Weight 1:", weight_11)
print("Weight 2:", weight_12)
print("Bias:", bias_1)

print()
print("Hidden Neuron 2:")
print("Weight 1:", weight_21)
print("Weight 2:", weight_22)
print("Bias:", bias_2)

print()
print("Output:")
print("Weight 1:", output_weight_1)
print("Weight 2:", output_weight_2)
print("Bias:", output_bias)

print()
print("XOR RESULTS")

for x1, x2, correct_answer in training_data:

    z1 = weight_11 * x1 + weight_12 * x2 + bias_1
    a1 = relu(z1)

    z2 = weight_21 * x1 + weight_22 * x2 + bias_2
    a2 = relu(z2)

    prediction = (
        output_weight_1 * a1
        + output_weight_2 * a2
        + output_bias
    )

    print(
        "Input:", (x1, x2),
        "Expected:", correct_answer,
        "Predicted:", prediction
    )
