def relu(value):
    return max(0, value)


def relu_derivative(value):
    if value > 0:
        return 1
    return 0


input_value = 2
correct_answer = 5

weight = 0.5
bias = 0.0

learning_rate = 0.001
steps = 20

for step in range(steps):
    z = weight * input_value + bias
    prediction = relu(z)

    error = prediction - correct_answer
    loss = error ** 2

    activation_gradient = relu_derivative(z)

    weight_gradient = 2 * error * input_value * activation_gradient
    bias_gradient = 2 * error * activation_gradient

    weight = weight - learning_rate * weight_gradient
    bias = bias - learning_rate * bias_gradient

    print(
        "Step:", step + 1,
        "z:", z,
        "Prediction:", prediction,
        "Loss:", loss,
        "Weight:", weight,
        "Bias:", bias
    )
