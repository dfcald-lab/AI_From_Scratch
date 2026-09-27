def relu(value):
    return max(0, value)


def relu_derivative(value):
    if value > 0:
        return 1
    return 0


examples = [
    ("Positive z", 2, 0.5, 0.0, 5),
    ("Negative z", 2, -1.0, 0.0, 5),
]

for name, input_value, weight, bias, correct_answer in examples:
    z = weight * input_value + bias
    prediction = relu(z)

    error = prediction - correct_answer
    loss = error ** 2

    activation_gradient = relu_derivative(z)

    weight_gradient = 2 * error * input_value * activation_gradient
    bias_gradient = 2 * error * activation_gradient

    print(name)
    print("z:", z)
    print("Prediction:", prediction)
    print("Error:", error)
    print("Loss:", loss)
    print("ReLU derivative:", activation_gradient)
    print("Weight gradient:", weight_gradient)
    print("Bias gradient:", bias_gradient)
    print()
