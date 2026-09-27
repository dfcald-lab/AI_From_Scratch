input_value = 2
correct_answer = 5

learning_rates = [
    0.0001,
    0.001,
    0.01,
    0.1,
    0.21,
]

steps = 20

for learning_rate in learning_rates:
    weight = 0.5
    bias = 0.0

    print()
    print("LEARNING RATE:", learning_rate)

    for step in range(steps):
        prediction = weight * input_value + bias
        error = prediction - correct_answer
        loss = error ** 2

        weight_gradient = 2 * error * input_value
        bias_gradient = 2 * error

        weight = weight - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

        if step == 0 or step == 4 or step == 9 or step == 19:
            print(
                "Step:", step + 1,
                "Prediction:", prediction,
                "Loss:", loss
            )

    print("Final Weight:", weight)
    print("Final Bias:  ", bias)
