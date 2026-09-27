input_value = 2
correct_answer = 5

weight = 0.5
bias = 0.0
learning_rate = 0.001
steps = 20

for step in range(steps):
    prediction = weight * input_value + bias
    error = prediction - correct_answer
    loss = error ** 2

    weight_gradient = 2 * error * input_value
    bias_gradient = 2 * error

    weight = weight - learning_rate * weight_gradient
    bias = bias - learning_rate * bias_gradient

    print(
        "Step:", step + 1,
        "Prediction:", prediction,
        "Loss:", loss,
        "Weight:", weight,
        "Bias:", bias
    )
