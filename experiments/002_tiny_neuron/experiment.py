training_data = [
    (1, 3),
    (2, 5),
    (3, 7),
    (4, 9),
    (5, 11),
]

weight = 0.5
bias = 0.0
learning_rate = 0.001
epochs = 1000

for epoch in range(epochs):
    for input_value, correct_answer in training_data:
        prediction = weight * input_value + bias

        error = prediction - correct_answer
        loss = error ** 2

        weight_gradient = 2 * error * input_value
        bias_gradient = 2 * error

        weight = weight - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

print("FINAL WEIGHT:", weight)
print("FINAL BIAS:  ", bias)

print()
print("TEST")

test_input = 7
prediction = weight * test_input + bias

print("Input:", test_input)
print("Expected:", 15)
print("Predicted:", prediction)
