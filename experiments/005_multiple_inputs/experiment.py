training_data = [
    (1, 1, 6),
    (2, 1, 8),
    (1, 2, 9),
    (3, 2, 13),
    (2, 3, 14),
]

weight_1 = 0.5
weight_2 = 0.5
bias = 0.0

learning_rate = 0.001
epochs = 1000

for epoch in range(epochs):
    for x1, x2, correct_answer in training_data:
        prediction = weight_1 * x1 + weight_2 * x2 + bias

        error = prediction - correct_answer
        loss = error ** 2

        weight_1_gradient = 2 * error * x1
        weight_2_gradient = 2 * error * x2
        bias_gradient = 2 * error

        weight_1 = weight_1 - learning_rate * weight_1_gradient
        weight_2 = weight_2 - learning_rate * weight_2_gradient
        bias = bias - learning_rate * bias_gradient

print("FINAL WEIGHT 1:", weight_1)
print("FINAL WEIGHT 2:", weight_2)
print("FINAL BIAS:    ", bias)

print()
print("TEST")

test_x1 = 4
test_x2 = 2
expected = 15

prediction = weight_1 * test_x1 + weight_2 * test_x2 + bias

print("Input 1:", test_x1)
print("Input 2:", test_x2)
print("Expected:", expected)
print("Predicted:", prediction)
