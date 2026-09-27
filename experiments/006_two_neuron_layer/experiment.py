training_data = [
    (1, 1, 3, 4),
    (2, 1, 5, 5),
    (1, 2, 4, 7),
    (3, 2, 8, 9),
    (2, 3, 7, 11),
]

weight_11 = 0.5
weight_12 = 0.5
bias_1 = 0.0

weight_21 = 0.5
weight_22 = 0.5
bias_2 = 0.0

learning_rate = 0.001
epochs = 2000

for epoch in range(epochs):
    for x1, x2, correct_1, correct_2 in training_data:

        prediction_1 = weight_11 * x1 + weight_12 * x2 + bias_1
        prediction_2 = weight_21 * x1 + weight_22 * x2 + bias_2

        error_1 = prediction_1 - correct_1
        error_2 = prediction_2 - correct_2

        weight_11_gradient = 2 * error_1 * x1
        weight_12_gradient = 2 * error_1 * x2
        bias_1_gradient = 2 * error_1

        weight_21_gradient = 2 * error_2 * x1
        weight_22_gradient = 2 * error_2 * x2
        bias_2_gradient = 2 * error_2

        weight_11 = weight_11 - learning_rate * weight_11_gradient
        weight_12 = weight_12 - learning_rate * weight_12_gradient
        bias_1 = bias_1 - learning_rate * bias_1_gradient

        weight_21 = weight_21 - learning_rate * weight_21_gradient
        weight_22 = weight_22 - learning_rate * weight_22_gradient
        bias_2 = bias_2 - learning_rate * bias_2_gradient


print("NEURON 1")
print("Weight 1:", weight_11)
print("Weight 2:", weight_12)
print("Bias:", bias_1)

print()
print("NEURON 2")
print("Weight 1:", weight_21)
print("Weight 2:", weight_22)
print("Bias:", bias_2)

print()
print("TEST")

test_x1 = 4
test_x2 = 2

prediction_1 = weight_11 * test_x1 + weight_12 * test_x2 + bias_1
prediction_2 = weight_21 * test_x1 + weight_22 * test_x2 + bias_2

print("Input 1:", test_x1)
print("Input 2:", test_x2)
print("Output 1 expected:", 10)
print("Output 1 predicted:", prediction_1)
print("Output 2 expected:", 10)
print("Output 2 predicted:", prediction_2)
