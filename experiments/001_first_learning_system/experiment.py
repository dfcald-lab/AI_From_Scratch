training_data = [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
    (5, 10),
]

test_data = [
    (6, 12),
    (7, 14),
    (8, 16),
    (9, 18),
    (10, 20),
]

weight = 0.5
learning_rate = 0.001
epochs = 50

for epoch in range(epochs):
    for input_value, correct_answer in training_data:
        prediction = weight * input_value
        error = prediction - correct_answer
        gradient = 2 * error * input_value

        weight = weight - learning_rate * gradient

print("FINAL WEIGHT:", weight)

print()
print("TEST RESULTS")

total_test_error = 0

for input_value, correct_answer in test_data:
    prediction = weight * input_value
    error = prediction - correct_answer
    total_test_error += abs(error)

    print(
        "Input:", input_value,
        "Expected:", correct_answer,
        "Predicted:", prediction,
        "Error:", error
    )

average_error = total_test_error / len(test_data)

print()
print("AVERAGE TEST ERROR:", average_error)
