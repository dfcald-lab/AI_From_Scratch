training_data = [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
    (5, 10),
]

weight = 0.5
learning_rate = 0.001
epochs = 50

for epoch in range(epochs):
    total_loss = 0

    for input_value, correct_answer in training_data:
        prediction = weight * input_value
        error = prediction - correct_answer
        loss = error ** 2
        gradient = 2 * error * input_value

        weight = weight - learning_rate * gradient
        total_loss += loss

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(
            "Epoch:",
            epoch + 1,
            "Weight:",
            weight,
            "Loss:",
            total_loss
        )

print()
print("FINAL WEIGHT:", weight)

test_input = 7
prediction = weight * test_input

print("TEST INPUT:", test_input)
print("PREDICTION:", prediction)
