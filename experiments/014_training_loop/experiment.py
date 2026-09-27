def relu(x):
    return max(0.0, x)


def relu_gradient(x):
    return 1.0 if x > 0 else 0.0


# Input
x1, x2 = 5.0, 2.0

# Target
y1, y2 = 10.0, 20.0

# Starting weights
w11, w12 = 2.0, 1.0
w21, w22 = 3.0, 4.0

# Starting biases
b1, b2 = 1.0, -2.0

# Learning rate
learning_rate = 0.01


for step in range(100):
    # -------------------------
    # Forward pass
    # -------------------------
    z1 = w11 * x1 + w12 * x2 + b1
    z2 = w21 * x1 + w22 * x2 + b2

    a1 = relu(z1)
    a2 = relu(z2)

    # -------------------------
    # Loss
    # -------------------------
    loss = 0.5 * ((a1 - y1) ** 2 + (a2 - y2) ** 2)

    # -------------------------
    # Backpropagation
    # -------------------------
    da1 = a1 - y1
    da2 = a2 - y2

    dz1 = da1 * relu_gradient(z1)
    dz2 = da2 * relu_gradient(z2)

    dw11 = dz1 * x1
    dw12 = dz1 * x2
    dw21 = dz2 * x1
    dw22 = dz2 * x2

    db1 = dz1
    db2 = dz2

    # -------------------------
    # Gradient descent
    # -------------------------
    w11 -= learning_rate * dw11
    w12 -= learning_rate * dw12
    w21 -= learning_rate * dw21
    w22 -= learning_rate * dw22

    b1 -= learning_rate * db1
    b2 -= learning_rate * db2

    if step % 10 == 0 or step == 99:
        print(
            f"step={step:3d} "
            f"loss={loss:8.4f} "
            f"output=[{a1:8.4f}, {a2:8.4f}]"
        )
