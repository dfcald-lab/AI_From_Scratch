from src.layer import Layer
from src.neuron import Neuron


layer = Layer(
    neurons=[
        Neuron(weights=[2.0, 1.0], bias=1.0),
        Neuron(weights=[3.0, 4.0], bias=-2.0),
    ]
)

inputs = [5.0, 2.0]
target = [10.0, 20.0]

learning_rate = 0.01


for step in range(100):
    # -------------------------
    # Forward pass
    # -------------------------
    raw_outputs, outputs = layer.forward(inputs)

    # -------------------------
    # Loss
    # -------------------------
    loss = sum(
        0.5 * (prediction - expected) ** 2
        for prediction, expected in zip(outputs, target)
    )

    # -------------------------
    # Gradient of loss
    # -------------------------
    output_gradients = [
        prediction - expected
        for prediction, expected in zip(outputs, target)
    ]

    # -------------------------
    # Backpropagation
    # -------------------------
    layer.backward(output_gradients)

    # -------------------------
    # Gradient descent
    # -------------------------
    layer.update(learning_rate)

    if step % 10 == 0 or step == 99:
        print(
            f"step={step:3d} "
            f"loss={loss:8.4f} "
            f"output={outputs}"
        )
