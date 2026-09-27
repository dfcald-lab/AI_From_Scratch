def relu(value):
    return max(0, value)


weight = 2
bias = -3

test_inputs = [-2, -1, 0, 1, 2, 3, 4]

print("WEIGHT:", weight)
print("BIAS:", bias)
print()

for input_value in test_inputs:
    raw_output = weight * input_value + bias
    activated_output = relu(raw_output)

    print(
        "Input:", input_value,
        "Raw:", raw_output,
        "ReLU:", activated_output
    )
