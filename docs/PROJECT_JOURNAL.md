# Project Journal

## Entry 001 — Project Founded

**Date:** 2026-09-26

### Decision

Begin a long-term project to build a self-owned AI from the ground up.

### Goals

- Learn the technology instead of treating it as a black box.
- Eventually train my own model.
- Build the surrounding AI system myself.
- Avoid mandatory subscriptions and unnecessary dependence on commercial AI APIs.
- Develop the project as a personal body of work that can grow over time.

### First Technical Principle

The first AI does not need to be powerful.

It needs to be understandable.

### Status

Phase 0 — Foundation

## Entry 002 — Development Environment Established

**Date:** 2026-09-26

### Environment


### Decision

Begin the first learning experiment using Python's standard library only.

No machine-learning framework will be used for Experiment 001.

### Reason

I want to see the learning process directly instead of hiding it behind a library.

### Status

Phase 0 — Foundation  
Preparing for Phase 1 — First Learning System

## Entry 003 — First Learning Experiment

**Date:** 2026-09-26

### Experiment

A one-parameter learning system was created using only Python's standard library.

The model:

prediction = weight × input

Training examples:

1 → 2  
2 → 4  
3 → 6  
4 → 8  
5 → 10

Initial weight:

0.5

Learning rate:

0.001

Training epochs:

50

### Result

Final weight:

1.9944572607243007

Test input:

7

Test prediction:

13.961200825070105

The value 7 was not included in the training examples.

### Observation

The system learned an approximation of the relationship y = 2x and generalized it to an unseen input.

### Lesson

A learning system can improve a parameter by measuring error and repeatedly adjusting that parameter.

This experiment introduced:

- prediction
- error
- loss
- gradient
- learning rate
- repeated training
- generalization
- evaluation

### Important Understanding

The model does not contain the rule "multiply by 2."

It contains a learned parameter that moved toward a value that minimizes its error on the examples.

## Entry 004 — Adding a Bias

**Date:** 2026-09-26

### Experiment

The model structure was expanded from:

prediction = weight × input

to:

prediction = weight × input + bias

This introduced a second learned parameter: bias.

Training examples:

1 → 3  
2 → 5  
3 → 7  
4 → 9  
5 → 11

The underlying relationship was y = 2x + 1.

### First Result

After 100 epochs:

- Weight: 2.1242365655296824
- Bias: 0.5447696447693232
- Test input: 7
- Prediction: 15.4144256034771

### Second Result

After increasing training to 1000 epochs:

- Weight: 2.026847653543838
- Bias: 0.9016324002141546
- Test input: 7
- Expected: 15
- Prediction: 15.089565975021019

### Observation

Increasing the number of training steps moved both learned parameters closer to the relationship represented by the training data.

### Lesson

A simple model can learn more than one parameter. Weight determines how strongly the input contributes, while bias shifts the prediction.

This is the first experiment in which the model behaves like a tiny neuron with a learned weight and bias.

## Entry 004 — Gradient Descent in Practice

**Date:** 2026-09-26

### Experiment

Tested repeated gradient descent updates using one training example:

x = 2
y = 5

Initial parameters:

w = 0.5
b = 0

Learning rate:

0.001

Training steps:

20

### Results

Step 1:

- Prediction: 1.0
- Loss: 16.0
- Weight: 0.516
- Bias: 0.008

Step 5:

- Prediction: 1.15761596
- Loss: 14.76391511084672

Step 10:

- Prediction: 1.3459310100654365
- Loss: 13.3522201832014

Step 20:

- Prediction: 1.695325504657653
- Loss: 10.920873520166197
- Weight: 0.7913488998444306
- Bias: 0.14567444992221532

### Observation

Repeated gradient descent caused the loss to decrease from 16.0 to 10.92 while the prediction moved from 1.0 toward the correct answer of 5.

### Lesson

Gradient descent repeatedly changes model parameters in the direction that reduces loss.

A small learning rate produces smaller parameter updates, making the process slower but more controlled.

## Entry 005 — Learning Rate Experiment

**Date:** 2026-09-26

### Experiment

Tested five learning rates using the same training example:

x = 2
y = 5

Initial parameters:

w = 0.5
b = 0

The learning rates tested were:

0.0001
0.001
0.01
0.1
0.21

### Results

#### Learning rate: 0.0001

After 20 steps:

- Prediction: 1.0753198605424035
- Loss: 15.403114197052899
- Weight: 0.5316978162727445
- Bias: 0.015848908136372256

Learning was very slow.

#### Learning rate: 0.001

After 20 steps:

- Prediction: 1.695325504657653
- Loss: 10.920873520166197
- Weight: 0.7913488998444306
- Bias: 0.14567444992221532

The model improved steadily but had not reached the target.

#### Learning rate: 0.01

After 20 steps:

- Prediction: 4.459659312930802
- Loss: 0.2919680581024125
- Weight: 1.905477352655089
- Bias: 0.7027386763275443

The model learned much faster and moved close to the target.

#### Learning rate: 0.1

The model reached the target:

Prediction = 5.0
Loss = 0.0

Final parameters:

- Weight: 2.1
- Bias: 0.8

#### Learning rate: 0.21

The model became unstable.

After 20 steps:

- Prediction: 29.46363617936579
- Loss: 598.4694951163748
- Weight: -8.663999918920945
- Bias: -4.581999959460473

### Observation

The learning rate controls how large each parameter update is.

A learning rate that is too small can make learning very slow.

A suitable learning rate can move the model toward the target efficiently.

A learning rate that is too large can cause the model to overshoot the target repeatedly and become unstable.

### Lesson

Gradient descent depends not only on the direction of the gradient, but also on the size of each step.

The learning rate determines that step size.

### Additional Observation

A negative prediction is not inherently an error. It simply means the model's current numerical output is below zero.

The problem occurs when the prediction moves farther from the target and the loss increases.

### Status

Experiment 004 complete.

## Entry 006 — Multiple Inputs

**Date:** 2026-09-27

### Experiment

Expanded the neuron from one input to two inputs.

The model structure became:

prediction = w1 × x1 + w2 × x2 + bias

Training data:

(1, 1) → 6
(2, 1) → 8
(1, 2) → 9
(3, 2) → 13
(2, 3) → 14

The underlying relationship was:

y = 2x1 + 3x2 + 1

The model was not given that relationship.

### Initial Parameters

w1 = 0.5
w2 = 0.5
bias = 0

Learning rate:

0.001

Training epochs:

1000

### Result

Final weight 1:

2.007437596601594

Final weight 2:

2.9896676735925904

Final bias:

1.005845009552969

### Unseen Test

Test inputs:

x1 = 4
x2 = 2

Expected:

15

Predicted:

15.014930743144525

### Observation

The neuron learned separate weights for separate inputs and combined them to produce its prediction.

The learned parameters were close to the relationship represented by the training data.

### Lesson

Each input has its own learned weight.

The weight determines how strongly that input contributes to the prediction.

With multiple inputs, the neuron can combine multiple signals before producing an output.

### Status

Experiment 005 complete.

## Entry 007 — Two-Neuron Layer

**Date:** 2026-09-27

### Experiment

Expanded from a single neuron to a layer containing two neurons.

Both neurons received the same two inputs:

x1
x2

Each neuron had its own two weights and bias.

The layer was trained to learn:

y1 = 2x1 + x2

y2 = x1 + 3x2

### Initial Parameters

All weights started at:

0.5

All biases started at:

0

Learning rate:

0.001

Training epochs:

2000

### Result — Neuron 1

Weight 1:

1.9809329330954377

Weight 2:

0.9810192332833941

Bias:

0.07573069152144327

Expected relationship:

y1 = 2x1 + x2

### Result — Neuron 2

Weight 1:

0.9721315321064297

Weight 2:

2.9711802993693683

Bias:

0.11283761404099299

Expected relationship:

y2 = x1 + 3x2

### Unseen Test

Test inputs:

x1 = 4
x2 = 2

Expected output 1:

10

Predicted output 1:

9.961500890469981

Expected output 2:

10

Predicted output 2:

9.943724341205447

### Observation

Two neurons can receive the same inputs while learning different relationships through separate weights and biases.

### Lesson

A layer allows multiple neurons to process the same information in different ways.

Each neuron has its own parameters and produces its own output.

### Next Direction

Explore activation functions and the role they play in allowing neural networks to represent nonlinear relationships.

### Status

Experiment 006 complete.

## Entry 008 — Activation Function: ReLU

**Date:** 2026-09-27

### Experiment

Introduced the ReLU activation function and observed how it transforms a neuron's raw output.

The raw neuron was:

z = wx + b

with:

w = 2
b = -3

The activation function was:

a = max(0, z)

### Results

Input -2:

z = -7
ReLU = 0

Input -1:

z = -5
ReLU = 0

Input 0:

z = -3
ReLU = 0

Input 1:

z = -1
ReLU = 0

Input 2:

z = 1
ReLU = 1

Input 3:

z = 3
ReLU = 3

Input 4:

z = 5
ReLU = 5

### Observation

ReLU passes positive values through unchanged and changes negative values to zero.

### Second Example

Using:

w = 3
b = -4

the raw outputs for inputs 1, 2, and 3 were:

-1
2
5

After ReLU:

0
2
5

### Lesson

An activation function transforms the output of a neuron.

ReLU itself does not learn parameters. It changes the forward signal and determines whether a gradient can pass backward.

For ReLU:

f(z) = max(0, z)

and its derivative is:

f'(z) = 0 when z < 0
f'(z) = 1 when z > 0

### Status

Experiment 007 complete.

## Entry 009 — Trainable ReLU

**Date:** 2026-09-27

### Experiment

Added ReLU to a trainable neuron.

The forward pass became:

z = weight × input + bias

prediction = max(0, z)

The gradient became dependent on the derivative of ReLU:

weight_gradient = 2 × error × input × ReLU_derivative(z)

bias_gradient = 2 × error × ReLU_derivative(z)

### Result

Using:

x = 2
y = 5
weight = 0.5
bias = 0
learning rate = 0.001

the neuron remained in the positive region during the 20 training steps.

The results matched the earlier gradient descent experiment because ReLU_derivative(z) remained 1.

### Observation

When z is positive, ReLU passes the value forward and its derivative is 1, allowing the gradient to pass backward.

When z is negative, ReLU outputs 0 and its derivative is 0, blocking the gradient from reaching the weight and bias.

### Lesson

An activation function affects both the forward output and the backward learning signal.

### Status

Experiment 008 complete.

## Entry 010 — ReLU Gradient Behavior

**Date:** 2026-09-27

### Experiment

Compared a positive and negative pre-activation value to observe how ReLU changes the gradient.

### Positive z

z = 1

Prediction = 1

Error = -4

Loss = 16

ReLU derivative = 1

Weight gradient = -16

Bias gradient = -8

### Negative z

z = -2

Prediction = 0

Error = -5

Loss = 25

ReLU derivative = 0

Weight gradient = 0

Bias gradient = 0

### Observation

A neuron can have a large prediction error while receiving a zero gradient when ReLU is inactive.

### Lesson

The size of the error alone does not determine the parameter update. The activation function can determine whether the gradient reaches the parameter.

### Status

Experiment 009 complete.

## Entry 011 — First Multi-Layer Network

**Date:** 2026-09-27

### Experiment

Built the first multi-layer neural network.

Architecture:

2 inputs → 2 ReLU hidden neurons → 1 output

The network was trained to learn XOR:

(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0

### Training

Learning rate:

0.05

Epochs:

1000

### Result

Final total loss:

6.409494854920721e-31

The final loss was effectively zero.

Predictions:

(0, 0) → 2.220446049250313e-16
(0, 1) → 0.9999999999999996
(1, 0) → 0.9999999999999996
(1, 1) → 4.440892098500626e-16

The values near zero are floating-point representations of values effectively equal to zero.

### Observation

The network successfully learned the XOR relationship using a hidden layer and ReLU activation.

### Lesson

A hidden layer combined with a nonlinear activation allows a neural network to represent relationships that a single linear neuron cannot.

This experiment also connected the forward pass and backward pass into one multi-layer training process.

The backward pass propagated gradients from the loss through the output neuron, through the ReLU activations, and into the hidden-layer parameters.

### Status

Experiment 010 complete.

## Entry 012 — Reusable Neuron

**Date:** 2026-09-27

### Experiment

Created a reusable Neuron class containing:

- weights
- bias
- ReLU
- ReLU derivative
- forward pass

The forward pass uses:

z = sum(weight × input) + bias

a = max(0, z)

### Test

Weights:

[0.5, 0.5]

Bias:

0.0

Inputs:

[2, 1]

### Result

Raw output:

1.5

ReLU output:

1.5

ReLU derivative:

1

### Lesson

The neuron can now be represented as a reusable component instead of rewriting its forward-pass logic in every experiment.

The neuron describes how an input is transformed. Training will remain a separate responsibility.

### Status

Experiment 011 complete.

## Entry 013 — Reusable Layer

**Date:** 2026-09-27

### Experiment

Created a reusable Layer class that contains multiple Neuron objects.

The layer sends the same inputs to each neuron and collects their raw and activated outputs.

### Test

Inputs:

[2, 1]

Neuron 1:

weights = [1.0, 2.0]
bias = 0.0

Neuron 2:

weights = [3.0, 1.0]
bias = 1.0

### Result

Raw outputs:

[4.0, 8.0]

Activated outputs:

[4.0, 8.0]

### Observation

The layer successfully processed the same inputs through multiple neurons and returned their outputs as a collection.

### Lesson

A neural-network layer can be represented as a reusable collection of neurons.

Each neuron has its own weights and bias while receiving the same input vector.

### Status

Experiment 012 complete.

## Entry 014 — Automatic Layer Construction

**Date:** 2026-09-27

### Experiment

Changed the Layer class so it can automatically create any requested number of neurons with the required number of input weights.

Test configuration:

- Inputs: 2
- Neurons: 3
- Random seed: 0
- Bias for each neuron: 0

### Parameter Count

Each neuron requires:

2 weights + 1 bias = 3 parameters

Three neurons therefore require:

3 × 3 = 9 parameters

### Test Input

[2, 1]

### Result

Neuron 1:

Weights = [0.6888437030500962, 0.515908805880605]

Raw output = 1.8935962119807974

ReLU output = 1.8935962119807974

Neuron 2:

Weights = [-0.15885683833831, -0.4821664994140733]

Raw output = -0.7998801760906933

ReLU output = 0

Neuron 3:

Weights = [0.02254944273721704, -0.19013172509917142]

Raw output = -0.14503283962473734

ReLU output = 0

### Observation

All neurons received the same input but produced different results because each neuron had its own independently initialized weights.

ReLU allowed one neuron to remain active while the other two produced zero outputs.

### Lesson

A layer can automatically create and manage multiple neurons.

Random initialization gives different neurons different starting parameters.

The layer can now be represented mathematically using weights, inputs, and biases as vectors and matrices.

### Status

Experiment 013 complete.


## Entry 015 — Training Loop

**Date:** 2026-09-27

### Experiment

Turned the manually derived forward pass, loss calculation, ReLU gradient, backpropagation, and gradient descent update into a repeated training loop.

The experiment used:

- 2 inputs
- 2 output neurons
- ReLU activation
- Mean squared error with a 1/2 factor
- Learning rate: 0.01
- Training steps: 100

Initial parameters:

W = [
[2.0, 1.0],
[3.0, 4.0]
]

b = [1.0, -2.0]

Input:

x = [5.0, 2.0]

Target:

y = [10.0, 20.0]

### Initial Forward Pass

The initial output was:

[13.0, 21.0]

Initial loss:

5.0

### Learning Process

The training loop repeatedly performed:

forward pass
→ loss
→ backpropagation
→ gradient calculation
→ gradient descent update

Using a learning rate of 0.01, the model quickly moved toward the target.

At step 10:

- Loss: approximately 0.0040
- Output: approximately [10.0847, 20.0282]

At step 20:

- Loss: approximately 0.0000 at the displayed precision
- Output: approximately [10.0024, 20.0008]

By step 30 and beyond, the printed output was effectively [10.0, 20.0].

### Observation

The model improved automatically when the same learning procedure was repeated.

The loss decreased from 5.0 to a value effectively zero at the displayed precision.

### Lesson

A neural network learns through repetition of a small set of operations:

1. Make a prediction.
2. Measure the error.
3. Calculate how the parameters contributed to that error.
4. Adjust the parameters.
5. Repeat.

This experiment turns the backpropagation math into an actual training loop.

### Important Understanding

The training loop is the mechanism that repeatedly applies the gradients.

The gradient determines the direction of the parameter update.

The learning rate determines the size of the update.

### Next Direction

Replace the manually named weights and biases in this experiment with the reusable Layer and Neuron components already developed in the project.

### Status

Experiment 014 complete.

## Entry 016 — Trainable Reusable Layer

**Date:** 2026-09-27

### Experiment

Extended the reusable Neuron and Layer components so they can participate in training.

The Neuron now stores:

- inputs from the forward pass
- raw output
- weight gradients
- bias gradient

The Neuron can now perform:

- forward pass
- backward pass
- parameter update

The Layer can now:

- forward inputs through all neurons
- propagate gradients backward through all neurons
- update all neuron parameters

### Test

Architecture:

2 inputs → 2 trainable neurons

Input:

[5.0, 2.0]

Target:

[10.0, 20.0]

Initial weights:

[
[2.0, 1.0],
[3.0, 4.0]
]

Initial biases:

[1.0, -2.0]

Learning rate:

0.01

Training steps:

100

### Result

Initial output:

[13.0, 21.0]

Initial loss:

5.0

At step 10:

- Loss: approximately 0.0040
- Output: approximately [10.0847, 20.0282]

At step 20:

- Output: approximately [10.0024, 20.0008]

By step 50, the output was effectively:

[10.0, 20.0]

### Observation

The reusable Layer produced the same learning behavior as the previous manually written training loop.

The difference is that the training process is now handled by reusable Neuron and Layer objects instead of manually naming every parameter.

### Lesson

Reusable neural-network components can contain both forward computation and the information required for backpropagation and parameter updates.

This is a major step toward building the network as a collection of reusable components rather than a collection of one-off experiments.

### Important Understanding

The Layer does not need to know the individual meaning of every weight.

It asks each Neuron to:

1. process its inputs,
2. calculate its gradients,
3. update its own parameters.

The Layer coordinates those operations across the neurons.

### Status

Experiment 015 complete.
