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

- Linux development environment
- Python 3.x
- Git repository

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

## Entry 017 — Reusable Multi-Layer Network

**Date:** 2026-09-27

### Experiment

Built a multi-layer neural network using the reusable Neuron and Layer components.

Architecture:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

This experiment replaced the manually named parameters from the earlier XOR experiment with reusable trainable Layer objects.

### Training Data

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

This is the XOR relationship.

### Training

Learning rate:

0.05

Epochs:

1000

The training loop performed:

forward pass
→ loss
→ output-layer backward pass
→ hidden-layer backward pass
→ parameter updates

### Results

Epoch 1:

Total loss ≈ 0.164122570751

Epoch 10:

Total loss ≈ 0.035717989313

Epoch 100:

Total loss ≈ 0.000920521315

Epoch 500:

Total loss ≈ 0.000000000041

Epoch 1000:

Total loss ≈ 0.000000000000

### Final XOR Results

Input [0, 0]:

Expected = 0.0
Predicted ≈ 0.000000000166

Input [0, 1]:

Expected = 1.0
Predicted ≈ 0.999999999923

Input [1, 0]:

Expected = 1.0
Predicted ≈ 0.999999999903

Input [1, 1]:

Expected = 0.0
Predicted ≈ 0.000000000001

The very small nonzero values are floating-point representations of values effectively equal to zero.

### Observation

The backward pass successfully propagated the learning signal from the output layer into the hidden layer.

The reusable Layer abstraction can therefore participate in a complete multi-layer training process.

### Lesson

A neural network can be constructed by connecting reusable layers.

Each layer performs its own forward computation and backward gradient calculation while passing information between layers.

This separates the network into reusable components instead of requiring every weight and gradient to be manually written.

### Important Understanding

Backpropagation is not limited to one layer.

The gradient can travel backward through multiple layers using the chain rule:

output loss
→ output layer
→ hidden layer
→ earlier layers

This experiment demonstrates that mechanism with the project's own code.

### Status

Experiment 016 complete.

## Entry 018 — Reusable Network

**Date:** 2026-09-27

### Experiment

Created a reusable Network class to coordinate multiple Layer objects.

The Network now handles:

- forward propagation through all layers
- backward propagation in reverse layer order
- parameter updates across all layers

The goal was to remove the need for each experiment to manually coordinate individual layers.

### Architecture

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

### Training Data

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

### Training

Learning rate:

0.05

Epochs:

1000

The training loop now uses:

network.forward()
→ loss
→ network.backward()
→ network.update()

### Results

Epoch 1:

Total loss ≈ 0.164122570751

Epoch 10:

Total loss ≈ 0.035717989313

Epoch 100:

Total loss ≈ 0.000920521315

Epoch 500:

Total loss ≈ 0.000000000041

Epoch 1000:

Total loss ≈ 0.000000000000

### Final XOR Results

Input [0, 0]:

Expected = 0.0
Predicted ≈ 0.000000000166

Input [0, 1]:

Expected = 1.0
Predicted ≈ 0.999999999923

Input [1, 0]:

Expected = 1.0
Predicted ≈ 0.999999999903

Input [1, 1]:

Expected = 0.0
Predicted ≈ 0.000000000001

### Observation

The reusable Network produced the same results as the previous multi-layer experiment.

The main change was architectural: the Network now coordinates the layers and hides the details of how many layers exist.

### Lesson

A neural network can be represented as a reusable sequence of layers.

Forward propagation moves information from the input toward the output.

Backward propagation moves gradients from the output toward the input.

The Network class provides the structure that connects those operations.

### Important Understanding

The network itself does not need to know the internal math of each neuron.

It coordinates the layers:

input
→ layer
→ layer
→ output

and then reverses that path during backpropagation.

### Status

Experiment 017 complete.

## Entry 019 — Automatic Network Construction

**Date:** 2026-09-27

### Experiment

Extended the reusable Network class so it can automatically construct its layers from a simple architecture description.

Instead of manually creating each Layer, the network can now be defined with:

number_of_inputs = 2
layer_sizes = [2, 1]
activations = ["relu", "linear"]

The Network creates the required layers and connects their input/output sizes automatically.

### Architecture

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

### Training Data

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

### First Run — Seed 0

The automatically constructed network did not successfully learn XOR.

Final loss:

0.350569675723

Final predictions:

[0, 0] → 0.683610867659947
[0, 1] → 0.683610867659947
[1, 0] → 0.683610867659947
[1, 1] → approximately 0

### Second Run — Seed 1

Changing the random seed produced a successful training run.

Epoch 1 loss:

0.203837789774

Epoch 10 loss:

0.079870619168

Epoch 100 loss:

0.000363064324

Epoch 500 loss:

0.000000000000

Epoch 1000 loss:

0.000000000000

Final predictions:

[0, 0] → approximately 0
[0, 1] → approximately 1
[1, 0] → approximately 1
[1, 1] → approximately 0

### Observation

The automatic Network construction worked in both runs.

The difference was the initial random parameters.

With seed 0, the network became stuck in a state where multiple inputs produced the same output and training stopped improving.

With seed 1, the same architecture and training process successfully learned XOR.

### Lesson

Network architecture and parameter initialization are separate concerns.

A correct architecture does not guarantee successful training from every random initialization.

Random initialization affects where optimization begins and can determine whether a small network successfully learns a particular problem.

### Important Understanding

The Network now separates three responsibilities:

1. Architecture — which layers and sizes exist.
2. Computation — forward and backward propagation.
3. Learning — updating parameters using gradients.

The architecture can now be described without manually constructing every layer.

### Status

Experiment 018 complete.

### Follow-up

Future experiments should investigate initialization more systematically rather than depending on a single seed.


## Entry 020 — Initialization Test

**Date:** 2026-09-27

### Experiment

Tested the effect of random parameter initialization by training the same automatically constructed XOR network with ten different random seeds.

Architecture:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

Training data:

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

Learning rate:

0.05

Training epochs:

1000

Seeds tested:

0 through 9

### Results

Seed 0:

Loss ≈ 0.333763992254
Success = False

Seed 1:

Loss ≈ 0.000000000000
Success = True

Seed 2:

Loss ≈ 0.333763992254
Success = False

Seed 3:

Loss ≈ 0.500000863377
Success = False

Seed 4:

Loss ≈ 0.500000863377
Success = False

Seed 5:

Loss ≈ 0.000000000053
Success = True

Seed 6:

Loss ≈ 0.333763992254
Success = False

Seed 7:

Loss ≈ 0.333487085225
Success = False

Seed 8:

Loss ≈ 0.333472106898
Success = False

Seed 9:

Loss ≈ 0.333472106898
Success = False

The experiment treated a total loss below 1e-8 as successful.

### Additional Investigation

Seed 5 produced:

[0, 0] → approximately 0.00000887
[0, 1] → approximately 0.99999674
[1, 0] → approximately 0.99999597
[1, 1] → approximately 0.00000017

Its total loss was approximately 5.3e-11.

### Observation

The same network architecture and training procedure produced different results depending only on the initial random parameters.

Two of the ten tested seeds reached the experiment's success threshold within 1000 epochs.

Several other seeds became stuck at nonzero loss values.

### Lesson

Random initialization is an important part of neural-network training.

The architecture alone does not determine the training result. The starting parameter values can affect whether optimization reaches a useful solution within a given number of training steps.

### Important Understanding

A random seed is useful during development because it makes an experiment reproducible.

Testing multiple seeds is also useful because a single successful run does not show how robust the training process is.

### Status

Experiment 019 complete.

### Next Direction

Investigate initialization strategies and how they affect gradient flow before moving to larger networks.



## Entry 021 — Initialization Scale

**Date:** 2026-09-27

### Experiment

Tested how the scale of the initial weights affects training.

The same automatically constructed XOR network was used for every run:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

Training data:

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

The random seed, learning rate, architecture, and training duration were held constant.

Learning rate:

0.05

Training epochs:

1000

The initial weights produced by the network were multiplied by different scale values.

### Results

Scale 0.0:

Loss ≈ 0.500000863377
Success = False

Scale 0.001:

Loss ≈ 0.000000000000
Success = True

Scale 0.01:

Loss ≈ 0.000000000000
Success = True

Scale 0.1:

Loss ≈ 0.000000000000
Success = True

Scale 1.0:

Loss ≈ 0.000000000000
Success = True

Scale 10.0:

Loss ≈ 0.500000863377
Success = False

Scale 100.0:

Loss ≈ 0.500000863377
Success = False

### Observation

The initialization scale affected whether the same network successfully learned XOR.

Zero initialization failed.

Very small through moderate initialization scales successfully learned the task.

Very large initialization scales failed under the same learning rate and training budget.

### Lesson

Initialization is not simply about making weights random.

The magnitude of the starting parameters also matters.

If all parameters begin at zero, neurons can remain identical and fail to learn different features.

If parameters begin too large, the same learning rate can produce unsuitable updates and prevent successful training.

### Important Understanding

Initialization interacts with optimization.

The starting values determine the initial activations and gradients, while the learning rate determines how far parameters move during each update.

A useful initialization strategy therefore needs to consider both the network architecture and the optimization process.

### Status

Experiment 020 complete.

### Next Direction

Investigate how initialization can be chosen systematically instead of selecting a scale manually.


## Entry 022 — He Initialization

**Date:** 2026-09-27

### Experiment

Compared the project's existing uniform random initialization with He initialization.

The same XOR network and training procedure were used across ten random seeds.

Architecture:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

Training data:

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

Learning rate:

0.05

Training epochs:

1000

The He initialization was applied to the ReLU hidden layer.

The linear output layer was left with its existing initialization.

### He Initialization

For the ReLU hidden layer, the standard deviation was calculated as:

sqrt(2 / fan_in)

Weights were sampled from a zero-centered normal distribution using that standard deviation.

### Results

Uniform initialization:

Successful runs = 2/10

He initialization:

Successful runs = 3/10

### Observation

He initialization produced one additional successful run in this ten-seed test.

However, most runs still did not reach the success threshold.

The result therefore shows that initialization strategy can affect training behavior, but this experiment does not establish that He initialization is universally better.

### Lesson

Initialization strategies are designed around the behavior of the activation function and the number of inputs to a layer.

For ReLU networks, He initialization uses a scale based on the layer's input size.

This gives a more principled starting point than selecting an arbitrary weight scale.

### Important Understanding

The initialization experiment also highlighted an important distinction:

- ReLU hidden layers use one type of initialization strategy.
- A linear output layer may require different treatment.

Initialization should therefore be considered together with the layer's activation function and architecture.

### Status

Experiment 021 complete.

### Next Direction

Measure the actual activation and gradient values produced by different initialization strategies to understand why some initializations train successfully while others become stuck.

## Entry 023 — Activation and Gradient Measurements

### Question

Why do different initialization strategies produce different training behavior?

Experiment 022 measured the initial hidden-layer ReLU activations and gradients produced by uniform and He initialization across the same ten seeds used in Experiment 021.

### Results

Uniform initialization:

- Active activation fraction = 0.325
- Mean activation = 0.215116
- Mean absolute gradient = 0.111849
- Zero-gradient fraction = 0.750000
- Successful runs = 2/10

He initialization:

- Active activation fraction = 0.4375
- Mean activation = 0.439705
- Mean absolute gradient = 0.153237
- Zero-gradient fraction = 0.666667
- Successful runs = 3/10

### Observation

He initialization produced more active hidden-layer ReLU outputs on average.

It also produced a lower fraction of zero gradients and a higher mean absolute gradient than uniform initialization.

One uniform-initialization run had a completely inactive hidden layer:

- active activation fraction = 0.000
- zero-gradient fraction = 1.000
- final loss = 0.500000863377
- success = False

This provides a concrete example of a network becoming stuck because the ReLU hidden layer did not pass useful gradients during the measured samples.

### Interpretation

The measurements support the idea that initialization affects training indirectly through the distribution of activations and gradients.

For this network, He initialization produced a healthier initial distribution than the tested uniform initialization, but the difference was not large enough to make every run successful.

The experiment therefore explains part of the behavior observed in Experiment 021 without establishing that activation statistics alone determine whether training succeeds.

### Lesson

Initialization is not only about the initial size of the weights.

It also affects:

- which ReLU units are active,
- how much signal reaches later layers,
- how much gradient can propagate backward,
- and whether a unit can become inactive and remain difficult to train.

### Important Understanding

A useful way to study neural-network training is to connect three levels of evidence:

1. Initialization values
2. Activation and gradient behavior
3. Final training outcome

Experiment 022 connects the first and second levels and provides a measurable explanation for part of the third.

### Status

Experiment 022 complete.

### Next Direction

Test whether activation and gradient behavior changes over the course of training by measuring these values at multiple checkpoints rather than only at initialization.

## Experiment 023 — Training Activation Checkpoints

Date: 2026-09-27

### Question
Does initialization affect activation survival and gradient flow during training?

### Setup
Compared uniform initialization against He initialization using:
- XOR training data
- 2 input neurons
- 2 hidden ReLU neurons
- 1 linear output neuron
- learning rate 0.05
- 1000 epochs
- seeds 0–9
- checkpoints 0, 1, 10, 100, 500, 1000

### Findings
He initialization produced stronger overall training behavior than uniform initialization, but remained highly seed-dependent.

Uniform initialization produced complete activation collapse for some seeds. He initialization reduced the frequency and severity of complete collapse, but did not eliminate it.

Successful He seeds reached near-zero loss, while several seeds converged to loss plateaus around 0.25 or 0.333.

### Conclusion
Initialization is a major contributor to training stability, but initialization alone does not explain the seed-dependent failures.

Next question: determine whether individual hidden neurons die during training or whether failures can occur while neurons remain active.

---

## Experiment 024 — Individual Neuron Lifetime

Date: 2026-09-27

### Question
When He initialization fails, do individual hidden ReLU neurons die at initialization or during training?

### Setup
Used the same network and training procedure as Experiment 023, with He initialization only.

Measured each hidden neuron independently at epochs 0, 1, 10, 100, 500, and 1000:
- active fraction
- mean activation
- mean absolute gradient
- zero-gradient fraction
- activation pattern across the four XOR inputs
- weight norm

### Findings
Two distinct failure modes appeared.

1. Dead-neuron failure:
Some neurons were already completely inactive at epoch 0, while others became permanently inactive during training.

Examples:
- Seed 5: neuron 0 was dead at initialization.
- Seed 7: neuron 1 was dead at initialization.
- Seed 6: neuron 1 died by epoch 1.
- Seed 0: neuron 0 died between epochs 10 and 100.
- Seed 3: neuron 1 died between epochs 10 and 100.

2. Non-dead failure:
Seeds 1 and 9 did not lose both neurons, yet still converged to a loss plateau near 0.25.

### Conclusion
Neuron death is a real failure mechanism, but it is not the complete explanation for failed training.

There is another failure mode in which hidden neurons remain active but learn a representation that does not allow the final linear neuron to solve XOR.

Next question: inspect the hidden representation produced by successful and unsuccessful seeds.

---

## Experiment 025 — Hidden Representation

Date: 2026-09-27

### Question
When neurons survive, does the learned hidden representation determine whether XOR can be solved?

### Setup
Used the same He-initialized 2-2-1 network and training procedure.

At epochs 0, 1, 10, 100, 500, and 1000, recorded:
- hidden neuron activations for all four XOR inputs
- output weights
- output bias
- predictions

### Findings
Successful runs developed hidden representations that allowed the linear output neuron to separate the XOR cases.

Seed 2 eventually produced:
- [0,0] -> prediction 0
- [0,1] -> prediction 1
- [1,0] -> prediction 1
- [1,1] -> prediction 0

Seed 4 and seed 8 similarly reached essentially perfect XOR predictions.

Failed runs can retain active neurons without producing a useful separable representation.

For example, seed 1 retains active hidden units but eventually predicts approximately:
- [0,0] -> 0.5128
- [0,1] -> 0.5128
- [1,0] -> 1.0000
- [1,1] -> 0.0000

This produces a loss plateau because two inputs remain indistinguishable to the learned representation.

### Conclusion
The experiments now show that seed-dependent failure has at least two mechanisms:

1. ReLU neurons can become permanently inactive.
2. Neurons can remain active but converge to a hidden representation that cannot support the required XOR separation.

Experiment 026 should isolate the geometry of the hidden representation and determine what property distinguishes successful representations from failed ones.

---

## Experiment 026 — Representation Geometry

Date: 2026-09-27

### Question
Does the geometric separability of the hidden representation distinguish successful and failed XOR training?

### Setup
Used the same He-initialized 2-2-1 network from Experiments 024 and 025.

The four XOR inputs were mapped into the two-dimensional hidden ReLU representation.

The positive class consisted of:
- [0,1]
- [1,0]

The negative class consisted of:
- [0,0]
- [1,1]

Measured the distance between the two class line segments in hidden space. A nonzero distance indicates that the two classes are linearly separable.

### Findings

At epoch 1000, every successful seed had a nonzero separation gap:

- Seed 2: gap 0.565872044
- Seed 4: gap 0.477392712
- Seed 8: gap 0.527839013

Every failed seed had a gap of 0.

The timing of separability was also important.

Seed 0 began with a separable representation (gap 0.344778929) but lost separability by epoch 100 and eventually converged to loss approximately 0.33347.

Seed 6 also began separable (gap 0.072356197) but lost separability by epoch 1 and eventually converged to loss approximately 0.33349.

Conversely, successful seeds could begin nonseparable and develop separability during training.

Seed 2 became separable between epochs 100 and 500.

Seed 4 became separable between epochs 100 and 500.

Seed 8 became separable between epochs 10 and 100.

Seeds 1 and 9 remained nonseparable and converged to loss plateaus near 0.25.

### Conclusion
Final hidden-space separability strongly corresponds with successful XOR training in this experiment.

Initialization determines the starting geometry, but training dynamics can either create the required separation or destroy it.

The next question is when the separation transition occurs and what changes immediately before the network becomes separable or loses separability.


---

## Experiment 027 — Separability Transition Timeline

Date: 2026-09-27

### Question
At what point during training does the hidden representation become separable, or lose separability?

### Setup
Used the same He-initialized 2-2-1 XOR network from Experiments 024–026.

Instead of checking only selected checkpoints, hidden-space separability was evaluated after every training epoch from 1 through 1000.

Recorded every transition between:
- separable
- nonseparable

For every transition, the hidden representation and loss were recorded.

### Findings

Successful runs developed separability during training:

- Seed 8: nonseparable -> separable at epoch 80.
- Seed 2: nonseparable -> separable at epoch 131.
- Seed 4: nonseparable -> separable at epoch 146.

All three ultimately reached essentially zero loss.

Failed runs could also lose an initially separable representation:

- Seed 6: separable -> nonseparable at epoch 1.
- Seed 0: separable -> nonseparable at epoch 27.

These runs eventually converged to loss near 0.333.

Four other failed seeds (1, 3, 5, and 7) never crossed into separability during the 1000-epoch run.

Seed 9 also remained nonseparable throughout training and converged to a loss near 0.25.

### Conclusion
Training success is associated with entering and maintaining a linearly separable hidden representation.

The experiment also shows that separability is not determined solely by initialization. Training can create the required representation or destroy one that existed initially.

The next question is what changes immediately before these separability transitions.


---

## Experiment 028 — Transition Microscope

Date: 2026-09-27

### Question
What changes immediately before and after the hidden representation becomes separable or loses separability?

### Setup
Used He initialization with the same 2-2-1 XOR network.

Focused on the five seeds with meaningful separability transitions from Experiment 027:
- Seed 0: YES -> NO at epoch 27
- Seed 2: NO -> YES at epoch 131
- Seed 4: NO -> YES at epoch 146
- Seed 6: YES -> NO at epoch 1
- Seed 8: NO -> YES at epoch 80

For each transition, inspected a five-epoch window before and after the transition.

Recorded:
- loss
- hidden-space separation gap
- hidden activations for all four inputs
- output weights and bias
- mean hidden-neuron gradients
- zero-gradient fractions

### Findings

#### Seed 0 — separability lost

Before the transition, the hidden representation was still separable but the gap was shrinking:

- epoch 22: gap 0.084172239
- epoch 23: gap 0.065868940
- epoch 24: gap 0.047404318
- epoch 25: gap 0.029383488
- epoch 26: gap 0.011260454

At epoch 27 the gap reached zero.

The important change was neuron 0 becoming inactive for the [1,0] input:

- epoch 26: neuron 0 activation for [1,0] = 0.011262
- epoch 27: neuron 0 activation for [1,0] = 0.000000

At the transition, neuron 0's mean gradient became exactly zero and its zero-gradient fraction became 1.000.

Afterward, neuron 0 remained permanently inactive and the representation remained nonseparable.

#### Seed 6 — separability lost immediately

The network began with a separable representation with gap 0.072356197.

After the first training epoch, the [0,1] activation of neuron 1 reached zero:

- epoch 0: neuron 1 [0,1] = 0.073127
- epoch 1: neuron 1 [0,1] = 0.000000

Neuron 1's mean gradient became exactly zero and its zero-gradient fraction became 1.000.

The separation gap immediately dropped to zero and remained there.

#### Seed 2 — separability formed without neuron death

The network was nonseparable through epoch 130.

At epoch 131, it became separable with gap 0.004320596.

The hidden points were:

- [0,0] -> (0.000000, 0.015828)
- [0,1] -> (0.000000, 0.011241)
- [1,0] -> (1.416142, 0.515809)
- [1,1] -> (0.096308, 0.511223)

Both hidden neurons remained trainable. Mean gradients were nonzero for both neurons.

The separation gap then increased during subsequent epochs.

#### Seed 4 — separability formed through a geometric boundary crossing

The representation remained nonseparable through epoch 145.

At epoch 146, the gap became positive:

0.002305559

The transition occurred while both neurons remained active and trainable.

The hidden points moved only slightly, but their geometric ordering changed enough for the two class segments to stop intersecting.

The gap then increased over subsequent epochs.

#### Seed 8 — gradual geometric separation

The network remained nonseparable through epoch 79.

At epoch 80 the gap became positive:

0.004042234

No hidden neuron became permanently inactive.

Between epochs 75 and 80, the hidden representations moved continuously while the gap changed from zero to positive.

The gap then grew rapidly:

- epoch 80: 0.004042234
- epoch 81: 0.019419898
- epoch 82: 0.034943003
- epoch 83: 0.050594217
- epoch 84: 0.066354879
- epoch 85: 0.082205062

### Conclusion

Experiment 028 identifies two different transition mechanisms.

1. A separable representation can be destroyed when a ReLU neuron crosses zero for an important training input. When this happens, the corresponding gradient becomes permanently zero and the neuron can no longer move that input back into the active region.

2. A separable representation can also form without neuron death. In successful seeds 2, 4, and 8, the hidden points gradually move until the positive and negative class segments become geometrically separable.

Therefore, the important variable is not simply whether neurons are alive.

The deeper variable is whether training dynamics move the hidden representation toward or away from a geometry that the final linear neuron can separate.

Next question: determine whether the gradient direction itself predicts whether the hidden representation will become more separable or less separable.


---

## Experiment 029 — Gradient Geometry

Date: 2026-09-27

### Question
Which individual training examples move the hidden representation toward or away from linear separability?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Focused on the informative transition seeds from Experiment 027:
- Seed 0
- Seed 2
- Seed 4
- Seed 6
- Seed 8

For the epochs surrounding each transition, each XOR training example was examined separately.

For every example, recorded:
- hidden-space separation gap before the update
- hidden-space separation gap after the update
- change in separation gap
- hidden-layer gradients for each neuron
- output-layer gradients
- hidden representation before and after the update

Training examples were processed in the existing order:
[0,0], [0,1], [1,0], [1,1].

### Findings

The four training examples exerted different and sometimes opposing effects on hidden-space geometry.

#### Seed 0

The representation was separable immediately before the failure.

At epoch 27, the [1,0] example produced the critical update:

- gap before: 0.011260454
- gap after: 0.000000000
- delta gap: -0.011260454

Neuron 0 changed its [1,0] hidden activation from approximately 0.011262 to 0.000000.

Its gradient then became exactly zero, leaving the neuron unable to recover that input.

#### Seed 6

The representation started separable.

At epoch 1, the [0,1] example produced:

- gap before: 0.072356197
- gap after: 0.000000000
- delta gap: -0.072356197

Neuron 1 changed its [0,1] activation from approximately 0.073127 to 0.000000.

Its gradient became exactly zero immediately afterward.

#### Seed 2

Near epoch 131, the [0,1] example created the first useful separation:

- gap before: 0
- gap after: 0.009122485

The following [1,0] and [1,1] examples produced competing changes, including negative delta-gap updates, but the representation remained separable.

The important point is that useful separation could be created while both hidden neurons remained trainable.

#### Seed 4

Near the transition, the [1,0] example was the update that created separation.

At epoch 145, [1,0] produced:

- gap before: 0
- gap after: 0.001440382

The subsequent [1,1] update removed that small gap.

At epoch 146, the [1,0] update produced a larger gap:

- gap before: 0
- gap after: 0.005315076

The [1,1] update again reduced the gap, but this time it remained positive:

- gap after [1,1]: 0.002305559

The network therefore crossed into the separable regime because the positive contribution from [1,0] exceeded the opposing update from [1,1].

#### Seed 8

Near epoch 79, the [1,0] example created a temporary separable representation:

- gap after [1,0]: 0.004898453

The following [1,1] update destroyed that gap:

- gap after [1,1]: 0

At epoch 80, the [1,0] update created a much larger gap:

- gap after [1,0]: 0.020289951

The following [1,1] update reduced it to:

- gap after [1,1]: 0.004042234

The representation therefore remained separable.

### Conclusion

Individual training examples can push the hidden representation in opposing geometric directions.

In the failed transitions for seeds 0 and 6, a single update moved a critical hidden activation across the ReLU boundary and permanently removed its gradient.

In successful transitions, particular examples created positive separation, while other examples sometimes reduced it without destroying it.

This suggests that the fixed training-example order may matter because the network parameters change after every example.

The next experiment should test whether changing or shuffling the presentation order changes the probability of reaching a separable hidden representation.


---

## Experiment 030 — Training Order Sensitivity

Date: 2026-09-27

### Question
Does the order in which XOR training examples are presented materially affect the seed-dependent training outcome?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Compared four deterministic presentation orders while keeping architecture, initialization, learning rate, and number of epochs unchanged:

- fixed: [0,1,2,3]
- reverse: [3,2,1,0]
- odd-even: [1,3,0,2]
- even-odd: [0,2,1,3]

Ten seeds were tested for each order.

A successful run was defined as final loss below 1e-6 with a nonzero hidden-space separation gap.

### Findings

Every tested order produced 3 successful runs out of 10.

Final aggregate results:

- fixed: 3/10 successful, mean loss 0.216863172405
- reverse: 3/10 successful, mean loss 0.216864670238
- odd-even: 3/10 successful, mean loss 0.216890863108
- even-odd: 3/10 successful, mean loss 0.216861674573

The reverse order changed the identity of one successful seed. Seed 1 failed under the fixed order but succeeded under the reverse order.

Seeds 4 and 8 succeeded under every tested order, while most other seeds remained failures.

The final mean separation gaps were also very similar across orders.

### Conclusion

Presentation order can alter individual training trajectories, but the tested orders did not materially change the overall probability of successful training in this 10-seed experiment.

Therefore, the seed-dependent instability is unlikely to be explained primarily by the ordering of the four XOR samples.

The next question is whether the instability comes from applying parameter updates after every individual example at all.

Experiment 031 should compare online per-example updates against full-batch updates using the same initialization, learning rate, architecture, and training data.


---

## Experiment 031 — Online vs Full-Batch Updates

Date: 2026-09-27

### Question
Does applying gradients after every individual example versus once per full batch affect the seed-dependent XOR training outcome?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Compared:

1. Online updates:
   - process one example
   - backpropagate
   - update parameters
   - repeat for all four examples

2. Full-batch updates:
   - compute gradients for all four examples at the same parameter state
   - average the gradients
   - perform one parameter update

The original batch implementation was first found to be invalid because `Neuron.backward()` overwrites stored gradients rather than accumulating them. The batch implementation was corrected to explicitly accumulate and average gradients before updating parameters.

Parameter-update counts were then equalized:

- online: 1000 epochs × 4 updates = 4000 parameter updates
- batch: 4000 epochs × 1 update = 4000 parameter updates

Ten He-initialized seeds were tested for each method.

### Findings

Online training:

- successful runs: 3/10
- mean final loss: 0.216863172405
- mean final separation gap: 0.157110377

Full-batch training:

- successful runs: 4/10
- mean final loss: 0.182302779168
- mean final separation gap: 0.242709185

Seeds 2, 4, and 8 succeeded under both methods.

Seed 1 failed under online training but succeeded under full-batch training.

Several failed seeds converged near the familiar loss plateau around 0.333333, while full-batch training produced a stronger hidden-space separation for several successful runs.

### Conclusion

Update granularity affects the optimization trajectory in this network.

With equalized parameter-update counts, full-batch training produced more successful runs and a larger mean final hidden-space separation gap than online training in this 10-seed experiment.

The result does not establish that full-batch training is generally superior. It establishes that the way Larry aggregates gradients is a meaningful experimental variable that can change whether a useful hidden representation emerges.

Next question: determine whether the difference comes from gradient averaging itself or from the different parameter trajectory produced by the two update rules.


---

## Experiment 032 — Batch Gradient Scaling

Date: 2026-09-27

### Question
Does batch gradient scaling affect the seed-dependent training outcome?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Compared three full-batch update rules:

1. Average:
   sum the four example gradients and divide by 4.

2. Sum:
   sum the four example gradients without dividing by 4.

3. Sum-scaled:
   sum the four example gradients without dividing by 4, but divide the learning rate by 4.

All conditions used 4000 batch parameter updates across seeds 0–9.

The average and sum-scaled conditions therefore apply mathematically equivalent parameter updates.

### Findings

Average:
- successful runs: 4/10
- mean final loss: 0.182302779168
- mean final separation gap: 0.242709185

Sum-scaled:
- successful runs: 4/10
- mean final loss: 0.182302779168
- mean final separation gap: 0.242709185

The identical results confirm that the two implementations are equivalent.

Sum:
- successful runs: 5/10
- mean final loss: 0.166666666667
- mean final separation gap: 0.272730891

The unscaled sum condition caused seed 9 to reach essentially zero loss, while seed 9 remained at a higher-loss state under average and sum-scaled updates.

### Conclusion

The magnitude of the batch parameter update materially changes the training trajectory.

The average and sum-scaled controls produced identical results, confirming that the observed difference is caused by the effective update size rather than the implementation of gradient accumulation itself.

In this experiment, the larger unscaled batch step produced more successful runs than the averaged batch step.

Therefore, learning-rate scale is now a strong candidate explanation for some of the seed-dependent behavior observed earlier.

Next question: determine whether there is a reproducible relationship between learning rate and the probability of reaching a separable hidden representation.


---

## Experiment 033 — Learning Rate Stability Window

Date: 2026-09-27

### Question
How does learning-rate magnitude affect the formation of a usable hidden representation and successful XOR training?

### Setup
Used the corrected full-batch implementation from Experiment 031.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- 4000 parameter updates
- seeds 0–9
- learning rates: 0.01, 0.025, 0.05, 0.10, 0.20

Two outcomes were tracked:

1. Hidden-space separability.
2. Near-zero final loss (< 1e-6) with nonzero separation.

### Findings

#### Learning rate 0.01

- separable runs: 2/10
- near-zero-loss runs: 0/10
- mean loss: 0.349609392756
- mean gap: 0.015550920

Seeds 2 and 4 developed nonzero separation but did not converge to near-zero loss.

#### Learning rate 0.025

- separable runs: 4/10
- near-zero-loss runs: 0/10
- mean loss: 0.222895230645
- mean gap: 0.160115760

Seeds 1, 2, 4, and 8 became separable, but their final losses remained above the near-zero threshold.

#### Learning rate 0.05

- separable runs: 5/10
- near-zero-loss runs: 4/10
- mean loss: 0.182302779168
- mean gap: 0.242709185

Seeds 1, 2, 4, 8, and 9 were separable. Seeds 1, 2, 4, and 8 reached near-zero loss.

#### Learning rate 0.10

- separable runs: 5/10
- near-zero-loss runs: 5/10
- mean loss: 0.166666666667
- mean gap: 0.272111421

Seeds 1, 2, 4, 8, and 9 reached near-zero loss and were separable.

#### Learning rate 0.20

- separable runs: 5/10
- near-zero-loss runs: 5/10
- mean loss: 0.166666666667
- mean gap: 0.272730891

The outcome was very similar to learning rate 0.10.

### Conclusion

Learning rate materially changes Larry's training trajectory.

At 0.01, some seeds formed separable representations but training did not reach near-zero loss.

Increasing the learning rate to 0.025 improved hidden-space separation and final loss, but still did not produce any near-zero-loss runs under the selected threshold.

At 0.05, five seeds became separable and four reached near-zero loss.

At 0.10 and 0.20, five seeds reached near-zero loss, with very similar aggregate behavior.

This suggests that, for the current full-batch setup, increasing the learning rate from 0.01 toward approximately 0.10 improves the ability to escape poor trajectories and form useful representations. The results between 0.10 and 0.20 are already very similar, so simply increasing the learning rate further may not explain the remaining seed dependence.

The next question should therefore examine what distinguishes the permanently failed seeds from the successful seeds after learning-rate effects have been accounted for.


## Experiment 034 — Early Trajectory Discriminator

### Question

Can the seeds that eventually solve XOR be distinguished from the permanently failed seeds early in training, after fixing the learning rate at a stable value?

### Setup

Used the corrected full-batch implementation from Experiment 031 and the learning rate selected from Experiment 033.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9
- checkpoints: epochs 0, 1, 2, 5, 10, 20, 50, 100, and 4000

At each checkpoint, measured:

- loss
- hidden-space separation gap
- hidden activation pattern
- mean hidden activation
- mean absolute hidden gradient
- zero-gradient fraction

Success remained defined as final loss < 1e-6 with nonzero hidden-space separation.

### Results

The final split was:

- successful seeds: 1, 2, 4, 8, 9
- failed seeds: 0, 3, 5, 6, 7
- success rate: 5/10

At initialization, the successful seeds were not simply the seeds with an immediately separable hidden representation. Seeds 1, 2, 4, 8, and 9 all had zero separation at epoch 0, while only seed 6 had a small positive gap among the eventual failures.

The clearest early discriminator was hidden-unit activity.

By epoch 50:

- successful seeds retained 6–7 active hidden-unit/example pairs
- failed seeds retained only 1–2 active pairs

By epoch 100:

- successful seeds retained 5–7 active hidden-unit/example pairs
- failed seeds retained only 1–2 active pairs

The failed trajectories therefore progressively lost hidden-unit activity while their separation gap remained zero. Seeds 0 and 6 also demonstrated explicit loss of an initially useful geometric state: seed 0's gap fell from 0.3448 at initialization to zero by epoch 100, while seed 6's initial gap of 0.0724 fell to zero by epoch 5.

The successful trajectories were different. Their hidden representations were not separable during the early checkpoints, but multiple hidden activations remained active while training continued. Their separation emerged later, reaching positive values only by the final checkpoint.

At epoch 4000, the successful seeds had:

- seed 1: gap 0.6441
- seed 2: gap 0.5722
- seed 4: gap 0.4756
- seed 8: gap 0.5293
- seed 9: gap 0.5000

All five reached near-zero loss.

### Conclusion

The remaining seed dependence is strongly associated with early hidden-unit activity rather than initial hidden-space separability alone.

Successful seeds preserved several active hidden responses through the early training trajectory, even while their hidden representations were still nonseparable. Failed seeds progressively reduced their active patterns to only 1–2 active hidden-unit/example pairs, after which their gradients weakened and the network converged to a loss of approximately 1/3 without developing a separable representation.

This suggests that preserving a sufficiently rich hidden representation early in training may be a prerequisite for later geometric separation and successful XOR learning.

The next experiment should isolate hidden-unit survival/activity as a causal variable rather than only measuring it as an outcome.

## Experiment 035 — Hidden Neuron Death Guard

### Question

Is complete hidden-neuron death itself responsible for the seed-dependent failure observed in Experiment 034?

### Setup

Compared the normal training process against an intervention that prevents a hidden ReLU neuron from becoming completely inactive.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9

The baseline used the standard training rule.

The guarded condition monitored the two hidden neurons after each update. If a hidden neuron was inactive on all four XOR examples, that neuron's weights and bias were restored to their values from immediately before the update.

Success remained defined as final loss < 1e-6 with nonzero hidden-space separation.

### Results

Baseline:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

Guarded:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

The intervention therefore did not increase the number of successful runs.

For the successful seeds, the guard was never triggered:

- seed 1: 0 interventions
- seed 2: 0 interventions
- seed 4: 0 interventions
- seed 8: 0 interventions
- seed 9: 0 interventions

For the failed seeds, the guard triggered repeatedly:

- seed 0: 3946 interventions
- seed 3: 3954 interventions
- seed 5: 4000 interventions
- seed 6: 3998 interventions
- seed 7: 4000 interventions

Seeds 5 and 7 became completely inactive immediately and remained so under the guard. Seeds 0, 3, and 6 were also prevented from permanently entering the exact all-dead state, but this did not produce successful learning.

The guarded results for seeds 0 and 6 retained small nonzero separation gaps, but their losses remained near 1/3 and therefore did not meet the success criterion.

### Conclusion

Preventing complete hidden-neuron death by itself did not improve XOR success.

The successful seeds never required the guard, while the failed seeds repeatedly attempted to enter a completely inactive hidden-neuron state. Blocking that state did not cause the failed trajectories to discover the useful hidden representation required for XOR.

This indicates that hidden-neuron death is likely associated with failed training but is not, by itself, the sole causal explanation for the remaining seed dependence.

The important distinction is between merely keeping a neuron numerically active and preserving a hidden representation that provides useful gradient information for separating the XOR classes.

The next experiment should therefore examine the quality and direction of the hidden gradients before neuron death, rather than treating neuron survival alone as the intervention target.

## Experiment 036 — Gradient Conflict and Support

### Question

Do opposing per-example hidden-layer gradients explain the seed-dependent failure observed in earlier experiments?

### Setup

Used the same training configuration as Experiments 033–035.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9
- checkpoints: epochs 0, 1, 2, 5, 10, 20, 50, 100, and 4000

For each checkpoint, the hidden-layer gradient was measured separately for each XOR training example.

Pairwise cosine similarity was used to measure gradient direction:

- positive cosine: aligned update directions
- negative cosine: opposing update directions

Because some ReLU examples produce zero hidden gradients, only nonzero gradient pairs were included in the cosine calculations.

The number of valid pairs was also recorded as a measure of gradient support.

### Results

The final split remained:

- successful seeds: 1, 2, 4, 8, 9
- failed seeds: 0, 3, 5, 6, 7
- success rate: 5/10

Gradient conflict alone did not explain success or failure.

Several successful seeds developed strongly negative mean cosine similarity:

- seed 1 reached approximately -0.26 by epochs 20–100
- seed 2 reached approximately -0.30 to -0.33
- seed 4 reached approximately -0.18 to -0.29
- seed 8 reached approximately -0.26 to -0.32
- seed 9 reached approximately -0.25 to -0.30

Therefore, successful learning can occur while individual examples push the hidden layer in substantially opposing directions.

The number of examples contributing usable hidden gradients was more strongly associated with the eventual outcome.

At epoch 5:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 3
- seeds 3, 5, and 7 had 1
- seed 6 had 0

At epoch 20:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 1
- seeds 3, 5, and 7 had 1
- seed 6 had 0

At epoch 50:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 1
- seed 3 had 1
- seeds 5, 6, and 7 had 0

At epoch 100:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 0
- seed 3 had 1
- seeds 5, 6, and 7 had 0

The successful trajectories therefore maintained broad hidden-gradient participation across the XOR dataset, while the failed trajectories rapidly became supported by only one or zero effective hidden-gradient examples.

### Conclusion

Per-example gradient conflict is not sufficient to explain Larry's seed dependence. Strongly opposing gradients occur in both successful and failed trajectories.

The more consistent distinction is gradient support.

Successful seeds preserve hidden gradients across many XOR examples during early training. Failed seeds rapidly lose that support, leaving the batch update determined by very few examples before eventually reaching a zero-gradient state.

This refines the earlier neuron-survival hypothesis. The important property may not be whether a hidden neuron remains numerically active, but whether the hidden layer continues receiving useful gradient information from a sufficiently broad portion of the training set.

The next experiment should test this activation/gradient-coverage hypothesis directly.

## Experiment 037 — Gradient Floor Rescue

### Question

Are failed seeds caused primarily by ReLU's zero derivative on negative hidden pre-activations?

### Setup

Compared the normal hidden ReLU against a controlled gradient-floor condition.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9

The baseline used the standard ReLU derivative:

- positive pre-activation: derivative 1
- nonpositive pre-activation: derivative 0

The gradient-floor condition kept the exact same ReLU forward function, but changed the hidden derivative for nonpositive pre-activations from 0 to 0.01.

This isolates the effect of removing the exact zero-gradient condition without changing the hidden activation values.

Success remained defined as final loss < 1e-6 with nonzero hidden-space separation.

### Results

Baseline:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

Gradient floor:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

The gradient floor changed the final hidden-space gap slightly for successful seeds but did not change which seeds succeeded.

Failed seeds remained failed:

- seed 0: loss 0.333335370, gap 0
- seed 3: loss 0.333335740, gap 0
- seed 5: loss 0.333359911, gap 0
- seed 6: loss 0.333345190, gap 0
- seed 7: loss 0.250019296, gap 0

Seed 7 showed the largest improvement in loss, decreasing from approximately 0.3333 to 0.2500, but it still failed to create a separable hidden representation.

All five successful seeds remained successful under the intervention.

### Conclusion

Eliminating exact zero hidden gradients is not sufficient to explain or prevent the seed-dependent failure.

The forward representation remained identical to ReLU, while only the negative-side hidden derivative was changed. Despite this intervention, none of the five failed seeds became successful.

This strengthens the conclusion from Experiments 035 and 036: the problem is not simply that gradients become exactly zero. Successful training appears to depend on whether the hidden representation develops in a useful direction and preserves enough structure across the training examples.

The improvement for seed 7 shows that a gradient floor can alter the trajectory, but the resulting representation still did not become separable.

The next experiment should investigate whether increasing hidden-layer capacity reduces the seed dependence by giving the optimizer more representational freedom.

## Experiment 038 — Hidden Capacity Sweep

### Question

Does increasing hidden-layer capacity reduce the seed dependence observed with the 2-neuron hidden layer?

### Setup

Varied only the number of hidden ReLU neurons.

- He initialization
- XOR training data
- one linear output neuron
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9
- hidden widths: 1, 2, 3, 4, and 6

Success was defined as final loss < 1e-6.

For width 2, hidden-space separation gap was also retained for comparison with earlier experiments.

### Results

The success rate increased substantially with hidden width:

| Hidden width | Successful runs | Mean final loss |
|---|---:|---:|
| 1 | 0/10 | 0.416666666667 |
| 2 | 5/10 | 0.166666666667 |
| 3 | 6/10 | 0.133333333333 |
| 4 | 7/10 | 0.083333333333 |
| 6 | 9/10 | 0.025000000000 |

Width 1 failed on every seed.

The original 2-neuron configuration retained the 5/10 success rate observed in Experiments 033–037.

Increasing the hidden width to 3 changed the successful seeds to:

- 1, 2, 4, 6, 8, 9

Width 4 succeeded on:

- 1, 2, 4, 6, 7, 8, 9

Width 6 succeeded on:

- 0, 1, 2, 3, 4, 6, 7, 8, 9

Only seed 5 failed at width 6, with final loss 0.25.

The successful width-2 runs continued to produce the same hidden-space gaps observed previously:

- seed 1: 0.644062744
- seed 2: 0.572161784
- seed 4: 0.475612787
- seed 8: 0.529266449
- seed 9: 0.500010441

### Conclusion

Hidden-layer capacity strongly affects the reliability of XOR learning under the current training setup.

A single hidden ReLU neuron could not solve XOR in any of the tested seeds. Two neurons were sufficient to solve XOR, but only 5/10 seeds converged successfully.

Increasing capacity progressively reduced seed dependence:

- width 3: 6/10
- width 4: 7/10
- width 6: 9/10

This indicates that the remaining failures are not explained solely by ReLU death or zero gradients. Additional hidden units provide more representational and optimization freedom, allowing more initializations to reach a useful solution.

However, increasing width also increases the number of trainable parameters. Therefore, this experiment establishes a strong association between capacity and reliability, but does not yet separate increased representational capacity from the optimization effects of having more parameters.

The next experiment should investigate whether the width improvement comes from additional representational degrees of freedom or simply from having more independent hidden units available during optimization.

## Experiment 039 — Effective Capacity by Pruning

### Question

Do successful width-6 networks actually require all six hidden neurons in their final learned representation?

### Setup

Used the width-6 configuration from Experiment 038.

- He initialization
- 6 hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9

After training, each successful network was tested with every possible subset of its six hidden neurons.

Pruning was performed by setting the output weight connected to a hidden neuron to zero while leaving the trained hidden-layer parameters unchanged.

For each seed, the smallest subset producing loss < 1e-6 was recorded.

This is a post-training compressibility test. It does not measure whether the narrower network could have reached the same solution if trained from scratch.

### Results

The width-6 network again succeeded on:

- 9/10 seeds

Seed 5 was the only failure.

Among the nine successful runs, the minimum number of contributing hidden neurons required after training was:

- seed 0: 3
- seed 1: 3
- seed 2: 3
- seed 3: 3
- seed 4: 5
- seed 6: 4
- seed 7: 4
- seed 8: 5
- seed 9: 5

Distribution:

- 3 neurons: 4 runs
- 4 neurons: 2 runs
- 5 neurons: 3 runs
- 6 neurons: 0 runs

Mean minimum contributing neurons: 3.89.

Therefore, none of the successful width-6 solutions required all six hidden neurons after training.

### Conclusion

The capacity improvement observed in Experiment 038 does not mean that six hidden neurons are required to represent the final XOR solutions.

Successful width-6 models could be compressed after training to between 3 and 5 contributing hidden neurons while retaining near-zero loss.

This suggests that additional hidden capacity primarily provides extra degrees of freedom during optimization rather than being fully required by the final solution.

The distinction is important: width 6 may make it easier for gradient descent to discover a useful representation, even when the resulting solution can later be represented with fewer neurons.

The post-training pruning result does not establish that a width-3 or width-4 network trained from scratch would reliably reproduce those solutions. Experiment 038 already showed that narrower networks have lower success rates.

The next experiment should test whether the extra capacity helps specifically by allowing multiple candidate hidden features to develop before some become unnecessary.

## Experiment 040 — Late Capacity Injection

### Question

Does additional hidden capacity need to be present from initialization, or can it rescue a width-2 trajectory after training has already begun?

### Setup

Started each run with the same 2-neuron hidden ReLU network used in Experiments 033–039.

- He initialization
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter-update steps
- seeds 0–9

The width-2 control trained for all 4000 updates without modification.

For the intervention conditions, four additional hidden ReLU neurons were added at one of five points:

- epoch 0
- epoch 5
- epoch 20
- epoch 50
- epoch 100

The original two hidden neurons were not modified.

The four new hidden neurons received fresh He-initialized weights and zero bias.

Their output-layer weights were initialized to zero so that adding the neurons did not immediately alter the network's prediction at the injection point. They then participated in normal training after injection.

Success was defined as final loss < 1e-6.

### Results

The width-2 control reproduced the previous result:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10
- mean final loss: 0.166666666667

Every capacity-injection condition succeeded on all ten seeds:

| Capacity injection | Successful runs | Mean final loss |
|---|---:|---:|
| none, width-2 control | 5/10 | 0.166666666667 |
| epoch 0 | 10/10 | 0.000000000000 |
| epoch 5 | 10/10 | 0.000000000000 |
| epoch 20 | 10/10 | 0.000000000000 |
| epoch 50 | 10/10 | 0.000000000000 |
| epoch 100 | 10/10 | 0.000000000000 |

Most importantly, the five seeds that consistently failed with width 2 in earlier experiments were all rescued by the late-capacity intervention.

This includes seeds whose width-2 trajectories had already developed poor or nearly inactive hidden representations before the additional neurons were introduced.

### Conclusion

Additional hidden capacity does not need to be present from the beginning for the current XOR task.

Even after 100 full-batch updates of the width-2 network, adding four new hidden neurons caused every tested seed to reach zero loss.

This shows that the poor width-2 trajectories are reversible rather than permanently trapped states. The failure is therefore not simply the consequence of an irreversible optimization collapse.

The result also strengthens the capacity hypothesis from Experiment 038. Extra neurons appear to provide the optimization process with additional directions in parameter space that can recover from trajectories that fail with only two hidden neurons.

The experiment does not yet establish how many additional neurons are necessary, nor whether the rescue is caused by the added representational capacity itself or by having multiple newly initialized hidden features available.

The next experiment should determine the minimum amount of additional capacity required to rescue the failed width-2 trajectories.

## Experiment 041 — Minimum Rescue Capacity

### Question

How many additional hidden neurons are required to rescue the width-2 trajectories that fail under the original configuration?

### Setup

Started every run with the same width-2 hidden ReLU network used in Experiments 033–040.

- He initialization
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- seeds 0–9
- capacity injection point: epoch 100
- additional neurons tested: 0, 1, 2, 3, and 4

At epoch 100, new hidden neurons were added without modifying the original two hidden neurons.

New neurons received fresh He initialization and zero bias.

Their output-layer weights were initialized to zero, so the network's predictions were unchanged immediately at the injection point.

Success was defined as final loss < 1e-6.

### Results

The width-2 control reproduced the previous result:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10
- mean final loss: 0.166666666667

Adding one hidden neuron at epoch 100 increased success to:

- successful seeds: 1, 2, 3, 4, 6, 8, 9
- success rate: 7/10
- mean final loss: 0.091666666667

Adding two hidden neurons increased success to:

- successful seeds: 0, 1, 2, 3, 4, 5, 6, 8, 9
- success rate: 9/10
- mean final loss: 0.025000000000

Adding three hidden neurons produced:

- successful seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- success rate: 10/10
- mean final loss: 0.000000000000

Adding four hidden neurons also produced:

- successful seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- success rate: 10/10
- mean final loss: 0.000000000000

The previously difficult seed 7 remained the only failure with two added neurons, reaching loss 0.25. Adding a third neuron rescued it.

### Results Summary

| Additional neurons at epoch 100 | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 5/10 | 0.166666666667 |
| 1 | 7/10 | 0.091666666667 |
| 2 | 9/10 | 0.025000000000 |
| 3 | 10/10 | 0.000000000000 |
| 4 | 10/10 | 0.000000000000 |

### Conclusion

The number of additional hidden neurons strongly affects the ability to rescue a failing width-2 trajectory.

At the fixed epoch-100 injection point:

- one additional neuron rescued two additional seeds
- two additional neurons rescued four additional seeds
- three additional neurons rescued all remaining failures
- a fourth additional neuron provided no further increase in success rate

This shows that the failed width-2 trajectories retain recoverable information even after 100 updates. The optimization does not require restarting from a new initialization; additional hidden capacity can supply enough new degrees of freedom for the remaining seeds to reach zero loss.

The result also provides a concrete capacity threshold for this specific experiment: adding three hidden neurons at epoch 100 was sufficient to achieve 10/10 success across seeds 0–9.

This threshold should not be interpreted as a universal requirement. It depends on the XOR task, network architecture, learning rate, initialization procedure, injection time, and deterministic initialization of the newly added neurons.

The next experiment should determine whether the rescue threshold depends on when the capacity is added, and whether the same small amount of added capacity can rescue trajectories much later in training.

## Experiment 042 — Capacity Injection Timing

### Question

Does the effectiveness of added hidden capacity depend strongly on when the capacity is introduced?

### Setup

Started every run with the same width-2 hidden ReLU network used in Experiments 040–041.

- He initialization
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- seeds 0–9
- injection epochs: 0, 25, 50, 100, 250, 500, 1000, 2000, 3000
- additional neurons tested: 1 and 2

At the selected injection epoch, new hidden neurons were added with fresh He initialization and zero bias.

Their output-layer weights were initialized to zero so that the intervention did not immediately change the network prediction at the injection point.

The width-2 network with no injection served as the control.

Success was defined as final loss < 1e-6.

### Results

The width-2 control reproduced the previous result:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10
- mean final loss: 0.166666666667

Adding one hidden neuron produced:

| Injection epoch | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 7/10 | 0.091666666667 |
| 25 | 7/10 | 0.091666666667 |
| 50 | 7/10 | 0.091666666667 |
| 100 | 7/10 | 0.091666666667 |
| 250 | 6/10 | 0.116666666667 |
| 500 | 6/10 | 0.116666666667 |
| 1000 | 6/10 | 0.116666666667 |
| 2000 | 6/10 | 0.116666666667 |
| 3000 | 6/10 | 0.116666667002 |

Adding two hidden neurons produced:

| Injection epoch | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 9/10 | 0.025000000000 |
| 25 | 9/10 | 0.025000000000 |
| 50 | 9/10 | 0.025000000000 |
| 100 | 9/10 | 0.025000000000 |
| 250 | 9/10 | 0.025000000000 |
| 500 | 9/10 | 0.025000000000 |
| 1000 | 9/10 | 0.025000000000 |
| 2000 | 9/10 | 0.025000000000 |
| 3000 | 9/10 | 0.025000010022 |

The identity of the rescued seeds was also highly stable.

With one added neuron at epochs 0–100, successful seeds were:

- 1, 2, 3, 4, 6, 8, 9

At epochs 250 and later, seed 3 was no longer rescued, reducing success to 6/10.

With two added neurons, successful seeds were consistently:

- 0, 1, 2, 3, 4, 5, 6, 8, 9

Seed 7 remained the only failure for every two-neuron injection timing.

At epoch 3000, seed 3 reached loss 0.000000100217 with two added neurons, which still satisfied the selected success threshold.

### Conclusion

Capacity injection timing has a smaller effect than the amount of added capacity.

Two additional hidden neurons rescued 9/10 seeds at every tested injection point, including very late injections at epochs 1000, 2000, and 3000. This shows that useful new capacity can remain effective even after most of the original width-2 training trajectory has already unfolded.

One additional neuron was effective through epoch 100, producing 7/10 success, but its effectiveness decreased after epoch 250 to 6/10. This indicates that a single new degree of freedom can become insufficient once the original trajectory has progressed further.

Seed 7 remained resistant to two-neuron rescue at every tested timing and continued to require the three-neuron intervention identified in Experiment 041.

These results reinforce the distinction between capacity and timing. Additional hidden units provide recoverable optimization freedom even late in training, but the amount of added capacity required can depend on the state of the trajectory.

The next experiment should investigate why seed 7 specifically requires more added capacity than the other failed seeds.

## Experiment 043 — Rescue Initialization Sensitivity

### Question

When two new hidden neurons are added to rescue a failed width-2 trajectory, is the outcome determined by the amount of capacity alone, or does the initialization of those new neurons matter?

### Setup

Used the width-2 network and late-capacity procedure from Experiment 040.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- new output weights initialized to zero
- seeds 0–9

Only the deterministic initialization of the two newly added neurons was varied.

The original network initialization and entire pre-injection trajectory remained unchanged.

Initialization offsets tested:

0, 1, 2, 3, 4, 5, 10, and 100.

Success was defined as final loss < 1e-6.

### Results

The original width-2 control is 5/10 successful.

With two neurons added at epoch 100, the success rate depended on the initialization offset:

| Initialization offset | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 9/10 | 0.025000000000 |
| 1 | 9/10 | 0.025000000000 |
| 2 | 10/10 | 0.000000000000 |
| 3 | 9/10 | 0.025000000000 |
| 4 | 9/10 | 0.025000000000 |
| 5 | 10/10 | 0.000000000000 |
| 10 | 8/10 | 0.050000000000 |
| 100 | 9/10 | 0.025000000000 |

Seed 7, which was the only seed not rescued by two added neurons across all injection timings in Experiment 042, was rescued under five of the eight initialization offsets tested here:

- offset 1
- offset 2
- offset 4
- offset 5
- offset 100

It failed under:

- offset 0
- offset 3
- offset 10

Seed 3 also changed outcome depending on initialization, succeeding under most offsets but failing under offsets 1, 4, and 10.

The best tested offsets, 2 and 5, produced 10/10 success.

### Conclusion

Two additional hidden neurons are sufficient to rescue seed 7, but the result depends on how those new neurons are initialized.

This rules out the interpretation that seed 7 fundamentally requires three additional neurons. Experiment 041 showed that three neurons guarantee rescue under the tested initialization, while Experiment 043 shows that two neurons can also rescue seed 7 under suitable initializations.

The results demonstrate that newly injected capacity creates additional optimization opportunities, but those opportunities depend on the starting location of the new neurons in parameter space.

This also helps explain why the width-6 experiment had a much higher success rate: more independently initialized hidden neurons provide more chances for useful features to emerge.

The next experiment should examine the initial activation coverage of the newly added neurons and determine whether successful rescue can be predicted from how the new neurons respond to the four XOR examples immediately after injection.

## Experiment 044 — Rescue Activation Coverage

### Question

Can the success of a two-neuron rescue be predicted from how broadly the newly added neurons activate across the XOR training examples immediately after injection?

### Setup

Used the rescue configuration from Experiment 043.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- new output weights initialized to zero
- seeds 0–9
- 8 deterministic initialization offsets for the newly added neurons

Immediately after injection, the activation pattern of each new neuron was recorded over:

- [0,0]
- [0,1]
- [1,0]
- [1,1]

The number of XOR examples activated by either new neuron was then counted as activation coverage.

Success was defined as final loss < 1e-6.

### Results

Across the 10 seeds and 8 initialization offsets:

- successful rescue cases: 73/80
- failed rescue cases: 7/80

Activation coverage among successful and failed cases was:

| Examples covered by the two new neurons | Successful | Failed |
|---|---:|---:|
| 0/4 | 1 | 0 |
| 1/4 | 3 | 1 |
| 2/4 | 29 | 5 |
| 3/4 | 40 | 1 |

Coverage therefore showed an association with rescue, particularly because most successful cases covered 3 of the 4 examples.

However, coverage was not sufficient to predict success.

Examples of this include:

- several successful cases with only 2/4 coverage
- three successful cases with 1/4 coverage
- one successful case with 0/4 coverage
- one failed case with 3/4 coverage

The 0/4 successful case occurred for a seed that was already capable of solving the task from the original width-2 trajectory, so the newly added neurons were not required for that run.

The failed cases also showed that simply covering more examples does not guarantee a successful rescue.

### Conclusion

Initial activation coverage of the newly added neurons is related to rescue reliability but is not by itself the determining factor.

The strongest concentration of successful cases occurred at 3/4 coverage, but 2/4 coverage was also frequently successful. Because the [0,0] input produces zero pre-activation for newly added neurons initialized with zero bias, the meaningful variation is primarily which of the other XOR examples the new neurons activate on and how those patterns combine.

This indicates that the structure of the activation pattern may matter more than raw coverage count.

The next experiment should therefore compare the exact activation patterns of successful and failed rescue cases and determine whether particular feature patterns are consistently associated with successful recovery.

## Experiment 045 — Rescue Activation Pattern Analysis

### Question

Does the exact activation pattern of the two newly added hidden neurons predict whether a failed width-2 trajectory will be rescued?

### Setup

Used the rescue configuration from Experiments 043–044.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- new output weights initialized to zero
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Immediately after injection, each newly added neuron was represented by a four-bit activation pattern over:

- [0,0]
- [0,1]
- [1,0]
- [1,1]

Neuron order was canonicalized, so swapping the two new neurons produced the same pattern representation.

Success was defined as final loss < 1e-6.

### Results

Across all 80 seed/initialization combinations:

- successful rescue cases: 73/80
- failed rescue cases: 7/80

The most common successful patterns included:

- 0010|0111: 15/15 successful
- 0100|0111: 10/10 successful
- 0101|0111: 10/10 successful
- 0010|0100: 10/10 successful

Other patterns were less reliable.

For example:

- 0000|0011: 6/7 successful
- 0000|0100: 3/4 successful
- 0000|0101: 11/14 successful
- 0000|0111: 2/3 successful
- 0100|0101: 2/3 successful

There was also a successful case with:

- 0000|0000: 1/1 successful

The seven failures occurred under several different patterns:

- 0000|0101: three failures
- 0000|0011: one failure
- 0000|0100: one failure
- 0100|0101: one failure
- 0000|0111: one failure

### Conclusion

The exact initial activation pattern of the two new neurons is associated with rescue reliability, but it is not sufficient to determine the outcome.

Several patterns were perfectly successful in the tested sample, while other patterns produced both successful and failed runs.

The successful 0000|0000 case is especially important: newly added neurons did not need to activate any of the four examples immediately after injection in order for the network to eventually reach zero loss.

Therefore, initial activation coverage and exact activation pattern are not complete explanations of rescue.

The results indicate that the newly added neurons can acquire useful behavior during subsequent optimization even when their initial activation is limited or absent.

The next experiment should examine the first few updates after injection and measure how quickly the newly added neurons acquire nonzero output weights, activation coverage, and useful gradients.

## Experiment 046 — Rescue Early Dynamics

### Question

What happens during the first few updates after capacity is injected, and can early new-neuron dynamics distinguish successful rescues from failed rescues?

### Setup

Used the configuration from Experiments 043–045.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

After injection, measurements were recorded at relative steps:

- 0
- 1
- 2
- 3
- 5
- 10
- 20

For the newly added neurons, the experiment measured:

- active examples per neuron
- mean absolute output weight
- mean absolute hidden gradient
- zero-gradient fraction

Success was defined as final loss < 1e-6.

### Results

Across the 80 seed/initialization combinations:

- successful rescues: 73
- failed rescues: 7

At the injection moment, new output weights were zero and consequently the newly added hidden neurons had zero hidden-layer gradient.

The successful and failed groups differed in initial activation support:

| Relative step | Successful mean active examples per new neuron | Failed mean active examples per new neuron |
|---|---:|---:|
| 0 | 1.575 | 1.071 |
| 1 | 1.575 | 1.071 |
| 2 | 2.226 | 1.571 |
| 3 | 2.055 | 1.214 |
| 5 | 2.048 | 1.214 |
| 10 | 2.089 | 1.214 |
| 20 | 2.110 | 1.143 |

The successful group also developed larger new-neuron output weights and hidden gradients during the first 20 updates.

At relative step 20:

- successful mean absolute new-neuron output weight: 0.057198
- failed mean absolute new-neuron output weight: 0.044954
- successful mean absolute new-neuron gradient: 0.011009
- failed mean absolute new-neuron gradient: 0.009282

The zero-gradient fraction showed a stronger difference:

- successful group: approximately 0.279
- failed group: approximately 0.476

### Failed Rescue Cases

The seven failed cases showed limited early support for the newly added neurons.

Examples included:

- seed 7, offset 0: activity remained approximately 2,0 across the first 20 updates
- seed 3, offset 1: activity remained approximately 0,2
- seed 7, offset 3: activity remained approximately 0,2
- seed 3, offset 4: activity remained approximately 2,0
- seed 3, offset 10: activity increased to approximately 2,2 but still failed
- seed 7, offset 10: activity remained limited to approximately 0,1–0,2
- seed 5, offset 100: activity remained approximately 3,0

These cases show that simply activating both neurons is not sufficient. For example, seed 3 with offset 10 eventually had both new neurons active on multiple examples but still failed.

### Conclusion

Successful rescue trajectories generally developed broader and stronger new-neuron participation during the first 20 updates after injection.

Compared with failed rescues, successful cases showed:

- more active examples per new neuron
- larger output weights
- larger hidden gradients
- a smaller fraction of zero hidden gradients

However, these measurements do not establish a single deterministic threshold. Some successful cases began with very limited activation, while at least one failed case reached relatively broad activation.

The results therefore identify early new-neuron participation as a useful correlate of successful rescue, but not yet a sufficient causal explanation.

The next experiment should test whether giving newly added neurons nonzero output connections at injection changes the result by allowing them to receive hidden-layer gradients immediately, while keeping their hidden initialization fixed.
