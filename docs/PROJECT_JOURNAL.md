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
