# Engineering Journal

## Entry 001 — Project Founded

**Date:** 2026-09-26

### Decision

Begin a long-term project to build a self-owned AI from the ground up.

### Goals

- Learn the technology instead of treating it as a black box.
- Eventually train our own model.
- Build the surrounding AI system ourselves.
- Avoid mandatory subscriptions and unnecessary dependence on commercial AI APIs.
- Design the repository so it can eventually be published publicly on GitHub.

### Security Decision

No passwords, API keys, tokens, credentials, or other secrets belong in the repository.

### First Technical Principle

Our first AI does not need to be powerful.

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

We want to see the learning process directly instead of hiding it behind a library.

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

### Important Understanding

The model does not contain the rule "multiply by 2."

It contains a learned parameter that moved toward a value that minimizes its error on the examples.

### Next Direction

Separate training from evaluation and test the learner on inputs it has never seen.
