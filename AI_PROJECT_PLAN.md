# AI From Scratch — Master Plan

## Project Vision

Build a genuinely self-owned AI system from the ground up.

The goal is not to reproduce ChatGPT, Claude, or another commercial AI. The goal is to understand the underlying technology and progressively build our own system, including its model, training process, memory, tools, and surrounding architecture.

## Core Principles

1. Learn while building.
2. Build in small, understandable pieces.
3. Understand the mathematics behind the important systems.
4. Prefer our own implementations when practical.
5. Clearly identify borrowed open-source components.
6. Design for local operation and ownership.
7. Keep private information and credentials outside the repository.
8. Document experiments, failures, decisions, and discoveries.

## GitHub / Security Rule

This project is intended for eventual public GitHub publication.

Never commit:

- passwords
- API keys
- access tokens
- private credentials
- private certificates
- secrets
- personal configuration
- private datasets

Use environment variables or excluded local configuration when secrets are ever required.

## Development Philosophy

We do not begin with a giant codebase.

We build:

understand → implement → test → break → investigate → improve → document

The first systems may be extremely small. Their purpose is understanding.

## Planned Learning Path

### Phase 0 — Foundation
Project structure, documentation, version control, security, and development environment.

### Phase 1 — First Learning System
Build a tiny model that learns from data.

### Phase 2 — Neural Network Foundations
Weights, layers, activation functions, loss, gradients, backpropagation, and optimization.

### Phase 3 — Language Foundations
Tokens, vocabulary, embeddings, sequence prediction, and language modeling.

### Phase 4 — Transformer Foundations
Attention and transformer architecture.

### Phase 5 — Our Own Language Model
Develop and train an increasingly capable model using data we can legally use and understand.

### Phase 6 — Memory
Working memory, long-term memory, retrieval, and persistence.

### Phase 7 — Reasoning and Planning
Explore architectures for multi-step reasoning and task planning.

### Phase 8 — Tools
Files, terminal, computer interaction, and other controlled capabilities.

### Phase 9 — Full System
Combine the model and surrounding systems into one coherent AI.

## Important

This plan is not a contract.

As we learn, we are allowed to change the architecture, reorder phases, remove ideas, and invent better approaches.
