# Problem Statement: LLM- and Graph Foundation Model-Based Framework for Campus Energy Management

Campus energy systems sit at the intersection of two open problems in current AI for energy research.

On one side, large language models have shown they can serve as flexible, natural-language interfaces to power system data and technical documents — answering questions, retrieving domain knowledge, and even attempting simplified optimization tasks. However, they are not physically grounded: they can approximate a small optimal power flow (OPF) problem by prompting alone, but their accuracy degrades as problems scale, and they have no principled way to enforce the physical laws (Kirchhoff's laws, power balance) that govern a real grid. On the other side, graph neural network-based foundation models for power flow (GridFMs) have been shown to solve grid physics directly and efficiently — reconstructing bus-level voltage, angle, and power injections at 3–4 orders of magnitude lower computational cost than conventional solvers. However, this line of work has so far targeted large-scale transmission systems using public benchmark datasets, not the building- and feeder-scale distribution networks that campuses actually operate, and it offers no interface a non-expert campus stakeholder (a facilities manager, a student, an administrator) could use directly.

## The Gap

There is currently no framework that combines a **physically grounded, computationally efficient model of a real distribution-scale network** with an **accessible natural-language interface**, applied to a setting where the underlying network topology and data are limited, partially proprietary, and much smaller in scale than the transmission-grid benchmarks the foundation model literature has focused on.

## Problem Statement

This project addresses the problem of **building and evaluating a hybrid AI pipeline for campus energy management**, in which a graph neural network — pretrained in a self-supervised manner to reconstruct masked power-flow states, following the grid foundation model (GridFM) paradigm — is fine-tuned on the campus's own network topology and energy data to provide fast, physically consistent estimates of energy usage and grid state. This model is then made accessible through a retrieval-augmented LLM interface that translates natural-language questions and optimization requests into queries against the model, rather than asking the LLM to reason about grid physics directly.

**Central research question:** Can a foundation-model approach designed for large transmission grids be adapted to work at a much smaller scale, with much less data, on a real (not benchmark) network — and does doing so produce a genuinely more capable and more trustworthy energy assistant than a generic LLM+RAG system alone?

## Objectives

1. Determine whether a GNN pretrained via masked feature reconstruction on public benchmark grids (IEEE test cases, PowerGraph) transfers usefully to a much smaller campus-scale network topology, and how much campus-specific data is needed to fine-tune it effectively.
2. Build the data and graph pipeline needed to represent the campus's electrical network and energy consumption data in the node/edge format required by the model.
3. Design and implement a retrieval-augmented LLM layer that answers natural-language questions about campus energy usage by querying the trained GNN's outputs (and campus documents/metering data), rather than attempting numerical reasoning itself.
4. Evaluate the combined system on:
   - (a) prediction accuracy against ground-truth power flow or metering data,
   - (b) computational speed versus a conventional solver, and
   - (c) usability and correctness of the natural-language answers it produces. 

## Significance

Most existing GridFM work is a "moonshot" proposal validated on public transmission benchmarks; this project would be one of the first attempts to test that vision at the scale and data-scarcity conditions a real institution (rather than a national grid operator) actually faces. It also directly addresses a limitation noted in the PES-GM paper's own discussion — that LLMs alone lack domain-specific knowledge and struggle beyond toy-scale OPF — by giving the LLM a physics-grounded model to defer to, instead of asking it to compute grid physics itself.

--
