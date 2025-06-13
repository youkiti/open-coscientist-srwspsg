# AI Co-Scientist: Multi-Agent Scientific Discovery System

A comprehensive implementation of the AI co-scientist concept from Google DeepMind's research, built using LangGraph for multi-agent orchestration. This system automates and accelerates scientific discovery through collaborative AI agents that generate, critique, rank, and evolve scientific hypotheses.

## 🧬 System Overview

The AI co-scientist employs a "generate, debate, and evolve" approach with specialized agents working together:

- **Generation Agents**: Create novel scientific hypotheses through literature synthesis and simulated debates
- **Reflection Agents**: Critically review hypotheses for correctness, novelty, and feasibility  
- **Ranking Agents**: Run ELO-based tournaments to identify the most promising ideas
- **Evolution Agents**: Refine and improve top-ranked hypotheses
- **Proximity Agents**: Analyze hypothesis similarity and cluster related ideas
- **Meta-review Agents**: Synthesize findings into comprehensive research overviews
- **Supervisor Agent**: Orchestrates the entire multi-agent research process