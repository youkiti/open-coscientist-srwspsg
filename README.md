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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd open-coscientist-agents

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import asyncio
from coscientist.framework import CoScientistFramework
from coscientist.configuration import goal_to_configuration
from langchain_openai import ChatOpenAI

# Initialize with your LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
framework = CoScientistFramework(llm)

# Define your research goal
research_goal = """
Develop a novel hypothesis for treating Alzheimer's disease 
by targeting protein aggregation mechanisms.
"""

# Generate research configuration
research_config = goal_to_configuration(llm, research_goal)

# Run the AI co-scientist system
async def run_research():
    results = await framework.run_framework(
        goal=research_goal,
        research_plan_config=research_config,
        max_iterations=10
    )
    return results

# Execute the research process
results = asyncio.run(run_research())
print(results["research_overview"])
```

### Demo Script

Run the included demo to see the system in action:

```bash
python examples/ai_coscientist_demo.py
```

## 🏗️ Architecture

### Multi-Agent Components

#### 1. Generation Agent (`coscientist/generation_agent.py`)
- **Independent Generation**: Single-agent hypothesis creation
- **Collaborative Generation**: Multi-agent scientific debates
- **Literature Integration**: Synthesizes existing research
- **Reasoning Types**: Supports deductive, inductive, and abductive reasoning

#### 2. Reflection Agent (`coscientist/reflection_agent.py`)  
- **Desk Rejection**: Initial quality filtering
- **Deep Verification**: Comprehensive hypothesis analysis
- **Observation Review**: Checks explanatory power against known data
- **Failure Scenarios**: Identifies potential weaknesses

#### 3. Ranking Agent (`coscientist/ranking_agent.py`)
- **ELO Tournament System**: Pairwise hypothesis comparisons
- **Round Robin Stage**: All hypotheses compete against each other
- **Bracket Stage**: Top hypotheses in elimination tournament
- **Dynamic Rating Updates**: Evolving quality assessments

#### 4. Evolution Agent (`coscientist/evolution_agent.py`)
- **Feasibility Refinement**: Improves practical implementability  
- **Inspiration Generation**: Creates new ideas from top hypotheses
- **Grounding Enhancement**: Strengthens hypotheses with literature
- **Constraint Satisfaction**: Adapts ideas to meet requirements

#### 5. Proximity Agent (`coscientist/proximity_agent.py`)
- **Semantic Embeddings**: Vector representations of hypotheses
- **Similarity Analysis**: Identifies related ideas
- **Community Detection**: Clusters similar hypotheses  
- **De-duplication**: Removes redundant ideas

#### 6. Meta-review Agent (`coscientist/meta_review_agent.py`)
- **Pattern Identification**: Finds recurring themes in reviews
- **Agent Optimization**: Suggests improvements for other agents
- **Research Synthesis**: Creates comprehensive research overviews
- **Strategic Planning**: Identifies next steps and collaborations

#### 7. Supervisor Agent (`coscientist/supervisor.py`)
- **Task Queue Management**: Orchestrates agent activities
- **Progress Assessment**: Monitors research advancement
- **Resource Allocation**: Optimizes computational resources
- **Termination Decisions**: Determines when objectives are met

### Supporting Infrastructure

#### Asynchronous Framework (`coscientist/framework.py`)
- **Concurrent Task Execution**: Parallel agent processing
- **Task Priority Management**: Intelligent resource allocation
- **State Synchronization**: Coordinated multi-agent state
- **Error Handling**: Robust failure recovery

#### Context Memory (`coscientist/context_memory.py`)
- **Persistent Storage**: SQLite-based data persistence
- **Session Management**: Track research sessions over time
- **Performance Metrics**: Agent effectiveness monitoring
- **State Recovery**: Resume interrupted research processes

#### External Tools (`coscientist/tools/`)
- **Web Search**: Literature discovery and fact-checking
- **Database Queries**: Structured data retrieval
- **API Integrations**: External service connections

## 📊 Features

### Core Capabilities
- ✅ **Multi-agent Collaboration**: Specialized agents work together seamlessly
- ✅ **Asynchronous Processing**: Efficient resource utilization
- ✅ **Persistent Memory**: Research state maintained across sessions
- ✅ **Literature Integration**: Automated literature review and synthesis
- ✅ **Hypothesis Evolution**: Iterative improvement of ideas
- ✅ **Quality Assessment**: Comprehensive evaluation and ranking
- ✅ **Research Planning**: Strategic next steps and collaboration suggestions

### LangGraph Integration
- **State Management**: TypedDict-based state definitions
- **Workflow Orchestration**: Graph-based agent coordination  
- **Conditional Routing**: Dynamic workflow adaptation
- **Node Isolation**: Independent agent processing
- **Error Recovery**: Graceful failure handling

### Extensibility
- **Modular Design**: Easy addition of new agent types
- **Configurable Parameters**: Customizable research preferences
- **Tool Integration**: Simple external tool addition
- **Prompt Engineering**: Easily modifiable agent prompts

## 🔧 Configuration

### Research Configuration
```python
from coscientist.custom_types import ResearchPlanConfig

config = ResearchPlanConfig(
    preferences="Focus on novel, testable hypotheses with clinical relevance",
    attributes=["Novelty", "Feasibility", "Impact", "Testability"],
    constraints=["Must be ethically sound", "Should be cost-effective"]
)
```

### Agent Customization
```python
from coscientist.reasoning_types import ReasoningType

# Configure generation agents
generation_agents = {
    "biologist": {
        "field": "molecular_biology", 
        "reasoning": ReasoningType.DEDUCTIVE
    },
    "chemist": {
        "field": "medicinal_chemistry",
        "reasoning": ReasoningType.INDUCTIVE  
    }
}
```

### Tournament Settings
```python
from coscientist.ranking_agent import EloTournament

tournament = EloTournament(
    llm=llm,
    goal=research_goal,
    preferences=config.preferences,
    notes="Focus on therapeutic potential",
    idea_attributes=config.attributes
)
```

## 📁 Project Structure

```
coscientist/
├── agents/
│   ├── generation_agent.py      # Hypothesis generation
│   ├── reflection_agent.py      # Critical review  
│   ├── ranking_agent.py         # ELO tournaments
│   ├── evolution_agent.py       # Hypothesis refinement
│   ├── proximity_agent.py       # Similarity analysis
│   ├── meta_review_agent.py     # Research synthesis
│   └── supervisor.py            # Process orchestration
├── framework.py                 # Asynchronous task system
├── context_memory.py           # Persistent state management
├── configuration.py            # Research planning
├── custom_types.py             # Data models
├── reasoning_types.py          # Reasoning frameworks
├── common.py                   # Shared utilities
├── prompts/                    # LLM prompt templates
│   ├── independent_generation.md
│   ├── collaborative_generation.md
│   ├── desk_reject.md
│   ├── deep_verification.md
│   └── observation_reflection.md
└── tools/                      # External integrations
    ├── search.py               # Literature search
    └── query_database.py       # Database queries
```

## 🧪 Example Use Cases

### 1. Drug Discovery
```python
research_goal = """
Identify novel drug targets for treating antibiotic-resistant 
bacterial infections by exploiting unique metabolic pathways.
"""
```

### 2. Climate Science  
```python
research_goal = """
Develop innovative approaches to carbon capture that could be 
implemented at industrial scale within the next decade.
"""
```

### 3. Neuroscience
```python
research_goal = """
Propose mechanisms for memory consolidation during sleep that 
could lead to treatments for memory disorders.
"""
```

## 🔬 Research Methodology

The system implements a rigorous scientific approach:

1. **Hypothesis Generation**: Creates diverse, novel scientific hypotheses
2. **Peer Review Simulation**: Multiple agents critically evaluate ideas  
3. **Competitive Ranking**: Tournament-style selection of best hypotheses
4. **Iterative Refinement**: Continuous improvement through evolution
5. **Meta-Analysis**: Synthesis of findings into actionable research plans
6. **Reproducibility**: All decisions and rationales are logged and traceable

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Adding new agent types
- Improving existing algorithms  
- Integrating external tools
- Enhancing prompt engineering
- Adding new reasoning frameworks

## 📚 References

Based on the "Towards an AI co-scientist" research from Google DeepMind and implements the multi-agent scientific discovery framework using modern LLM and graph-based orchestration techniques.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to accelerate scientific discovery? Start exploring with the AI co-scientist system!** 🚀
