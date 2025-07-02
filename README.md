# 🧪 Open CoScientist Agents

A comprehensive multi-agent system for AI-driven scientific discovery based on Google DeepMind's [AI co-scientist](https://arxiv.org/abs/2502.18864), built with LangGraph and [GPT Researcher](https://github.com/assafelovic/gpt-researcher). The aim is for this system to accelerate scientific research through collaborative AI agents that generate, critique, rank, and evolve scientific hypotheses using tournament-style competition.

This implementation makes use of all best reasoning models from frontier labs: `Gemini 2.5 Pro`, `Claude Sonnet 4`, and `o3`.

![App Demo](assets/app_demo.gif)

## Key Features

### Multi-Agent Architecture
- **Literature Review Agent**: Systematically decomposes research goals and conducts comprehensive literature analysis
- **Generation Agents**: Create novel scientific hypotheses using multiple reasoning approaches
- **Reflection Agents**: Perform deep verification and causal reasoning analysis
- **Evolution Agents**: Refine and improve hypotheses based on feedback and competition
- **Meta-Review Agent**: Synthesizes insights across multiple research directions
- **Supervisor Agent**: Orchestrates the entire research workflow -- decides which actions to take next and when to finish the research.
- **Final Report Agent**: Generates comprehensive research summaries

### Tournament-Style Hypothesis Competition
- **ELO Rating System**: Ranks hypotheses through head-to-head competitive analysis
- **Debate Transcripts**: Full records of why one hypothesis outperforms another
- **Win-Loss Statistics**: Track performance across multiple evaluation rounds
- **Hypothesis Evolution**: See how ideas improve through iterative refinement

### Interactive Web Interface
- **Streamlit Dashboard**: Comprehensive visualization of research results
- **Real-time Monitoring**: Track research progress and agent activities
- **Hypothesis Explorer**: Deep dive into individual hypotheses and their reasoning
- **Tournament Viewer**: Analyze competitive dynamics between ideas

## Installation

### Prerequisites
- Python 3.12 or higher
- A boatload of API keys

### Install from PyPI (Coming Soon)
```bash
pip install open-coscientist-agents
```

### Install from Source
```bash
git clone https://github.com/conradry/open-coscientist-agents.git
cd open-coscientist-agents
pip install -e .
```

## Configuration

### Environment Variables
Set up your API keys for model providers:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

Set up your API key for Tavily search:
```bash
export TAVILY_API_KEY='your-api-key'
```

Optional, but highly recommended for monitoring and debugging, set up API keys for LangSmith:
```bash
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="your-langsmith-project"
```

### Web Interface
Launch the interactive dashboard:
```bash
cd app
pip install -r viewer_requirements.txt
streamlit run tournament_viewer.py
```

Features include:
- **Configuration Agent**: Set up research parameters
- **Literature Review**: Explore research foundation
- **Tournament Rankings**: View hypothesis competition results
- **Proximity Graph**: Semantic relationship visualization
- **Meta-Reviews**: Synthesized research insights
- **Supervisor Decisions**: Workflow orchestration logs
- **Final Report**: Comprehensive research summary

### Start a research run in Python
```python
from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager

goal = "How does the gut microbiome influence rheumatoid arthritis and can probiotics help to mitigate symptoms? If so, which ones are promising?"
initial_state = CoscientistState(goal=goal)

config = CoscientistConfig()
state_manager = CoscientistStateManager(initial_state)
cosci = CoscientistFramework(config, state_manager)

final_report, final_meta_review = await cosci.run()
```

## Performance & Scalability

In principle, this system can be easily scaled with asynchronous execution of many tasks. In practice, API rate limits make it difficult to run in parallel. Future work will explore ways to get around this by smartly allocating work to different providers.

Currently designed to work with 20-30 hypotheses in a tournament. Scaling that to more will require optimizations like smarter prioritization of head-to-head matches, summarizing context to make meta-review tractable, and actually supporting asynchronous execution.


## Caveats and sharp edges

- The system isn't fully configurable and there are fields that are hardcoded (like number of hypotheses, subtopics for literature review, etc.).
- Obviously no tests or evaluations yet. Getting feedback will help to steer this project in the right direction for research usefulness.

## Contributing

We welcome contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Google DeepMind's research on AI-assisted scientific discovery
- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Uses [GPT Researcher](https://github.com/assafelovic/gpt-researcher) for literature analysis
- Visualization powered by [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/)
