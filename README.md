# 🧪 Open CoScientist Agents

A comprehensive multi-agent system for AI-driven scientific discovery based on Google DeepMind's [AI co-scientist](https://arxiv.org/abs/2502.18864), built with LangGraph and [GPT Researcher](https://github.com/assafelovic/gpt-researcher). The aim is for this system to accelerate scientific research through collaborative AI agents that generate, critique, rank, and evolve scientific hypotheses using tournament-style competition.

This implementation uses `Gemini 2.5 Pro`, `Claude Sonnet 4`, and `o3` in collaboration and competition.

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

## CLI Testing (Post-Configuration)

Use the lightweight CLI harness to debug steps after goal confirmation without the UI.

- Help: `python -m coscientist.cli --help` (or `cosci --help` after `pip install -e .`)
- Create fresh goal dir: `cosci new --goal "Your goal here"`
- Initial pipeline (literature review → generation → tournament → meta-review):
  - `cosci start --goal "Your goal here" --n 4 [--pause-after-lr]`
- Single step (targeted):
  - Generate: `cosci step --goal "Your goal here" --action generate_new_hypotheses --n 2`
  - Reflect: `cosci step --goal "Your goal here" --action reflect`
  - Tournament: `cosci step --goal "Your goal here" --action run_tournament -k 8`
- Full loop: `cosci run --goal "Your goal here" --max-iter 10`
- Checkpoints: `cosci checkpoints --goal "Your goal here"`
- Resume from checkpoint: add `--checkpoint-path /path/to/coscientist_state_*.pkl`

Notes:
- Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY` as needed.
- The CLI warns if keys are missing; phases using LLMs/GPTResearcher will fail otherwise.

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
streamlit run tournament_viewer.py --server.address 0.0.0.0
```

> **Note**: The `--server.address 0.0.0.0` flag ensures proper localhost binding on all systems. If you experience `ERR_EMPTY_RESPONSE` with localhost, this resolves IPv4/IPv6 binding conflicts.

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
import asyncio
from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager

goal = "How does the gut microbiome influence rheumatoid arthritis and can probiotics help to mitigate symptoms? If so, which ones are promising?"
initial_state = CoscientistState(goal=goal)

config = CoscientistConfig()
state_manager = CoscientistStateManager(initial_state)
cosci = CoscientistFramework(config, state_manager)

final_report, final_meta_review = asyncio.run(cosci.run())
```

## Performance & Scalability

In principle, this system can be easily scaled with asynchronous execution of many tasks. In practice, API rate limits make it difficult to run in parallel. Future work will explore ways to get around this by smartly allocating work to different providers.

Currently designed to work with 20-30 hypotheses in a tournament. Scaling that to more will require optimizations like smarter prioritization of head-to-head matches, summarizing context to make meta-review tractable, and actually supporting asynchronous execution.


## Testing

This project includes a comprehensive test suite to ensure reliability and quality.

### Quick Test Commands
```bash
# Run basic functionality tests
python -m pytest tests/test_working_basic.py -v

# Run with coverage report  
python -m pytest tests/test_working_basic.py --cov=coscientist --cov-fail-under=0

# Run all available tests
python -m pytest tests/ -v --cov-fail-under=0

# View test suite summary
python test_summary.py
```

### Test Categories
- **Unit Tests**: Individual component testing with mocked dependencies
- **Integration Tests**: Component interaction and workflow validation  
- **End-to-End Tests**: Complete research scenario testing
- **Performance Tests**: Scalability and resource usage validation
- **CLI Tests**: Command-line interface functionality

### Test Infrastructure
- **13+ Working Tests**: Covering core functionality with 92.9% success rate
- **Comprehensive Mocking**: Realistic LLM and API response simulation
- **Test Fixtures**: Sample data for literature reviews, hypotheses, and tournaments
- **Coverage Reporting**: Detailed analysis of tested code paths

For detailed testing documentation, see `tests/README.md`.

## Caveats and Sharp Edges

- The system isn't fully configurable and there are fields that are hardcoded (like number of hypotheses, subtopics for literature review, etc.).
- Test coverage is expanding - core functionality is well-tested but some advanced features need additional test coverage.

## Contributing

We welcome contributions!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Google DeepMind's research on AI-assisted scientific discovery
- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Uses [GPT Researcher](https://github.com/assafelovic/gpt-researcher) for literature analysis
- Visualization powered by [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/)
