# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open CoScientist Agents is a multi-agent system for AI-driven scientific discovery. It orchestrates multiple AI agents that collaborate and compete to generate, critique, rank, and evolve scientific hypotheses using a tournament-style approach. The system is built on LangGraph and integrates GPT Researcher for literature analysis.

## Development Commands

### Environment Setup
```bash
# Install package in development mode
pip install -e .

# Install with development dependencies  
pip install -e .[dev]

# Set up environment variables (required)
# Copy .env.example to .env and fill in API keys:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY  
# - GOOGLE_API_KEY
# - TAVILY_API_KEY
# - LANGSMITH_API_KEY (optional, for monitoring)
```

### Start Test Run in Viewer
- Open the Streamlit viewer and use the "🧪 Start Test Run" button at the top.
- It uses the last confirmed refined goal (CQ) saved at `~/.coscientist/last_confirmed_goal.txt`.
- The panel shows PID, recent log tail, status (running/done/error), and a Stop control.
- The last confirmed goal is saved automatically when the Configuration Agent completes; selecting a goal in the sidebar also works as fallback.

### CLI Testing (Headless, Post-Configuration)

For granular, headless debugging of phases after goal confirmation, use the CLI harness:

```bash
# Help
python -m coscientist.cli --help   # or `cosci --help` after pip install -e .

# Create fresh state dir for a goal
cosci new --goal "Your goal here"

# Initial pipeline (LR → gen → tournament → meta-review)
cosci start --goal "Your goal here" --n 4 --pause-after-lr  # pause after LR to inspect

# Single step (targeted)
cosci step --goal "Your goal here" --action generate_new_hypotheses --n 2
cosci step --goal "Your goal here" --action reflect
cosci step --goal "Your goal here" --action run_tournament -k 8

# Full supervisor loop with cap
cosci run --goal "Your goal here" --max-iter 10

# List and resume from checkpoints
cosci checkpoints --goal "Your goal here"
cosci start --goal "Your goal here" --checkpoint-path /path/to/coscientist_state_*.pkl
```

Notes:
- Set provider/search keys as needed: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`.
- The `reflect` action processes the verification queue and updates proximity graph.

### Running the Application
```bash
# Launch Streamlit web interface
cd app
pip install -r viewer_requirements.txt
streamlit run tournament_viewer.py --server.address 0.0.0.0

# NOTE: --server.address 0.0.0.0 prevents localhost ERR_EMPTY_RESPONSE issues
# by ensuring proper IPv4 binding on all interfaces

# Run a research session programmatically
python -c "
import asyncio
from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager

goal = 'Your research question here'
state = CoscientistState(goal=goal)
config = CoscientistConfig()
state_manager = CoscientistStateManager(state)
framework = CoscientistFramework(config, state_manager)

final_report, meta_review = asyncio.run(framework.run())
"
```

### Testing
```bash
# Run tests from project root
python tests/test_claude_opus_integration.py

# Run individual agent tests
python -m pytest tests/ -v

# Test specific functionality
python -c "from coscientist.framework import _SMARTER_LLM_POOL; print(list(_SMARTER_LLM_POOL.keys()))"
```

### Development Tools
```bash
# Code formatting (if black is installed)
black coscientist/ app/

# Type checking (if mypy is installed)  
mypy coscientist/

# Linting (if ruff is installed)
ruff check coscientist/ app/
```

## Architecture

### Core Components

**Framework (`coscientist/framework.py`)**
- Central orchestrator that manages the multi-agent system
- Configures LLM pools: `_SMARTER_LLM_POOL` for reasoning-heavy tasks, `_CHEAPER_LLM_POOL` for lightweight operations
- Supports multiple AI providers: OpenAI (GPT-5, o3, o4-mini), Anthropic (Claude Opus 4.1, Sonnet 4), Google (Gemini 2.5 Pro/Flash)
- Custom OpenAI client wrapper at `coscientist/openai_client.py` provides access to new responses.create() API with reasoning capabilities

**State Management (`coscientist/global_state.py`)**
- Persistent state across research sessions using `CoscientistState` and `CoscientistStateManager`
- Auto-saves progress with configurable intervals
- State files stored in `~/.coscientist/` by default (configurable via `COSCIENTIST_DIR` env var)

**Agent Architecture**
Each agent is implemented as a LangGraph node with specific responsibilities:
- **Literature Review Agent**: Decomposes research goals, conducts comprehensive literature analysis
- **Generation Agents**: Create hypotheses using independent and collaborative reasoning approaches  
- **Reflection Agents**: Perform verification, causal reasoning, assumption analysis
- **Evolution Agents**: Refine hypotheses based on feedback and tournament results
- **Supervisor Agent**: Makes workflow decisions, determines when research is complete
- **Ranking Agent**: Implements ELO tournament system for hypothesis competition
- **Meta-Review Agent**: Synthesizes insights across research directions
- **Final Report Agent**: Generates comprehensive research summaries

### Key Data Structures

**Custom Types (`coscientist/custom_types.py`)**
- `ParsedHypothesis`: Core hypothesis representation with metadata
- `ReviewedHypothesis`: Extended hypothesis with review feedback
- Agent-specific state classes for maintaining context across operations

**Tournament System (`coscientist/ranking_agent.py`)**
- ELO rating system for hypothesis ranking
- Head-to-head comparison with debate transcripts
- Win-loss statistics and performance tracking

**Proximity Analysis (`coscientist/proximity_agent.py`)**
- Semantic similarity analysis using OpenAI embeddings
- NetworkX-based graph construction for hypothesis relationships
- Community detection using Louvain method

### Configuration

**Model Configuration**
- Primary models configured in `_SMARTER_LLM_POOL` and `_CHEAPER_LLM_POOL` dictionaries
- Research-specific settings in `coscientist/researcher_config.json`
- Default assignments: Claude Opus 4.1 for critical agents, Gemini Flash for meta-review

**Prompt Templates**
- Modular prompts stored in `coscientist/prompts/` directory
- Each agent type has dedicated prompt files for consistent behavior
- Templates support scientific reasoning patterns and evaluation criteria

### Web Interface (`app/`)

**Streamlit Dashboard (`app/tournament_viewer.py`)**
- Modular page architecture with dedicated files for each view
- Real-time monitoring of research progress
- Interactive visualization of tournament results and hypothesis relationships
- Quick debug: "Start Test Run" button to launch agents using the last confirmed CQ (refined goal)

**Background Processing (`app/background.py`)**  
- Handles long-running research sessions
- Process management for concurrent research workflows

## Important Implementation Notes

### Environment Variables
The system requires `.env` file with API keys. The `load_dotenv()` call in `coscientist/__init__.py` ensures environment variables are loaded before any LLM clients are initialized.

### LLM Provider Integration
- OpenAI: Uses both standard LangChain client and custom `openai_client.py` for GPT-5 responses.create() API
- Anthropic: Standard LangChain integration for Claude models
- Google: LangChain integration for Gemini models
- Configured with retry logic and explicit timeouts/loop breakers where applicable

### State Persistence
Research sessions automatically save state to disk. State files include:
- Hypothesis generation history
- Tournament results and rankings  
- Literature review findings
- Agent decision logs

### Scalability Considerations
- System designed for 20-30 hypotheses in tournament mode
- API rate limiting affects parallelization
- Asynchronous execution planned but currently limited by provider constraints

### Research Configuration
Default parameters in `researcher_config.json` control:
- Model assignments (FAST_LLM, SMART_LLM, STRATEGIC_LLM)
- Token limits and search parameters
- Research depth and breadth settings
- Output formatting preferences

## Recent Stability Updates

### Loop Breakers and Timeouts
- Supervisor loop safeguards:
  - Max iteration cap (default 50) via `CoscientistConfig(max_supervisor_iterations=...)`.
  - Repeated-action detector via env `COSCI_REPEAT_ACTION_LIMIT` (default 6) to abort if the same decision repeats.
- GPT Researcher calls (Tavily/web) use explicit async timeouts and lightweight retries:
  - `COSCI_RESEARCH_TIMEOUT_SECONDS` (default 420)
  - `COSCI_WRITE_TIMEOUT_SECONDS` (default 240)
  - `COSCI_RESEARCH_MAX_RETRIES` (default 0)
- On timeout/error, agents return placeholder text so the pipeline continues instead of hanging.

### Embedding Pipeline Limits
- To avoid OpenAI Embedding API token overflows:
  - `EMBEDDING_CHUNK_SIZE` set to 800 and `EMBEDDING_KWARGS.chunk_size=800` in `coscientist/researcher_config.json`.
  - Do not pass `batch_size` to `OpenAIEmbeddings` (some SDK versions reject it). Leave batching to the library defaults.
- Proximity embeddings use `text-embedding-3-small` with 256 dimensions.

### Test Run Button
- Viewer adds a "Start Test Run" button (top of page) for fast debugging with the last confirmed refined goal (CQ).
- The last goal is persisted at `~/.coscientist/last_confirmed_goal.txt` by `app/common.py` when Configuration completes.

## Quick Troubleshooting
- Error: `Embeddings.create() got an unexpected keyword argument 'batch_size'`
  - Remove `batch_size` from `OpenAIEmbeddings(...)` and from `EMBEDDING_KWARGS`.
- Error: `Requested N tokens, max 300000 tokens per request`
  - Lower `EMBEDDING_CHUNK_SIZE` to 512 (or less); avoid sending too many large chunks in one call.
- Viewer run hangs on Tavily/web fetch
  - Use the timeout envs above; check log tail in the viewer; set `COSCI_REPEAT_ACTION_LIMIT` smaller to break sooner.

### Order
- Use `test/` folder for test codes.
