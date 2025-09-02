Cosci CLI
=========

A lightweight CLI harness to exercise “post-goal” research steps from the terminal. It is useful to reproduce issues that occur after goal confirmation without going through the UI.

Install (optional)
------------------

- Use directly via module: `python -m coscientist.cli --help`
- Or install entry point locally: `pip install -e .` then use `cosci` command

Common Commands
---------------

- List goals discovered under `~/.coscientist`:
  - `python -m coscientist.cli goals`
- Create a fresh directory for a goal:
  - `python -m coscientist.cli new --goal "Your goal here"`
- Run initial pipeline (literature review → generation → tournament → meta-review):
  - `python -m coscientist.cli start --goal "Your goal here" --n 4`
  - Add `--pause-after-lr` to stop after literature review for debugging
- Run a single action for targeted testing:
  - `python -m coscientist.cli step --goal "Your goal here" --action generate_new_hypotheses --n 2`
  - Supported actions: `generate_new_hypotheses`, `reflect`, `evolve_hypotheses`, `expand_literature_review`, `run_tournament`, `run_meta_review`, `finish`
- Full supervisor-controlled loop:
  - `python -m coscientist.cli run --goal "Your goal here" --max-iter 10`
- Checkpoints for a goal:
  - `python -m coscientist.cli checkpoints --goal "Your goal here"`
- Resume from a specific checkpoint:
  - add `--checkpoint-path /path/to/coscientist_state_*.pkl` to `start`, `step`, or `run`
- Clear goal directory:
  - `python -m coscientist.cli clear --goal "Your goal here"`

Environment
-----------

- Set required API keys in environment or `.env` in repo root:
  - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`
- The CLI prints a warning if keys are missing; phases that invoke LLMs or GPTResearcher will then fail.

Tips
----

- For faster smoke tests: reduce `--n`, enable `--pause-after-lr`, or test with `step` subcommand.
- Check logs under `log/` for detailed progress and errors.
