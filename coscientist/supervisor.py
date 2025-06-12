"""
Supervisor agent
---------------
- Manages a task queue, and assigns tasks to Agents.
- Assesses current progress and decides when to halt.
- Periodically computes and writes to the context memory, a
suite of statistics, including number of hypotheses generated and
those requiring review, and the progress of the tournament.
- It also summarizes the effectiveness of different Agents (e.g,
are new ideas from the Generation agent better than refined ideas
from the Evolution agent?), and gives them more work if they are
performing well.
"""
