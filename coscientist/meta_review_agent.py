"""
Meta review agent
-----------------
- Formulates a research overview with memory
- Feedback from this agent is appended to the prompts of the
others in subsequent rounds.

More details:
- Takes in the tournament state with all debates and ELO ratings,
summarizes common patterns in the reviews and debates to synthesize
the meta-review feedback.
- Feedback helps to steer the Reflection agent so that it accounts
for common reasoning failures.
- Writes top hypotheses into a research overview that highlights
areas to follow up with real and specific experiments. This
gets fed to the Generation agent in later rounds. Format of the
overview can match the style of a review paper or a grant proposal
(like an NIH Specific Aims Page).
- Creates a list of research contacts from authors of papers
that are relevant to the research plan.
"""

REVIEW_GENERATION_PROMPT = """
You are an expert in scientific research and meta-analysis.
Synthesize a comprehensive meta-review of provided reviews
pertaining to the following research goal:

Goal: {goal}

Preferences:
{preferences}

Additional instructions:
{instructions}

Provided reviews for meta-analysis:
{reviews}

Instructions:
* Generate a structured meta-analysis report of the provided reviews.
* Focus on identifying recurring critique points and common issues raised by reviewers.
* The generated meta-analysis should provide actionable insights for researchers
developing future proposals.
* Refrain from evaluating individual proposals or reviews;
focus on producing a synthesized meta-analysis.

Response:
"""
