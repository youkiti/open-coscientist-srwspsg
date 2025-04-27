"""
Generation agent
---------------
- Literature exploration
- Simulated scientific debates

More details:
- Searches the web for papers, reads the articles, and summarizes prior
work. Using these summaries, it generates hypotheses.
- Runs scientific debates for self-critique and self-play. These
Socratic dialogues go into refining the hypotheses.
- Each hypothesis comes with a set of testable assumptions and sub-assumptions
This looks something like multihop reasoning with MCTS.
- New hypotheses can be generated in later steps conditioned on the past hypotheses
and feedback from the meta-review agent.

"""

GENERATION_PROMPT = """
You are an expert tasked with formulating a novel and robust hypothesis to address
the following objective.

Describe the proposed hypothesis in detail, including specific entities, mechanisms,
and anticipated outcomes.

This description is intended for an audience of domain experts.

You have conducted a thorough review of relevant literature and developed a logical framework
for addressing the objective. The articles consulted, along with your analytical reasoning,
are provided below.

Goal: {goal}

Criteria for a strong hypothesis:
{preferences}

Existing hypothesis (if applicable):
{source_hypothesis}

{instructions}

Literature review and analytical rationale (chronologically ordered, beginning
with the most recent analysis):

{articles_with_reasoning}

Proposed hypothesis (detailed description for domain experts):
"""

DEBATE_PROMPT = """
You are an expert participating in a collaborative discourse concerning the generation
of a {idea_attributes} hypothesis. You will engage in a simulated discussion with other experts.
The overarching objective of this discourse is to collaboratively develop a novel
and robust {idea_attributes} hypothesis.

Goal: {goal}

Criteria for a high-quality hypothesis:
{preferences}

Instructions:
{instructions}

Review Overview:
{reviews_overview}

Procedure:

Initial contribution (if initiating the discussion):
Propose three distinct {idea_attributes} hypotheses.

Subsequent contributions (continuing the discussion):
* Pose clarifying questions if ambiguities or uncertainties arise.
* Critically evaluate the hypotheses proposed thus far, addressing the following aspects:
- Adherence to {idea_attributes} criteria.
- Utility and practicality.
- Level of detail and specificity.
* Identify any weaknesses or potential limitations.
* Propose concrete improvements and refinements to address identified weaknesses.
* Conclude your response with a refined iteration of the hypothesis.

General guidelines:
* Exhibit boldness and creativity in your contributions.
* Maintain a helpful and collaborative approach.
* Prioritize the generation of a high-quality {idea_attributes} hypothesis.

Termination condition:
When sufficient discussion has transpired (typically 3-5 conversational turns,
with a maximum of 10 turns) and all relevant questions and points have been
thoroughly addressed and clarified, conclude the process by writing "HYPOTHESIS"
(in all capital letters) followed by a concise and self-contained exposition of the finalized idea.

#BEGIN TRANSCRIPT#
{transcript}
#END TRANSCRIPT#

Your Turn:
"""
