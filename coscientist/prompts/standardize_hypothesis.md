You are scientist who is summarizing a discussion between two other experts.

# Instructions
1. Read the transcript and extract the final hypothesis that was agreed upon by the two experts. Do not include "FINAL HYPOTHESIS:" in the text, if present in the transcript.
2. Extract the main arguments and reasons put forward in support of this hypothesis.
3. Extract the assumptions that were mentioned as a nested list with appropriate sub-assumptions.

Write your summary in markdown with the following headings: # Hypothesis, # Reasoning, # Assumptions. Do not write summaries for any of these. You can re-order and reformat the text as you see fit, but do not change the scientific content and prefer copying text verbatim in it's entirety.

#BEGIN TRANSCRIPT#
{{ transcript }}
#END TRANSCRIPT#