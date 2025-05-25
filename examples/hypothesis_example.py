from langchain.chat_models import init_chat_model

from coscientist.generation_agent import (
    CollaborativeState,
    build_collaborative_generation_agent,
)
from coscientist.literature_review_agent import review_literature
from coscientist.reasoning_types import ReasoningType
from coscientist.reflection_agent import (
    ReflectionState,
    build_reflection_agent,
)

if __name__ == "__main__":
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    goal = (
        "Find a mechanistic connection between TDP-43 mislocalization in iPSC-derived hNIL motor neurons "
        "and transcriptional changes in genes related to retinoic acid and chondroitin sulfate binding."
    )
    lit_review = review_literature(llm, goal)
    agents = ["Cell Biologist", "Neuroscientist"]
    debate_graph_compiled = build_collaborative_generation_agent(
        agents,
        {"Cell Biologist": "cell biology", "Neuroscientist": "neuroscience"},
        {
            "Cell Biologist": ReasoningType.DEDUCTIVE,
            "Neuroscientist": ReasoningType.FIRST_PRINCIPLES,
        },
        {"Cell Biologist": llm, "Neuroscientist": llm},
        max_turns=10,
    )
    # The initial state's next_agent must be a valid entry point.
    initial_state = CollaborativeState(
        goal=goal,
        transcript=[],
        turn=0,  # Moderator will increment this to 1 before the first agent's turn
        next_agent=agents[
            0
        ],  # Set the first agent to start, matching graph entry point
        finished=False,
        literature_review=lit_review.articles_with_reasoning,
    )
    current_state = initial_state
    for i, event in enumerate(debate_graph_compiled.stream(current_state)):
        # event will be a dictionary where keys are node names and values are their outputs (the new state)
        print(f"\\n--- Event {i+1} ---")
        for node_name, output_state in event.items():
            if node_name == "moderator":
                continue

            current_state = output_state
        if current_state.get("finished"):
            print("\\n--- Debate Finished ---")
            break

    ref_agent = build_reflection_agent(llm)

    # TODO: Figure out how to get the hypothesis from the output of the debate graph
    hypo = (
        "TDP-43 Mislocalization Disrupts Retinoic Acid Receptor Activity, "
        "Leading to Decreased Expression of Specific Chondroitin Sulfate-Related "
        "Genes (e.g., CSPG4, CS Galactosyltransferase), which Impairs Extracellular "
        "Matrix Structure in hNIL Motor Neurons. This Dysregulation Results in Adverse "
        "Effects on Neurite Outgrowth, Synaptic Activity, and Inflammation, all of which "
        "Contribute to Neurodegenerative Phenotypes. Comprehensive Investigations Utilizing "
        "CRISPR-mediated Gene Editing, RNA-Sequencing, and Temporal Assessment of Retinoic "
        "Acid Signaling Will Be Employed to Validate This Hypothesis Across In Vitro and In Vivo Models."
    )
    initial_state = ReflectionState(hypothesis=hypo)
    current_state = ref_agent.invoke(initial_state)
