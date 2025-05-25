"""
AI Co-Scientist Demo
==================

This script demonstrates the complete AI co-scientist system in action,
showing how the multi-agent framework works to generate, review, rank,
and evolve scientific hypotheses.

Run this script to see the system tackle a research question about ALS
and protein phosphorylation.
"""

import asyncio
import json
from pathlib import Path


# Assuming we have a simple LLM implementation for demo
# In practice, you'd use LangChain with actual LLM providers
class MockLLM:
    """Mock LLM for demonstration purposes."""

    def invoke(self, prompt):
        class MockResponse:
            def __init__(self, content):
                self.content = content

        # Simple mock responses based on prompt content
        if "research plan configuration" in prompt.lower():
            return MockResponse(
                json.dumps(
                    {
                        "preferences": "Focus on novel hypotheses with detailed mechanistic explanations",
                        "attributes": [
                            "Novelty",
                            "Feasibility",
                            "Testability",
                            "Correctness",
                        ],
                        "constraints": [
                            "Must be biologically plausible",
                            "Should be testable",
                        ],
                    }
                )
            )

        elif (
            "hypothesis generation" in prompt.lower()
            or "independent_generation" in prompt.lower()
        ):
            return MockResponse("""
            ## Evidence
            ALS is characterized by selective motor neuron degeneration. Recent studies show nuclear pore complex dysfunction in ALS models.
            
            ## Hypothesis
            Hyperphosphorylation of nucleoporin Nup62 by stress-activated kinases disrupts nuclear transport selectivity, leading to toxic protein accumulation and motor neuron death in ALS.
            
            ## Reasoning
            Nuclear pore complexes control molecular traffic between nucleus and cytoplasm. Nup62 is a key component for transport selectivity. Stress conditions in ALS could activate kinases that phosphorylate Nup62, compromising transport function.
            
            ## Assumptions Table
            | Assumption | Evidence Level | Testability |
            |------------|----------------|-------------|
            | Nup62 hyperphosphorylation occurs in ALS | Medium | High |
            | Phosphorylation affects transport selectivity | Low | High |
            | Transport disruption causes toxicity | Medium | Medium |
            """)

        elif "desk_reject" in prompt.lower():
            return MockResponse("""
            This hypothesis presents a novel and plausible mechanism for ALS pathogenesis. 
            The connection between nuclear pore dysfunction and neurodegeneration is well-established.
            The specific focus on Nup62 phosphorylation is testable and could lead to therapeutic targets.
            
            FINAL EVALUATION: PASS
            """)

        elif "deep_verification" in prompt.lower():
            return MockResponse("""
            ## Detailed Analysis
            
            **Strengths:**
            - Novel mechanistic insight into ALS pathogenesis
            - Builds on established NPC dysfunction in neurodegeneration
            - Testable predictions with current techniques
            - Potential therapeutic implications
            
            **Weaknesses:**
            - Limited direct evidence for Nup62 phosphorylation in ALS
            - Unclear which specific kinases are involved
            - Transport selectivity mechanisms not fully characterized
            
            **Experimental Validation:**
            1. Analyze Nup62 phosphorylation in ALS patient samples
            2. Test transport function in phosphorylation mimetic mutants
            3. Identify responsible kinases through proteomics
            
            **Overall Assessment:** Promising hypothesis with strong biological rationale requiring experimental validation.
            """)

        elif "tournament" in prompt.lower() or "better" in prompt.lower():
            return MockResponse("better hypothesis: 1")

        elif "pattern" in prompt.lower():
            return MockResponse("""
            ## Pattern Analysis
            
            **Common Strengths:**
            - Focus on protein modifications and cellular dysfunction
            - Good mechanistic detail
            - Testable predictions
            
            **Recurring Weaknesses:**
            - Limited supporting evidence
            - Unclear therapeutic pathways
            - Need for more specific molecular details
            
            **Evaluation Trends:**
            - High value placed on novelty and testability
            - Mechanistic explanations preferred over phenomenological
            """)

        elif "optimization" in prompt.lower():
            return MockResponse("""
            ## Agent Optimization Suggestions
            
            **Generation Agent:**
            - Include more literature context in hypothesis generation
            - Focus on specific molecular mechanisms
            - Provide clearer experimental predictions
            
            **Reflection Agent:**
            - Emphasize therapeutic potential in reviews
            - Consider technical feasibility more thoroughly
            - Include cost-benefit analysis for experiments
            
            **Ranking Agent:**
            - Weight testability higher in comparisons
            - Consider broader impact potential
            - Include interdisciplinary perspectives
            """)

        elif "research overview" in prompt.lower():
            return MockResponse("""
            # Research Overview: ALS and Nuclear Pore Complex Dysfunction
            
            ## Executive Summary
            This research investigation identified novel mechanistic hypotheses linking nuclear pore complex (NPC) dysfunction to ALS pathogenesis, with particular focus on nucleoporin phosphorylation.
            
            ## Key Hypotheses
            
            ### Top-Ranked Hypothesis: Nup62 Hyperphosphorylation
            **Core Mechanism:** Stress-activated kinases hyperphosphorylate nucleoporin Nup62, disrupting nuclear transport selectivity and causing toxic protein accumulation in motor neurons.
            
            **Strengths:**
            - Novel and testable mechanism
            - Strong biological rationale
            - Clear experimental pathway
            
            ## Recommended Next Steps
            
            ### Immediate Experiments (6-12 months)
            1. **Phosphorylation Analysis:** Mass spectrometry analysis of Nup62 phosphorylation in ALS patient samples vs. controls
            2. **Functional Assays:** Nuclear transport assays using phosphorylation-mimetic Nup62 mutants
            3. **Kinase Identification:** Proteomics screen to identify kinases responsible for Nup62 phosphorylation
            
            ### Medium-term Studies (1-2 years)
            1. **Animal Models:** Test hypothesis in ALS mouse models
            2. **Drug Screening:** Screen for kinase inhibitors that prevent Nup62 hyperphosphorylation
            3. **Biomarker Development:** Validate Nup62 phosphorylation as ALS biomarker
            
            ## Knowledge Gaps
            - Specific kinases involved in Nup62 phosphorylation
            - Temporal progression of NPC dysfunction in ALS
            - Reversibility of transport defects
            
            ## Suggested Collaborations
            - Nuclear transport experts (e.g., Görlich lab, MPI)
            - ALS researchers with patient samples
            - Kinase biology specialists
            - Drug discovery teams
            
            ## Risk Assessment
            **High Risk:** Limited evidence for Nup62 phosphorylation in ALS
            **Medium Risk:** Technical challenges in transport assays
            **Low Risk:** Basic biological concepts well-established
            
            ## Funding Opportunities
            - NIH R01 for mechanism studies
            - ALS Association grants for translational research
            - European Research Council for innovative approaches
            """)

        else:
            return MockResponse("Mock response for prompt analysis.")

    def with_structured_output(self, schema):
        class MockStructuredLLM:
            def __init__(self, parent_llm):
                self.parent_llm = parent_llm

            def invoke(self, prompt):
                # For demo, just use the parent's invoke and parse JSON if possible
                response = self.parent_llm.invoke(prompt)
                try:
                    # Try to parse as JSON for structured output
                    return schema(**json.loads(response.content))
                except Exception:
                    return response

        return MockStructuredLLM(self)


async def main():
    """Run the AI co-scientist demo."""
    print("🧬 AI Co-Scientist System Demo")
    print("=" * 50)

    # Initialize the system
    print("\n1. Initializing AI Co-Scientist Framework...")

    from coscientist.configuration_agent import goal_to_configuration
    from coscientist.context_memory import get_context_memory
    from coscientist.framework import CoScientistFramework

    # Use mock LLM for demo
    llm = MockLLM()
    framework = CoScientistFramework(llm, max_concurrent_tasks=2)

    # Define research goal
    research_goal = """
    Develop a novel hypothesis for the key factor or process which causes ALS 
    related to phosphorylation of a Nuclear Pore Complex (NPC) nucleoporin. 
    Explain mechanism of action in detail. Include also a feasible experiment 
    to test the hypothesis.
    """

    print(f"\n2. Research Goal:")
    print(f"   {research_goal}")

    # Generate research configuration
    print("\n3. Generating Research Configuration...")
    research_config = goal_to_configuration(llm, research_goal)
    print(f"   ✓ Preferences: {research_config.preferences}")
    print(f"   ✓ Attributes: {research_config.attributes}")
    print(f"   ✓ Constraints: {research_config.constraints}")

    # Initialize context memory
    print("\n4. Setting up Context Memory...")
    memory = get_context_memory("demo_memory.db")
    session_id = memory.create_session(research_goal, research_config)
    print(f"   ✓ Created session {session_id}")

    # Run the framework
    print("\n5. Running Multi-Agent Research Process...")
    print("   This may take a few minutes as agents collaborate...")

    try:
        final_results = await framework.run_framework(
            goal=research_goal,
            research_plan_config=research_config,
            max_iterations=5,  # Reduced for demo
        )

        print("\n6. Research Process Complete! 🎉")

        # Display results
        print("\n" + "=" * 50)
        print("FINAL RESEARCH RESULTS")
        print("=" * 50)

        if "research_overview" in final_results:
            print(final_results["research_overview"])

        if "optimization_suggestions" in final_results:
            print("\n" + "-" * 30)
            print("SYSTEM OPTIMIZATION SUGGESTIONS")
            print("-" * 30)
            print(final_results["optimization_suggestions"])

        # Show session summary
        print("\n" + "-" * 30)
        print("SESSION SUMMARY")
        print("-" * 30)

        summary = memory.get_session_summary(session_id)
        stats = summary.get("statistics", {})

        print(f"Total Hypotheses Generated: {stats.get('total_hypotheses', 0)}")
        print(f"Total Tasks Executed: {stats.get('total_tasks', 0)}")
        print(f"Tournament Matches: {stats.get('total_matches', 0)}")

        if stats.get("task_breakdown"):
            print("\nTask Breakdown:")
            for status, count in stats["task_breakdown"].items():
                print(f"  {status}: {count}")

        print(f"\n✓ All data saved to session {session_id}")
        print(f"✓ Database: demo_memory.db")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("This is expected in demo mode with mock components.")

    print("\n" + "=" * 50)
    print("Demo Complete!")
    print("=" * 50)

    print("""
🔬 What Just Happened?

The AI Co-Scientist system demonstrated its multi-agent approach:

1. **Generation Agents** created novel hypotheses about ALS and NPC dysfunction
2. **Reflection Agents** critically reviewed hypotheses for quality and feasibility  
3. **Ranking Agents** ran tournaments to identify the most promising ideas
4. **Evolution Agents** refined top hypotheses and generated new variants
5. **Proximity Agents** analyzed similarity between hypotheses
6. **Meta-review Agents** synthesized findings into actionable research plans
7. **Supervisor Agent** orchestrated the entire process

The system used LangGraph for workflow management, enabling:
- Asynchronous task execution
- Multi-agent collaboration  
- Persistent memory across sessions
- Iterative hypothesis refinement

In a real deployment, this would connect to:
- Live literature databases (PubMed, Google Scholar)
- Advanced LLMs (GPT-4, Claude, Gemini)
- Specialized scientific tools
- Collaborative research platforms

🚀 Next Steps:
- Explore the generated hypotheses
- Review the session data in the database
- Modify the research goal and run again
- Integrate with real LLMs and databases
    """)


if __name__ == "__main__":
    asyncio.run(main())
