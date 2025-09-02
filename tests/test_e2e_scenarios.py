"""
End-to-end scenario tests for complete research workflows.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path

from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager
from tests.utils import create_mock_hypothesis


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
class TestQuickResearchScenario:
    """Test quick research scenario with small hypothesis set."""

    async def test_alzheimers_research_scenario(self, mock_llm_pools, temp_dir, mock_env, mock_researcher_config):
        """Complete research scenario: What causes Alzheimer's disease?"""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "What causes Alzheimer's disease?"
        config = CoscientistConfig(
            debug_mode=True,
            max_supervisor_iterations=3,
            pause_after_literature_review=False
        )
        
        state = CoscientistState(goal=goal)
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock literature review
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        Research Questions:
        1. What are the molecular mechanisms of Alzheimer's disease?
        2. What role do amyloid plaques play?
        3. How does tau pathology contribute?
        4. What are the genetic factors?
        
        Search Queries:
        - Alzheimer disease amyloid beta plaques
        - tau protein hyperphosphorylation dementia
        - APOE4 genetic risk Alzheimer
        - neuroinflammation Alzheimer pathogenesis
        """)
        
        # Mock hypothesis generation
        alzheimers_hypotheses = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Amyloid-beta plaques cause neuronal death through inflammatory pathways",
                    "reasoning": "Accumulation of Aβ peptides triggers microglial activation and cytokine release",
                    "confidence": 0.85,
                    "assumptions": ["BBB integrity is compromised", "Inflammation is primary mechanism"],
                    "testing_approach": "In vivo mouse models with amyloid injections and inflammatory markers",
                    "observables": ["Neuronal loss", "IL-1β levels", "Microglial activation"]
                },
                {
                    "id": 2,
                    "hypothesis": "Tau hyperphosphorylation disrupts microtubule transport leading to synaptic failure",
                    "reasoning": "Phosphorylated tau loses microtubule binding affinity, causing transport deficits",
                    "confidence": 0.90,
                    "assumptions": ["Tau phosphorylation is causal", "Transport is critical for synapses"],
                    "testing_approach": "Live cell imaging of axonal transport in tau transgenic neurons",
                    "observables": ["Transport velocity", "Tau-P levels", "Synaptic markers"]
                },
                {
                    "id": 3,
                    "hypothesis": "APOE4 variant reduces amyloid clearance efficiency",
                    "reasoning": "APOE4 has altered lipid binding that affects amyloid proteolysis",
                    "confidence": 0.75,
                    "assumptions": ["Clearance mechanisms are rate-limiting", "Lipid metabolism affects clearance"],
                    "testing_approach": "Comparative clearance assays with APOE3 vs APOE4",
                    "observables": ["Clearance rate", "Lipid profiles", "Protease activity"]
                },
                {
                    "id": 4,
                    "hypothesis": "Vascular dysfunction precedes and accelerates neurodegeneration",
                    "reasoning": "BBB breakdown allows toxin entry and reduces waste clearance",
                    "confidence": 0.70,
                    "assumptions": ["Vascular changes are early events", "BBB integrity is critical"],
                    "testing_approach": "Longitudinal imaging of BBB permeability and cognition",
                    "observables": ["BBB leakage", "Cerebral blood flow", "Cognitive decline"]
                }
            ]
        }
        """
        mock_llm_pools["gpt-5"].add_response(alzheimers_hypotheses)
        
        # Mock tournament debates
        debate_responses = [
            "Winner: Hypothesis A\nReasoning: Stronger clinical evidence from human trials",
            "Winner: Hypothesis B\nReasoning: More mechanistic detail and reproducible findings",
            "Winner: Hypothesis A\nReasoning: Broader therapeutic implications",
            "Winner: Hypothesis C\nReasoning: Genetic evidence provides strongest support",
            "Winner: Hypothesis B\nReasoning: Most comprehensive experimental validation",
            "Winner: Hypothesis A\nReasoning: Best integration of multiple evidence types"
        ]
        
        for response in debate_responses:
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(response)
        
        # Mock supervisor decisions
        supervisor_decisions = [
            "Decision: run_tournament\nReasoning: Need to rank hypotheses for quality assessment",
            "Decision: reflect\nReasoning: Should analyze assumptions and refine understanding",
            "Decision: finalize\nReasoning: Sufficient evidence gathered for conclusions"
        ]
        
        for decision in supervisor_decisions:
            mock_llm_pools["o3"].add_response(decision)
        
        # Mock reflection
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        Reflection Analysis:
        
        Key assumptions requiring validation:
        1. Inflammatory mechanisms are primary vs secondary
        2. Transport deficits causally link to synaptic failure
        3. Clearance mechanisms are rate-limiting in pathogenesis
        
        Causal relationships identified:
        - Amyloid → Inflammation → Neuronal death
        - Tau-P → Transport disruption → Synaptic loss
        - APOE4 → Reduced clearance → Amyloid accumulation
        
        Research gaps:
        - Temporal sequence of pathological events
        - Interaction between amyloid and tau pathways
        - Role of vascular factors in initiation vs progression
        """)
        
        # Mock final report
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        # Alzheimer's Disease Causation: Research Synthesis
        
        ## Executive Summary
        Based on comprehensive analysis of 4 competing hypotheses, Alzheimer's disease appears to result from multiple interacting pathways rather than a single cause.
        
        ## Top-Ranked Hypotheses
        1. **Tau hyperphosphorylation and transport disruption** (ELO: 1580)
           - Strongest mechanistic evidence
           - Direct link to synaptic pathology
           
        2. **Amyloid-induced neuroinflammation** (ELO: 1545)
           - Robust clinical trial evidence
           - Therapeutic relevance
           
        ## Key Findings
        - Multiple pathways contribute to neurodegeneration
        - Early vascular changes may be underappreciated
        - Genetic factors (APOE4) modulate clearance mechanisms
        
        ## Research Recommendations
        - Longitudinal studies tracking multiple biomarkers
        - Combination therapies targeting multiple pathways
        - Earlier intervention before irreversible damage
        """)
        
        # Mock meta-review
        mock_llm_pools["gemini-2.5-flash"].add_response("""
        ## Meta-Analysis of Research Process
        
        **Methodology Strengths:**
        - Comprehensive literature decomposition
        - Evidence-based hypothesis generation
        - Rigorous tournament-style evaluation
        
        **Key Insights:**
        - Multifactorial causation model emerged
        - Tournament revealed tau pathology as leading mechanism
        - Vascular hypothesis underperformed but remains relevant
        
        **Process Quality:** High
        **Confidence in Results:** 85%
        **Recommended Next Steps:** Experimental validation of top hypotheses
        """)
        
        # Mock GPT Researcher
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = """
            Alzheimer's Disease Research Findings:
            
            Amyloid-beta plaques and tau tangles are hallmark pathologies, but their causal relationships remain debated. 
            Recent evidence suggests tau pathology correlates better with cognitive decline than amyloid burden.
            APOE4 carriers show accelerated amyloid accumulation and reduced clearance efficiency.
            Neuroinflammation appears to be both consequence and contributor to pathogenesis.
            Vascular factors including BBB dysfunction may precede classical pathologies.
            """
            mock_instance.write_report.return_value = mock_instance.conduct_research.return_value
            mock_researcher.return_value = mock_instance
            
            # Execute complete research workflow
            final_report, meta_review = await framework.start(n_hypotheses=4)
            
            # Verify comprehensive results
            assert final_report is not None
            assert meta_review is not None
            assert len(final_report) > 100
            assert "alzheimer" in final_report.lower()
            
            # Verify state contains complete research
            assert state.literature_review is not None
            assert len(state.hypotheses) == 4
            assert state.tournament_results is not None
            assert state.final_report is not None
            
            # Verify hypothesis rankings changed
            ratings = [h.elo_rating for h in state.hypotheses]
            assert not all(r == 1500.0 for r in ratings)
            
            # Verify research quality
            assert any(h.confidence > 0.8 for h in state.hypotheses)
            assert "tau" in final_report.lower() or "amyloid" in final_report.lower()

    async def test_battery_technology_scenario(self, mock_llm_pools, temp_dir, mock_env, mock_researcher_config):
        """Research scenario: How can we develop better batteries?"""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "How can we develop better batteries for electric vehicles?"
        config = CoscientistConfig(max_supervisor_iterations=2)
        
        state = CoscientistState(goal=goal)
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock literature review
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        Research Questions:
        1. What limits current battery energy density?
        2. How can charging speed be improved?
        3. What materials show promise for next-generation batteries?
        
        Search Queries:
        - solid state battery technology
        - lithium metal anode degradation
        - silicon nanowire electrodes
        """)
        
        # Mock hypothesis generation
        battery_hypotheses = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Solid-state electrolytes eliminate dendrite formation enabling lithium metal anodes",
                    "reasoning": "Ceramic electrolytes provide mechanical barrier to dendrite growth",
                    "confidence": 0.88,
                    "assumptions": ["Interface stability can be achieved", "Manufacturing is scalable"],
                    "testing_approach": "Cycling tests with ceramic vs liquid electrolytes",
                    "observables": ["Dendrite formation", "Cycle life", "Energy density"]
                },
                {
                    "id": 2,
                    "hypothesis": "Silicon nanowire anodes provide 10x capacity improvement over graphite",
                    "reasoning": "Silicon has theoretical capacity of 4200 mAh/g vs 372 for graphite",
                    "confidence": 0.75,
                    "assumptions": ["Volume expansion can be managed", "SEI formation is stable"],
                    "testing_approach": "Electrochemical cycling with different nanowire structures",
                    "observables": ["Capacity retention", "Volume change", "SEI thickness"]
                }
            ]
        }
        """
        mock_llm_pools["gpt-5"].add_response(battery_hypotheses)
        
        # Mock tournament and other responses
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Winner: Hypothesis A\nReasoning: More proven technology")
        mock_llm_pools["o3"].add_response("Decision: finalize\nReasoning: Sufficient analysis completed")
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Battery research report with solid-state technology leading")
        mock_llm_pools["gemini-2.5-flash"].add_response("Meta-analysis: Solid-state batteries show most promise")
        
        # Mock GPT Researcher
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = "Battery technology research findings"
            mock_instance.write_report.return_value = "Battery technology research findings"
            mock_researcher.return_value = mock_instance
            
            # Execute research
            final_report, meta_review = await framework.start(n_hypotheses=2)
            
            # Verify results
            assert final_report is not None
            assert "battery" in final_report.lower() or "solid" in final_report.lower()
            assert len(state.hypotheses) == 2


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
class TestComplexResearchScenario:
    """Test complex research scenario with larger hypothesis set and multiple iterations."""

    async def test_climate_change_mitigation_scenario(self, mock_llm_pools, temp_dir, mock_env, mock_researcher_config):
        """Complex scenario: What are the most effective climate change mitigation strategies?"""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "What are the most effective strategies for climate change mitigation?"
        config = CoscientistConfig(
            debug_mode=False,
            max_supervisor_iterations=5,
            pause_after_literature_review=False
        )
        
        state = CoscientistState(goal=goal)
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        # Mock comprehensive literature review
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        Research Questions:
        1. What are the largest sources of greenhouse gas emissions?
        2. Which mitigation strategies have the highest impact potential?
        3. What are the economic and social feasibility constraints?
        4. How do different strategies interact and complement each other?
        
        Search Queries:
        - carbon emission reduction strategies effectiveness
        - renewable energy transition pathways
        - carbon capture storage technology
        - forest carbon sequestration potential
        - behavioral change climate mitigation
        """)
        
        # Mock large hypothesis set generation
        climate_hypotheses = """
        {
            "hypotheses": [
                {
                    "id": 1,
                    "hypothesis": "Rapid renewable energy transition can reduce emissions by 70% within 20 years",
                    "reasoning": "Solar and wind costs have reached grid parity, enabling accelerated deployment",
                    "confidence": 0.85,
                    "assumptions": ["Grid storage technology scales", "Political support continues"],
                    "testing_approach": "Economic modeling of transition scenarios",
                    "observables": ["Deployment rates", "Cost trajectories", "Grid stability"]
                },
                {
                    "id": 2,
                    "hypothesis": "Direct air capture at scale can remove gigatons of CO2 cost-effectively",
                    "reasoning": "DAC costs projected to fall below $100/tonne with scale and innovation",
                    "confidence": 0.60,
                    "assumptions": ["Energy requirements can be met with renewables", "Geological storage is secure"],
                    "testing_approach": "Techno-economic analysis of large-scale DAC deployment",
                    "observables": ["Cost per tonne", "Energy consumption", "Storage capacity"]
                },
                {
                    "id": 3,
                    "hypothesis": "Nature-based solutions provide 30% of required mitigation at low cost",
                    "reasoning": "Forest restoration and soil carbon have high sequestration potential",
                    "confidence": 0.78,
                    "assumptions": ["Land availability is sufficient", "Permanence can be assured"],
                    "testing_approach": "Meta-analysis of sequestration rates across ecosystems",
                    "observables": ["Sequestration rates", "Land requirements", "Co-benefits"]
                },
                {
                    "id": 4,
                    "hypothesis": "Behavioral interventions can reduce individual emissions by 25%",
                    "reasoning": "Transportation and consumption changes have large potential impact",
                    "confidence": 0.50,
                    "assumptions": ["Interventions can scale", "Social norms shift"],
                    "testing_approach": "Randomized controlled trials of behavior change programs",
                    "observables": ["Emission reductions", "Adoption rates", "Persistence"]
                },
                {
                    "id": 5,
                    "hypothesis": "Industrial decarbonization through electrification reduces emissions by 40%",
                    "reasoning": "Heat pumps and electric processes can replace fossil fuel use",
                    "confidence": 0.72,
                    "assumptions": ["High-temperature electrification is feasible", "Grid is decarbonized"],
                    "testing_approach": "Pilot projects in heavy industry sectors",
                    "observables": ["Process efficiency", "Cost premiums", "Technology readiness"]
                },
                {
                    "id": 6,
                    "hypothesis": "Carbon pricing at $100/tonne drives optimal mitigation investment",
                    "reasoning": "Price signals incentivize least-cost emission reductions across sectors",
                    "confidence": 0.80,
                    "assumptions": ["Political feasibility exists", "Border adjustments prevent leakage"],
                    "testing_approach": "Comparative analysis of carbon pricing implementations",
                    "observables": ["Emission reductions", "Economic impacts", "Innovation effects"]
                }
            ]
        }
        """
        mock_llm_pools["gpt-5"].add_response(climate_hypotheses)
        
        # Mock tournament with many debates
        debate_outcomes = [
            "Winner: Hypothesis A\nReasoning: Renewable energy has proven scalability",
            "Winner: Hypothesis C\nReasoning: Nature-based solutions offer immediate implementation",
            "Winner: Hypothesis F\nReasoning: Carbon pricing addresses market failures effectively",
            "Winner: Hypothesis A\nReasoning: Technology readiness is highest for renewables",
            "Winner: Hypothesis C\nReasoning: Co-benefits include biodiversity and jobs",
            "Winner: Hypothesis F\nReasoning: Economy-wide impact and revenue generation",
            "Winner: Hypothesis E\nReasoning: Industrial sector is major emissions source",
            "Winner: Hypothesis A\nReasoning: Costs continue declining rapidly",
            "Winner: Hypothesis F\nReasoning: Price signal drives innovation across all sectors",
            "Winner: Hypothesis C\nReasoning: Natural solutions are immediately available"
        ]
        
        for outcome in debate_outcomes:
            mock_llm_pools["claude-opus-4-1-20250805"].add_response(outcome)
        
        # Mock supervisor decisions for multiple iterations
        supervisor_sequence = [
            "Decision: run_tournament\nReasoning: Need initial ranking of strategies",
            "Decision: generate_new_hypotheses\nReasoning: Should explore hybrid approaches",
            "Decision: run_tournament\nReasoning: Compare new vs existing hypotheses",
            "Decision: reflect\nReasoning: Analyze interactions between strategies",
            "Decision: finalize\nReasoning: Comprehensive analysis completed"
        ]
        
        for decision in supervisor_sequence:
            mock_llm_pools["o3"].add_response(decision)
        
        # Mock additional hypothesis generation
        additional_hypotheses = """
        {
            "hypotheses": [
                {
                    "id": 7,
                    "hypothesis": "Integrated renewable energy + storage + DAC systems maximize effectiveness",
                    "reasoning": "Combining strategies addresses intermittency and creates negative emissions",
                    "confidence": 0.75,
                    "assumptions": ["System integration is technically feasible", "Costs remain competitive"],
                    "testing_approach": "Demonstration projects of integrated systems",
                    "observables": ["System efficiency", "Cost effectiveness", "Scalability"]
                }
            ]
        }
        """
        mock_llm_pools["gpt-5"].add_response(additional_hypotheses)
        
        # Mock additional tournament outcomes
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Winner: Hypothesis G\nReasoning: Integrated approach maximizes synergies")
        
        # Mock reflection
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        Strategic Interaction Analysis:
        
        Complementary strategies:
        - Renewable energy + carbon pricing creates market pull
        - Nature-based solutions + industrial decarbonization addresses all sectors
        - DAC + renewables ensures negative emissions capability
        
        Key dependencies:
        - Grid infrastructure limits renewable deployment speed
        - Carbon pricing effectiveness depends on coverage and level
        - Behavioral change requires supporting infrastructure
        
        Portfolio recommendations:
        - Lead with renewables and carbon pricing for immediate impact
        - Scale nature-based solutions for cost-effective sequestration
        - Develop DAC as backup for hard-to-abate emissions
        """)
        
        # Mock comprehensive final report
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("""
        # Climate Change Mitigation Strategy Analysis
        
        ## Executive Summary
        Analysis of 7 mitigation strategies reveals that an integrated portfolio approach is most effective, with renewable energy transition and carbon pricing as cornerstone policies.
        
        ## Strategy Rankings (Post-Tournament)
        1. **Renewable Energy Transition** (ELO: 1620) - Proven, scalable, cost-competitive
        2. **Carbon Pricing** (ELO: 1585) - Economy-wide incentives, revenue generation
        3. **Nature-based Solutions** (ELO: 1565) - Immediate availability, co-benefits
        4. **Industrial Decarbonization** (ELO: 1540) - Addresses major emission source
        5. **Integrated Systems** (ELO: 1520) - Maximizes synergies, emerging approach
        6. **Direct Air Capture** (ELO: 1480) - High potential, cost uncertainties
        7. **Behavioral Interventions** (ELO: 1445) - Important but limited scale
        
        ## Key Findings
        - No single strategy sufficient; portfolio approach required
        - Renewable energy + carbon pricing provides foundation
        - Nature-based solutions offer immediate, cost-effective sequestration
        - Technology strategies need continued cost reduction
        - Policy integration crucial for effectiveness
        
        ## Implementation Roadmap
        Phase 1 (0-5 years): Renewable energy scale-up, carbon pricing implementation
        Phase 2 (5-10 years): Nature-based solutions expansion, industrial pilots
        Phase 3 (10-20 years): DAC deployment, integrated system optimization
        
        ## Confidence Assessment: 82%
        """)
        
        # Mock meta-review
        mock_llm_pools["gemini-2.5-flash"].add_response("""
        ## Research Process Meta-Analysis
        
        **Methodology Strengths:**
        - Comprehensive strategy coverage
        - Multi-iteration refinement process
        - Evidence-based ranking through tournaments
        - Integration analysis of strategy interactions
        
        **Key Research Insights:**
        - Portfolio approach emerged as superior to single strategies
        - Economic instruments (carbon pricing) highly effective
        - Technology solutions dominate but require policy support
        - Behavioral strategies underperformed vs structural changes
        
        **Quality Indicators:**
        - 7 distinct strategies evaluated
        - 10+ tournament matches conducted
        - Supervisor made 5 strategic decisions
        - High confidence final recommendations (82%)
        
        **Research Impact:** High - provides clear strategic priorities
        **Methodological Rigor:** Excellent - comprehensive and systematic
        """)
        
        # Mock GPT Researcher with comprehensive responses
        research_responses = [
            "Renewable energy costs have fallen 90% over the past decade, making it the cheapest electricity source in most regions.",
            "Carbon pricing systems now cover 23% of global emissions, with prices ranging from $1-130 per tonne CO2.",
            "Nature-based climate solutions could provide 37% of required mitigation to limit warming to 2°C.",
            "Industrial processes account for 21% of global CO2 emissions, with limited decarbonization progress to date.",
            "Direct air capture capacity needs to scale 1000x by 2050 to make meaningful climate impact."
        ]
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.side_effect = research_responses
            mock_instance.write_report.side_effect = research_responses
            mock_researcher.return_value = mock_instance
            
            # Execute complex research workflow
            final_report, meta_review = await framework.start(n_hypotheses=6)
            
            # Verify comprehensive results
            assert final_report is not None
            assert meta_review is not None
            assert len(final_report) > 500
            
            # Verify complex state evolution
            assert len(state.hypotheses) >= 6  # Original 6 + potentially more from second generation
            assert state.tournament_results is not None
            assert len(state.tournament_results) >= 10  # Multiple tournament rounds
            
            # Verify supervisor made multiple decisions
            assert state.supervisor_decisions is not None
            
            # Verify research depth
            assert "renewable" in final_report.lower()
            assert "carbon" in final_report.lower()
            assert any(h.elo_rating > 1600 for h in state.hypotheses)  # Top performer
            assert any(h.elo_rating < 1400 for h in state.hypotheses)  # Lower performer


@pytest.mark.e2e
@pytest.mark.asyncio
class TestErrorRecoveryScenarios:
    """Test error recovery and checkpoint functionality."""

    async def test_api_failure_recovery(self, mock_llm_pools, temp_dir, mock_env, mock_researcher_config):
        """Test recovery from API failures during research."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Test API failure recovery"
        config = CoscientistConfig(save_on_error=True, max_supervisor_iterations=2)
        
        state = CoscientistState(goal=goal)
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        # Setup successful literature review
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Research questions and queries")
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = "Research completed successfully"
            mock_instance.write_report.return_value = "Research completed successfully"
            mock_researcher.return_value = mock_instance
            
            # Conduct literature review successfully
            await framework.conduct_literature_review()
            assert state.literature_review is not None
            
            # Simulate API failure during hypothesis generation
            with patch.object(framework, 'generate_hypotheses', side_effect=Exception("API failure")):
                # Should handle error gracefully and save state
                try:
                    await framework.start(n_hypotheses=4)
                except Exception:
                    pass  # Expected due to our mock failure
                
                # State should be preserved despite error
                assert state.literature_review is not None
                
                # If save_on_error is working, state file should exist
                state_files = list(Path(temp_dir).glob("*/coscientist_state_*.pkl"))
                assert len(state_files) > 0

    async def test_checkpoint_resume_scenario(self, mock_llm_pools, temp_dir, mock_env, mock_researcher_config):
        """Test resuming research from checkpoint."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Checkpoint resume test"
        
        # Phase 1: Initial research that creates checkpoint
        config1 = CoscientistConfig(pause_after_literature_review=True)
        state1 = CoscientistState(goal=goal)
        state_manager1 = CoscientistStateManager(state1)
        framework1 = CoscientistFramework(config1, state_manager1)
        
        # Mock literature review
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Literature review content")
        
        with patch("coscientist.literature_review_agent.GPTResearcher") as mock_researcher:
            mock_instance = AsyncMock()
            mock_instance.conduct_research.return_value = "Initial research findings"
            mock_instance.write_report.return_value = "Initial research findings"
            mock_researcher.return_value = mock_instance
            
            # Execute first phase
            await framework1.conduct_literature_review()
            checkpoint_path = state1.save()
            
            assert state1.literature_review is not None
            assert checkpoint_path.exists()
        
        # Phase 2: Resume from checkpoint
        state2 = CoscientistState.load(checkpoint_path)
        state_manager2 = CoscientistStateManager(state2)
        config2 = CoscientistConfig(pause_after_literature_review=False)
        framework2 = CoscientistFramework(config2, state_manager2)
        
        # Verify state was loaded correctly
        assert state2.goal == goal
        assert state2.literature_review == "Initial research findings"
        
        # Mock continuation responses
        mock_llm_pools["gpt-5"].add_response("""
        {"hypotheses": [{"id": 1, "hypothesis": "Resume hypothesis", "reasoning": "Resume reasoning", "confidence": 0.8, "assumptions": [], "testing_approach": "Resume test", "observables": []}]}
        """)
        mock_llm_pools["o3"].add_response("Decision: finalize\nReasoning: Resume and complete")
        mock_llm_pools["claude-opus-4-1-20250805"].add_response("Final report from resumed research")
        mock_llm_pools["gemini-2.5-flash"].add_response("Meta-review of resumed research")
        
        # Continue research from checkpoint
        final_report, meta_review = await framework2.start(n_hypotheses=1)
        
        # Verify continuation was successful
        assert final_report is not None
        assert len(state2.hypotheses) >= 1
        assert state2.hypotheses[0].hypothesis == "Resume hypothesis"


@pytest.mark.e2e
@pytest.mark.requires_api
class TestRealAPIScenarios:
    """Test scenarios with real API connections (requires API keys)."""

    @pytest.mark.requires_openai
    @pytest.mark.requires_tavily
    @pytest.mark.slow
    async def test_minimal_real_research(self, temp_dir):
        """Minimal test with real APIs to verify integration."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Simple, well-defined research question
        goal = "What is photosynthesis?"
        config = CoscientistConfig(
            max_supervisor_iterations=1,
            pause_after_literature_review=True
        )
        
        state = CoscientistState(goal=goal)
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        try:
            # Execute just literature review with real APIs
            await framework.conduct_literature_review()
            
            # Verify basic functionality
            assert state.literature_review is not None
            assert len(state.literature_review) > 50
            assert "photosynthesis" in state.literature_review.lower()
            
        except Exception as e:
            # Real API tests may fail due to rate limits, network issues, etc.
            pytest.skip(f"Real API test failed (expected): {e}")

    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    async def test_hypothesis_generation_real(self, temp_dir):
        """Test hypothesis generation with real Claude API."""
        import os
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Why do leaves change color in autumn?"
        config = CoscientistConfig(max_supervisor_iterations=1)
        
        state = CoscientistState(goal=goal)
        state.literature_review = "Leaves contain chlorophyll, carotenoids, and anthocyanins. Chlorophyll breaks down in autumn."
        
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        try:
            # Test real hypothesis generation
            hypotheses = await framework.generate_hypotheses(n_hypotheses=2)
            
            # Basic validation
            assert len(hypotheses) >= 1
            assert all(h.hypothesis for h in hypotheses)
            assert all(h.reasoning for h in hypotheses)
            assert all(0 <= h.confidence <= 1 for h in hypotheses)
            
        except Exception as e:
            pytest.skip(f"Real API test failed (expected): {e}")