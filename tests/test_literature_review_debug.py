#!/usr/bin/env python3
"""
Test script for debugging literature review with a specific research goal.
This allows testing the literature review phase in isolation.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from coscientist.framework import CoscientistConfig, _SMARTER_LLM_POOL
from coscientist.global_state import CoscientistState, CoscientistStateManager
from coscientist.literature_review_agent import build_literature_review_agent

# Setup comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Your specific research goal
RESEARCH_GOAL = """
アンケートで得た収入カテゴリとレセプトデータを連結し、主要脆弱性骨折（大腿骨近位部、臨床椎体、上腕骨近位部、橈骨遠位部）時点で骨粗鬆症未診断かつ骨折前に骨粗鬆症治療歴のない50歳以上を対象に、骨折後6カ月以内の(1)骨粗鬆症の新規診断付与、(2)二次予防の抗骨粗鬆症薬（経口・注射を含む）の開始、ならびに骨折後12カ月時点での治療継続の有無が収入カテゴリによって異なるかを検証する。
"""


async def test_literature_review_only():
    """Test only the literature review phase with debug output."""
    
    print("=" * 80)
    print("LITERATURE REVIEW DEBUG TEST")
    print("=" * 80)
    print(f"Research Goal: {RESEARCH_GOAL[:200]}...")
    print("=" * 80)
    
    try:
        # First, clear any existing directory for this goal
        print("\n1. Clearing any existing goal directory...")
        result = CoscientistState.clear_goal_directory(RESEARCH_GOAL)
        print(f"   {result}")
        
        # Create fresh state
        print("\n2. Creating fresh CoscientistState...")
        state = CoscientistState(goal=RESEARCH_GOAL)
        state_manager = CoscientistStateManager(state)
        print(f"   State created with output directory: {state._output_dir}")
        
        # Configure the system to use Claude Opus (with fixed max_tokens)
        print("\n3. Configuring literature review agent...")
        config = CoscientistConfig(
            literature_review_agent_llm=_SMARTER_LLM_POOL["claude-opus-4-1-20250805"]
        )
        print(f"   Using LLM: {type(config.literature_review_agent_llm).__name__}")
        print(f"   Max tokens: {config.literature_review_agent_llm.max_tokens}")
        
        # Build literature review agent
        print("\n4. Building literature review agent...")
        literature_review_agent = build_literature_review_agent(
            config.literature_review_agent_llm
        )
        
        # Prepare initial state for literature review
        print("\n5. Preparing initial literature review state...")
        initial_lit_review_state = state_manager.next_literature_review_state(
            max_subtopics=3  # Reduced for faster testing
        )
        print(f"   Max subtopics set to: 3")
        
        # Run literature review
        print("\n6. Running literature review (this may take a while)...")
        print("   Starting topic decomposition...")
        
        final_lit_review_state = await literature_review_agent.ainvoke(
            initial_lit_review_state
        )
        
        # Process results
        print("\n7. Literature review completed successfully!")
        subtopics = final_lit_review_state.get('subtopics', [])
        print(f"   Found {len(subtopics)} subtopics:")
        for i, topic in enumerate(subtopics, 1):
            print(f"   {i}. {topic[:100]}...")
        
        # Save the state
        print("\n8. Saving checkpoint...")
        checkpoint_path = state.save()
        print(f"   Checkpoint saved to: {checkpoint_path}")
        
        # Update state manager
        state_manager.update_literature_review(final_lit_review_state)
        
        print("\n" + "=" * 80)
        print("LITERATURE REVIEW TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return final_lit_review_state
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("ERROR OCCURRED DURING LITERATURE REVIEW TEST")
        print("!" * 80)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        # Try to save state even on error
        try:
            print("\nAttempting to save error checkpoint...")
            checkpoint_path = state.save()
            print(f"Error checkpoint saved to: {checkpoint_path}")
        except:
            print("Failed to save error checkpoint")
        
        raise


def main():
    """Main entry point."""
    print("\nStarting Literature Review Debug Test...\n")
    
    try:
        result = asyncio.run(test_literature_review_only())
        print("\nTest completed successfully!")
        
        # Print summary of results
        if result and 'subtopic_reports' in result:
            print(f"\nGenerated {len(result['subtopic_reports'])} research reports")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()