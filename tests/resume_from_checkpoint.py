#!/usr/bin/env python3
"""
Example script showing how to resume research from a saved checkpoint.
This is useful for debugging and continuing interrupted research sessions.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Your research goal (must match the original to find checkpoints)
RESEARCH_GOAL = """
アンケートで得た収入カテゴリとレセプトデータを連結し、主要脆弱性骨折（大腿骨近位部、臨床椎体、上腕骨近位部、橈骨遠位部）時点で骨粗鬆症未診断かつ骨折前に骨粗鬆症治療歴のない50歳以上を対象に、骨折後6カ月以内の(1)骨粗鬆症の新規診断付与、(2)二次予防の抗骨粗鬆症薬（経口・注射を含む）の開始、ならびに骨折後12カ月時点での治療継続の有無が収入カテゴリによって異なるかを検証する。
"""


async def resume_from_checkpoint():
    """Resume research from the latest checkpoint."""
    
    print("=" * 80)
    print("RESUME FROM CHECKPOINT EXAMPLE")
    print("=" * 80)
    
    try:
        # List available checkpoints for this goal
        print("\n1. Listing available checkpoints...")
        checkpoints = CoscientistState.list_checkpoints(goal=RESEARCH_GOAL)
        
        if not checkpoints:
            print("   No checkpoints found for this research goal.")
            print("   Run the research first to create checkpoints.")
            return
        
        print(f"   Found {len(checkpoints)} checkpoints:")
        for i, checkpoint in enumerate(checkpoints[-5:], 1):  # Show last 5
            print(f"   {i}. {Path(checkpoint).name}")
        
        # Load the latest checkpoint  
        print("\n2. Loading latest checkpoint...")
        latest_checkpoint = checkpoints[0]  # checkpoints are sorted newest first
        print(f"   Loading: {Path(latest_checkpoint).name}")
        
        state = CoscientistState.load(latest_checkpoint)
        print(f"   Checkpoint loaded successfully!")
        print(f"   Research goal: {state.goal[:100]}...")
        
        # Show current state
        print("\n3. Current research state:")
        print(f"   - Literature review: {'Complete' if state.literature_review else 'Not started'}")
        print(f"   - Generated hypotheses: {len(state.generated_hypotheses)}")
        print(f"   - Reviewed hypotheses: {len(state.reviewed_hypotheses)}")
        print(f"   - Tournament completed: {'Yes' if state.tournament else 'No'}")
        print(f"   - Meta-reviews: {len(state.meta_reviews)}")
        print(f"   - Final report: {'Complete' if state.final_report else 'Not complete'}")
        
        # Resume with framework
        print("\n4. Resuming research with framework...")
        
        # Create state manager with loaded state
        state_manager = CoscientistStateManager(state)
        
        # Configure framework with debug options
        config = CoscientistConfig(
            debug_mode=True,  # Enable debug mode for verbose logging
            save_on_error=True,  # Save on errors
            pause_after_literature_review=False  # Don't pause since we're resuming
        )
        
        # Create framework
        framework = CoscientistFramework(config, state_manager)
        
        print("\n5. Continuing research from checkpoint...")
        print("   Note: The framework will automatically continue from where it left off")
        
        # Continue research
        if not state.final_report:
            final_report, meta_review = await framework.run()
            
            print("\n" + "=" * 80)
            print("RESEARCH COMPLETED!")
            print("=" * 80)
            
            if final_report:
                print("\nFinal Report Preview:")
                print(final_report[:500] + "..." if len(final_report) > 500 else final_report)
        else:
            print("\n   Research is already complete!")
            print("\n   Final Report Preview:")
            preview = state.final_report['content'] if isinstance(state.final_report, dict) else str(state.final_report)
            print(preview[:500] + "..." if len(preview) > 500 else preview)
        
    except FileNotFoundError as e:
        print(f"\nError: No existing research found for this goal")
        print("Please run the research first to create checkpoints")
    except Exception as e:
        print(f"\nError resuming from checkpoint: {e}")
        raise


async def load_specific_checkpoint(checkpoint_path: str):
    """Load and examine a specific checkpoint file."""
    
    print(f"\nLoading specific checkpoint: {checkpoint_path}")
    
    try:
        state = CoscientistState.load(checkpoint_path)
        
        print("\nCheckpoint details:")
        print(f"  Goal: {state.goal[:100]}...")
        print(f"  Iteration: {state._iteration}")
        print(f"  Output directory: {state._output_dir}")
        
        # Show literature review topics if available
        if state.literature_review and 'subtopics' in state.literature_review:
            print("\n  Literature Review Subtopics:")
            for i, topic in enumerate(state.literature_review['subtopics'], 1):
                print(f"    {i}. {topic[:80]}...")
        
        # Show hypothesis count by type
        if state.generated_hypotheses:
            print(f"\n  Generated Hypotheses: {len(state.generated_hypotheses)}")
            
        return state
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        # Load specific checkpoint if path provided
        checkpoint_path = sys.argv[1]
        print(f"Loading specific checkpoint: {checkpoint_path}")
        asyncio.run(load_specific_checkpoint(checkpoint_path))
    else:
        # Resume from latest checkpoint
        print("Resuming from latest checkpoint...")
        asyncio.run(resume_from_checkpoint())


if __name__ == "__main__":
    main()