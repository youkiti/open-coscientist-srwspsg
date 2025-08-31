#!/usr/bin/env python3
"""
Test script to verify that the logging system is working correctly.
Run this to generate log files and see the logging in action.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import after path is set up
from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager


async def test_logging():
    """Test the logging system with a simple research goal."""
    
    # Define a simple research goal
    test_goal = "What are the key mechanisms of neural plasticity in learning and memory formation?"
    
    print(f"\n{'='*60}")
    print("Testing Coscientist Logging System")
    print(f"{'='*60}")
    print(f"Research Goal: {test_goal}")
    print(f"{'='*60}\n")
    
    # Check if log directory was created
    log_dir = project_root / "log"
    if log_dir.exists():
        print(f"✓ Log directory created at: {log_dir}")
    else:
        print(f"✗ Log directory not found at: {log_dir}")
        return
    
    # Initialize the framework
    try:
        print("\nInitializing Coscientist framework...")
        
        # Create state and config
        state = CoscientistState(goal=test_goal)
        config = CoscientistConfig()
        state_manager = CoscientistStateManager(state)
        framework = CoscientistFramework(config, state_manager)
        
        print("✓ Framework initialized successfully")
        
        # Start the research (just the literature review phase)
        print("\nStarting research (literature review phase only)...")
        print("This will test the logging during the literature review phase...")
        print("(Note: This may take a few minutes to complete)\n")
        
        # Run just the start method to test literature review logging
        await framework.start(n_hypotheses=2)
        
        print("\n✓ Research phase completed")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        logging.error(f"Test error: {str(e)}", exc_info=True)
    
    # Check generated log files
    print(f"\n{'='*60}")
    print("Generated Log Files:")
    print(f"{'='*60}")
    
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        for log_file in sorted(log_files):
            size = log_file.stat().st_size
            print(f"  • {log_file.name} ({size:,} bytes)")
            
            # Show first few lines of main log
            if "main_" in log_file.name:
                print(f"\n  First few lines from {log_file.name}:")
                with open(log_file, 'r') as f:
                    lines = f.readlines()[:5]
                    for line in lines:
                        print(f"    {line.rstrip()}")
    else:
        print("  No log files found")
    
    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}")
    print("\nTo view detailed logs, check the files in: log/")
    print("Main log: log/main_*.log")
    print("Literature review details: log/literature_review_*.log")
    print("GPT Researcher details: log/gpt_researcher_*.log")
    print("Debug information: log/debug_*.log")


if __name__ == "__main__":
    # Check for API keys
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"Error: Missing required environment variables: {', '.join(missing_keys)}")
        print("Please set these in your .env file or environment")
        sys.exit(1)
    
    # Run the test
    try:
        asyncio.run(test_logging())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)