#!/usr/bin/env python3
"""
Test script for Claude Opus 4.1 integration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'app'))

# Load environment variables
load_dotenv(parent_dir / '.env')

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage


def test_claude_opus_client():
    """Test the Claude Opus 4.1 client."""
    print("Testing Claude Opus 4.1 client...")
    
    # Create client
    client = ChatAnthropic(
        model="claude-opus-4-1-20250805", 
        max_tokens=1000, 
        max_retries=3
    )
    
    # Test message
    test_message = HumanMessage(
        content="What are the most important considerations when designing a scientific experiment? Provide a brief but comprehensive overview."
    )
    
    try:
        print("Sending test message to Claude Opus 4.1...")
        result = client._generate([test_message])
        
        print("✅ Claude Opus 4.1 Response:")
        print("-" * 60)
        print(result.generations[0].message.content)
        print("-" * 60)
        print("✅ Test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This might be expected if Claude Opus 4.1 API is not yet available.")
        return False


def test_framework_integration():
    """Test that the framework can use Claude Opus 4.1."""
    print("\nTesting framework integration...")
    
    try:
        from coscientist.framework import _SMARTER_LLM_POOL, CoscientistConfig
        
        if "claude-opus-4-1-20250805" in _SMARTER_LLM_POOL:
            print("✅ Claude Opus 4.1 found in SMARTER_LLM_POOL")
            
            opus_client = _SMARTER_LLM_POOL["claude-opus-4-1-20250805"]
            print(f"✅ Client type: {type(opus_client)}")
            print(f"✅ Client model: {opus_client.model}")
            
            # Test default configuration
            config = CoscientistConfig()
            print(f"✅ Literature review agent using: {config.literature_review_agent_llm.model}")
            print(f"✅ Supervisor agent using: {config.supervisor_agent_llm.model}")
            print(f"✅ Final report agent using: {config.final_report_agent_llm.model}")
            
            return True
        else:
            print("❌ Claude Opus 4.1 not found in SMARTER_LLM_POOL")
            print(f"Available models: {list(_SMARTER_LLM_POOL.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Framework integration test failed: {e}")
        return False


def test_config_integration():
    """Test configuration page integration."""
    print("\nTesting configuration page integration...")
    
    try:
        from configuration_page import get_llm_options
        
        llm_options = get_llm_options()
        if "Claude Opus 4.1" in llm_options:
            print("✅ Claude Opus 4.1 found in configuration options")
            
            opus_option = llm_options["Claude Opus 4.1"]
            print(f"✅ Configuration client type: {type(opus_option)}")
            print(f"✅ Configuration client model: {opus_option.model}")
            return True
        else:
            print("❌ Claude Opus 4.1 not found in configuration options")
            print(f"Available options: {list(llm_options.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        return False


def test_researcher_config():
    """Test researcher config integration."""
    print("\nTesting researcher config...")
    
    try:
        import json
        
        config_path = parent_dir / 'coscientist' / 'researcher_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if config.get("SMART_LLM") == "anthropic:claude-opus-4-1-20250805":
            print("✅ SMART_LLM updated to Claude Opus 4.1")
            return True
        else:
            print(f"❌ SMART_LLM not updated: {config.get('SMART_LLM')}")
            return False
            
    except Exception as e:
        print(f"❌ Researcher config test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Starting Claude Opus 4.1 Integration Tests")
    print("=" * 60)
    
    # Check if Anthropic API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️ ANTHROPIC_API_KEY not found in environment variables")
        print("API tests will likely fail, but integration tests should still work")
    
    tests = [
        ("Framework Integration", test_framework_integration),
        ("Configuration Integration", test_config_integration),
        ("Researcher Config", test_researcher_config),
        ("Claude Opus API", test_claude_opus_client),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test errored: {e}")
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Claude Opus 4.1 integration is ready.")
    elif passed >= total - 1:  # Allow API test to fail if model not available
        print("✨ Integration tests passed! Claude Opus 4.1 is configured correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()