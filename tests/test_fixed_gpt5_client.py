#!/usr/bin/env python3
"""
Test the fixed GPT-5 client to verify correct content extraction
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables
load_dotenv(parent_dir / '.env')

from coscientist.openai_client import create_openai_responses_client
from langchain_core.messages import HumanMessage


def test_fixed_client():
    """Test the fixed OpenAI responses client."""
    print("🧪 Testing Fixed GPT-5 Client")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found")
        return False
    
    try:
        # Create client
        client = create_openai_responses_client(
            model="gpt-5",
            max_tokens=1000,
            reasoning_effort="medium"
        )
        
        # Test message
        test_message = HumanMessage(
            content="Please respond with exactly: 'GPT-5 content extraction is working perfectly!'"
        )
        
        print("📝 Sending test message to fixed GPT-5 client...")
        result = client._generate([test_message])
        
        content = result.generations[0].message.content
        print(f"✅ Extracted content: {content}")
        
        # Check if we got the actual response text instead of config object
        if "GPT-5" in content and "working" in content and "ResponseTextConfig" not in content:
            print("🎉 SUCCESS! Content extraction is working correctly!")
            return True
        else:
            print("❌ Content extraction still not working correctly")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_multiple_scenarios():
    """Test multiple scenarios with the fixed client."""
    print("\n🧪 Testing Multiple Scenarios")
    print("=" * 50)
    
    client = create_openai_responses_client(
        model="gpt-5",
        max_tokens=500,
        reasoning_effort="low"
    )
    
    scenarios = [
        "What is 2+2?",
        "Explain photosynthesis in one sentence.",
        "List three colors.",
        "Write a haiku about science."
    ]
    
    success_count = 0
    
    for i, scenario in enumerate(scenarios, 1):
        try:
            print(f"\n📋 Test {i}: {scenario}")
            message = HumanMessage(content=scenario)
            result = client._generate([message])
            content = result.generations[0].message.content
            
            if "ResponseTextConfig" not in content and len(content.strip()) > 0:
                print(f"✅ Response: {content[:100]}...")
                success_count += 1
            else:
                print(f"❌ Bad response: {content}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Results: {success_count}/{len(scenarios)} scenarios passed")
    return success_count == len(scenarios)


def main():
    """Run all tests."""
    print("🚀 Testing Fixed GPT-5 Client Implementation")
    print("=" * 80)
    
    # Test 1: Basic functionality
    basic_success = test_fixed_client()
    
    # Test 2: Multiple scenarios
    if basic_success:
        multi_success = test_multiple_scenarios()
        
        if multi_success:
            print("\n🎉 ALL TESTS PASSED! GPT-5 client is ready for browser testing.")
        else:
            print("\n⚠️ Some multi-scenario tests failed.")
    else:
        print("\n❌ Basic test failed. Need to investigate further.")


if __name__ == "__main__":
    main()