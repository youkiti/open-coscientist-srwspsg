#!/usr/bin/env python3
"""
Test script to investigate GPT-5 response structure and fix response extraction
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import inspect

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables
load_dotenv(parent_dir / '.env')

from openai import OpenAI
from coscientist.openai_client import OpenAIResponsesClient


def investigate_direct_openai_api():
    """Test the direct OpenAI API to understand response structure."""
    print("🔍 Investigating Direct OpenAI API Response Structure")
    print("=" * 60)
    
    try:
        client = OpenAI()
        
        # Test the responses.create API directly
        print("📝 Testing client.responses.create() directly...")
        response = client.responses.create(
            model="gpt-5",
            input=[{"role": "user", "content": "Hello! Please respond with 'GPT-5 is working correctly.'"}],
            text={
                "format": {"type": "text"},
                "verbosity": "medium"
            },
            reasoning={
                "effort": "medium", 
                "summary": "auto"
            },
            tools=[],
            store=False,
            include=[]
        )
        
        print(f"\n✅ Response received! Type: {type(response)}")
        print(f"📋 Response attributes: {dir(response)}")
        
        # Deep inspection of response structure
        print(f"\n🔍 Raw response: {response}")
        
        if hasattr(response, 'text'):
            print(f"\n📄 response.text: {response.text}")
            print(f"📄 response.text type: {type(response.text)}")
            print(f"📄 response.text attributes: {dir(response.text)}")
            
            if hasattr(response.text, 'content'):
                print(f"✅ response.text.content: {response.text.content}")
            if hasattr(response.text, 'value'):
                print(f"✅ response.text.value: {response.text.value}")
            if hasattr(response.text, 'data'):
                print(f"✅ response.text.data: {response.text.data}")
        
        if hasattr(response, 'choices'):
            print(f"\n🎯 response.choices: {response.choices}")
            if response.choices:
                choice = response.choices[0]
                print(f"🎯 First choice: {choice}")
                print(f"🎯 Choice attributes: {dir(choice)}")
                if hasattr(choice, 'message'):
                    print(f"✅ choice.message.content: {choice.message.content}")
        
        if hasattr(response, 'content'):
            print(f"\n📝 response.content: {response.content}")
            
        if hasattr(response, 'data'):
            print(f"\n📊 response.data: {response.data}")
            
        # Try to convert to string representation
        print(f"\n📋 String representation: {str(response)}")
        
        return response
        
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")
        return None


def test_current_extraction_method():
    """Test the current extraction method to see what's going wrong."""
    print("\n🔧 Testing Current OpenAI Responses Client")
    print("=" * 60)
    
    try:
        from coscientist.openai_client import create_openai_responses_client
        from langchain_core.messages import HumanMessage
        
        client = create_openai_responses_client(
            model="gpt-5",
            max_tokens=1000,
            reasoning_effort="medium"
        )
        
        test_message = HumanMessage(content="Hello! Please respond with 'GPT-5 is working correctly via LangChain wrapper.'")
        
        print("📝 Testing current client implementation...")
        result = client._generate([test_message])
        
        print(f"✅ LangChain result received!")
        print(f"📋 Result type: {type(result)}")
        print(f"📋 Result: {result}")
        print(f"📋 Generations: {result.generations}")
        print(f"📋 First generation: {result.generations[0]}")
        print(f"📋 Message content: {result.generations[0].message.content}")
        
        return result
        
    except Exception as e:
        print(f"❌ Current client test failed: {e}")
        return None


def analyze_response_object(response):
    """Perform deep analysis of response object."""
    print(f"\n🔬 Deep Analysis of Response Object")
    print("=" * 60)
    
    if response is None:
        print("❌ No response object to analyze")
        return
    
    print(f"📊 Object type: {type(response)}")
    print(f"📊 Object repr: {repr(response)}")
    
    # Try different attribute access patterns
    test_paths = [
        'text.content',
        'text.value', 
        'text.data',
        'text.text',
        'content',
        'data',
        'value',
        'message.content',
        'choices[0].message.content',
        'choices[0].text',
        'output.content',
        'output.text',
        'result.content',
        'result.text'
    ]
    
    print(f"\n🧪 Testing different attribute access patterns:")
    for path in test_paths:
        try:
            # Build the attribute access chain
            obj = response
            parts = path.split('.')
            
            for part in parts:
                if '[' in part and ']' in part:
                    # Handle array access like choices[0]
                    attr_name = part.split('[')[0]
                    index = int(part.split('[')[1].split(']')[0])
                    obj = getattr(obj, attr_name)[index]
                else:
                    obj = getattr(obj, part)
            
            print(f"✅ {path}: {obj}")
            print(f"   Type: {type(obj)}")
            
        except Exception as e:
            print(f"❌ {path}: Failed - {e}")


def create_test_extraction_function(response):
    """Create and test different extraction functions."""
    print(f"\n🛠️ Creating Test Extraction Functions")
    print("=" * 60)
    
    def extract_method_1(resp):
        """Method 1: Try text.content"""
        if hasattr(resp, 'text') and resp.text:
            if hasattr(resp.text, 'content'):
                return resp.text.content
        return None
    
    def extract_method_2(resp):
        """Method 2: Try direct string conversion"""
        if hasattr(resp, 'text'):
            return str(resp.text)
        return None
        
    def extract_method_3(resp):
        """Method 3: Try choices pattern"""
        if hasattr(resp, 'choices') and resp.choices:
            return resp.choices[0].message.content
        return None
        
    def extract_method_4(resp):
        """Method 4: Try content attribute"""
        if hasattr(resp, 'content'):
            return resp.content
        return None
        
    def extract_method_5(resp):
        """Method 5: Try data attribute"""
        if hasattr(resp, 'data') and hasattr(resp.data, 'content'):
            return resp.data.content
        return None
        
    def extract_method_6(resp):
        """Method 6: Look for any content-like attribute"""
        content_attrs = ['content', 'text', 'value', 'data', 'output', 'result']
        for attr in content_attrs:
            if hasattr(resp, attr):
                obj = getattr(resp, attr)
                if isinstance(obj, str) and len(obj) > 0 and not obj.startswith('Response'):
                    return obj
        return None
    
    methods = [
        ("text.content", extract_method_1),
        ("string conversion", extract_method_2), 
        ("choices pattern", extract_method_3),
        ("content attr", extract_method_4),
        ("data.content", extract_method_5),
        ("smart search", extract_method_6)
    ]
    
    successful_method = None
    
    for name, method in methods:
        try:
            result = method(response)
            if result and isinstance(result, str) and len(result.strip()) > 0:
                print(f"✅ {name}: SUCCESS -> '{result[:100]}...'")
                if successful_method is None:
                    successful_method = (name, method)
            else:
                print(f"❌ {name}: No valid content - {result}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    return successful_method


def main():
    """Run all tests to understand and fix GPT-5 response format."""
    print("🧪 GPT-5 Response Format Investigation")
    print("=" * 80)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found in environment variables")
        return
    
    # Test 1: Direct API
    response = investigate_direct_openai_api()
    
    # Test 2: Current client
    current_result = test_current_extraction_method()
    
    # Test 3: Deep analysis
    if response:
        analyze_response_object(response)
        
        # Test 4: Find working extraction method
        successful_method = create_test_extraction_function(response)
        
        if successful_method:
            name, method = successful_method
            print(f"\n🎉 FOUND WORKING METHOD: {name}")
            print(f"✅ Extracted content: {method(response)}")
            
            # Generate the fix code
            print(f"\n💡 SUGGESTED FIX for _extract_content method:")
            print("=" * 60)
            print("""
def _extract_content(self, response: Any) -> str:
    \"\"\"Extract content from OpenAI response.\"\"\"
    try:""")
            
            if name == "text.content":
                print("""        # GPT-5 responses.create() API response format
        if hasattr(response, 'text') and response.text:
            if hasattr(response.text, 'content'):
                return response.text.content
            return str(response.text)""")
            elif name == "choices pattern":
                print("""        # Standard OpenAI chat completion format  
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content""")
            elif name == "content attr":
                print("""        # Direct content attribute
        if hasattr(response, 'content'):
            return response.content""")
            
            print("""        
        # Fallback
        return str(response)
    except Exception as e:
        logger.error(f"Failed to extract content from response: {e}")
        return f"Response extraction error: {str(e)}"
""")
        else:
            print(f"\n❌ No working extraction method found!")
    
    print(f"\n🏁 Investigation complete!")


if __name__ == "__main__":
    main()