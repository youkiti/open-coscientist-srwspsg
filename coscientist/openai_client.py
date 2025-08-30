"""
OpenAI Client Wrapper for new responses.create() API
---------------------------------------------------
This module provides a wrapper around the new OpenAI client.responses.create() API
to integrate with the existing LangChain-based architecture.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

logger = logging.getLogger(__name__)


class OpenAIResponsesClient(BaseChatModel):
    """
    OpenAI client wrapper that uses the new responses.create() API while maintaining
    compatibility with LangChain's BaseChatModel interface.
    """
    
    client: OpenAI = Field(default_factory=OpenAI)
    model: str = Field(default="gpt-5")
    max_tokens: int = Field(default=50000)
    max_retries: int = Field(default=3)
    temperature: float = Field(default=0.7)
    reasoning_effort: str = Field(default="medium")
    verbosity: str = Field(default="medium")
    include_reasoning: bool = Field(default=True)
    include_web_search: bool = Field(default=True)
    store: bool = Field(default=False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not hasattr(self, 'client'):
            self.client = OpenAI()
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using OpenAI's new responses.create() API."""
        
        # Convert LangChain messages to OpenAI format
        input_messages = self._convert_messages(messages)
        
        # Prepare include list for response
        include_list = []
        if self.include_reasoning:
            include_list.append("reasoning.encrypted_content")
        if self.include_web_search:
            include_list.append("web_search_call.action.sources")
        
        try:
            # Prepare the request parameters according to the new API
            request_params = {
                "model": self.model,
                "input": input_messages,
                "text": {
                    "format": {
                        "type": "text"
                    },
                    "verbosity": self.verbosity
                },
                "reasoning": {
                    "effort": self.reasoning_effort,
                    "summary": "auto"
                },
                "tools": [],  # Can be extended to support tools
                "store": self.store,
                "include": include_list,
            }
            
            # Add optional parameters if they exist in the API
            if 'max_tokens' in kwargs:
                # max_tokens might be part of text config in the new API
                request_params["text"]["max_tokens"] = kwargs.pop('max_tokens', self.max_tokens)
            if 'temperature' in kwargs:
                request_params["temperature"] = kwargs.pop('temperature', self.temperature)
                
            # Add any remaining kwargs
            request_params.update(kwargs)
            
            response = self.client.responses.create(**request_params)
            
            # Extract the response content
            content = self._extract_content(response)
            
            # Create LangChain compatible response
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            # Fallback to error message
            message = AIMessage(content=f"Error: {str(e)}")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI format."""
        converted = []
        for message in messages:
            if isinstance(message, HumanMessage):
                converted.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                converted.append({
                    "role": "assistant", 
                    "content": message.content
                })
            else:
                # Handle other message types as system messages
                converted.append({
                    "role": "system",
                    "content": str(message.content)
                })
        return converted
    
    def _extract_content(self, response: Any) -> str:
        """Extract content from OpenAI response."""
        try:
            # GPT-5 responses.create() API format
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    # Look for ResponseOutputMessage with assistant role
                    if (hasattr(output_item, 'type') and 
                        output_item.type == 'message' and 
                        hasattr(output_item, 'role') and 
                        output_item.role == 'assistant' and
                        hasattr(output_item, 'content')):
                        
                        # Extract text from content array
                        for content_item in output_item.content:
                            if (hasattr(content_item, 'type') and 
                                content_item.type == 'output_text' and
                                hasattr(content_item, 'text')):
                                return content_item.text
            
            # Fallback: Standard OpenAI chat completion format
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
                
            # Last resort fallback
            return str(response)
            
        except Exception as e:
            logger.error(f"Failed to extract content from response: {e}")
            return f"Response extraction error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "openai_responses"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }


def create_openai_responses_client(
    model: str = "gpt-5",
    max_tokens: int = 50000,
    max_retries: int = 3,
    temperature: float = 0.7,
    reasoning_effort: str = "medium",
    **kwargs
) -> OpenAIResponsesClient:
    """
    Factory function to create an OpenAI responses client.
    
    Args:
        model: The OpenAI model to use (e.g., "gpt-5")
        max_tokens: Maximum tokens in response
        max_retries: Maximum number of API retries
        temperature: Sampling temperature
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
        **kwargs: Additional parameters
    
    Returns:
        OpenAIResponsesClient instance
    """
    return OpenAIResponsesClient(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        **kwargs
    )