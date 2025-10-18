import requests
from typing import Dict, Any, Optional, List, Union
import json


class OpenAIClient:
    """
    OpenAI API client for GPT-5 model family (GPT-5, GPT-5-mini, GPT-5-nano).
    Supports text generation and structured outputs with JSON schemas.
    """

    def __init__(self, api_key: str):
        """
        Initialize the OpenAI client.

        Args:
            api_key: Your OpenAI API key
        """
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Model information for GPT-5 family
        self.models = {
            "gpt-5": {
                "name": "gpt-5",
                "input_price": 1.25,  # per 1M tokens
                "output_price": 10.0,  # per 1M tokens
                "context_window": 272000,
                "max_output": 128000,
                "knowledge_cutoff": "2024-09-30"
            },
            "gpt-5-mini": {
                "name": "gpt-5-mini",
                "input_price": 0.25,  # per 1M tokens
                "output_price": 2.0,  # per 1M tokens
                "context_window": 272000,
                "max_output": 128000,
                "knowledge_cutoff": "2024-05-30"
            },
            "gpt-5-nano": {
                "name": "gpt-5-nano",
                "input_price": 0.05,  # per 1M tokens
                "output_price": 0.40,  # per 1M tokens
                "context_window": 272000,
                "max_output": 128000,
                "knowledge_cutoff": "2024-05-30"
            }
        }

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        # Basic parameters (not all supported by GPT-5 models)
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        n: Optional[int] = None,
        stream: bool = False,
        # GPT-5 specific parameters
        reasoning_effort: Optional[str] = None,  # "minimal", "low", "medium", "high"
        verbosity: Optional[str] = None,  # "low", "medium", "high"
        # Structured output parameters
        response_format: Optional[Dict[str, Any]] = None,
        # Function calling parameters
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        # Additional parameters
        seed: Optional[int] = None,
        user: Optional[str] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI API.

        Args:
            model: The model to use (e.g., "gpt-5", "gpt-5-mini", "gpt-5-nano")
            messages: List of message dictionaries with 'role' and 'content' keys
                     Can include image inputs for vision capabilities
            temperature: Sampling temperature (0-2), higher = more random
            max_tokens: Maximum tokens in response (up to model's limit)
            top_p: Nucleus sampling parameter (0-1)
            frequency_penalty: Reduce repetition of token sequences (-2 to 2)
            presence_penalty: Reduce repetition of topics (-2 to 2)
            stop: Stop sequence(s) to end generation
            n: Number of completions to generate
            stream: Whether to stream the response
            reasoning_effort: GPT-5 specific - control reasoning depth
            verbosity: GPT-5 specific - control response detail level
            response_format: JSON schema for structured outputs
            tools: List of available function/tool definitions
            tool_choice: Control function calling behavior
            parallel_tool_calls: Enable/disable parallel function calls
            seed: For deterministic outputs
            user: Unique user identifier for abuse monitoring
            logit_bias: Modify likelihood of specific tokens
            logprobs: Return log probabilities of output tokens
            top_logprobs: Number of most likely tokens to return

        Returns:
            API response as a dictionary
        """
        endpoint = f"{self.base_url}/chat/completions"

        # Check if this is a GPT-5 model (reasoning model)
        is_gpt5_model = model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]

        # Build the request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        # Add standard parameters if provided (but skip unsupported ones for GPT-5)
        if temperature is not None:
            if is_gpt5_model:
                # GPT-5 models only support default temperature of 1
                if temperature != 1.0:
                    print(f"Warning: GPT-5 models only support temperature=1.0, ignoring provided value of {temperature}")
                # Don't add temperature parameter for GPT-5 models (uses default)
            else:
                payload["temperature"] = temperature

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # These parameters are NOT supported by GPT-5 models
        if not is_gpt5_model:
            if top_p is not None:
                payload["top_p"] = top_p

            if frequency_penalty is not None:
                payload["frequency_penalty"] = frequency_penalty

            if presence_penalty is not None:
                payload["presence_penalty"] = presence_penalty

            if logit_bias is not None:
                payload["logit_bias"] = logit_bias

            if logprobs is not None:
                payload["logprobs"] = logprobs

            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs
        else:
            # Warn if unsupported parameters are provided for GPT-5
            if top_p is not None:
                print(f"Warning: top_p is not supported by {model}, parameter ignored")
            if frequency_penalty is not None:
                print(f"Warning: frequency_penalty is not supported by {model}, parameter ignored")
            if presence_penalty is not None:
                print(f"Warning: presence_penalty is not supported by {model}, parameter ignored")
            if logit_bias is not None:
                print(f"Warning: logit_bias is not supported by {model}, parameter ignored")
            if logprobs is not None:
                print(f"Warning: logprobs is not supported by {model}, parameter ignored")
            if top_logprobs is not None:
                print(f"Warning: top_logprobs is not supported by {model}, parameter ignored")

        if stop is not None:
            payload["stop"] = stop

        if n is not None:
            payload["n"] = n

        # GPT-5 specific parameters
        if reasoning_effort is not None:
            if reasoning_effort in ["minimal", "low", "medium", "high"]:
                payload["reasoning_effort"] = reasoning_effort
            else:
                raise ValueError(f"Invalid reasoning_effort: {reasoning_effort}. Must be 'minimal', 'low', 'medium', or 'high'")

        if verbosity is not None:
            if verbosity in ["low", "medium", "high"]:
                payload["verbosity"] = verbosity
            else:
                raise ValueError(f"Invalid verbosity: {verbosity}. Must be 'low', 'medium', or 'high'")

        # Structured output support
        if response_format is not None:
            # Support both simple JSON and JSON schema formats
            if response_format.get("type") == "json_object":
                # Simple JSON mode
                payload["response_format"] = {"type": "json_object"}
            elif response_format.get("type") == "json_schema":
                # Structured output with schema - pass through the entire response_format
                payload["response_format"] = response_format
            elif response_format.get("type") == "text":
                # Default text format
                payload["response_format"] = {"type": "text"}
            else:
                # If no type specified, assume it's the complete response_format object
                payload["response_format"] = response_format

        # Function calling / Tools support
        if tools is not None:
            payload["tools"] = tools

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if parallel_tool_calls is not None:
            # Note: Structured Outputs are not compatible with parallel function calls
            payload["parallel_tool_calls"] = parallel_tool_calls

        # Additional parameters
        if seed is not None:
            payload["seed"] = seed

        if user is not None:
            payload["user"] = user

        # Note: logit_bias, logprobs, and top_logprobs are handled above based on model type

        # Make the API request
        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Handle API errors with detailed information
            error_message = f"OpenAI API request failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message += f": {error_data['error'].get('message', 'Unknown error')}"
                    error_type = error_data['error'].get('type', 'unknown')
                    error_message += f" (Type: {error_type})"
            except:
                error_message += f": {e.response.text}"
            raise Exception(error_message)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error during API request: {str(e)}")

    def create_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        schema_name: str = "response",
        schema_description: Optional[str] = None,
        strict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method for creating structured outputs with JSON schemas.

        Args:
            model: The model to use
            messages: List of message dictionaries
            json_schema: The JSON schema that the output must conform to
            schema_name: Name for the schema (default: "response")
            schema_description: Optional description of the schema
            strict: Whether to enforce strict schema compliance (default: True)
            **kwargs: Additional parameters to pass to chat_completion

        Returns:
            API response with structured output
        """
        # Build the response format for structured output
        # The schema should be directly under json_schema, not wrapped in a "schema" field
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": strict,
                **json_schema  # Spread the schema properties directly
            }
        }

        if schema_description:
            response_format["json_schema"]["description"] = schema_description

        # Disable parallel tool calls for structured outputs
        if "parallel_tool_calls" in kwargs:
            kwargs["parallel_tool_calls"] = False

        return self.chat_completion(
            model=model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )

    def create_function_call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method for function calling.

        Args:
            model: The model to use
            messages: List of message dictionaries
            functions: List of function definitions
            function_call: Control which function to call ("auto", "none", or specific function)
            **kwargs: Additional parameters to pass to chat_completion

        Returns:
            API response with function call
        """
        # Convert functions to tools format (OpenAI's newer format)
        tools = [{"type": "function", "function": func} for func in functions]

        # Convert function_call to tool_choice format
        tool_choice = None
        if function_call is not None:
            if function_call == "auto":
                tool_choice = "auto"
            elif function_call == "none":
                tool_choice = "none"
            elif isinstance(function_call, dict):
                tool_choice = {
                    "type": "function",
                    "function": {"name": function_call["name"]}
                }

        return self.chat_completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )

    def parse_structured_response(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse structured output from API response.

        Args:
            response: The API response dictionary

        Returns:
            Parsed JSON content or None if refusal detected
        """
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            message = choice.get("message", {})

            # Check for refusal
            if message.get("refusal"):
                return None

            # Get the content
            content = message.get("content", "")

            # Try to parse as JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, return as-is
                return {"content": content}

        return None

    def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model: The model name

        Returns:
            Dictionary with model information or None if not found
        """
        return self.models.get(model)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Estimate the cost for a given number of tokens.

        Args:
            model: The model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with cost breakdown
        """
        model_info = self.get_model_info(model)
        if not model_info:
            raise ValueError(f"Unknown model: {model}")

        input_cost = (input_tokens / 1_000_000) * model_info["input_price"]
        output_cost = (output_tokens / 1_000_000) * model_info["output_price"]

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }