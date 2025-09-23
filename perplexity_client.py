import requests
from typing import Dict, Any, Optional, List


class PerplexityAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Perplexity API.

        Args:
            model: The model to use (sonar, sonar-pro, sonar-reasoning, sonar-deep-research)
            messages: List of message dictionaries with 'role' and 'content' keys
            response_format: Optional structured output format specification
            url: Optional URL to reference specific webpage content

        Returns:
            API response as a dictionary
        """
        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages
        }

        if response_format:
            payload["response_format"] = response_format

        if url:
            payload["url"] = url

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Request failed: {str(e)}")