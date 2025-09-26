import requests
from typing import Dict, Any, Optional, List


class PerplexityAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        url: Optional[str] = None,
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,
        search_after_date_filter: Optional[str] = None,
        search_before_date_filter: Optional[str] = None,
        search_context_size: Optional[str] = None,
        return_images: Optional[bool] = None,
        return_related_questions: Optional[bool] = None,
        user_location: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Perplexity Grounded LLM API.

        Args:
            model: The model to use (sonar, sonar-pro, sonar-reasoning, sonar-deep-research)
            messages: List of message dictionaries with 'role' and 'content' keys
            response_format: Optional JSON response format (only JSON type supported)
            url: Optional URL to reference specific webpage content
            search_domain_filter: List of domains to include/exclude (max 3, prefix with '-' to exclude)
            search_recency_filter: Filter by time ('hour', 'day', 'week', 'month')
            search_after_date_filter: Filter results after date (MM/DD/YYYY format)
            search_before_date_filter: Filter results before date (MM/DD/YYYY format)
            search_context_size: Amount of search context ('low', 'medium', 'high')
            return_images: Whether to include images in response
            return_related_questions: Whether to return related follow-up questions
            user_location: Location for localized search (lat, lon, country)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stream: Whether to stream the response

        Returns:
            API response as a dictionary
        """
        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        # Add optional parameters only if provided
        if response_format:
            # Support both simple JSON and JSON schema formats
            if response_format.get("type") in ["json", "json_schema"]:
                payload["response_format"] = response_format

        if url:
            payload["url"] = url

        # Search filtering parameters
        if search_domain_filter is not None:
            # Limit to 3 domains as per API restrictions
            payload["search_domain_filter"] = search_domain_filter[:3]

        if search_recency_filter:
            if search_recency_filter in ['hour', 'day', 'week', 'month']:
                payload["search_recency_filter"] = search_recency_filter

        # Date filtering
        if search_after_date_filter:
            payload["search_after_date_filter"] = search_after_date_filter

        if search_before_date_filter:
            payload["search_before_date_filter"] = search_before_date_filter

        # Search options
        if search_context_size:
            if search_context_size in ['low', 'medium', 'high']:
                payload["search_context_size"] = search_context_size

        if return_images is not None:
            payload["return_images"] = return_images

        if return_related_questions is not None:
            payload["return_related_questions"] = return_related_questions

        # Location for localized search
        if user_location:
            payload["user_location"] = user_location

        # Standard LLM parameters
        if temperature is not None:
            payload["temperature"] = temperature

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if top_p is not None:
            payload["top_p"] = top_p

        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty

        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Request failed: {str(e)}")