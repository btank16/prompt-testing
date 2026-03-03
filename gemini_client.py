from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.models = {
            "gemini-3-pro-preview": {
                "name": "Gemini 3 Pro Preview",
                "input_price_per_1m": 1.25,
                "output_price_per_1m": 10.0,
                "thinking_levels": ["none", "low", "medium", "high"],
                "rpm_limit": 150,
            },
            "gemini-3-flash-preview": {
                "name": "Gemini 3 Flash Preview",
                "input_price_per_1m": 0.15,
                "output_price_per_1m": 0.60,
                "thinking_levels": ["none", "low", "medium", "high"],
                "rpm_limit": 500,
            },
        }

    def get_model_info(self, model: str) -> dict:
        return self.models.get(model, {})

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        info = self.models.get(model, {})
        input_cost = (input_tokens / 1_000_000) * info.get("input_price_per_1m", 0)
        output_cost = (output_tokens / 1_000_000) * info.get("output_price_per_1m", 0)
        return input_cost + output_cost

    def generate_content(
        self,
        model: str,
        prompt: str,
        system_prompt: str = None,
        json_schema: dict = None,
        temperature: float = None,
        top_p: float = None,
        max_output_tokens: int = None,
        thinking_level: str = None,
        use_google_search: bool = False,
        use_url_context: bool = False,
    ) -> dict:
        config_kwargs = {}

        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if top_p is not None:
            config_kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        if json_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema

        if thinking_level and thinking_level != "none":
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level.upper()
            )

        # Grounding tools
        tools = []
        if use_google_search:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        if use_url_context:
            tools.append(types.Tool(url_context=types.UrlContext))
        if tools:
            config_kwargs["tools"] = tools

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            usage = {}
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0,
                }

            return {
                "content": response.text,
                "usage": usage,
                "model": model,
            }

        except Exception as e:
            raise RuntimeError(f"Gemini API error ({model}): {str(e)}") from e
