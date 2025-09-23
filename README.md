# LLM Prompt Tester - Perplexity Sonar API

A Python GUI application for testing prompts with Perplexity's Sonar API models and validating JSON responses.

## Features

- **Multiple Sonar Models**: Test with sonar, sonar-pro, sonar-reasoning, and sonar-deep-research
- **JSON Validation**: Define expected JSON schemas and validate responses
- **Token Tracking**: Monitor input, output, and total token usage
- **Response Time**: Track API response times
- **Test Management**: Save and load test cases
- **Export Results**: Export test history to JSON
- **Secure API Key Storage**: Store API keys in .env file (git-ignored)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

3. Add your Perplexity API key to `.env`:

```
PERPLEXITY_API_KEY=your_actual_api_key_here
```

4. Run the application:

```bash
python llm_prompt_tester.py
```

## Usage

1. **Configure API Key**: The app will automatically load your API key from `.env`
2. **Select Model**: Choose from available Sonar models
3. **Enter Prompt**: Type your test prompt in the input field
4. **Set Expected JSON** (Optional): Define a JSON schema for validation
5. **Add System Prompt** (Optional): Customize the system message
6. **Run Test**: Click "Run Test" to send the request
7. **View Results**: See the response, token usage, and validation status

## File Structure

- `llm_prompt_tester.py` - Main application
- `.env` - API key storage (git-ignored)
- `.gitignore` - Excludes sensitive files from git
- `requirements.txt` - Python dependencies

## Security Note

- Never commit your `.env` file to git
- The `.gitignore` file is configured to exclude all sensitive files
- Keep your API key secure and rotate it regularly

## Available Models

- **sonar**: Fast, general-purpose model
- **sonar-pro**: Enhanced model with more citations and context
- **sonar-reasoning**: Multi-step reasoning capabilities
- **sonar-deep-research**: Comprehensive research reports
