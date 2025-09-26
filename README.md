# LLM Prompt Tester - Perplexity Grounded LLM API

A comprehensive Python GUI application for testing prompts with Perplexity's Grounded LLM API, featuring advanced search capabilities, filtering options, and JSON response support.

## Features

### Core Capabilities
- **Multiple Sonar Models**: Test with sonar, sonar-pro, sonar-reasoning, and sonar-deep-research
- **Grounded Web Search**: Real-time web search integration with citation tracking
- **JSON Response Support**: Request and validate JSON-formatted responses
- **Token & Cost Tracking**: Monitor input, output, total tokens, and source citations
- **Response Time Monitoring**: Track API response latency
- **Test Management**: Save, load, and export test configurations
- **Secure API Key Storage**: Store API keys in .env file (git-ignored)

### Advanced Search Parameters
- **Domain Filtering**: Whitelist or blacklist up to 3 domains (use '-' prefix to exclude)
- **Recency Filter**: Filter results by time (hour, day, week, month)
- **Date Range Filtering**: Specify before/after dates in MM/DD/YYYY format
- **Search Context Size**: Control search depth (low, medium, high)
- **Location-Based Search**: Provide latitude, longitude, and country for localized results
- **Return Options**: Toggle images and related questions in responses

### LLM Parameters
- **Temperature Control**: Adjustable creativity (0-2, default 0.2)
- **Max Tokens**: Control response length
- **Top-p Sampling**: Nucleus sampling parameter
- **Frequency & Presence Penalties**: Fine-tune response generation

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

### Basic Setup
1. **Configure API Key**: The app will automatically load your API key from `.env`
2. **Select Model**: Choose from available Sonar models
3. **Enter Prompt**: Type your test prompt in the input field

### Search Configuration
1. **Reference URL** (Optional): Provide a specific webpage URL for context
2. **Domain Filter**: Enter up to 3 domains separated by commas (e.g., `docs.com, -spam.com`)
3. **Recency Filter**: Select time range for search results
4. **Date Filters**: Set before/after dates for precise filtering
5. **Search Context**: Choose between low, medium, or high context retrieval
6. **Additional Options**: Enable images or related questions in responses

### LLM Configuration
1. **Temperature**: Adjust response creativity (default 0.2)
2. **Max Tokens**: Set maximum response length
3. **Advanced Parameters**: Configure top-p, frequency penalty, and presence penalty

### JSON Response
1. **Enable JSON Mode**: Check "Request JSON Response" to receive structured data
2. **Custom Format**: Optionally provide expected JSON structure for validation

### Location Settings
1. **Coordinates**: Enter latitude and longitude for location-based search
2. **Country Code**: Specify ISO country code (e.g., US, UK, FR)

### Running Tests
1. **Run Test**: Click to send request with all configured parameters
2. **View Results**:
   - Main response content
   - Search results with citations
   - Related questions (if enabled)
   - Token usage and response time
3. **Save/Load Tests**: Store and retrieve test configurations for reuse

## File Structure

- `llm_prompt_tester.py` - Main GUI application
- `perplexity_client.py` - Perplexity API client implementation
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

## API Integration

The application leverages Perplexity's Grounded LLM API with comprehensive parameter support:

### Supported Parameters
- **Search Filtering**: Domain filters, recency filters, date ranges
- **Search Options**: Context size, images, related questions
- **Location**: Latitude, longitude, country for localized search
- **LLM Controls**: Temperature, max tokens, top-p, penalties
- **Response Format**: JSON-only responses with validation

### Response Structure
The API returns:
- **Main Content**: The generated response
- **Search Results**: Array of sources with titles, URLs, and dates
- **Related Questions**: Follow-up questions for continued exploration
- **Usage Statistics**: Token counts and costs
- **Citations**: Embedded references to source materials

### Extensibility
The modular architecture with separate `perplexity_client.py` allows easy integration of additional LLM providers or API endpoints.
