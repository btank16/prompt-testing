# LLM Prompt Tester

A Streamlit web application for testing and comparing prompts across multiple LLM providers — Perplexity, OpenAI, and Google Gemini — with batch processing, JSON schema support, and content extraction from websites and PDFs.

## Features

### Multi-Provider Support
- **Perplexity** (Sonar models): Grounded web search with citation tracking, domain filtering, recency filters, date ranges, location-based search
- **OpenAI** (GPT-5 family): Structured outputs, function calling, reasoning effort control
- **Google Gemini**: Google Search grounding, URL context, configurable thinking levels

### Batch Processing
- Template variables (`{city}`, `{country}`, etc.) with CSV upload or manual entry
- Concurrent API execution with configurable workers and retry logic
- Results displayed as DataFrames with export to CSV/TSV

### JSON Schema Support
- Define response schemas for structured JSON output
- Provider-specific schema wrapping handled automatically
- JSON validation and expansion of parallel arrays into rows

### Content Extraction
- **Website extraction** via crawl4ai — extracts web pages as clean Markdown
- **PDF extraction** via Docling — AI-powered layout analysis, table reconstruction, OCR support
- Extract standalone or inject content into prompt templates as variables
- Export extracted content as Markdown or plain text

### LLM Parameters
- Temperature, top-p, max tokens, frequency/presence penalties
- Provider-specific controls (reasoning effort, verbosity, thinking level)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
crawl4ai-setup  # one-time browser setup for web extraction
```

2. Create a `.env` file in the project root with your API keys:

```
PERPLEXITY_API_KEY=your_perplexity_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

3. Run the application:

```bash
streamlit run app.py
```

Or use the launch script (macOS):

```bash
./launch.command
```

## Usage

1. **Connect API Keys**: Keys auto-load from `.env`, or enter them in the sidebar
2. **Select a Tab**: Gemini Batch, Perplexity, OpenAI, or Content Extraction
3. **Configure Prompt**: Enter system prompt, prompt template with `{variables}`, and optional JSON schema
4. **Add Variables**: Upload CSV or manually enter variable values for batch runs
5. **Run**: Execute batch with progress tracking and retry logic
6. **Export**: Download results as CSV/TSV

### Content Extraction Workflow
1. Go to the **Content Extraction** tab
2. Choose source: Website URL, PDF upload, or PDF URL
3. Click Extract to get Markdown content
4. Optionally click **Send to Tab** to inject content into a provider's variable table
5. Reference it in your prompt template with `{extracted_content}`

## File Structure

- `app.py` - Streamlit web application (main entry point)
- `gemini_client.py` - Google Gemini API client
- `perplexity_client.py` - Perplexity API client
- `openai_client.py` - OpenAI API client
- `content_extractor.py` - Website (crawl4ai) and PDF (Docling) extraction
- `batch_helpers.py` - Batch execution, retry logic, results rendering
- `utils.py` - Variable detection/substitution, JSON normalization, response formatting
- `Good_prompts/` - Saved prompt configurations (JSON)
- `launch.command` - macOS launch script
- `.env` - API key storage (git-ignored)

## Security Note

- Never commit your `.env` file to git
- The `.gitignore` is configured to exclude all sensitive files
- API keys are masked in the UI with password inputs
