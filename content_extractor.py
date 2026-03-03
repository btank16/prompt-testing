"""Content extraction module for websites (crawl4ai) and PDFs (Docling).

Both tools run locally — no API keys required.
"""

import asyncio
import tempfile
import os
import time


# ---------------------------------------------------------------------------
# Website extraction via crawl4ai
# ---------------------------------------------------------------------------

def extract_website(url: str, timeout: int = 60) -> dict:
    """Extract content from a webpage using crawl4ai.

    Returns dict with keys:
        content (str): Extracted Markdown text.
        title (str): Page title.
        source_url (str): The URL extracted from.
        content_type (str): Always "website".
        word_count (int): Word count of extracted content.
        success (bool): Whether extraction succeeded.
        error (str | None): Error message if failed.
        elapsed (float): Seconds taken.
    """
    start = time.time()
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        result = asyncio.run(
            _async_extract_website(url, timeout, BrowserConfig, CrawlerRunConfig, AsyncWebCrawler)
        )
        result["elapsed"] = round(time.time() - start, 2)
        return result
    except ImportError:
        return _error_result(url, "website", time.time() - start,
                             "crawl4ai is not installed. Run: pip install crawl4ai && crawl4ai-setup")
    except Exception as e:
        return _error_result(url, "website", time.time() - start, str(e))


async def _async_extract_website(url, timeout, BrowserConfig, CrawlerRunConfig, AsyncWebCrawler):
    browser_config = BrowserConfig(headless=True)
    crawler_config = CrawlerRunConfig(
        word_count_threshold=10,
        excluded_tags=["nav", "footer", "header"],
        exclude_external_links=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawler_config)

        if result.success:
            content = result.markdown or ""
            title = ""
            if result.metadata and isinstance(result.metadata, dict):
                title = result.metadata.get("title", "")
            return {
                "content": content,
                "title": title,
                "source_url": url,
                "content_type": "website",
                "word_count": len(content.split()),
                "success": True,
                "error": None,
            }
        else:
            error_msg = getattr(result, "error_message", None) or "Extraction failed"
            return {
                "content": "",
                "title": "",
                "source_url": url,
                "content_type": "website",
                "word_count": 0,
                "success": False,
                "error": error_msg,
            }


# ---------------------------------------------------------------------------
# PDF extraction via Docling
# ---------------------------------------------------------------------------

MAX_PDF_SIZE_MB = 50


def extract_pdf_from_upload(file_bytes: bytes, filename: str = "upload.pdf") -> dict:
    """Extract content from uploaded PDF bytes using Docling.

    Returns dict with keys:
        content (str): Extracted Markdown text.
        title (str): Filename.
        source_url (str | None): None for uploads.
        content_type (str): Always "pdf".
        page_count (int): Number of pages.
        word_count (int): Word count of extracted content.
        success (bool): Whether extraction succeeded.
        error (str | None): Error message if failed.
        elapsed (float): Seconds taken.
    """
    start = time.time()

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        return _error_result(
            filename, "pdf", time.time() - start,
            f"File too large ({size_mb:.1f}MB). Maximum is {MAX_PDF_SIZE_MB}MB."
        )

    tmp_path = None
    try:
        from docling.document_converter import DocumentConverter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        result = _run_docling(tmp_path, DocumentConverter)
        result["title"] = filename
        result["source_url"] = None
        result["elapsed"] = round(time.time() - start, 2)
        return result
    except ImportError:
        return _error_result(filename, "pdf", time.time() - start,
                             "docling is not installed. Run: pip install docling")
    except Exception as e:
        return _error_result(filename, "pdf", time.time() - start, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_pdf_from_url(url: str) -> dict:
    """Extract content from a PDF at a URL using Docling.

    Docling supports URL input directly.
    """
    start = time.time()
    try:
        from docling.document_converter import DocumentConverter

        result = _run_docling(url, DocumentConverter)
        result["title"] = url.rsplit("/", 1)[-1] if "/" in url else url
        result["source_url"] = url
        result["elapsed"] = round(time.time() - start, 2)
        return result
    except ImportError:
        return _error_result(url, "pdf", time.time() - start,
                             "docling is not installed. Run: pip install docling")
    except Exception as e:
        return _error_result(url, "pdf", time.time() - start, str(e))


def _run_docling(source, DocumentConverter):
    """Run Docling conversion and return a partial result dict."""
    converter = DocumentConverter()
    conv_result = converter.convert(source)
    content = conv_result.document.export_to_markdown()

    page_count = 0
    if hasattr(conv_result.document, "pages"):
        page_count = len(conv_result.document.pages)

    return {
        "content": content,
        "content_type": "pdf",
        "page_count": page_count,
        "word_count": len(content.split()),
        "success": True,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(source: str, content_type: str, elapsed: float, error: str) -> dict:
    return {
        "content": "",
        "title": "",
        "source_url": source,
        "content_type": content_type,
        "page_count": 0,
        "word_count": 0,
        "success": False,
        "error": error,
        "elapsed": round(elapsed, 2),
    }
