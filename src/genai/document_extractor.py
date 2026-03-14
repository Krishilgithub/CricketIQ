"""
Document Extractor — Extract structured info from PDFs, URLs, and text documents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import GEMINI_API_KEY


def extract_from_text(text: str, extraction_type: str = "summary") -> str:
    """
    Extract structured information from text using Gemini.

    Args:
        text: Raw text content
        extraction_type: 'summary', 'stats', 'players', 'match_report'
    """
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY not set."

    try:
        import google.generativeai as genai
    except ImportError:
        return "❌ google-generativeai not installed."

    prompts = {
        "summary": f"""Summarize the following cricket-related document concisely,
highlighting key facts, statistics, and outcomes:

{text[:5000]}

Summary:""",
        "stats": f"""Extract all cricket statistics from the following text.
Format as a structured list with categories (batting, bowling, fielding, match results):

{text[:5000]}

Statistics:""",
        "players": f"""Extract all player names and their associated statistics
from the following text. Format as a table:

{text[:5000]}

Players:""",
        "match_report": f"""Extract a structured match report from the following text:
- Match: (teams, venue, date)
- Result: (winner, margin)
- Top Performers: (batting, bowling)
- Key Highlights:

{text[:5000]}

Match Report:""",
    }

    prompt = prompts.get(extraction_type, prompts["summary"])

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text


def extract_from_url(url: str, extraction_type: str = "summary") -> str:
    """Fetch URL content and extract information."""
    try:
        import requests
        from html.parser import HTMLParser

        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Cricket Stats Bot)"
        })
        resp.raise_for_status()

        # Simple HTML text extraction
        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self.text.append(stripped)

        extractor = TextExtractor()
        extractor.feed(resp.text)
        text = " ".join(extractor.text)

        return extract_from_text(text, extraction_type)

    except Exception as e:
        return f"Error fetching URL: {e}"


def extract_from_pdf(pdf_path: str, extraction_type: str = "summary") -> str:
    """Extract text from PDF and analyze."""
    try:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        except ImportError:
            return "❌ PyPDF2 not installed. pip install PyPDF2"

        if not text.strip():
            return "⚠️ Could not extract text from PDF."

        return extract_from_text(text, extraction_type)

    except Exception as e:
        return f"Error reading PDF: {e}"
