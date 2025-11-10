import fitz  # PyMuPDF
import re
import contextlib
import os

def is_mostly_alphabetic(text, threshold=0.8):
    """Check if more than `threshold` proportion of characters in text are alphabetical."""
    if not text.strip():
        return False
    alphabetic_count = sum(c.isalpha() for c in text)
    total_count = len(text.replace(" ", ""))  # Exclude spaces from total count
    if total_count == 0:
        return False
    return alphabetic_count / total_count > threshold

def count_words(text):
    """Count the number of words in the text."""
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def remove_citations(text):
    """Remove citations like [1], [1-2], [1,2], (Wang et al., 2023e), or (OpenAI, 2023; Reid et al., 2024)."""
    # Remove bracketed citations like [1], [1-2], [1,2], [1, 2, 3]
    text = re.sub(r'\[\d+(?:-\d+)?(?:,\s*\d+)*\]', '', text)
    # Remove parenthetical citations like (Wang et al., 2023e) or (OpenAI, 2023; Reid et al., 2024)
    # text = re.sub(r'\([^()]+?,\s*\d{4}[a-z]?(?:;\s*[^()]+?,\s*\d{4}[a-z]?)*\)', '', text)
    # Using a simpler, more efficient pattern to avoid catastrophic backtracking
    text = re.sub(r'\([^()]*\d{4}[a-z]?[^()]*\)', '', text)
    # Clean up extra spaces left by citation removal
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_reference(text):
    """Check if a paragraph is likely a reference based on common patterns."""
    # Common reference indicators
    reference_patterns = [
        r'\b\d{4}\b.*(?:Journal|Proceedings|Conference|CoRR|arXiv|NeurIPS|Trans-?\s*actions)\b',  # Year and publication terms
        r'\b(?:https?://|doi:|URL\s+https?://)',  # URLs or DOIs
        r'\b(?:Vol\.|pp\.|eds\.)',  # Volume, pages, or editors
        r'[^.]*\b\d{4}[a-z]?\.\s*$',  # Ends with a year (e.g., 2023.)
        r'(?:[A-Z]\.\s*){2,}|(?:\w+,\s*[A-Z]\.\s*){3,}'  # Multiple initials or author names with commas
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in reference_patterns)

def chunk_pdf(pdf_path, min_words=30, alphabetic_threshold=0.8) -> list[str]:
    """Extract and filter paragraphs from a PDF, removing citations and references."""
    try:
        # Suppress MuPDF stderr warnings
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                doc = fitz.open(pdf_path)
                paragraphs = []

                for page in doc:
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        text = block[4].strip()
                        if not text:
                            continue
                        if is_reference(text):
                            continue
                        text = remove_citations(text)
                        if count_words(text) >= min_words and is_mostly_alphabetic(text, alphabetic_threshold):
                            paragraphs.append(text)

                doc.close()
        return paragraphs

    except Exception:
        # Silently handle errors - return empty list
        return []
