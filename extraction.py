# Text extraction from files, URL, and Anki export.

import csv
import html
import ipaddress
import os
import re
import socket
import sqlite3
import tempfile
from io import StringIO
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

import constants
from errors import ErrorHandler
from resources import get_file_parsers
from utils import detect_file_encoding

logger = __import__("logging").getLogger(__name__)

_RE_CLOZE = re.compile(r'\{\{c\d+::(.*?)(?::.*?)?\}\}')
_RE_HTML  = re.compile(r'<[^>]+>')
_RE_SOUND = re.compile(r'\[sound:.*?\]')
_RE_IMAGE = re.compile(r'\[Image:.*?\]')


def clean_anki_field(text: str) -> str:
    """Helper to clean a single Anki field content."""
    text = _RE_CLOZE.sub(r'\1', text)
    text = _RE_HTML.sub(' ', text)
    text = html.unescape(text)
    text = _RE_SOUND.sub('', text)
    text = _RE_IMAGE.sub('', text)
    return text.strip()


def _extract_df_columns_text(df: pd.DataFrame) -> list:
    """Flatten non-empty string values from all columns of a DataFrame."""
    parts = []
    for col in df.columns:
        try:
            col_text = df[col].astype(str)
        except Exception:
            col_text = df[col].apply(lambda x: str(x) if x is not None else "")
        col_text = col_text[col_text.notna() & (col_text != '') & (col_text != 'nan')]
        parts.extend(col_text.tolist())
    return parts


def parse_anki_txt_export(uploaded_file: Any) -> str:
    """Robustly parse Anki export txt files."""
    try:
        bytes_data = uploaded_file.getvalue()
        encoding = detect_file_encoding(bytes_data)
        content = bytes_data.decode(encoding, errors='ignore')

        extracted_words = []
        f_io = StringIO(content)
        reader = csv.reader(f_io, delimiter='\t')

        for row in reader:
            if not row:
                continue
            if row[0].startswith('#'):
                continue

            target_text = ""
            if len(row) > 1 and (re.match(r'^\[sound:.*\]$', row[0].strip()) or row[0].strip().isdigit()):
                target_text = row[1]
            else:
                target_text = row[0]

            clean_text = clean_anki_field(target_text)
            clean_word = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', clean_text)
            if clean_word and len(clean_word) >= 2:
                extracted_words.append(clean_word)

        return "\n".join(extracted_words)

    except Exception as e:
        return ErrorHandler.handle_file_error(e, "Anki Export Import")


def _hostname_resolves_to_public_ip(hostname: str) -> bool:
    """Return True only when hostname resolves and every IP is globally routable."""
    h = (hostname or "").strip().strip("[]")
    if not h:
        return False
    try:
        ip_obj = ipaddress.ip_address(h)
        return ip_obj.is_global
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False

    resolved = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_text = sockaddr[0]
        try:
            resolved.add(ipaddress.ip_address(ip_text))
        except ValueError:
            continue
    if not resolved:
        return False
    return all(ip_obj.is_global for ip_obj in resolved)


def _is_safe_url(url: str) -> bool:
    """Block URLs pointing to localhost/private IP space or non-HTTP schemes."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = (parsed.hostname or "").lower()
    blocked = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "metadata.google")
    for b in blocked:
        if hostname == b or hostname.endswith("." + b):
            return False
    return _hostname_resolves_to_public_ip(hostname)


def _safe_get_with_redirects(
    url: str,
    headers: dict,
    timeout: int,
    max_redirects: int = 5,
) -> requests.Response:
    """Fetch URL while validating every redirect target."""
    current_url = url
    for _ in range(max_redirects + 1):
        if not _is_safe_url(current_url):
            raise requests.RequestException("URL blocked for security reasons (private/local address).")

        response = requests.get(
            current_url,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location", "").strip()
            if not location:
                return response
            current_url = urljoin(current_url, location)
            continue
        return response

    raise requests.RequestException(f"Too many redirects (>{max_redirects})")


def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL."""
    if not _is_safe_url(url):
        return "Error: URL blocked for security reasons (private/local address)."

    _, _, _, _, BeautifulSoup = get_file_parsers()

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = _safe_get_with_redirects(
            url,
            headers=headers,
            timeout=constants.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        # Guard against downloading enormous pages
        max_content = constants.MAX_UPLOAD_BYTES
        if len(response.content) > max_content:
            return f"Error: page too large ({len(response.content)} bytes)"

        soup = BeautifulSoup(response.content, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)

        # If extracted text is very short, try Jina Reader as fallback (handles JS-rendered pages)
        if len(text.split()) < 100:
            try:
                jina_url = f"https://r.jina.ai/{url}"
                jina_resp = requests.get(
                    jina_url,
                    headers={"Accept": "text/plain"},
                    timeout=constants.REQUEST_TIMEOUT_SECONDS,
                )
                jina_resp.raise_for_status()
                if len(jina_resp.text.split()) > len(text.split()):
                    return jina_resp.text
            except Exception:
                pass  # Fall through to original text

        return text
    except requests.RequestException as e:
        return ErrorHandler.handle_file_error(e, "URL")
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "URL parsing")


def extract_from_txt(uploaded_file: Any) -> str:
    """Extract text from TXT file."""
    bytes_data = uploaded_file.getvalue()
    encoding = detect_file_encoding(bytes_data)

    try:
        return bytes_data.decode(encoding)
    except UnicodeDecodeError:
        logger.warning("Decode failed with %s, trying latin-1", encoding)
        return bytes_data.decode('latin-1', errors='ignore')


def extract_from_pdf(uploaded_file: Any) -> str:
    """Extract text from PDF file. Only first PDF_MAX_PAGES pages to keep speed."""
    pypdf, _, _, _, _ = get_file_parsers()

    try:
        reader = pypdf.PdfReader(uploaded_file)
        pages = reader.pages[: constants.PDF_MAX_PAGES]
        pages_text = []
        for p in pages:
            txt = p.extract_text()
            if txt:
                pages_text.append(txt)
        return "\n".join(pages_text)
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "PDF")


def extract_from_docx(uploaded_file: Any) -> str:
    """Extract text from DOCX file."""
    _, docx, _, _, _ = get_file_parsers()

    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "DOCX")


def extract_from_epub(uploaded_file: Any) -> str:
    """Extract text from EPUB file."""
    _, _, ebooklib, epub, BeautifulSoup = get_file_parsers()

    try:
        book = epub.read_epub(uploaded_file)
        text_parts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_parts.append(soup.get_text(separator=' ', strip=True))
        return "\n".join(text_parts)
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "EPUB")


def extract_from_csv(uploaded_file: Any) -> str:
    """Extract text from CSV file."""
    bytes_data = uploaded_file.getvalue()
    encoding = detect_file_encoding(bytes_data)

    try:
        content = bytes_data.decode(encoding)
        df = pd.read_csv(StringIO(content))
        if df.empty or len(df.columns) == 0:
            return "Error: CSV 文件为空或无数据行。"
        text_parts = _extract_df_columns_text(df)
        return " ".join(text_parts)
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "CSV")


def extract_from_excel(uploaded_file: Any) -> str:
    """Extract text from Excel file."""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')
    except Exception:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=None, engine='xlrd')
        except Exception as e:
            return ErrorHandler.handle_file_error(e, "Excel")

    try:
        if not df:
            return "Error: Excel 文件无法读取任何工作表。"
        text_parts = []
        for sheet_name, sheet_df in df.items():
            if sheet_df.empty:
                continue
            text_parts.extend(_extract_df_columns_text(sheet_df))
        return " ".join(text_parts)
    except Exception as e:
        return ErrorHandler.handle_file_error(e, "Excel")


def extract_from_sqlite(uploaded_file: Any) -> str:
    """Extract text from SQLite database file."""
    tmp_db_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_db:
            tmp_db.write(uploaded_file.getvalue())
            tmp_db_path = tmp_db.name

        with sqlite3.connect(tmp_db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT stem FROM WORDS WHERE stem IS NOT NULL")
                rows = cursor.fetchall()
                text = " ".join([row[0] for row in rows if row[0]])

                if not text:
                    cursor.execute("SELECT word FROM WORDS")
                    rows = cursor.fetchall()
                    text = " ".join([row[0] for row in rows if row[0]])

                return text
            except sqlite3.OperationalError as e:
                return f"Error reading DB schema: {e}"

    except Exception as e:
        return ErrorHandler.handle_file_error(e, "SQLite DB")
    finally:
        if tmp_db_path and os.path.exists(tmp_db_path):
            try:
                os.remove(tmp_db_path)
            except OSError as e:
                logger.warning("Could not remove temp DB: %s", e)


def extract_text_from_file(uploaded_file: Any) -> str:
    """Main function to extract text from uploaded files."""
    file_type = uploaded_file.name.split('.')[-1].lower()

    extractors = {
        'txt': extract_from_txt,
        'pdf': extract_from_pdf,
        'docx': extract_from_docx,
        'epub': extract_from_epub,
        'db': extract_from_sqlite,
        'sqlite': extract_from_sqlite,
        'csv': extract_from_csv,
        'xlsx': extract_from_excel,
        'xls': extract_from_excel,
    }

    extractor = extractors.get(file_type)
    if extractor:
        return extractor(uploaded_file)

    return f"Unsupported file type: {file_type}"


def is_upload_too_large(uploaded_file: Any) -> bool:
    """Check if uploaded file exceeds size limit."""
    if not uploaded_file:
        return False
    size = getattr(uploaded_file, "size", None)
    if size is None:
        return False
    return size > constants.MAX_UPLOAD_BYTES
