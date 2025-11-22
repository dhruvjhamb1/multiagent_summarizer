import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException
import aiofiles
import pdfplumber
import PyPDF2
from ..config import settings


logger = logging.getLogger(__name__)

def validate_file(file: UploadFile) -> bool:
    """
    Validate uploaded file type and size.
    Raises HTTPException for invalid files.
    """
    # Check file extension
    allowed_extensions = {'.pdf', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and TXT files are allowed."
        )

    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    if hasattr(file, 'size') and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB."
        )

    return True

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using pdfplumber as primary method,
    with PyPDF2 as fallback. Handles corrupted PDFs gracefully.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info(f"Successfully extracted text from PDF: {file_path}")
        return text.strip()
    except Exception as e:
        logger.warning(f"pdfplumber failed for {file_path}: {e}, trying PyPDF2")
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"Successfully extracted text from PDF using PyPDF2: {file_path}")
            return text.strip()
        except Exception as e2:
            logger.error(f"Both pdfplumber and PyPDF2 failed for {file_path}: {e2}")
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from PDF file. The file may be corrupted or password-protected."
            )

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from TXT file with multiple encoding attempts.
    Handles special characters.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            logger.info(f"Successfully read TXT file with encoding {encoding}: {file_path}")
            return content
        except UnicodeDecodeError:
            logger.debug(f"Failed to read with encoding {encoding}, trying next...")
            continue
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            raise HTTPException(
                status_code=400,
                detail="Failed to read text file."
            )

    # If all encodings fail
    raise HTTPException(
        status_code=400,
        detail="Failed to read text file with supported encodings (UTF-8, Latin-1, CP1252)."
    )

async def save_uploaded_file(file: UploadFile, document_id: str) -> str:
    """
    Save uploaded file to uploads directory using aiofiles.
    Returns the file path.
    """
    upload_dir = Path(settings.storage_path) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename
    safe_filename = f"{document_id}_{file.filename}"
    file_path = upload_dir / safe_filename

    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        logger.info(f"Successfully saved file: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file."
        )