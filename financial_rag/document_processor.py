"""
financial_rag/document_processor.py
===================================
Document processing and ingestion utilities for financial PDFs.
Supports:
  1. PDF page count detection
  2. Memory-efficient page streaming as PIL Images (for multimodal embeddings)
  3. Text extraction and chunking (for text retrieval and DeepEval evaluation)
  4. Base64 encoding for vision models and UI rendering
"""

import os
import gc
import base64
from io import BytesIO
from PIL import Image
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


def get_pdf_page_count(pdf_path: str) -> int:
    """Returns the total number of pages in the PDF file."""
    if PDF2IMAGE_AVAILABLE:
        try:
            info = pdfinfo_from_path(pdf_path)
            return int(info.get("Pages", 0))
        except Exception:
            pass
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF page count for {pdf_path}: {e}")
        return 0


def iter_pdf_pages(pdf_path: str, dpi: int = 130, max_dim: int = 1400):
    """
    Generator that yields (page_num, total_pages, PIL Image) one page at a time.
    Keeps memory footprint constant regardless of PDF length.
    """
    if not PDF2IMAGE_AVAILABLE:
        raise ImportError("pdf2image and poppler are required to stream PDF page images.")

    total_pages = get_pdf_page_count(pdf_path)
    print(f"Streaming PDF pages from: {pdf_path} (Total pages: {total_pages})")
    for page_num in range(1, total_pages + 1):
        try:
            pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
            if not pages:
                continue
            img = pages[0]
            w, h = img.size
            if w > max_dim or h > max_dim:
                if w > h:
                    new_w = max_dim
                    new_h = int(h * (max_dim / w))
                else:
                    new_h = max_dim
                    new_w = int(w * (max_dim / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            yield page_num, total_pages, img
            del pages
            del img
            gc.collect()
        except Exception as e:
            print(f"Error converting page {page_num}: {e}")
            continue


def pdf_to_images(pdf_path: str, max_pages: int = None) -> list[Image.Image]:
    """Converts a PDF file into a list of PIL Images, one per page."""
    images = []
    for page_num, total_pages, img in iter_pdf_pages(pdf_path):
        images.append(img)
        if max_pages and len(images) >= max_pages:
            break
    return images


def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a JPEG Base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_text_from_pdf(pdf_path: str, max_pages: int = None) -> list[Document]:
    """
    Extracts text from each page of a PDF file using pypdf.
    Returns a list of LangChain Document objects with rich metadata
    (filename, page_number, total_pages, doc_type).
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    filename = os.path.basename(pdf_path)
    documents = []

    pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
    for idx in range(pages_to_process):
        page = reader.pages[idx]
        text = page.extract_text() or ""
        cleaned_text = " ".join(text.split())
        if cleaned_text:
            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "page": idx + 1,
                        "total_pages": total_pages,
                        "doc_type": "annual_report",
                    },
                )
            )
    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """
    Chunks extracted documents using RecursiveCharacterTextSplitter,
    preserving page and source metadata on each individual chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
