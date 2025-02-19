# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/usr/bin/env python3

# @yaml
# signature: parse_pdf_url <pdf_url>
# docstring: Download the corresponding pdf url, and parse the text out of it.
# arguments:
#   pdf_url:
#       type: string
#       description: the url of the pdf to get
#       required: true

import argparse
import requests
import tempfile
import json
from pypdf import PdfReader
import pymupdf
import pymupdf4llm
import os


def download_pdf(pdf_url: str, output_filename: str) -> None:
    """
    Download a PDF file from a URL and save it to a specified file
    Args:
        pdf_url (str): The URL of the PDF file.
    """
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")


def load_pdf(pdf_path, num_pages=None, min_size=100) -> str:
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)  # open a document
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # iterate the document pages
                text = text + page.get_text()  # get plain text encoded as UTF-8
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")

    return text

parser = argparse.ArgumentParser(
    description="Download a pdf and parse it as text"
)
parser.add_argument("pdf_url", type=str, help="The location of the pdf to download and parse")
args = parser.parse_args()


pdf_filename = "temporary_file"
download_pdf(args.pdf_url, pdf_filename)
pdf_text = load_pdf(pdf_filename)
print(pdf_text)
os.remove(pdf_filename)
