import os
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf, lit
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType

import fitz  # PyMuPDF
import sys

# Force Spark to use the current Python executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
# ... rest of your code

# ── 1. Initialize Spark Session ───────────────────────────────────────────────
def get_spark():
    return SparkSession.builder \
        .appName("FinancialRAG") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()

# ── 2. Schema for extracted data ──────────────────────────────────────────────
schema = StructType([
    StructField("file_name", StringType(), True),
    StructField("ticker",    StringType(), True),   # e.g. NVDA
    StructField("year",      StringType(), True),   # e.g. 2024
    StructField("page",      IntegerType(), True),
    StructField("text",      StringType(), True),
])

# ── 3. Filename metadata parser ───────────────────────────────────────────────
def parse_filename_metadata(filename: str) -> tuple[str, str]:
    """
    Extract ticker and year from SEC filing filename.
    Pattern: {ticker}-{YYYYMMDD}.pdf
    Examples:
        nvda-20250126.pdf  → ("NVDA", "2025")
        goog-20241231.pdf  → ("GOOG", "2024")
        tsla-20251231.pdf  → ("TSLA", "2025")
    """
    match = re.match(r"([a-zA-Z]+)-(\d{4})\d{4}\.pdf", filename, re.IGNORECASE)
    if match:
        ticker = match.group(1).upper()
        year   = match.group(2)
        return ticker, year
    return "UNKNOWN", "UNKNOWN"

# ── 4. PDF Extraction Logic (PyMuPDF) ─────────────────────────────────────────
def get_pdf_content(pdf_folder: str) -> list[tuple]:
    """
    Extract text from all PDFs in a folder.
    Returns list of (file_name, ticker, year, page_num, text) tuples.
    """
    all_data = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠ No PDF files found in {pdf_folder}")
        return all_data

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    for file in pdf_files:
        ticker, year = parse_filename_metadata(file)
        path = os.path.join(pdf_folder, file)

        try:
            doc = fitz.open(path)
            print(f"  Processing {file} ({ticker}, {year}) — {len(doc)} pages")

            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:  # Skip blank pages
                    all_data.append((file, ticker, year, page_num + 1, text))

            doc.close()
        except Exception as e:
            print(f"  ✗ Failed to parse {file}: {e}")

    print(f"Extracted {len(all_data)} pages total")
    return all_data

# ── 5. Chunking UDF ───────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks for embedding.
    chunk_size=1000 chars, overlap=100 chars.
    """
    if not text:
        return []
    chunk_size = 1000
    overlap    = 100
    chunks     = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()
        if len(chunk) > 50:   # Skip tiny fragments
            chunks.append(chunk)
    return chunks

chunk_udf = udf(chunk_text, ArrayType(StringType()))

# ── 6. Main Processing Function ───────────────────────────────────────────────
def process_all_pdfs(input_path: str):
    """
    Full pipeline: extract → load into Spark → chunk → return DataFrame.

    Output columns: file_name, ticker, year, page, chunk_text
    """
    # Extract text from all PDFs
    raw_list = get_pdf_content(input_path)

    if not raw_list:
        print("No data extracted. Check your data/pdfs/ folder.")
        return None

    # Load into Spark DataFrame
    df = get_spark().createDataFrame(raw_list, schema=schema)

    # Chunk text and explode into one row per chunk
    df_chunks = (
        df.withColumn("chunk_text", explode(chunk_udf(col("text"))))
          .select("file_name", "ticker", "year", "page", "chunk_text")
          .filter(col("chunk_text").isNotNull())
          .filter(col("chunk_text") != "")
    )

    total = df_chunks.count()
    print(f"\n✓ Generated {total} chunks across {df.count()} pages")

    return df_chunks


if __name__ == "__main__":
    pdf_dir = "data/pdfs/"
    processed_df = process_all_pdfs(pdf_dir)
    if processed_df:
        print("\nSample chunks:")
        processed_df.show(10, truncate=80)

        print("\nChunks per ticker:")
        processed_df.groupBy("ticker", "year") \
                    .count() \
                    .orderBy("ticker", "year") \
                    .show()
