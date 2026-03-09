import pandas as pd
from pypdf import PdfReader
from pathlib import Path
from langchain_core.documents import Document
from typing import List

def get_chunks_from_csv(csv_path: str) -> List[Document]:
    """
    Загружает CSV файл(ы) и возвращает список документов.

    Если передан путь к файлу — обрабатывается один CSV.
    Если передана директория — обрабатываются все *.csv внутри (без рекурсии).
    """
    base_path = Path(csv_path)

    if not base_path.exists():
        raise FileNotFoundError(f"CSV файл или директория не найдены: {csv_path}")

    csv_files: List[Path]
    if base_path.is_dir():
        csv_files = [p for p in base_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    else:
        csv_files = [base_path]

    all_documents: List[Document] = []

    for file_path in csv_files:
        documents: List[Document] = []

        df = pd.read_csv(file_path, encoding="windows-1251")

        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])

            doc = Document(
                page_content=row_text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "row": idx + 1,
                    "file_path": str(file_path),
                    "columns": list(df.columns),
                    "doc_type": "csv",
                },
            )
            documents.append(doc)

        if not documents:
            doc = Document(
                page_content=f"CSV file: {file_path.name}\nColumns: {', '.join(df.columns)}",
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "columns": list(df.columns),
                    "doc_type": "csv",
                    "empty": True,
                },
            )
            documents.append(doc)

        print(f"✅ Обработано {len(documents)} строк из CSV: {file_path.name}")
        all_documents.extend(documents)

    print(f"📦 Всего документов из CSV: {len(all_documents)} (путь: {base_path})")
    return all_documents


def get_chunks_from_pdf(pdf_path: str) -> List[Document]:
    """
    Загружает PDF файл(ы) и возвращает список документов (по одному на страницу).

    Если передан путь к файлу — обрабатывается один PDF.
    Если передана директория — обрабатываются все *.pdf внутри (без рекурсии).
    """
    base_path = Path(pdf_path)

    if not base_path.exists():
        raise FileNotFoundError(f"PDF файл или директория не найдены: {pdf_path}")

    pdf_files: List[Path]
    if base_path.is_dir():
        pdf_files = [p for p in base_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    else:
        pdf_files = [base_path]

    all_documents: List[Document] = []

    for file_path in pdf_files:
        documents: List[Document] = []

        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "filename": file_path.name,
                            "page": page_num + 1,
                            "total_pages": len(pdf_reader.pages),
                            "file_path": str(file_path),
                            "doc_type": "pdf",
                        },
                    )
                    documents.append(doc)

        print(f"✅ Обработано {len(documents)} страниц из PDF: {file_path.name}")
        all_documents.extend(documents)

    print(f"📦 Всего документов из PDF: {len(all_documents)} (путь: {base_path})")
    return all_documents


def get_chunks_from_sql(sql_path: str) -> List[Document]:
    """
    Загружает SQL файл(ы) и возвращает список документов (по одному на запрос).

    Если передан путь к файлу — обрабатывается один SQL.
    Если передана директория — обрабатываются все *.sql внутри (без рекурсии).
    """
    base_path = Path(sql_path)

    if not base_path.exists():
        raise FileNotFoundError(f"SQL файл или директория не найдены: {sql_path}")

    sql_files: List[Path]
    if base_path.is_dir():
        sql_files = [p for p in base_path.iterdir() if p.is_file() and p.suffix.lower() == ".sql"]
    else:
        sql_files = [base_path]

    all_documents: List[Document] = []

    for file_path in sql_files:
        documents: List[Document] = []

        with open(file_path, "r", encoding="utf-8") as file:
            sql_content = file.read()

        sql_queries = sql_content.split(";")

        non_empty_queries = [q for q in sql_queries if q.strip()]

        for idx, query in enumerate(sql_queries):
            query = query.strip()
            if query:
                doc = Document(
                    page_content=query,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "query_num": idx + 1,
                        "total_queries": len(non_empty_queries),
                        "file_path": str(file_path),
                        "doc_type": "sql",
                    },
                )
                documents.append(doc)

        if not documents and sql_content.strip():
            doc = Document(
                page_content=sql_content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "doc_type": "sql",
                },
            )
            documents.append(doc)

        print(f"✅ Обработано {len(documents)} запросов из SQL: {file_path.name}")
        all_documents.extend(documents)

    print(f"📦 Всего документов из SQL: {len(all_documents)} (путь: {base_path})")
    return all_documents


def get_chunks_from_txt(txt_path: str) -> List[Document]:
    """
    Загружает TXT файл(ы) и возвращает список документов.

    Если передан путь к файлу — возвращается один Document.
    Если передана директория — собираются все *.txt внутри директории (без рекурсии).
    """
    base_path = Path(txt_path)

    if not base_path.exists():
        raise FileNotFoundError(f"TXT файл или директория не найдены: {txt_path}")

    documents: List[Document] = []

    def _read_text(file_path: Path) -> str:
        # Пытаемся прочитать в UTF-8, если не получается — в windows-1251
        for enc in ("utf-8", "windows-1251"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # На крайний случай читаем с игнорированием ошибок
        with open(file_path, "r", errors="ignore") as f:
            return f.read()

    txt_files: List[Path]
    if base_path.is_dir():
        txt_files = [p for p in base_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    else:
        txt_files = [base_path]

    for file_path in txt_files:
        text = _read_text(file_path).strip()
        if not text:
            # Пустой файл — всё равно создаём минимум один Document, чтобы можно было отследить
            doc = Document(
                page_content="",
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "doc_type": "txt",
                    "empty": True,
                },
            )
        else:
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "doc_type": "txt",
                },
            )
        documents.append(doc)

    print(f"✅ Обработано {len(documents)} TXT файлов/документов из: {base_path}")
    return documents