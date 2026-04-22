"""
Индексация таблиц и колонок из main.xdic в ChromaDB.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
import ollama
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

XDIC_PATH = os.getenv("XDIC_PATH", str(PROJECT_ROOT / "data" / "xdic" / "main.xdic"))
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma"))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b")
TABLE_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "table_schemas")
COLUMN_COLLECTION_NAME = os.getenv("CHROMA_COLUMN_COLLECTION", "column_schemas")

BATCH_SIZE = 20

XDIC_TYPE_MAP = {
    "Текст": "text",
    "Строка": "varchar",
    "Целое": "integer",
    "Длинное целое": "bigint",
    "Вещественное": "double precision",
    "Деньги": "numeric(15,2)",
    "Дата": "date",
    "ДатаВремя": "timestamp",
    "Время": "time",
    "Флаги": "integer",
    "Перечисляемое": "smallint",
    "Переключатель": "smallint",
    "Иерархия": "integer",
    "Внешний ключ": "integer",
    "Один к одному": "integer",
    "Многие к одному": "integer",
    "Указатель": "integer",
    "Двоичные данные": "bytea",
    "Гуид": "uuid",
    "xml": "xml",
    "json": "jsonb",
}


@dataclass
class FieldInfo:
    name: str
    field_type: str
    description: str = ""
    referenced_table: str = ""
    enum_values: list[str] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    description: str = ""
    category: str = ""
    view_type: str = ""
    fields: list[FieldInfo] = field(default_factory=list)

    @property
    def is_temporary(self) -> bool:
        return self.name.startswith("#") or self.view_type in ("Временная", "ВременнаяДляПользователя")


def parse_xdic(path: str) -> list[TableInfo]:
    print(f"📖 Парсинг {path}...")
    tree = ET.parse(path)
    root = tree.getroot()
    tables = []

    for table_el in root.iter("Table"):
        name = table_el.get("Имя", "")
        if not name:
            continue

        tbl = TableInfo(
            name=name,
            description=table_el.get("Описание", ""),
            category=table_el.get("Категория", ""),
            view_type=table_el.get("Вид", ""),
        )

        fields_node = table_el.find("Поля")
        if fields_node is not None:
            for field_el in fields_node.findall("Поле"):
                fname = field_el.get("Имя", "")
                if not fname:
                    continue
                ftype = field_el.get("Тип", "")
                enum_vals = []
                if ftype in ("Перечисляемое", "Флаги", "Переключатель"):
                    raw = field_el.get("Поля", "")
                    if raw:
                        enum_vals = [v.strip() for v in raw.split("\\n") if v.strip()]

                tbl.fields.append(
                    FieldInfo(
                        name=fname,
                        field_type=ftype,
                        description=field_el.get("Описание", ""),
                        referenced_table=field_el.get("Таблица", ""),
                        enum_values=enum_vals,
                    )
                )

        tables.append(tbl)

    print(f"   Найдено таблиц: {len(tables)}")
    return tables


def table_to_document(tbl: TableInfo) -> str:
    parts = [f"Таблица: {tbl.name}"]
    if tbl.description:
        parts.append(f"Описание: {tbl.description}")

    if tbl.fields:
        parts.append("Поля:")
        for fi in tbl.fields[:50]:
            pg_type = XDIC_TYPE_MAP.get(fi.field_type, fi.field_type)
            line = f"  - {fi.name} ({pg_type})"
            if fi.description:
                line += f": {fi.description}"
            if fi.referenced_table:
                line += f" [FK → {fi.referenced_table}]"
            parts.append(line)
        if len(tbl.fields) > 50:
            parts.append(f"  ... и ещё {len(tbl.fields) - 50} полей")

    return "\n".join(parts)


def table_to_metadata(tbl: TableInfo) -> dict:
    fk_tables = [f.referenced_table for f in tbl.fields if f.field_type == "Внешний ключ" and f.referenced_table]
    return {
        "table_name": tbl.name,
        "name": tbl.name,
        "description": (tbl.description or "")[:500],
        "field_count": len(tbl.fields),
        "fk_count": len(fk_tables),
        "fk_tables": ", ".join(fk_tables)[:500],
        "is_service": tbl.view_type == "Служебная" or tbl.category == "Служебная",
        "view_type": tbl.view_type or "regular",
        "business_category": (tbl.category or "")[:200],
    }


def build_canonical_phrase(tbl: TableInfo, field: FieldInfo) -> str:
    hint = field.description.strip() if field.description else ""
    if hint:
        return f"{tbl.name}.{field.name} {hint}"[:240]
    return f"{tbl.name}.{field.name} column"[:240]


def column_to_document(tbl: TableInfo, field: FieldInfo) -> str:
    pg_type = XDIC_TYPE_MAP.get(field.field_type, field.field_type)
    fk_target = field.referenced_table or "none"
    canonical = build_canonical_phrase(tbl, field)
    return "\n".join(
        [
            f"Таблица: {tbl.name}",
            f"Колонка: {field.name}",
            f"Описание колонки: {field.description or 'нет описания'}",
            f"Тип: {pg_type}",
            f"FK target: {fk_target}",
            f"Canonical phrase: {canonical}",
        ]
    )


def column_to_metadata(tbl: TableInfo, field: FieldInfo) -> dict:
    return {
        "table_name": tbl.name,
        "column_name": field.name,
        "is_fk": bool(field.referenced_table),
        "referenced_table": field.referenced_table or "",
        "view_type": tbl.view_type or "regular",
        "business_category": (tbl.category or "")[:200],
    }


def get_embeddings(texts: list[str]) -> list[list[float]]:
    client = ollama.Client(host=OLLAMA_URL, timeout=120.0)
    response = client.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]


def check_ollama() -> bool:
    try:
        client = ollama.Client(host=OLLAMA_URL, timeout=10.0)
        resp = client.list()
        models = [model.get("name", model.get("model", "")) for model in resp.get("models", [])]
        if not any(EMBED_MODEL.split(":")[0] in m for m in models):
            print(f"❌ Модель {EMBED_MODEL} не найдена в Ollama! Доступные: {models}")
            return False
        test = get_embeddings(["тест"])
        print(f"   Размерность embedding: {len(test[0])}")
        return True
    except Exception:
        print(f"❌ Ollama недоступна по адресу {OLLAMA_URL}")
        return False


def init_chroma() -> tuple[chromadb.Collection, chromadb.Collection]:
    print(f"🗄️  ChromaDB: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    for collection_name in (TABLE_COLLECTION_NAME, COLUMN_COLLECTION_NAME):
        try:
            client.delete_collection(collection_name)
            print(f"   Удалена старая коллекция '{collection_name}'")
        except Exception:
            pass

    table_collection = client.create_collection(
        name=TABLE_COLLECTION_NAME,
        metadata={"description": "Schema tables from xdic dictionary"},
    )
    column_collection = client.create_collection(
        name=COLUMN_COLLECTION_NAME,
        metadata={"description": "Schema columns from xdic dictionary"},
    )
    return table_collection, column_collection


def embed_with_progress(documents: list[str], labels: list[str], title: str) -> list[list[float]]:
    print(f"🧠 Генерация embeddings для {title} (батч по {BATCH_SIZE})...")
    all_embeddings: list[list[float]] = []
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    vector_dim = 0

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        try:
            embeddings = get_embeddings(batch)
            if embeddings and not vector_dim:
                vector_dim = len(embeddings[0])
            all_embeddings.extend(embeddings)
        except Exception as exc:
            print(f"   ❌ Ошибка в батче {batch_num}: {exc}")
            for j, doc in enumerate(batch):
                try:
                    emb = get_embeddings([doc])[0]
                    if not vector_dim:
                        vector_dim = len(emb)
                    all_embeddings.append(emb)
                except Exception as one_exc:
                    if not vector_dim:
                        vector_dim = len(get_embeddings(["тест"])[0])
                    print(f"      Пропущен объект: {labels[i + j]} — {one_exc}")
                    all_embeddings.append([0.0] * vector_dim)

        elapsed = time.time() - start_time
        speed = len(all_embeddings) / elapsed if elapsed > 0 else 0
        print(f"   Батч {batch_num}/{total_batches} — {len(all_embeddings)}/{len(documents)} ({speed:.1f}/сек)")

    return all_embeddings


def add_to_collection(
    collection: chromadb.Collection,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
    embeddings: list[list[float]],
) -> None:
    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
            ids=ids[i : i + BATCH_SIZE],
        )


def run_test_queries(
    table_collection: chromadb.Collection,
    column_collection: chromadb.Collection,
) -> None:
    print("🔎 Тестовый поиск по таблицам и колонкам...")
    test_queries = [
        "лицевые счета",
        "приборы учёта",
        "параметры площадь",
        "адресная иерархия улицы дома",
        "начисления оплата",
    ]
    for query in test_queries:
        query_embedding = get_embeddings([query])
        table_hits = table_collection.query(query_embeddings=query_embedding, n_results=3)
        column_hits = column_collection.query(query_embeddings=query_embedding, n_results=3)
        table_names = table_hits["ids"][0] if table_hits["ids"] else []
        column_names = column_hits["ids"][0] if column_hits["ids"] else []
        print(f"   «{query}»")
        print(f"      таблицы: {table_names}")
        print(f"      колонки: {column_names}")


def main() -> None:
    start_time = time.time()
    print("=" * 60)
    print("  Text2SQL ЖКХ — Индексация схемы в ChromaDB")
    print("=" * 60)

    if not Path(XDIC_PATH).exists():
        print(f"❌ Файл не найден: {XDIC_PATH}")
        sys.exit(1)

    print("🔍 Проверка Ollama...")
    if not check_ollama():
        sys.exit(1)

    all_tables = parse_xdic(XDIC_PATH)
    tables = [t for t in all_tables if not t.is_temporary]
    print(f"   Пропущено временных: {len(all_tables) - len(tables)}")
    print(f"   К индексации таблиц: {len(tables)}")

    table_documents: list[str] = []
    table_metadatas: list[dict] = []
    table_ids: list[str] = []

    column_documents: list[str] = []
    column_metadatas: list[dict] = []
    column_ids: list[str] = []

    for tbl in tables:
        table_documents.append(table_to_document(tbl))
        table_metadatas.append(table_to_metadata(tbl))
        table_ids.append(tbl.name)

        for field in tbl.fields:
            column_documents.append(column_to_document(tbl, field))
            column_metadatas.append(column_to_metadata(tbl, field))
            column_ids.append(f"{tbl.name}.{field.name}")

    print(f"📝 Табличных документов: {len(table_documents)}")
    print(f"📝 Документов по колонкам: {len(column_documents)}")

    table_embeddings = embed_with_progress(table_documents, table_ids, "таблиц")
    column_embeddings = embed_with_progress(column_documents, column_ids, "колонок")

    table_collection, column_collection = init_chroma()

    print("💾 Сохранение таблиц в ChromaDB...")
    add_to_collection(table_collection, table_documents, table_metadatas, table_ids, table_embeddings)
    print("💾 Сохранение колонок в ChromaDB...")
    add_to_collection(column_collection, column_documents, column_metadatas, column_ids, column_embeddings)

    print(f"   ✅ Таблиц сохранено: {table_collection.count()}")
    print(f"   ✅ Колонок сохранено: {column_collection.count()}")
    run_test_queries(table_collection, column_collection)
    total_time = time.time() - start_time
    print(f"⏱️  Общее время: {total_time:.1f} сек")


if __name__ == "__main__":
    main()
