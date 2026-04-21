"""
Индексация таблиц из main.xdic в ChromaDB.

Что делает:
1. Парсит main.xdic → извлекает все таблицы с описаниями
2. Для каждой таблицы формирует текстовый документ
3. Получает embedding через Ollama (qwen3-embedding:8b)
4. Сохраняет в ChromaDB (persistent storage)

Запуск:
    cd ~/text2sql-zhkh
    source ../text2sql-env/bin/activate
    python scripts/index_schema.py

Время: ~5-15 минут на 1313 таблиц.
"""

import os
import sys
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import chromadb
import ollama
from dotenv import load_dotenv


# ─── Конфигурация ────────────────────────────────────────

# Загружаем .env из корня проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

XDIC_PATH = os.getenv("XDIC_PATH", str(PROJECT_ROOT / "data" / "xdic" / "main.xdic"))
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma"))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "table_schemas")

# Размер батча для embedding (сколько таблиц за один запрос к Ollama)
BATCH_SIZE = 20


# ─── Маппинг типов xdic → PostgreSQL ────────────────────

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


# ─── Простые модели для парсинга ─────────────────────────

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
    view_type: str = ""        # Служебная, Временная и т.д.
    fields: list[FieldInfo] = field(default_factory=list)

    @property
    def is_temporary(self) -> bool:
        return (
            self.name.startswith("#")
            or self.view_type in ("Временная", "ВременнаяДляПользователя")
        )


# ─── Парсинг xdic ───────────────────────────────────────

def parse_xdic(path: str) -> list[TableInfo]:
    """Парсит main.xdic и возвращает список таблиц."""
    print(f"📖 Парсинг {path}...")

    tree = ET.parse(path)
    root = tree.getroot()
    tables = []

    # Ищем все <Table> в любом месте дерева
    for table_el in root.iter("Table"):
        name = table_el.get("Имя", "")
        if not name:
            continue

        tbl = TableInfo(
            name=name,
            description=table_el.get("Описание", ""),
            view_type=table_el.get("Вид", ""),
        )

        # Парсим поля
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

                fi = FieldInfo(
                    name=fname,
                    field_type=ftype,
                    description=field_el.get("Описание", ""),
                    referenced_table=field_el.get("Таблица", ""),
                    enum_values=enum_vals,
                )
                tbl.fields.append(fi)

        tables.append(tbl)

    print(f"   Найдено таблиц: {len(tables)}")
    return tables


# ─── Формирование текстового описания ───────────────────

def table_to_document(tbl: TableInfo) -> str:
    """
    Формирует текстовый документ для одной таблицы.
    Этот текст будет embedded и сохранён в ChromaDB.
    Чем качественнее текст — тем лучше семантический поиск.
    """
    parts = []

    # Заголовок
    parts.append(f"Таблица: {tbl.name}")
    if tbl.description:
        parts.append(f"Описание: {tbl.description}")

    # Поля (ограничиваем, чтобы документ не был слишком длинным)
    if tbl.fields:
        field_lines = []
        for fi in tbl.fields[:50]:  # Макс 50 полей
            pg_type = XDIC_TYPE_MAP.get(fi.field_type, fi.field_type)
            line = f"  - {fi.name} ({pg_type})"
            if fi.description:
                line += f": {fi.description}"
            if fi.referenced_table:
                line += f" [FK → {fi.referenced_table}]"
            if fi.enum_values:
                line += f" [значения: {', '.join(fi.enum_values)}]"
            field_lines.append(line)

        parts.append("Поля:")
        parts.extend(field_lines)

        if len(tbl.fields) > 50:
            parts.append(f"  ... и ещё {len(tbl.fields) - 50} полей")

    # Связи (FK)
    fk_fields = [f for f in tbl.fields if f.field_type == "Внешний ключ"]
    if fk_fields:
        parts.append("Связи:")
        for f in fk_fields:
            parts.append(f"  - {tbl.name}.{f.name} → {f.referenced_table}")

    return "\n".join(parts)


def table_to_metadata(tbl: TableInfo) -> dict:
    """Метаданные для ChromaDB — используются при фильтрации."""
    fk_tables = [f.referenced_table for f in tbl.fields
                 if f.field_type == "Внешний ключ" and f.referenced_table]
    return {
        "table_name": tbl.name,
        "name": tbl.name,
        "description": (tbl.description or "")[:500],
        "field_count": len(tbl.fields),
        "fk_count": len(fk_tables),
        "fk_tables": ", ".join(fk_tables)[:500],  # ChromaDB ограничивает длину
        "is_service": tbl.view_type == "Служебная",
        "view_type": tbl.view_type or "regular",
    }


# ─── Ollama Embedding ───────────────────────────────────

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Получает embeddings через Ollama API."""
    client = ollama.Client(host=OLLAMA_URL, timeout=120.0)
    response = client.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]


def check_ollama() -> bool:
    """Проверяет доступность Ollama и наличие модели."""
    try:
        client = ollama.Client(host=OLLAMA_URL, timeout=10.0)
        resp = client.list()
        models = [
            model.get("name", model.get("model", ""))
            for model in resp.get("models", [])
        ]

        if not any(EMBED_MODEL.split(":")[0] in m for m in models):
            print(f"❌ Модель {EMBED_MODEL} не найдена в Ollama!")
            print(f"   Доступные модели: {models}")
            print(f"   Выполните: ollama pull {EMBED_MODEL}")
            return False

        # Тестовый embedding
        test = get_embeddings(["тест"])
        print(f"   Размерность embedding: {len(test[0])}")
        return True
    except ConnectionError:
        print(f"❌ Ollama недоступна по адресу {OLLAMA_URL}")
        print(f"   Убедитесь, что Ollama запущена: ollama serve")
        return False


# ─── ChromaDB ───────────────────────────────────────────

def init_chroma() -> chromadb.Collection:
    """Инициализирует ChromaDB и возвращает коллекцию."""
    print(f"🗄️  ChromaDB: {CHROMA_DIR}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Удаляем старую коллекцию (если есть) для чистой переиндексации
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Удалена старая коллекция '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Schema tables from xdic dictionary"},
    )
    print(f"   Создана коллекция '{COLLECTION_NAME}'")
    return collection


# ─── Основной процесс ──────────────────────────────────

def main():
    print("=" * 60)
    print("  Text2SQL ЖКХ — Индексация схемы в ChromaDB")
    print("=" * 60)
    print()

    # 1. Проверяем xdic
    if not Path(XDIC_PATH).exists():
        print(f"❌ Файл не найден: {XDIC_PATH}")
        print(f"   Скопируйте main.xdic в {Path(XDIC_PATH).parent}/")
        sys.exit(1)

    # 2. Проверяем Ollama + модель
    print("🔍 Проверка Ollama...")
    if not check_ollama():
        sys.exit(1)
    print("   ✅ Ollama готова")
    print()

    # 3. Парсим xdic
    all_tables = parse_xdic(XDIC_PATH)

    # Фильтруем временные таблицы
    tables = [t for t in all_tables if not t.is_temporary]
    skipped = len(all_tables) - len(tables)
    print(f"   Пропущено временных: {skipped}")
    print(f"   К индексации: {len(tables)}")
    print()

    # 4. Формируем документы
    print("📝 Формирование документов...")
    documents = []
    metadatas = []
    ids = []

    for tbl in tables:
        doc = table_to_document(tbl)
        meta = table_to_metadata(tbl)
        documents.append(doc)
        metadatas.append(meta)
        ids.append(tbl.name)

    print(f"   Документов: {len(documents)}")
    avg_len = sum(len(d) for d in documents) / len(documents) if documents else 0
    print(f"   Средняя длина: {avg_len:.0f} символов")
    print()

    # 5. Получаем embeddings батчами
    print(f"🧠 Генерация embeddings (батч по {BATCH_SIZE})...")
    all_embeddings = []
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

    start_time = time.time()
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        try:
            embeddings = get_embeddings(batch)
            all_embeddings.extend(embeddings)
            elapsed = time.time() - start_time
            speed = len(all_embeddings) / elapsed if elapsed > 0 else 0
            print(f"   Батч {batch_num}/{total_batches} — "
                  f"{len(all_embeddings)}/{len(documents)} "
                  f"({speed:.1f} табл/сек)")
        except Exception as e:
            print(f"   ❌ Ошибка в батче {batch_num}: {e}")
            print(f"   Таблицы: {[t.name for t in tables[i : i + BATCH_SIZE]]}")
            # Пробуем по одной
            for j, doc in enumerate(batch):
                try:
                    emb = get_embeddings([doc])
                    all_embeddings.extend(emb)
                except Exception as e2:
                    print(f"      Пропущена: {tables[i + j].name} — {e2}")
                    # Добавляем нулевой вектор, чтобы не сбить индексы
                    all_embeddings.append([0.0] * len(all_embeddings[0]))

    embed_time = time.time() - start_time
    print(f"   ✅ Embeddings готовы за {embed_time:.1f} сек")
    print()

    # 6. Сохраняем в ChromaDB
    print("💾 Сохранение в ChromaDB...")
    collection = init_chroma()

    # ChromaDB тоже лучше добавлять батчами
    for i in range(0, len(documents), BATCH_SIZE):
        collection.add(
            documents=documents[i : i + BATCH_SIZE],
            embeddings=all_embeddings[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
            ids=ids[i : i + BATCH_SIZE],
        )

    print(f"   ✅ Сохранено: {collection.count()} таблиц")
    print()

    # 7. Тестовый поиск
    print("🔎 Тестовый поиск...")
    test_queries = [
        "лицевые счета",
        "приборы учёта",
        "параметры площадь",
        "адресная иерархия улицы дома",
        "начисления оплата",
    ]

    for q in test_queries:
        q_emb = get_embeddings([q])
        results = collection.query(
            query_embeddings=q_emb,
            n_results=3,
        )
        names = results["ids"][0] if results["ids"] else []
        print(f"   «{q}» → {names}")

    print()
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"  ✅ Индексация завершена за {total_time:.1f} сек")
    print(f"  📊 Таблиц в ChromaDB: {collection.count()}")
    print(f"  📁 Данные: {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
