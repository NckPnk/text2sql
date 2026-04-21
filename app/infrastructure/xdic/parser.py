"""
Парсер словаря БД (.xdic) для системы Text2SQL.

Основная идея:
- main.xdic — смысловая прослойка (описания, связи, бизнес-логика)
- PostgreSQL — источник реальных данных (типы, размеры, примеры, статистика)

Парсер объединяет оба источника для предоставления максимально полной
информации о структуре БД.
"""

import xml.etree.ElementTree as ET
import re
from dataclasses import dataclass, field
from typing import Optional
import psycopg
from psycopg.rows import dict_row


# ─────────────────────────────────────────────
# Dataclass-модели
# ─────────────────────────────────────────────

@dataclass
class FieldInfo:
    """Информация о поле таблицы."""
    name: str
    field_type: str  # Тип из xdic (Текст, Целое, Внешний ключ и т.д.)
    description: str = ""
    referenced_table: str = ""  # Для внешних ключей
    on_delete: str = ""
    length: Optional[int] = None
    precision: Optional[int] = None
    default_value: str = ""
    pg_type: str = ""  # Переопределённый тип PostgreSQL из xdic
    enum_values: list = field(default_factory=list)  # Для перечисляемых полей
    is_hierarchy: bool = False
    is_identity: bool = False

    # Поля, заполняемые из БД
    db_type: str = ""  # Реальный тип из information_schema
    db_nullable: bool = True
    db_default: str = ""
    db_column_position: int = 0

    @property
    def is_foreign_key(self) -> bool:
        return self.field_type == "Внешний ключ"

    @property
    def is_primary_key(self) -> bool:
        return self.name == "row_id"


@dataclass
class IndexInfo:
    """Информация об индексе."""
    name: str
    columns: str
    is_unique: bool = False
    is_clustered: bool = False
    expression: str = ""
    include_columns: str = ""


@dataclass
class TableInfo:
    """Полная информация о таблице."""
    name: str
    description: str = ""
    category: str = ""
    view_type: str = ""  # Вид: Служебная, Временная и т.д.
    schema: str = "public"

    fields: dict[str, FieldInfo] = field(default_factory=dict)
    indexes: list[IndexInfo] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)

    # Заполняется из БД
    db_row_count: Optional[int] = None
    db_size: str = ""
    db_real_name: str = ""  # Реальное имя в PostgreSQL (может отличаться)

    @property
    def is_temporary(self) -> bool:
        return self.view_type in ("Временная", "ВременнаяДляПользователя")

    @property
    def is_service(self) -> bool:
        return self.view_type == "Служебная"

    @property
    def foreign_keys(self) -> dict[str, FieldInfo]:
        return {n: f for n, f in self.fields.items() if f.is_foreign_key}

    @property
    def field_names(self) -> list[str]:
        return list(self.fields.keys())


# ─────────────────────────────────────────────
# Маппинг типов xdic → PostgreSQL
# ─────────────────────────────────────────────

XDIC_TYPE_MAP = {
    "Текст": "varchar",
    "Строка": "varchar",
    "Целое": "integer",
    "Длинное целое": "bigint",
    "Вещественное": "numeric",
    "Деньги": "numeric(15,2)",
    "Дата": "date",
    "ДатаВремя": "timestamp",
    "Время": "time",
    "Двоичные данные": "bytea",
    "Флаги": "integer",
    "Перечисляемое": "smallint",
    "Переключатель": "smallint",
    "Иерархия": "integer",
    "Внешний ключ": "integer",
    "Один к одному": "integer",
    "Многие к одному": "integer",
    "Указатель": "integer",
    "Гуид": "uuid",
    "xml": "xml",
    "json": "jsonb",
}


# ─────────────────────────────────────────────
# Основной парсер
# ─────────────────────────────────────────────

class XdicParser:
    """
    Парсер .xdic файла с возможностью обогащения данными из PostgreSQL.

    Использование:
        parser = XdicParser("main.xdic")
        parser.parse()
        parser.connect_db("host=localhost dbname=mydb user=user password=pass")
        parser.enrich_from_db()

        # Для text2sql:
        tables = parser.search_tables("лицевые счета")
        context = parser.get_table_context("Лицевые счета")
        schema_ddl = parser.get_create_table_sql("Лицевые счета")
    """

    def __init__(self, xdic_path: str):
        self.xdic_path = xdic_path
        self.tables: dict[str, TableInfo] = {}
        self.views: dict[str, dict] = {}
        self.functions: list[str] = []
        self._conn = None
        self._tree = None

    # ─── XML-парсинг ───────────────────────────

    def parse(self) -> "XdicParser":
        """Парсит .xdic файл и заполняет внутренние структуры."""
        self._tree = ET.parse(self.xdic_path)
        root = self._tree.getroot()

        # Таблицы
        tables_node = root.find("Tables")
        if tables_node is not None:
            for table_el in tables_node.findall("Table"):
                tbl = self._parse_table(table_el)
                if tbl:
                    self.tables[tbl.name] = tbl

        # Представления
        views_node = root.find("Views")
        if views_node is not None:
            for view_el in views_node.findall("View"):
                name = view_el.get("Имя", "")
                self.views[name] = {
                    "name": name,
                    "file": view_el.get("Файл", ""),
                    "version": view_el.get("Версия", ""),
                }

        # Функции
        funcs_node = root.find("Functions")
        if funcs_node is not None:
            for func_el in funcs_node.findall("Function"):
                self.functions.append(func_el.get("Имя", ""))

        return self

    def _parse_table(self, el: ET.Element) -> Optional[TableInfo]:
        """Парсит один элемент <Table>."""
        name = el.get("Имя", "")
        if not name:
            return None

        tbl = TableInfo(
            name=name,
            description=el.get("Описание", ""),
            category=el.get("Категория", ""),
            view_type=el.get("Вид", ""),
            schema=el.get("Схема", "public"),
        )

        # Поля
        fields_node = el.find("Поля")
        if fields_node is not None:
            for field_el in fields_node.findall("Поле"):
                fi = self._parse_field(field_el)
                if fi:
                    tbl.fields[fi.name] = fi

        # Индексы
        idx_node = el.find("Индексы_базы")
        if idx_node is not None:
            for idx_el in idx_node.findall("Индекс"):
                idx = IndexInfo(
                    name=idx_el.get("Имя", ""),
                    columns=idx_el.get("Поля", ""),
                    is_unique="UNIQUE" in idx_el.get("Опции", ""),
                    is_clustered="CLUSTERED" in idx_el.get("Опции", ""),
                    expression=idx_el.get("Выражение", ""),
                    include_columns=idx_el.get("ДопПоля", ""),
                )
                tbl.indexes.append(idx)

        # Триггеры
        trig_node = el.find("Триггеры")
        if trig_node is not None:
            for trig_el in trig_node.findall("Триггер"):
                tbl.triggers.append(trig_el.get("Имя", ""))

        return tbl

    def _parse_field(self, el: ET.Element) -> Optional[FieldInfo]:
        """Парсит один элемент <Поле>."""
        name = el.get("Имя", "")
        if not name:
            return None

        ftype = el.get("Тип", "")
        length_str = el.get("Длина", "") or el.get("Размер", "")

        fi = FieldInfo(
            name=name,
            field_type=ftype,
            description=el.get("Описание", ""),
            referenced_table=el.get("Таблица", ""),
            on_delete=el.get("При_удалении", ""),
            pg_type=el.get("PostgreSQL", ""),
            default_value=el.get("Значение_по_умолчанию", "")
                          or el.get("Значение_по_умолчанию_PostgreSQL", ""),
            is_hierarchy=(ftype == "Иерархия"),
            is_identity="Identity" in el.get("Опции", ""),
        )

        # Длина
        if length_str:
            try:
                fi.length = int(length_str)
            except ValueError:
                pass

        # Точность
        prec = el.get("Точность", "")
        if prec:
            try:
                fi.precision = int(prec)
            except ValueError:
                pass

        # Перечисляемые значения
        enum_str = el.get("Поля", "")
        if enum_str and ftype in ("Перечисляемое", "Флаги", "Переключатель"):
            fi.enum_values = [v.strip() for v in enum_str.split("\\n") if v.strip()]

        return fi

    # ─── Подключение к БД ──────────────────────

    def connect_db(self, dsn: str) -> "XdicParser":
        """
        Подключается к PostgreSQL.
        dsn — строка подключения, например:
          "host=localhost port=5432 dbname=mydb user=admin password=secret"
        """
        self._conn = psycopg.connect(dsn)
        return self

    def close_db(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Обогащение из БД ──────────────────────

    def enrich_from_db(self, tables: list[str] | None = None):
        """
        Обогащает метаданные реальными данными из PostgreSQL:
        - реальные типы колонок
        - количество строк
        - размер таблицы
        Если tables=None — обогащает все таблицы.
        """
        if not self._conn:
            raise RuntimeError("Нет подключения к БД. Вызовите connect_db() сначала.")

        target = tables or list(self.tables.keys())

        for tname in target:
            tbl = self.tables.get(tname)
            if not tbl or tbl.is_temporary:
                continue

            real_name = self._resolve_real_table_name(tname)
            if not real_name:
                continue

            tbl.db_real_name = real_name
            self._enrich_columns(tbl, real_name)
            self._enrich_stats(tbl, real_name)

    def _resolve_real_table_name(self, xdic_name: str) -> str:
        """Пытается найти реальное имя таблицы в БД (с учётом кавычек)."""
        cur = self._conn.cursor()
        try:
            # Пробуем точное совпадение
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
                  AND table_name = %s
                LIMIT 1
            """, (xdic_name,))
            row = cur.fetchone()
            if row:
                return row[0]

            # Пробуем в нижнем регистре
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
                  AND lower(table_name) = lower(%s)
                LIMIT 1
            """, (xdic_name,))
            row = cur.fetchone()
            return row[0] if row else ""
        finally:
            cur.close()

    def _enrich_columns(self, tbl: TableInfo, real_name: str):
        """Заполняет реальные типы колонок из information_schema."""
        cur = self._conn.cursor(row_factory=dict_row)
        try:
            cur.execute("""
                SELECT column_name, data_type, is_nullable,
                       column_default, ordinal_position,
                       character_maximum_length, numeric_precision
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (real_name,))

            db_cols = {row["column_name"]: row for row in cur.fetchall()}

            for fname, fi in tbl.fields.items():
                # Ищем колонку (имена в PG могут быть в кавычках)
                col = db_cols.get(fname) or db_cols.get(fname.lower())
                if col:
                    fi.db_type = col["data_type"]
                    fi.db_nullable = col["is_nullable"] == "YES"
                    fi.db_default = col["column_default"] or ""
                    fi.db_column_position = col["ordinal_position"]
        finally:
            cur.close()

    def _enrich_stats(self, tbl: TableInfo, real_name: str):
        """Получает статистику таблицы (кол-во строк, размер)."""
        cur = self._conn.cursor()
        try:
            # Приблизительное кол-во строк (быстро)
            cur.execute("""
                SELECT reltuples::bigint
                FROM pg_class
                WHERE relname = %s
            """, (real_name,))
            row = cur.fetchone()
            if row:
                tbl.db_row_count = max(0, row[0])

            # Размер таблицы
            cur.execute(
                "SELECT pg_size_pretty(pg_total_relation_size(%s))",
                (real_name,)
            )
            row = cur.fetchone()
            if row:
                tbl.db_size = row[0]
        except Exception:
            pass
        finally:
            cur.close()

    # ─── Запросы к БД для text2sql ─────────────

    def get_sample_values(self, table_name: str, column_name: str,
                          limit: int = 5) -> list:
        """
        Получает примеры уникальных значений колонки из БД.
        Полезно для text2sql — чтобы модель понимала, какие данные в поле.
        """
        if not self._conn:
            return []

        tbl = self.tables.get(table_name)
        if not tbl or not tbl.db_real_name:
            return []

        cur = self._conn.cursor()
        try:
            query = f"""
                SELECT DISTINCT "{column_name}"
                FROM "{tbl.db_real_name}"
                WHERE "{column_name}" IS NOT NULL
                LIMIT %s
            """
            cur.execute(query, (limit,))
            return [row[0] for row in cur.fetchall()]
        except Exception:
            return []
        finally:
            cur.close()

    def get_column_stats(self, table_name: str, column_name: str) -> dict:
        """
        Получает статистику по колонке: min, max, count distinct и т.д.
        """
        if not self._conn:
            return {}

        tbl = self.tables.get(table_name)
        if not tbl or not tbl.db_real_name:
            return {}

        cur = self._conn.cursor(row_factory=dict_row)
        try:
            query = f"""
                SELECT
                    COUNT(DISTINCT "{column_name}") as distinct_count,
                    COUNT(*) as total_count,
                    MIN("{column_name}"::text) as min_val,
                    MAX("{column_name}"::text) as max_val
                FROM "{tbl.db_real_name}"
            """
            cur.execute(query)
            return dict(cur.fetchone() or {})
        except Exception:
            return {}
        finally:
            cur.close()

    # ─── API для Text2SQL ──────────────────────

    def search_tables(self, query: str, include_temp: bool = False) -> list[TableInfo]:
        """
        Ищет таблицы по имени или описанию.
        Используется для определения релевантных таблиц по NL-запросу.
        """
        query_lower = query.lower()
        results = []

        for tbl in self.tables.values():
            if not include_temp and tbl.is_temporary:
                continue

            score = 0
            # Точное совпадение имени
            if query_lower == tbl.name.lower():
                score += 100
            # Частичное совпадение имени
            elif query_lower in tbl.name.lower():
                score += 50
            # Совпадение в описании
            if tbl.description and query_lower in tbl.description.lower():
                score += 30
            # Совпадение в именах полей
            for fname in tbl.fields:
                if query_lower in fname.lower():
                    score += 10
                    break
            # Совпадение в описаниях полей
            for fi in tbl.fields.values():
                if fi.description and query_lower in fi.description.lower():
                    score += 5
                    break

            if score > 0:
                results.append((score, tbl))

        results.sort(key=lambda x: -x[0])
        return [t for _, t in results]

    def search_fields(self, query: str) -> list[tuple[str, FieldInfo]]:
        """Ищет поля по имени или описанию. Возвращает (table_name, FieldInfo)."""
        query_lower = query.lower()
        results = []

        for tbl in self.tables.values():
            if tbl.is_temporary:
                continue
            for fi in tbl.fields.values():
                score = 0
                if query_lower == fi.name.lower():
                    score += 100
                elif query_lower in fi.name.lower():
                    score += 50
                if fi.description and query_lower in fi.description.lower():
                    score += 30
                if score > 0:
                    results.append((score, tbl.name, fi))

        results.sort(key=lambda x: -x[0])
        return [(t, f) for _, t, f in results]

    def get_table_context(self, table_name: str,
                          include_related: bool = True,
                          max_depth: int = 1) -> dict:
        """
        Формирует полный контекст таблицы для LLM.
        Включает поля, связи, описания и опционально связанные таблицы.
        """
        tbl = self.tables.get(table_name)
        if not tbl:
            return {}

        ctx = {
            "table": table_name,
            "description": tbl.description,
            "row_count": tbl.db_row_count,
            "columns": [],
            "foreign_keys": [],
            "referenced_by": [],
        }

        # Колонки
        for fi in tbl.fields.values():
            col_info = {
                "name": fi.name,
                "type": fi.db_type or XDIC_TYPE_MAP.get(fi.field_type, fi.field_type),
                "description": fi.description,
                "nullable": fi.db_nullable,
            }
            if fi.is_foreign_key:
                col_info["references"] = fi.referenced_table
            if fi.enum_values:
                col_info["enum_values"] = fi.enum_values
            ctx["columns"].append(col_info)

        # Внешние ключи (эта таблица → другие)
        for fi in tbl.fields.values():
            if fi.is_foreign_key and fi.referenced_table:
                ctx["foreign_keys"].append({
                    "column": fi.name,
                    "references_table": fi.referenced_table,
                    "on_delete": fi.on_delete,
                })

        # Обратные связи (другие таблицы → эта)
        if include_related:
            for other_tbl in self.tables.values():
                if other_tbl.is_temporary or other_tbl.name == table_name:
                    continue
                for fi in other_tbl.fields.values():
                    if fi.is_foreign_key and fi.referenced_table == table_name:
                        ctx["referenced_by"].append({
                            "table": other_tbl.name,
                            "column": fi.name,
                        })

        return ctx

    def get_related_tables(self, table_name: str,
                           max_depth: int = 1) -> set[str]:
        """
        Возвращает множество таблиц, связанных с данной через FK.
        """
        visited = set()
        self._walk_relations(table_name, max_depth, 0, visited)
        visited.discard(table_name)
        return visited

    def _walk_relations(self, table_name: str, max_depth: int,
                        current_depth: int, visited: set):
        if current_depth > max_depth or table_name in visited:
            return
        visited.add(table_name)

        tbl = self.tables.get(table_name)
        if not tbl:
            return

        # FK из этой таблицы
        for fi in tbl.fields.values():
            if fi.is_foreign_key and fi.referenced_table:
                self._walk_relations(
                    fi.referenced_table, max_depth,
                    current_depth + 1, visited
                )

        # FK в эту таблицу
        for other in self.tables.values():
            for fi in other.fields.values():
                if fi.is_foreign_key and fi.referenced_table == table_name:
                    self._walk_relations(
                        other.name, max_depth,
                        current_depth + 1, visited
                    )

    def get_join_path(self, table_a: str, table_b: str) -> list[dict]:
        """
        Находит путь JOIN между двумя таблицами.
        Возвращает список шагов: [{from_table, from_col, to_table, to_col}].
        """
        # BFS
        from collections import deque
        adj = self._build_adjacency()

        queue = deque([(table_a, [table_a])])
        visited = {table_a}

        while queue:
            current, path = queue.popleft()
            if current == table_b:
                return self._path_to_joins(path)

            for neighbor, _ in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # Путь не найден

    def _build_adjacency(self) -> dict:
        """Строит граф смежности таблиц по FK."""
        adj: dict[str, list[tuple[str, str]]] = {}
        for tbl in self.tables.values():
            if tbl.is_temporary:
                continue
            for fi in tbl.fields.values():
                if fi.is_foreign_key and fi.referenced_table:
                    adj.setdefault(tbl.name, []).append(
                        (fi.referenced_table, fi.name)
                    )
                    adj.setdefault(fi.referenced_table, []).append(
                        (tbl.name, fi.name)
                    )
        return adj

    def _path_to_joins(self, path: list[str]) -> list[dict]:
        """Преобразует путь таблиц в список JOIN-условий."""
        joins = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            tbl_a = self.tables.get(a)
            tbl_b = self.tables.get(b)
            if not tbl_a or not tbl_b:
                continue

            # Ищем FK из a → b
            for fi in tbl_a.fields.values():
                if fi.is_foreign_key and fi.referenced_table == b:
                    joins.append({
                        "from_table": a,
                        "from_column": fi.name,
                        "to_table": b,
                        "to_column": "row_id",
                    })
                    break
            else:
                # Ищем FK из b → a
                for fi in tbl_b.fields.values():
                    if fi.is_foreign_key and fi.referenced_table == a:
                        joins.append({
                            "from_table": b,
                            "from_column": fi.name,
                            "to_table": a,
                            "to_column": "row_id",
                        })
                        break

        return joins

    # ─── Генерация SQL ─────────────────────────

    def get_create_table_sql(self, table_name: str,
                             with_comments: bool = True) -> str:
        """
        Генерирует CREATE TABLE DDL для контекста LLM.
        Использует реальные типы из БД, если доступны, иначе — маппинг из xdic.
        """
        tbl = self.tables.get(table_name)
        if not tbl:
            return ""

        lines = []
        safe_name = f'"{tbl.name}"'

        if with_comments and tbl.description:
            lines.append(f"-- {tbl.description}")

        lines.append(f"CREATE TABLE {safe_name} (")

        col_lines = []
        # row_id всегда первый
        col_lines.append('    "row_id" serial PRIMARY KEY')

        for fi in tbl.fields.values():
            pg_type = fi.db_type or fi.pg_type or XDIC_TYPE_MAP.get(
                fi.field_type, "text"
            )

            nullable = "" if fi.db_nullable else " NOT NULL"
            default = f" DEFAULT {fi.default_value}" if fi.default_value else ""
            fk_comment = ""
            if fi.is_foreign_key and fi.referenced_table:
                fk_comment = f'  -- FK → "{fi.referenced_table}"'

            desc_comment = ""
            if with_comments and fi.description:
                desc_comment = f"  -- {fi.description}"

            comment = fk_comment or desc_comment

            col_lines.append(
                f'    "{fi.name}" {pg_type}{nullable}{default}{"," if True else ""}'
                f'{comment}'
            )

        lines.append(",\n".join(col_lines))
        lines.append(");")

        return "\n".join(lines)

    def get_schema_summary(self, table_names: list[str] | None = None,
                           compact: bool = False) -> str:
        """
        Генерирует текстовое описание схемы для промпта text2sql.
        compact=True — только имя таблицы + колонки без типов.
        """
        targets = table_names or [
            t for t, tbl in self.tables.items()
            if not tbl.is_temporary
        ]

        parts = []
        for tname in targets:
            tbl = self.tables.get(tname)
            if not tbl:
                continue

            if compact:
                cols = ", ".join(tbl.field_names[:20])
                desc = f" — {tbl.description}" if tbl.description else ""
                parts.append(f"{tname}({cols}){desc}")
            else:
                parts.append(self.get_create_table_sql(tname))

        return "\n\n".join(parts)

    # ─── Экспорт для text2sql pipeline ─────────

    def export_for_text2sql(self) -> dict:
        """
        Экспортирует всю схему в формате, удобном для text2sql-системы.
        """
        schema = {
            "tables": {},
            "relationships": [],
            "table_descriptions": {},
        }

        for tbl in self.tables.values():
            if tbl.is_temporary:
                continue

            schema["tables"][tbl.name] = {
                "columns": {
                    fi.name: {
                        "type": fi.db_type or XDIC_TYPE_MAP.get(
                            fi.field_type, "text"
                        ),
                        "description": fi.description,
                        "is_pk": fi.is_primary_key,
                        "is_fk": fi.is_foreign_key,
                        "fk_table": fi.referenced_table if fi.is_foreign_key else None,
                    }
                    for fi in tbl.fields.values()
                },
                "description": tbl.description,
                "row_count": tbl.db_row_count,
            }

            if tbl.description:
                schema["table_descriptions"][tbl.name] = tbl.description

            for fi in tbl.fields.values():
                if fi.is_foreign_key and fi.referenced_table:
                    schema["relationships"].append({
                        "from_table": tbl.name,
                        "from_column": fi.name,
                        "to_table": fi.referenced_table,
                        "to_column": "row_id",
                        "type": "many_to_one",
                    })

        return schema

    # ─── Вспомогательные ───────────────────────

    def get_table(self, name: str) -> Optional[TableInfo]:
        """Получить таблицу по имени."""
        return self.tables.get(name)

    def get_all_table_names(self, include_temp: bool = False) -> list[str]:
        """Список всех имён таблиц."""
        return [
            t for t, tbl in self.tables.items()
            if include_temp or not tbl.is_temporary
        ]

    def get_tables_by_category(self, category: str) -> list[TableInfo]:
        """Таблицы по категории (Начислительные, Квитанции, Журнал и т.д.)."""
        return [
            tbl for tbl in self.tables.values()
            if tbl.category == category
        ]

    def __repr__(self):
        total = len(self.tables)
        non_temp = sum(1 for t in self.tables.values() if not t.is_temporary)
        return (
            f"XdicParser(tables={total}, non_temp={non_temp}, "
            f"views={len(self.views)}, functions={len(self.functions)})"
        )


# ─────────────────────────────────────────────
# Пример использования
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Парсинг XML
    parser = XdicParser("main.xdic")
    parser.parse()
    print(parser)

    # 2. Подключение к БД и обогащение (раскомментировать при наличии БД)
    # parser.connect_db("host=localhost dbname=mydb user=admin password=secret")
    # parser.enrich_from_db()

    # 3. Поиск таблиц
    results = parser.search_tables("лицевые счета")
    for tbl in results[:5]:
        print(f"  {tbl.name}: {tbl.description[:80]}...")

    # 4. Контекст для LLM
    ctx = parser.get_table_context("Лицевые счета")
    print(f"\nКолонок: {len(ctx['columns'])}")
    print(f"FK: {len(ctx['foreign_keys'])}")
    print(f"Ссылаются: {len(ctx['referenced_by'])} таблиц")

    # 5. Путь JOIN
    path = parser.get_join_path("Список оплаты", "Лицевые счета")
    for step in path:
        print(f"  {step['from_table']}.{step['from_column']} → "
              f"{step['to_table']}.{step['to_column']}")

    # 6. DDL
    ddl = parser.get_create_table_sql("Банки")
    print(f"\n{ddl}")

    # 7. Экспорт
    export = parser.export_for_text2sql()
    print(f"\nЭкспортировано таблиц: {len(export['tables'])}")
    print(f"Связей: {len(export['relationships'])}")
