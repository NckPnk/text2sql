from __future__ import annotations

from app.core.models import TableContext


def test_build_prunes_wide_table_ddl(prompt_builder) -> None:
    columns = [f"    col_{i} text," for i in range(1, 26)]
    ddl = "\n".join([
        "CREATE TABLE stack.wide_table (",
        *columns,
        ");",
    ])
    table = TableContext(
        name="stack.wide_table",
        ddl=ddl,
        description="Wide table",
        relevance_score=1.0,
        matched_columns=["col_2", "col_20"],
    )

    _, user_prompt = prompt_builder.build("test question", [table])

    assert "-- pruned" in user_prompt
    assert "col_2" in user_prompt
    assert "col_20" in user_prompt


def test_build_keeps_small_table_ddl_unchanged(prompt_builder) -> None:
    ddl = (
        "CREATE TABLE stack.small_table (\n"
        "    id bigint,\n"
        "    status text,\n"
        "    created_at timestamp\n"
        ");"
    )
    table = TableContext(
        name="stack.small_table",
        ddl=ddl,
        description="Small table",
        relevance_score=1.0,
    )

    _, user_prompt = prompt_builder.build("test question", [table])

    assert "-- pruned" not in user_prompt
    assert ddl in user_prompt
