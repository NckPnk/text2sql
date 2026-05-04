from __future__ import annotations

import asyncio


def test_retrieve_adds_joinpath_bridge_table(
    schema_retrieval_service,
    chroma_client_mock,
    register_xdic_table,
    make_search_result,
) -> None:
    register_xdic_table(
        "stack.orders",
        ddl="CREATE TABLE stack.orders (id bigint, customer_id bigint);",
        relations=[{"from_field": "customer_id", "to_table": "stack.customers"}],
    )
    register_xdic_table(
        "stack.customers",
        ddl="CREATE TABLE stack.customers (id bigint, city_id bigint);",
        relations=[{"from_field": "city_id", "to_table": "stack.cities"}],
    )
    register_xdic_table(
        "stack.cities",
        ddl="CREATE TABLE stack.cities (id bigint, name text);",
    )

    chroma_client_mock.search.return_value = [
        make_search_result("stack.orders", 0.99),
        make_search_result("stack.cities", 0.97),
    ]
    chroma_client_mock.search_columns.return_value = []

    contexts = asyncio.run(schema_retrieval_service.retrieve("заказы по городам", max_tables=3))
    names = [context.name for context in contexts]

    assert "stack.orders" in names
    assert "stack.cities" in names
    assert "stack.customers" in names


def test_retrieve_does_not_add_bridge_when_no_capacity(
    schema_retrieval_service,
    chroma_client_mock,
    register_xdic_table,
    make_search_result,
) -> None:
    register_xdic_table(
        "stack.orders",
        ddl="CREATE TABLE stack.orders (id bigint, customer_id bigint);",
        relations=[{"from_field": "customer_id", "to_table": "stack.customers"}],
    )
    register_xdic_table(
        "stack.customers",
        ddl="CREATE TABLE stack.customers (id bigint, city_id bigint);",
        relations=[{"from_field": "city_id", "to_table": "stack.cities"}],
    )
    register_xdic_table(
        "stack.cities",
        ddl="CREATE TABLE stack.cities (id bigint, name text);",
    )

    chroma_client_mock.search.return_value = [
        make_search_result("stack.orders", 0.99),
        make_search_result("stack.cities", 0.97),
    ]
    chroma_client_mock.search_columns.return_value = []

    contexts = asyncio.run(schema_retrieval_service.retrieve("заказы по городам", max_tables=2))
    names = [context.name for context in contexts]

    assert names == ["stack.orders", "stack.cities"]
