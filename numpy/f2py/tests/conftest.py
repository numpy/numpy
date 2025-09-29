import pytest


def pytest_itemcollected(item):
    item.add_marker(pytest.mark.thread_unsafe(reason="f2py is thread-unsafe"))
