import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_itemcollected(item):
    item.add_marker(pytest.mark.thread_unsafe(reason="f2py tests are not thread-unsafe"))
