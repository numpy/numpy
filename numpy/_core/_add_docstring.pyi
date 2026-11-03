__all__ = ["add_newdoc"]

def add_newdoc(
    place: str,
    obj: str,
    doc: str | tuple[str, str] | list[tuple[str, str]],
    warn_on_python: bool = True,
) -> None: ...
