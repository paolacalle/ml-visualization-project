from ._bulk import TableLoader


class ThreadsLoader(TableLoader):
    """Insert outcome threads after pairs, preserving pair and post ownership."""
    table = "threads"
