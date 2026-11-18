from db_loaders._bulk import TableLoader


class PairsLoader(TableLoader):
    """Insert pair/source-line records; reject conflicting rows for an existing ID."""
    table = "pairs"
