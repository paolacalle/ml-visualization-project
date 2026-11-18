from db_loaders._bulk import TableLoader


class ThreadCommentsLoader(TableLoader):
    """Insert ordered memberships after threads/comments, preserving their post."""
    table = "thread_comments"
