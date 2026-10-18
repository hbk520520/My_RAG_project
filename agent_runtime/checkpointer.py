"""Fence checkpoint writes with the same lease authority as run results."""
from langgraph.checkpoint.sqlite import SqliteSaver


class FencedSqliteSaver(SqliteSaver):
    def __init__(self, connection, context):
        super().__init__(connection)
        self.context = context

    def put(self, *args, **kwargs):
        context = self.context
        with context.store.connection() as authority:
            authority.execute("BEGIN IMMEDIATE")
            context.store._active(authority, context.run_id, context.epoch, context.token)
            return super().put(*args, **kwargs)

    def put_writes(self, *args, **kwargs):
        context = self.context
        with context.store.connection() as authority:
            authority.execute("BEGIN IMMEDIATE")
            context.store._active(authority, context.run_id, context.epoch, context.token)
            return super().put_writes(*args, **kwargs)
