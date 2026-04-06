"""Identity query transformer (no-op)."""

from core.interfaces import QueryTransformer


class IdentityTransformer(QueryTransformer):
    """Pass-through transformer that returns the query unchanged."""

    def transform(self, query: str) -> str:
        """Return the query unchanged."""
        return query
