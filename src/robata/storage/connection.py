"""Database connection and session management."""

from __future__ import annotations

from typing import Any

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session, sessionmaker

    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False

    class Session:  # type: ignore[no-redef]
        """Stub Session when SQLAlchemy is not installed."""

        pass


def get_engine(database_url: str | None = None) -> Any:
    """Create a SQLAlchemy engine.

    Args:
        database_url: Database connection URL. Defaults to SQLite in-memory.

    Returns:
        SQLAlchemy engine instance.
    """
    if not _HAS_SQLALCHEMY:
        raise ImportError("SQLAlchemy is required. Install with: pip install sqlalchemy")

    url = database_url or "sqlite:///:memory:"
    return create_engine(url, echo=False)


def get_session_maker(engine: Any) -> Any:
    """Create a session maker bound to an engine.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        Session maker.
    """
    if not _HAS_SQLALCHEMY:
        raise ImportError("SQLAlchemy is required. Install with: pip install sqlalchemy")

    return sessionmaker(bind=engine)


__all__ = [
    "Session",
    "get_engine",
    "get_session_maker",
]
