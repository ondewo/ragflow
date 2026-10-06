"""
Tests for the PostgreSQL advisory lock wait behaviour.

``PostgresDatabaseLock`` guards initialization, migrations and the task
executor's progress writes, so a contended lock has to wait for its timeout
budget rather than report a failure the caller cannot tell apart from a real
error. The waiting is done by a blocking ``pg_advisory_lock`` bounded by the
session ``lock_timeout``; ``GaussDBDatabaseLock`` polls instead and must keep
its own statements.
"""

import pytest
from peewee import OperationalError

import api.db.db_models as db_models
from api.db.db_models import GaussDBDatabaseLock, PostgresDatabaseLock

pytestmark = pytest.mark.p2

SET_LOCK_TIMEOUT = "SELECT set_config('lock_timeout', %s, false)"
ADVISORY_LOCK = "SELECT pg_advisory_lock(%s)"
TRY_ADVISORY_LOCK = "SELECT pg_try_advisory_lock(%s)"
RESET_LOCK_TIMEOUT = "RESET lock_timeout"


class _Cursor:
    def __init__(self, row: tuple | None) -> None:
        self._row = row

    def fetchone(self) -> tuple | None:
        return self._row


class _RecordingDB:
    """Records the statements a lock issues, optionally failing one of them."""

    def __init__(self, fail_on: str | None = None, row: tuple | None = (1,)) -> None:
        self.statements: list[tuple[str, tuple | None]] = []
        self._fail_on = fail_on
        self._row = row

    @property
    def sql(self) -> list[str]:
        return [statement for statement, _ in self.statements]

    def execute_sql(self, sql, params=None, commit=True):
        self.statements.append((sql, params))
        if self._fail_on is not None and self._fail_on in sql:
            raise OperationalError("canceling statement due to lock timeout")
        return _Cursor(self._row)


@pytest.fixture
def no_retry_delay(monkeypatch) -> None:
    """Skip the exponential backoff ``with_retry`` sleeps between attempts."""
    monkeypatch.setattr(db_models.time, "sleep", lambda _seconds: None)


class TestPostgresDatabaseLockWait:
    def test_bounded_timeout_waits_under_a_statement_lock_timeout(self):
        db = _RecordingDB()
        lock = PostgresDatabaseLock("unit_test_lock", 7, db=db)

        assert lock.lock() is True
        assert db.sql == [SET_LOCK_TIMEOUT, ADVISORY_LOCK, RESET_LOCK_TIMEOUT]
        assert db.statements[0][1] == (str(lock.timeout * 1000),)
        assert db.statements[1][1] == (lock.lock_id,)

    def test_bounded_timeout_does_not_use_the_non_blocking_variant(self):
        db = _RecordingDB()

        PostgresDatabaseLock("unit_test_lock", 10, db=db).lock()

        assert not any(TRY_ADVISORY_LOCK in statement for statement in db.sql)

    @pytest.mark.parametrize("timeout", [-1, -60])
    def test_negative_timeout_waits_without_a_lock_timeout(self, timeout):
        db = _RecordingDB()
        lock = PostgresDatabaseLock("unit_test_lock", timeout, db=db)

        assert lock.lock() is True
        assert db.sql == [ADVISORY_LOCK]
        assert db.statements[0][1] == (lock.lock_id,)

    def test_lock_timeout_is_reset_and_the_error_surfaces(self, no_retry_delay):
        db = _RecordingDB(fail_on="pg_advisory_lock")
        lock = PostgresDatabaseLock("unit_test_lock", 1, db=db)

        with pytest.raises(OperationalError):
            lock.lock()

        # Three with_retry attempts, each of which restores the session setting.
        assert db.sql == [SET_LOCK_TIMEOUT, ADVISORY_LOCK, RESET_LOCK_TIMEOUT] * 3


class TestGaussDBDatabaseLockKeepsPolling:
    def test_bounded_timeout_polls_the_non_blocking_variant(self):
        db = _RecordingDB(row=(1,))
        lock = GaussDBDatabaseLock("unit_test_lock", 5, db=db)

        assert lock.lock() is True
        assert db.sql == [TRY_ADVISORY_LOCK]
        assert db.statements[0][1] == (lock.lock_id,)
