"""Phase 0: secret/env handling works without requiring a real key."""

import pytest

from stealth.config import get_secret


def test_missing_required_secret_raises_clear_error():
    with pytest.raises(RuntimeError, match="Missing required secret"):
        get_secret("DEFINITELY_NOT_A_REAL_SECRET_XYZ", required=True)


def test_optional_missing_secret_returns_none():
    assert get_secret("DEFINITELY_NOT_A_REAL_SECRET_XYZ", required=False) is None


def test_existing_env_var_is_returned(monkeypatch):
    monkeypatch.setenv("STEALTH_TEST_SECRET", "value123")
    assert get_secret("STEALTH_TEST_SECRET") == "value123"
