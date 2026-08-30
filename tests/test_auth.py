"""Password hashing and session tokens. No database needed."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.auth import (
    AuthError,
    hash_password,
    issue_token,
    needs_rehash,
    read_token,
    secret_key,
    validate_email,
    validate_password,
    verify_password,
)
from app.core.errors import ConfigError

SECRET = "x" * 48


def test_a_hash_does_not_contain_the_password():
    digest = hash_password("correct horse battery staple")

    assert "correct horse" not in digest
    assert digest.startswith("$argon2id$"), "argon2id, not bcrypt or a plain digest"


def test_the_same_password_hashes_differently_every_time():
    """Per-hash salt: identical passwords must not produce identical rows, or
    the table itself tells you which accounts share a password."""
    assert hash_password("correct horse battery staple") != hash_password(
        "correct horse battery staple")


def test_verify_accepts_the_right_password_and_rejects_the_wrong_one():
    digest = hash_password("correct horse battery staple")

    assert verify_password(digest, "correct horse battery staple") is True
    assert verify_password(digest, "Correct horse battery staple") is False


def test_verify_against_a_missing_account_still_returns_false():
    """And does the work anyway -- see the timing test below."""
    assert verify_password(None, "any password at all") is False


def test_a_missing_account_is_not_measurably_faster_than_a_wrong_password():
    """Otherwise the response time is an account-enumeration oracle: an
    attacker learns which addresses are registered without ever logging in."""
    import time

    digest = hash_password("correct horse battery staple")

    def elapsed(fn) -> float:
        runs = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            runs.append(time.perf_counter() - start)
        return sorted(runs)[len(runs) // 2]   # median, to survive a noisy machine

    wrong_password = elapsed(lambda: verify_password(digest, "wrong password entirely"))
    no_account = elapsed(lambda: verify_password(None, "wrong password entirely"))

    ratio = no_account / wrong_password
    assert 0.3 < ratio < 3.0, f"timing differs by {ratio:.1f}x -- enumeration oracle"


def test_a_garbage_hash_is_rejected_rather_than_raising():
    """A corrupt row must fail the login, not crash the endpoint."""
    assert verify_password("not-a-hash", "whatever") is False
    assert needs_rehash("not-a-hash") is True


@pytest.mark.parametrize("bad", ["", "short", "a" * 11])
def test_passwords_below_the_minimum_are_refused(bad):
    with pytest.raises(AuthError):
        validate_password(bad)


def test_absurdly_long_passwords_are_refused():
    """argon2 will happily hash a megabyte; that is free CPU for whoever sends
    it."""
    with pytest.raises(AuthError):
        validate_password("a" * 2000)


@pytest.mark.parametrize("bad", ["nope", "no@domain", "@acme.rs", "a b@acme.rs", ""])
def test_malformed_emails_are_refused(bad):
    with pytest.raises(AuthError):
        validate_email(bad)


def test_emails_are_normalised():
    assert validate_email("  PeRa@Acme.RS ") == "pera@acme.rs"


def test_a_token_round_trips_to_its_user():
    token = issue_token("user-123", secret=SECRET)

    assert read_token(token.value, secret=SECRET) == "user-123"


def test_a_token_signed_with_another_secret_is_rejected():
    token = issue_token("user-123", secret=SECRET)

    with pytest.raises(AuthError):
        read_token(token.value, secret="y" * 48)


def test_an_expired_token_is_rejected():
    long_ago = datetime.now(timezone.utc) - timedelta(days=2)
    token = issue_token("user-123", secret=SECRET, now=long_ago)

    with pytest.raises(AuthError):
        read_token(token.value, secret=SECRET)


def test_an_unsigned_token_is_rejected():
    """alg=none is the classic JWT bypass: the token names its own algorithm
    and a verifier that trusts that field accepts anything."""
    import jwt as pyjwt

    forged = pyjwt.encode({"sub": "user-123"}, key="", algorithm="none")

    with pytest.raises(AuthError):
        read_token(forged, secret=SECRET)


def test_a_tampered_payload_is_rejected():
    import base64
    import json

    token = issue_token("user-123", secret=SECRET).value
    header, payload, signature = token.split(".")
    body = json.loads(base64.urlsafe_b64decode(payload + "=="))
    body["sub"] = "somebody-else"
    swapped = base64.urlsafe_b64encode(json.dumps(body).encode()).rstrip(b"=").decode()

    with pytest.raises(AuthError):
        read_token(f"{header}.{swapped}.{signature}", secret=SECRET)


def test_garbage_is_rejected_without_crashing():
    for junk in ("", "not.a.token", "a.b.c", "x" * 500):
        with pytest.raises(AuthError):
            read_token(junk, secret=SECRET)


def test_the_signing_secret_has_no_default(monkeypatch):
    """A fallback secret is what reaches production, because nothing ever
    forces anyone to set a real one -- and a shared default means every
    deployment can forge every other one's sessions."""
    monkeypatch.delenv("APP_AUTH_SECRET", raising=False)
    with pytest.raises(ConfigError, match="APP_AUTH_SECRET"):
        secret_key()

    monkeypatch.setenv("APP_AUTH_SECRET", "too-short")
    with pytest.raises(ConfigError):
        secret_key()

    monkeypatch.setenv("APP_AUTH_SECRET", SECRET)
    assert secret_key() == SECRET
