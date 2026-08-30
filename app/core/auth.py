"""Password hashing and session tokens.

Deliberately thin, and deliberately built on argon2-cffi and PyJWT rather than
anything hand-rolled. There is nothing to gain by being clever here and a great
deal to lose, so this module's job is to make the boring choices in one place
and leave no room for a caller to make a different one.
"""
from __future__ import annotations

import hmac
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from .errors import ConfigError

# argon2id at the library's defaults, which track the current OWASP guidance.
# Not bcrypt: bcrypt silently truncates at 72 bytes, so two different long
# passwords can be the same password.
_hasher = PasswordHasher()

TOKEN_ALGORITHM = "HS256"
TOKEN_TTL = timedelta(hours=12)

# Long enough that a leaked short password is not what breaks this. The upper
# bound exists because argon2 hashes whatever it is handed, so a megabyte of
# input is a free CPU burn for whoever sends it.
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 1024

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Wrong credentials, expired session, malformed token -- all of it.

    One exception type on purpose: the caller must not be able to tell "no such
    account" from "wrong password", because that difference is how an attacker
    enumerates who has an account.
    """


@dataclass(frozen=True)
class SessionToken:
    value: str
    expires_at: datetime


def secret_key(env: dict[str, str] | None = None) -> str:
    """The signing key, from the environment and nowhere else.

    No default, not even in development. A fallback secret is the kind of thing
    that reaches production precisely because nothing ever forced anyone to set
    a real one, and every deployment sharing a default key means every
    deployment can forge every other one's sessions.
    """
    env = dict(os.environ if env is None else env)
    value = env.get("APP_AUTH_SECRET", "")
    if len(value) < 32:
        raise ConfigError(
            "APP_AUTH_SECRET must be set to at least 32 characters before authentication "
            "can be used. Generate one with: python -c \"import secrets; "
            "print(secrets.token_urlsafe(48))\""
        )
    return value


def validate_email(email: str) -> str:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email) or len(email) > 320:
        raise AuthError("that does not look like an email address")
    return email


def validate_password(password: str) -> str:
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise AuthError(
            f"password must be between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} "
            "characters")
    return password


def hash_password(password: str) -> str:
    return _hasher.hash(validate_password(password))


def verify_password(stored_hash: str | None, password: str) -> bool:
    """Constant-ish time regardless of whether the account exists.

    When stored_hash is None -- no such user -- this still runs a full argon2
    verification against a dummy hash before returning False. Skipping it would
    make "no such account" measurably faster than "wrong password", which
    hands out a list of valid addresses to anyone with a stopwatch.
    """
    if stored_hash is None:
        try:
            _hasher.verify(_DUMMY_HASH, password)
        except (VerifyMismatchError, InvalidHashError):
            pass
        return False
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


_DUMMY_HASH = _hasher.hash("not-a-real-password-just-for-timing")


def needs_rehash(stored_hash: str) -> bool:
    """True once the tuning parameters move on, so a password can be upgraded
    on the next successful login rather than staying at old settings forever."""
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True


def issue_token(user_id: str, *, secret: str, now: datetime | None = None) -> SessionToken:
    """`now` is injectable so expiry can be tested without waiting twelve hours."""
    now = now or datetime.now(timezone.utc)
    expires = now + TOKEN_TTL
    payload = {"sub": user_id, "iat": int(now.timestamp()), "exp": int(expires.timestamp())}
    return SessionToken(jwt.encode(payload, secret, algorithm=TOKEN_ALGORITHM), expires)


def read_token(token: str, *, secret: str) -> str:
    """Return the user id a token vouches for, or raise.

    `algorithms` is pinned to one value rather than read from the token's own
    header: a token that names its own algorithm can name "none", and a
    verifier that believes it will accept anything.
    """
    try:
        payload: dict[str, Any] = jwt.decode(token, secret, algorithms=[TOKEN_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise AuthError("invalid or expired session") from exc
    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise AuthError("token carries no subject")
    return sub


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)
