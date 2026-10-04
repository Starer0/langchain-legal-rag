"""Password hashing without plaintext storage or secret-bearing return objects."""

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError
from argon2.low_level import Type


_HASHER = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1,
                         hash_len=32, salt_len=16, type=Type.ID)
_MAX_PASSWORD_BYTES = 1024


def _password_size(password):
    if not isinstance(password, str):
        return None
    try:
        return len(password.encode('utf8'))
    except UnicodeError:
        return None


def validate_password(password, *, minimum_length=12):
    size = _password_size(password)
    if size is None or len(password) < minimum_length or size > _MAX_PASSWORD_BYTES:
        raise ValueError(f'Password must contain at least {minimum_length} characters and at most 1024 UTF-8 bytes')


def hash_password(password, *, minimum_length=12):
    validate_password(password, minimum_length=minimum_length)
    return _HASHER.hash(password)


def verify_password(encoded, password):
    if not isinstance(encoded, str) or not encoded.startswith('$argon2id$'):
        return False
    size = _password_size(password)
    if size is None or size > _MAX_PASSWORD_BYTES:
        return False
    try:
        return _HASHER.verify(encoded, password)
    except (VerificationError, InvalidHash):
        return False


# Unknown users still perform a real hash verification; this is not an account.
DUMMY_PASSWORD_HASH = _HASHER.hash('unknown-account-verification-padding')
