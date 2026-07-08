from src.services.types import User

# Hardcoded for this mock scaffold. Replace with a real user store later —
# the login() signature stays the same.
_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "analyst": {"password": "pass123", "role": "user"},
}


def login(username: str, password: str) -> User | None:
    record = _USERS.get(username)
    if record is None or record["password"] != password:
        return None
    return User(username=username, role=record["role"])
