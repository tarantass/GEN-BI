from pathlib import Path
from typing import Dict


def load_env(path: str = ".env") -> Dict[str, str]:
    """
    Простая загрузка значений из .env файла формата KEY=VALUE.
    Строки с # и пустые строки игнорируются.
    """
    env: Dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return env

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()

    return env


ENV: Dict[str, str] = load_env()

