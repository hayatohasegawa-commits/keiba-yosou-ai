"""HTTPレスポンス用ファイルキャッシュ + レート制限。

netkeibaへの負荷を最小化するため、すべてのGETは本モジュール経由にする。
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class RateLimitedCache:
    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        rate_limit_seconds: float = 2.0,
        timeout_seconds: int = 30,
        ttl_days: int = 30,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout_seconds = timeout_seconds
        self.ttl = timedelta(days=ttl_days)
        self._last_request_at: float = 0.0

    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.html"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime < self.ttl

    def _wait(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_at = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def _fetch(self, url: str, encoding: Optional[str] = None) -> str:
        self._wait()
        resp = requests.get(
            url,
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        if encoding:
            resp.encoding = encoding
        else:
            resp.encoding = resp.apparent_encoding
        return resp.text

    def get(self, url: str, encoding: Optional[str] = None, force: bool = False) -> str:
        path = self._cache_path(url)
        if not force and self._is_fresh(path):
            return path.read_text(encoding="utf-8")
        html = self._fetch(url, encoding=encoding)
        path.write_text(html, encoding="utf-8")
        return html
