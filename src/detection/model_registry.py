"""
Async model registry with atomic swap support.

Active model for each category lives at:
  {MODELS_DIR}/{category}/model.joblib

Retraining writes to:
  {MODELS_DIR}/{category}/model.joblib.pending
then os.replace() → model.joblib  (atomic on POSIX / Linux containers)

FastAPI caches the loaded bundle in memory and refreshes if mtime changes.
A background task (poll_for_updates) checks every 60 s.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

CATEGORIES = ("construction", "IT", "logistics", "facilities")


class ModelBundle:
    """All artifacts needed to score one vendor category."""

    __slots__ = (
        "model", "score_min", "score_max",
        "background",
        "kmeans", "scaler", "peer_norm_p95",
        "mtime", "version",
    )

    def __init__(
        self,
        model: IsolationForest,
        score_min: float,
        score_max: float,
        background: Optional[np.ndarray],
        kmeans: Optional[KMeans],
        scaler: Optional[StandardScaler],
        peer_norm_p95: float,
        mtime: float,
        version: str,
    ):
        self.model         = model
        self.score_min     = score_min
        self.score_max     = score_max
        self.background    = background
        self.kmeans        = kmeans
        self.scaler        = scaler
        self.peer_norm_p95 = peer_norm_p95
        self.mtime         = mtime
        self.version       = version


class ModelRegistry:
    """Thread-safe async model cache. Never serves a partially-loaded bundle."""

    def __init__(self, models_dir: str):
        self._dir     = models_dir
        self._bundles: Dict[str, Optional[ModelBundle]] = {c: None for c in CATEGORIES}
        self._locks:   Dict[str, asyncio.Lock]          = {c: asyncio.Lock() for c in CATEGORIES}

    def _model_path(self, category: str) -> str:
        return os.path.join(self._dir, category, "model.joblib")

    def _meta_path(self, category: str) -> str:
        return os.path.join(self._dir, category, "meta.json")

    async def load_category(self, category: str) -> bool:
        path = self._model_path(category)
        if not os.path.exists(path):
            return False

        mtime   = os.path.getmtime(path)
        current = self._bundles.get(category)
        if current and abs(current.mtime - mtime) < 1e-3:
            return True  # already up-to-date

        async with self._locks[category]:
            # Re-check under lock (another coroutine may have just loaded it)
            mtime   = os.path.getmtime(path)
            current = self._bundles.get(category)
            if current and abs(current.mtime - mtime) < 1e-3:
                return True

            try:
                obj = await asyncio.get_event_loop().run_in_executor(
                    None, joblib.load, path
                )
                version = "unknown"
                meta_path = self._meta_path(category)
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    version = meta.get("version", "unknown")

                self._bundles[category] = ModelBundle(
                    model         = obj["model"],
                    score_min     = obj["score_min"],
                    score_max     = obj["score_max"],
                    background    = obj.get("background"),
                    kmeans        = obj.get("kmeans"),
                    scaler        = obj.get("scaler"),
                    peer_norm_p95 = float(obj.get("peer_norm_p95", 1.0)),
                    mtime         = mtime,
                    version       = version,
                )
                log.info(
                    "Loaded model category=%s version=%s",
                    category, version,
                )
                return True
            except Exception as exc:
                log.error("Failed to load model for %s: %s", category, exc)
                return False

    async def load_all(self) -> None:
        await asyncio.gather(*[self.load_category(c) for c in CATEGORIES])

    async def get(self, category: str) -> Optional[ModelBundle]:
        """Return current bundle, refreshing from disk if stale."""
        await self.load_category(category)
        return self._bundles.get(category)

    async def poll_for_updates(self) -> None:
        """Background task: reload models whose files have changed on disk."""
        while True:
            await asyncio.sleep(60)
            for category in CATEGORIES:
                path = self._model_path(category)
                if not os.path.exists(path):
                    continue
                current = self._bundles.get(category)
                mtime   = os.path.getmtime(path)
                if current is None or abs(current.mtime - mtime) > 1e-3:
                    await self.load_category(category)


# Module-level singleton — initialised in FastAPI lifespan
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    if _registry is None:
        raise RuntimeError("ModelRegistry not initialised. Call init_registry() first.")
    return _registry


def init_registry(models_dir: str) -> ModelRegistry:
    global _registry
    _registry = ModelRegistry(models_dir)
    return _registry
