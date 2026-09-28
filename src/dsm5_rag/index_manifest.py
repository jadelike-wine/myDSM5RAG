"""索引清单（manifest）：记录每个文档文件的 hash/mtime，用于判断索引是否需要重建。

背景：`_scan_document_files()` 过去只在进程启动时跑一次，`build_index(force_rebuild=False)`
只要磁盘上存在索引目录就直接加载 —— 往 `dsm5_documents/` 放新文件后调用 `build_index`
会返回成功，但新内容一条都没进索引。现在每次构建都写清单，加载前先比对：
文件集合、内容 sha256、以及会影响切分结果的配置（chunk_size / chunk_overlap /
BM25 语言）任一变化，就判定为需要重建。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "index_manifest.json"
_CHUNK_SIZE = 1024 * 1024


def hash_file(path: Path) -> str:
    """流式计算文件 sha256（大 PDF 也不会把整个文件读进内存）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_files(files: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    """把文件列表整理成 {绝对路径: {name, size, mtime, sha256}}。"""
    described: Dict[str, Dict[str, Any]] = {}
    for path in files:
        resolved = Path(path).resolve()
        try:
            stat = resolved.stat()
            described[str(resolved)] = {
                "file_name": resolved.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "sha256": hash_file(resolved),
            }
        except OSError as e:
            logger.warning("无法读取文件信息 %s: %s", resolved, e)
    return described


class IndexManifest:
    """索引清单：读写 persist_dir/index_manifest.json 并做差异比对。"""

    def __init__(
        self,
        persist_dir: Path | str,
        files: Optional[Dict[str, Dict[str, Any]]] = None,
        build_config: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.files: Dict[str, Dict[str, Any]] = files or {}
        self.build_config: Dict[str, Any] = build_config or {}
        self.stats: Dict[str, Any] = stats or {}
        self.updated_at = updated_at or datetime.now().isoformat()

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.persist_dir / MANIFEST_FILENAME

    @classmethod
    def load(cls, persist_dir: Path | str) -> Optional["IndexManifest"]:
        path = Path(persist_dir) / MANIFEST_FILENAME
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("索引清单读取失败 %s: %s", path, e)
            return None
        return cls(
            persist_dir=persist_dir,
            files=payload.get("files") or {},
            build_config=payload.get("build_config") or {},
            stats=payload.get("stats") or {},
            updated_at=payload.get("updated_at"),
        )

    def save(self) -> Path:
        self.updated_at = datetime.now().isoformat()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": self.updated_at,
            "build_config": self.build_config,
            "stats": self.stats,
            "files": self.files,
        }
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)
        return self.path

    # ------------------------------------------------------------------
    @staticmethod
    def build_config_signature(config: Any) -> Dict[str, Any]:
        """影响索引内容的配置项；这些变了就必须重建。"""
        return {
            "chunk_size": getattr(config, "chunk_size", None),
            "chunk_overlap": getattr(config, "chunk_overlap", None),
            "bm25_language": getattr(config, "bm25_language", None),
            "bm25_skip_stemming": getattr(config, "bm25_skip_stemming", None),
            "embedding_model": getattr(config, "embedding_model", None),
        }

    def diff(self, current_files: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """返回新增/变更/删除的文件清单。"""
        previous = self.files
        added = sorted(set(current_files) - set(previous))
        removed = sorted(set(previous) - set(current_files))
        changed = sorted(
            path
            for path in set(current_files) & set(previous)
            if current_files[path].get("sha256") != previous[path].get("sha256")
        )
        return {"added": added, "changed": changed, "removed": removed}

    def needs_rebuild(
        self,
        current_files: Dict[str, Dict[str, Any]],
        current_config: Dict[str, Any],
    ) -> tuple[bool, Dict[str, Any]]:
        """判断是否需要重建，并给出原因明细。"""
        reasons: Dict[str, Any] = {}
        file_diff = self.diff(current_files)
        if file_diff["added"]:
            reasons["added"] = [Path(p).name for p in file_diff["added"]]
        if file_diff["changed"]:
            reasons["changed"] = [Path(p).name for p in file_diff["changed"]]
        if file_diff["removed"]:
            reasons["removed"] = [Path(p).name for p in file_diff["removed"]]
        config_diff = {
            key: {"old": self.build_config.get(key), "new": value}
            for key, value in current_config.items()
            if self.build_config.get(key) != value
        }
        if config_diff:
            reasons["config"] = config_diff

        needs = bool(reasons) or not self.files
        if not self.files:
            reasons = {"reason": "清单里没有已索引文件记录"}
        return needs, reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "updated_at": self.updated_at,
            "build_config": self.build_config,
            "stats": self.stats,
            "file_count": len(self.files),
            "files": self.files,
        }


def status_of(persist_dir: Path | str, files: Iterable[Path], config: Any) -> Dict[str, Any]:
    """给状态端点用：当前索引与磁盘文档是否一致。"""
    current = describe_files(files)
    manifest = IndexManifest.load(persist_dir)
    if manifest is None:
        return {
            "manifest_present": False,
            "up_to_date": False,
            "reason": "尚未生成索引清单",
            "file_count": len(current),
        }
    needs, reasons = manifest.needs_rebuild(current, IndexManifest.build_config_signature(config))
    return {
        "manifest_present": True,
        "up_to_date": not needs,
        "indexed_file_count": len(manifest.files),
        "current_file_count": len(current),
        "changed_since_index": reasons if needs else {},
        "manifest_updated_at": manifest.updated_at,
    }
