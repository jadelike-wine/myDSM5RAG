"""索引清单（manifest）单元测试。"""

from datetime import datetime, timedelta

import pytest

from dsm5_rag.index_manifest import (
    MANIFEST_FILENAME,
    IndexManifest,
    describe_files,
    hash_file,
    status_of,
)


@pytest.fixture
def docs(tmp_path):
    """造两个文档文件，返回 (目录, [路径])。"""
    directory = tmp_path / "documents"
    directory.mkdir()
    a = directory / "a.txt"
    b = directory / "b.txt"
    a.write_text("Major Depressive Disorder 296.3x", encoding="utf-8")
    b.write_text("Generalized Anxiety Disorder 300.02", encoding="utf-8")
    return directory, [a, b]


class TestFileHashing:
    def test_hash_is_stable_for_same_content(self, docs):
        _directory, files = docs
        assert hash_file(files[0]) == hash_file(files[0])

    def test_hash_changes_with_content(self, docs):
        _directory, files = docs
        before = hash_file(files[0])
        files[0].write_text("changed content", encoding="utf-8")
        assert hash_file(files[0]) != before

    def test_describe_files_records_size_mtime_hash(self, docs):
        _directory, files = docs
        described = describe_files(files)
        assert len(described) == 2
        entry = described[str(files[0].resolve())]
        assert entry["file_name"] == "a.txt"
        assert entry["size"] > 0
        assert len(entry["sha256"]) == 64
        datetime.fromisoformat(entry["mtime"])

    def test_describe_files_skips_missing(self, tmp_path):
        assert describe_files([tmp_path / "nope.txt"]) == {}


class TestDiffAndRebuildDecision:
    def test_no_change_means_no_rebuild(self, docs):
        _directory, files = docs
        current = describe_files(files)
        config = {"chunk_size": 1024, "bm25_language": "en"}
        manifest = IndexManifest(
            persist_dir=_directory, files=current, build_config=config
        )
        needs, reasons = manifest.needs_rebuild(current, config)
        assert needs is False
        assert reasons == {}

    def test_added_file_triggers_rebuild(self, docs, tmp_path):
        directory, files = docs
        manifest = IndexManifest(
            persist_dir=directory,
            files=describe_files(files),
            build_config={"chunk_size": 1024},
        )
        new_file = directory / "c.txt"
        new_file.write_text("Panic Disorder 300.01", encoding="utf-8")
        current = describe_files(files + [new_file])

        needs, reasons = manifest.needs_rebuild(current, {"chunk_size": 1024})
        assert needs is True
        assert reasons["added"] == ["c.txt"]

    def test_removed_and_changed_files(self, docs):
        directory, files = docs
        manifest = IndexManifest(
            persist_dir=directory,
            files=describe_files(files),
            build_config={},
        )
        files[0].write_text("edited", encoding="utf-8")
        current = describe_files([files[0]])  # b.txt 被删除

        needs, reasons = manifest.needs_rebuild(current, {})
        assert needs is True
        assert reasons["changed"] == ["a.txt"]
        assert reasons["removed"] == ["b.txt"]

    def test_config_change_triggers_rebuild(self, docs):
        _directory, files = docs
        current = describe_files(files)
        manifest = IndexManifest(
            persist_dir=_directory, files=current, build_config={"chunk_size": 1024}
        )
        needs, reasons = manifest.needs_rebuild(current, {"chunk_size": 512})
        assert needs is True
        assert reasons["config"]["chunk_size"] == {"old": 1024, "new": 512}

    def test_empty_manifest_always_rebuilds(self, docs):
        _directory, files = docs
        manifest = IndexManifest(persist_dir=_directory, files={}, build_config={})
        needs, reasons = manifest.needs_rebuild(describe_files(files), {})
        assert needs is True

    def test_mtime_only_change_needs_rebuild(self, docs):
        """只改 mtime 不改内容：sha256 相同 → 判定为未变化（避免无谓重建）。"""
        directory, files = docs
        manifest = IndexManifest(
            persist_dir=directory,
            files=describe_files(files),
            build_config={},
        )
        future = datetime.now() + timedelta(hours=1)
        stamp = future.timestamp()
        import os

        os.utime(files[0], (stamp, stamp))
        current = describe_files(files)
        needs, _reasons = manifest.needs_rebuild(current, {})
        assert needs is False
        # mtime 字段本身确实变了
        key = str(files[0].resolve())
        assert current[key]["mtime"] != manifest.files[key]["mtime"]


class TestPersistence:
    def test_save_and_load_roundtrip(self, docs):
        _directory, files = docs
        current = describe_files(files)
        manifest = IndexManifest(
            persist_dir=_directory,
            files=current,
            build_config={"chunk_size": 1024},
            stats={"nodes": 12},
        )
        path = manifest.save()
        assert path == _directory / MANIFEST_FILENAME
        assert path.is_file()

        loaded = IndexManifest.load(_directory)
        assert loaded is not None
        assert loaded.files == current
        assert loaded.build_config == {"chunk_size": 1024}
        assert loaded.stats["nodes"] == 12

    def test_load_missing_returns_none(self, tmp_path):
        assert IndexManifest.load(tmp_path) is None

    def test_load_corrupted_returns_none(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("{ not json", encoding="utf-8")
        assert IndexManifest.load(tmp_path) is None

    def test_save_leaves_no_temp_file(self, docs):
        _directory, files = docs
        IndexManifest(persist_dir=_directory, files=describe_files(files)).save()
        assert not (_directory / (MANIFEST_FILENAME + ".tmp")).exists()


class TestStatusOf:
    def test_without_manifest(self, docs):
        _directory, files = docs
        result = status_of(_directory / "nonexistent", files, object())
        assert result["manifest_present"] is False
        assert result["up_to_date"] is False

    def test_with_manifest_and_pending_change(self, docs):
        directory, files = docs
        config = type("C", (), {"chunk_size": 1024, "chunk_overlap": 100,
                                "bm25_language": "en", "bm25_skip_stemming": False,
                                "embedding_model": "m"})()
        IndexManifest(
            persist_dir=directory,
            files=describe_files(files),
            build_config=IndexManifest.build_config_signature(config),
        ).save()

        assert status_of(directory, files, config)["up_to_date"] is True

        extra = directory / "d.txt"
        extra.write_text("new", encoding="utf-8")
        result = status_of(directory, files + [extra], config)
        assert result["up_to_date"] is False
        assert result["changed_since_index"]["added"] == ["d.txt"]

    def test_config_signature_keys(self):
        config = type(
            "C",
            (),
            {
                "chunk_size": 1,
                "chunk_overlap": 2,
                "bm25_language": "en",
                "bm25_skip_stemming": True,
                "embedding_model": "x",
            },
        )()
        signature = IndexManifest.build_config_signature(config)
        assert set(signature) == {
            "chunk_size",
            "chunk_overlap",
            "bm25_language",
            "bm25_skip_stemming",
            "embedding_model",
        }
