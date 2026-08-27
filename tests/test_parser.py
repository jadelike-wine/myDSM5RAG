"""DocumentParser 单元测试"""

import tempfile
from pathlib import Path

from dsm5_rag.parser import DocumentParser


class TestDocumentParser:
    """测试 DocumentParser 文档解析器"""

    def test_parse_txt_success(self):
        """测试 TXT 文件解析"""
        content = "这是测试内容\n第二行内容"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = Path(f.name)

        try:
            docs = DocumentParser.parse_txt(tmp_path)
            assert len(docs) == 1
            assert docs[0].text == content
            assert docs[0].metadata["file_name"] == tmp_path.name
            assert docs[0].metadata["file_type"] == "txt"
        finally:
            tmp_path.unlink()

    def test_parse_txt_empty(self):
        """测试空 TXT 文件解析"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("   \n  \n")
            tmp_path = Path(f.name)

        try:
            docs = DocumentParser.parse_txt(tmp_path)
            assert len(docs) == 0
        finally:
            tmp_path.unlink()

    def test_parse_txt_non_existent(self):
        """测试不存在的文件"""
        docs = DocumentParser.parse_txt(Path("/nonexistent/file.txt"))
        assert len(docs) == 0

    def test_parse_pdf_non_existent(self):
        """测试不存在的PDF文件"""
        docs = DocumentParser.parse_pdf(Path("/nonexistent/file.pdf"))
        assert len(docs) == 0

    def test_parse_docx_non_existent(self):
        """测试不存在的DOCX文件"""
        docs = DocumentParser.parse_docx(Path("/nonexistent/file.docx"))
        assert len(docs) == 0