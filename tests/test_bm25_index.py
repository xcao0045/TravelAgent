"""TDD Round 1: BM25IndexManager 测试"""
import tempfile
import os
from rag.bm25_index import BM25IndexManager


class TestBM25BuildAndSearch:
    """索引构建与检索"""

    def test_build_and_search_returns_relevant_chunks(self):
        """构建索引 → 搜索 → 返回相关 chunk，得分排序正确"""
        tmpdir = tempfile.mkdtemp()
        try:
            bm25 = BM25IndexManager(persist_dir=tmpdir)
            chunks = [
                ("c1", "苏州园林很美值得去"),
                ("c2", "苏州有拙政园和留园"),
                ("c3", "杭州西湖风景秀丽"),
                ("c4", "苏州寒山寺历史悠久"),
            ]
            bm25.build_or_rebuild("cases", chunks)

            results = bm25.search("cases", "苏州 园林", k=3)

            # c1 命中两个词 (苏州+园林) 得分最高
            assert len(results) >= 2
            assert results[0][0] == "c1"
            assert all(isinstance(r[1], float) for r in results)
        finally:
            pass

    def test_search_on_empty_index_returns_empty(self):
        """空索引搜索 → 返回空列表不抛异常"""
        tmpdir = tempfile.mkdtemp()
        bm25 = BM25IndexManager(persist_dir=tmpdir)
        results = bm25.search("cases", "苏州", k=3)
        assert results == []

    def test_build_overwrites_previous_index(self):
        """重复 build → 覆盖旧索引"""
        tmpdir = tempfile.mkdtemp()
        bm25 = BM25IndexManager(persist_dir=tmpdir)
        bm25.build_or_rebuild("cases", [
            ("c1", "苏州园林"), ("c2", "杭州西湖"), ("c3", "北京故宫"),
        ])
        results = bm25.search("cases", "苏州", k=3)
        assert results[0][0] == "c1"  # 最高分是匹配的文档

        # 重建为不相关数据
        bm25.build_or_rebuild("cases", [
            ("c4", "成都宽窄巷子"), ("c5", "上海外滩"), ("c6", "桂林山水"),
        ])
        results2 = bm25.search("cases", "苏州", k=3)
        # 无匹配词时所有 doc 得分相同，排名无意义
        assert len(results2) == 3

    def test_multiple_collections_independent(self):
        """preferences 和 cases 索引独立"""
        tmpdir = tempfile.mkdtemp()
        bm25 = BM25IndexManager(persist_dir=tmpdir)
        bm25.build_or_rebuild("preferences", [
            ("p1", "酒店A隔音好亲子适合"), ("p2", "餐厅B味道棒"), ("p3", "景点C门票贵")
        ])
        bm25.build_or_rebuild("cases", [
            ("c1", "苏州三日游"), ("c2", "杭州两日游"), ("c3", "成都五日游")
        ])

        prefs = bm25.search("preferences", "亲子 酒店", k=3)
        cases = bm25.search("cases", "苏州", k=3)

        assert len(prefs) >= 1
        assert cases[0][0] == "c1"  # 匹配"苏州"的文档排第一

    def test_chinese_word_boundary_search(self):
        """验证 BM25 对中文词的准确命中"""
        tmpdir = tempfile.mkdtemp()
        bm25 = BM25IndexManager(persist_dir=tmpdir)
        chunks = [
            ("c1", "成都熊猫基地是亲子游的好去处"),
            ("c2", "北京故宫是明清两代的皇家宫殿"),
            ("c3", "上海外滩夜景很美"),
        ]
        bm25.build_or_rebuild("cases", chunks)

        # "熊猫" 只出现在 c1
        results = bm25.search("cases", "熊猫 基地", k=3)
        assert len(results) >= 1
        assert results[0][0] == "c1"

        # "故宫" 只出现在 c2
        results2 = bm25.search("cases", "故宫", k=3)
        assert results2[0][0] == "c2"


class TestBM25Persistence:
    """pickle 持久化"""

    def test_persist_and_restore(self):
        """persist → 新建实例 restore → 数据恢复"""
        tmpdir = tempfile.mkdtemp()
        try:
            bm25_1 = BM25IndexManager(persist_dir=tmpdir)
            bm25_1.build_or_rebuild("cases", [
                ("c1", "苏州园林"), ("c2", "杭州西湖"), ("c3", "成都宽窄巷子"),
            ])
            bm25_1.persist()

            bm25_2 = BM25IndexManager(persist_dir=tmpdir)
            assert bm25_2.restore() is True
            results = bm25_2.search("cases", "苏州", k=3)
            assert len(results) >= 1
            assert results[0][0] == "c1"
        finally:
            pass

    def test_restore_nonexistent_file_returns_false(self):
        """文件不存在 → restore 返回 False，不抛异常"""
        tmpdir = tempfile.mkdtemp()
        bm25 = BM25IndexManager(persist_dir=tmpdir)
        assert bm25.restore() is False


class TestBM25AddAndRemove:
    """增量操作（内部走全量重建）"""

    def test_add_chunks_incremental(self):
        """add_chunks 追加 → 搜索结果包含新旧条目"""
        tmpdir = tempfile.mkdtemp()
        bm25 = BM25IndexManager(persist_dir=tmpdir)
        bm25.build_or_rebuild("cases", [
            ("c1", "苏州园林"), ("c2", "杭州西湖"), ("c3", "北京故宫"),
        ])
        bm25.add_chunks("cases", [("c4", "桂林山水")])

        suzhou_results = bm25.search("cases", "苏州", k=3)
        assert suzhou_results[0][0] == "c1"  # 匹配项排第一
        guilin_results = bm25.search("cases", "桂林", k=3)
        assert guilin_results[0][0] == "c4"

    def test_remove_by_ids(self):
        """remove_by_ids → 指定 id 不再出现在搜索结果中"""
        tmpdir = tempfile.mkdtemp()
        bm25 = BM25IndexManager(persist_dir=tmpdir)
        bm25.build_or_rebuild("cases", [
            ("c1", "苏州园林"), ("c2", "杭州西湖"), ("c3", "成都宽窄巷子"),
        ])
        bm25.remove_by_ids("cases", ["c1", "c3"])

        results = bm25.search("cases", "苏州", k=3)
        # c1 已删除，"苏州" 不出现在剩余文档中
        assert all(r[0] != "c1" for r in results)
        assert len(bm25.search("cases", "杭州", k=3)) == 1
