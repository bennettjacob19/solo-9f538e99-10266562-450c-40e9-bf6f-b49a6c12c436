"""GPT 检查与分区读取的端到端测试。

覆盖：样盘检查、主/备表单独失效、双表失效、布局不一致、分区扇区读取、
越界拒绝、表指纹过期拒绝（409 语义）、只读所需块、覆盖/丢弃与检查读取交错、
跨盘并发（一盘的 GPT 操作不阻塞另一盘的单块读取）。
"""
import hashlib
import threading
import time
import uuid

import pytest

from app import gpt
from app.storage import (
    BLOCK_SIZE, BlockService, StorageError, TableChangedError,
)

BLOCK_COUNT = 32
DISK_SECTORS = BLOCK_COUNT * 8  # 256
# 满配 128 项时一次检查应读取的块：主表区 0-4，备份表区 27-31
EXPECTED_CHECK_BLOCKS = [0, 1, 2, 3, 4, 27, 28, 29, 30, 31]


@pytest.fixture()
def svc(tmp_path):
    s = BlockService(tmp_path / "data", BLOCK_COUNT)
    s.initialize()
    yield s
    s.stop_worker()


def load_sample(svc, name="clone-a"):
    """经块写接口把样盘装入克隆盘，返回块列表。"""
    blocks = gpt.build_sample_gpt_blocks(svc.block_count)
    for i, b in enumerate(blocks):
        svc.overwrite_block(name, i, b)
    return blocks


def sample_image(svc):
    return b"".join(gpt.build_sample_gpt_blocks(svc.block_count))


# ------------------------------------------------------------------ 检查
def test_sample_disk_checks_ok(svc):
    load_sample(svc)
    r = svc.gpt_check("clone-a")
    assert r["status"] == "ok"
    assert r["used_table"] == "primary"
    assert r["pmbr"]["ok"] is True
    assert r["primary"]["valid"] and r["backup"]["valid"]
    assert r["layout_consistent"] is True
    assert r["table_digest"]
    # 分区编号与起止 LBA
    parts = {p["no"]: p for p in r["partitions"]}
    assert set(parts) == {1, 2, 3}
    assert (parts[1]["first_lba"], parts[1]["last_lba"]) == (34, 96)
    assert (parts[2]["first_lba"], parts[2]["last_lba"]) == (97, 159)
    assert (parts[3]["first_lba"], parts[3]["last_lba"]) == (160, 222)
    assert parts[1]["sectors"] == 63
    # 只读表所在的块，而不是整盘
    assert r["blocks_read"] == EXPECTED_CHECK_BLOCKS


def test_check_on_blank_disk_reports_both_invalid(svc):
    r = svc.gpt_check("clone-a")  # 模板内容不是 GPT
    assert r["status"] == "invalid"
    assert r["used_table"] is None
    assert r["table_digest"] is None
    assert r["partitions"] == []
    assert not r["primary"]["valid"] and not r["backup"]["valid"]
    assert r["primary"]["errors"] and r["backup"]["errors"]


def test_primary_corrupted_falls_back_to_backup(svc):
    blocks = load_sample(svc)
    # 破坏主表头（块 0 内 LBA 1）的一个字节 -> 头 CRC 失败
    b0 = bytearray(blocks[0])
    b0[512 + 10] ^= 0xFF
    svc.overwrite_block("clone-a", 0, bytes(b0))

    r = svc.gpt_check("clone-a")
    assert r["status"] == "degraded"
    assert r["used_table"] == "backup"
    assert not r["primary"]["valid"]
    assert any("CRC" in e for e in r["primary"]["errors"])
    assert "主表失效" in r["message"]
    assert len(r["partitions"]) == 3  # 备份表照常给出分区


def test_backup_corrupted_uses_primary(svc):
    blocks = load_sample(svc)
    # 破坏备份表项数组（块 27 内）的一个字节 -> 备份数组 CRC 失败
    b27 = bytearray(blocks[27])
    b27[7 * 512 + 100] ^= 0x01
    svc.overwrite_block("clone-a", 27, bytes(b27))

    r = svc.gpt_check("clone-a")
    assert r["status"] == "degraded"
    assert r["used_table"] == "primary"
    assert not r["backup"]["valid"]
    assert any("CRC" in e for e in r["backup"]["errors"])
    assert "备份表失效" in r["message"]


def test_both_tables_invalid_disables_reads(svc):
    blocks = load_sample(svc)
    b0 = bytearray(blocks[0])
    b0[512] ^= 0xFF  # 主表头签名
    svc.overwrite_block("clone-a", 0, bytes(b0))
    b31 = bytearray(blocks[31])
    b31[7 * 512] ^= 0xFF  # 备份表头签名
    svc.overwrite_block("clone-a", 31, bytes(b31))

    r = svc.gpt_check("clone-a")
    assert r["status"] == "invalid"
    assert r["table_digest"] is None
    assert "已停用分区读取" in r["message"]
    # 无可用表：读取被拒绝并提示重新检查
    with pytest.raises(TableChangedError):
        svc.gpt_read_partition_sector("clone-a", 1, 0, "whatever")


def test_valid_but_inconsistent_layouts_disable_reads(svc):
    blocks = load_sample(svc)
    img = bytearray(b"".join(blocks))
    # 只改备份表的布局：把分区 2 的结束 LBA 改小，并重算备份表两处 CRC，
    # 使备份表本身有效但与主表布局不一致
    array = bytearray(img[223 * 512:223 * 512 + 128 * 128])
    unique2 = uuid.UUID(bytes=hashlib.sha256(b"gpt-sample-part-2").digest()[:16])
    array[128:256] = gpt.build_entry(gpt.GUID_LINUX_FS, unique2, 97, 150, "Linux root")
    new_crc = gpt.crc32(bytes(array))
    img[223 * 512:223 * 512 + len(array)] = array
    img[255 * 512:256 * 512] = gpt.build_header(
        current_lba=255, backup_lba=1, first_usable_lba=34, last_usable_lba=222,
        disk_guid=hashlib.sha256(b"gpt-sample-disk").digest()[:16],
        entries_lba=223, num_entries=128, entry_size=128, array_crc=new_crc)
    for i in range(27, 32):
        svc.overwrite_block("clone-a", i, bytes(img[i * 4096:(i + 1) * 4096]))

    r = svc.gpt_check("clone-a")
    assert r["primary"]["valid"] and r["backup"]["valid"]
    assert r["status"] == "inconsistent"
    assert r["table_digest"] is None
    assert r["consistency_problems"]
    assert "布局不一致" in r["message"]
    with pytest.raises(TableChangedError):
        svc.gpt_read_partition_sector("clone-a", 1, 0, "whatever")


def test_table_position_is_checked(svc):
    blocks = load_sample(svc)
    img = bytearray(b"".join(blocks))
    # 把主表头里的表项数组位置改错（重算 CRC 使头部自身有效）
    hdr = bytearray(img[512:1024])
    import struct
    struct.pack_into("<Q", hdr, 72, 5)  # entries_lba = 5（应为 2）
    struct.pack_into("<I", hdr, 16, 0)
    import zlib
    struct.pack_into("<I", hdr, 16, zlib.crc32(bytes(hdr[:92])) & 0xFFFFFFFF)
    img[512:1024] = hdr
    svc.overwrite_block("clone-a", 0, bytes(img[0:4096]))

    r = svc.gpt_check("clone-a")
    assert r["status"] == "degraded"
    assert r["used_table"] == "backup"
    assert any("表位置不符" in e or "位置" in e for e in r["primary"]["errors"])


# ------------------------------------------------------------------ 分区读取
def test_partition_sector_read(svc):
    load_sample(svc)
    check = svc.gpt_check("clone-a")
    digest = check["table_digest"]
    img = sample_image(svc)

    r = svc.gpt_read_partition_sector("clone-a", 1, 0, digest)
    assert r["lba"] == 34
    assert r["length"] == 512
    assert r["data_hex"] == img[34 * 512:35 * 512].hex()
    assert b"SAMPLE GPT DISK" in bytes.fromhex(r["data_hex"])
    # 复核表 + 目标扇区：块 4 含 LBA 34，已在表区中，无额外块
    assert r["blocks_read"] == EXPECTED_CHECK_BLOCKS

    # 分区内最后一扇区
    last = svc.gpt_read_partition_sector("clone-a", 1, 62, digest)
    assert last["lba"] == 96


def test_partition_read_only_reads_target_block_beyond_tables(svc):
    load_sample(svc)
    digest = svc.gpt_check("clone-a")["table_digest"]
    # 分区 2 的第 10 扇区 -> LBA 107 -> 块 13（不在表区）
    r = svc.gpt_read_partition_sector("clone-a", 2, 10, digest)
    assert r["lba"] == 107
    assert r["blocks_read"] == sorted(EXPECTED_CHECK_BLOCKS + [13])


def test_out_of_bounds_sector_rejected(svc):
    load_sample(svc)
    digest = svc.gpt_check("clone-a")["table_digest"]
    with pytest.raises(StorageError, match="越界"):
        svc.gpt_read_partition_sector("clone-a", 1, 63, digest)  # 分区 1 共 63 扇区
    with pytest.raises(StorageError, match="越界"):
        svc.gpt_read_partition_sector("clone-a", 1, -1, digest)
    with pytest.raises(StorageError, match="不存在"):
        svc.gpt_read_partition_sector("clone-a", 9, 0, digest)
    with pytest.raises(StorageError, match="digest"):
        svc.gpt_read_partition_sector("clone-a", 1, 0, "")


def test_stale_digest_rejected_and_recheck_restores(svc):
    load_sample(svc)
    digest = svc.gpt_check("clone-a")["table_digest"]

    # 交错执行：覆盖 GPT 区域（块 0）-> 表指纹失效，读取被拒并提示重新检查
    svc.overwrite_block("clone-a", 0, b"\x00" * BLOCK_SIZE)
    with pytest.raises(TableChangedError, match="重新检查"):
        svc.gpt_read_partition_sector("clone-a", 1, 0, digest)

    # 重新检查后主表失效、采用备份表，读取恢复
    r2 = svc.gpt_check("clone-a")
    assert r2["status"] == "degraded" and r2["used_table"] == "backup"
    out = svc.gpt_read_partition_sector("clone-a", 1, 0, r2["table_digest"])
    assert out["lba"] == 34
    assert out["table"] == "backup"


def test_interleaved_writes_outside_gpt_keep_digest(svc):
    """覆盖/丢弃普通数据块不改变分区表：指纹不变，读取看到新内容。"""
    load_sample(svc)
    digest = svc.gpt_check("clone-a")["table_digest"]

    payload = b"INTERLEAVED!" + b"\xAA" * (BLOCK_SIZE - 12)
    svc.overwrite_block("clone-a", 13, payload)  # 块 13 在分区 2 数据区
    svc.discard_block("clone-a", 14)

    r = svc.gpt_read_partition_sector("clone-a", 2, 10, digest)  # LBA 107 在块 13
    assert bytes.fromhex(r["data_hex"]) == payload[3 * 512:4 * 512]
    z = svc.gpt_read_partition_sector("clone-a", 2, 15, digest)  # LBA 112 在块 14
    assert bytes.fromhex(z["data_hex"]) == b"\x00" * 512


def test_discard_of_gpt_block_invalidates_digest(svc):
    load_sample(svc)
    digest = svc.gpt_check("clone-a")["table_digest"]
    svc.discard_block("clone-a", 31)  # 备份表头所在块
    with pytest.raises(TableChangedError):
        svc.gpt_read_partition_sector("clone-a", 1, 0, digest)
    r = svc.gpt_check("clone-a")
    assert r["status"] == "degraded" and r["used_table"] == "primary"


# ------------------------------------------------------------------ 并发
def test_gpt_check_on_one_disk_does_not_block_other_disk(svc):
    load_sample(svc, "clone-a")
    svc.disk_delay = 0.2  # 模拟慢盘：每块 200ms，检查约 10 块 ≈ 2s
    started = threading.Event()

    def hook(name, _no):
        if name == "clone-a":
            started.set()

    svc.block_read_hook = hook
    result = {}
    t = threading.Thread(target=lambda: result.update(svc.gpt_check("clone-a")))
    t.start()
    assert started.wait(5), "GPT check did not start reading"

    t0 = time.monotonic()
    r = svc.read_block("clone-b", 0)  # 另一盘的单块读取应照常完成
    elapsed = time.monotonic() - t0
    t.join(10)

    assert not t.is_alive()
    assert result["status"] == "ok"
    assert len(r.data) == BLOCK_SIZE
    # 若被 clone-a 的检查阻塞，耗时会接近 2s；正常只付自己的一次盘延迟
    assert elapsed < 1.5


def test_partition_read_on_one_disk_does_not_block_other_disk(svc):
    load_sample(svc, "clone-a")
    digest = svc.gpt_check("clone-a")["table_digest"]
    svc.disk_delay = 0.2
    started = threading.Event()

    def hook(name, _no):
        if name == "clone-a":
            started.set()

    svc.block_read_hook = hook
    result = {}
    t = threading.Thread(
        target=lambda: result.update(
            svc.gpt_read_partition_sector("clone-a", 3, 5, digest)))
    t.start()
    assert started.wait(5)

    t0 = time.monotonic()
    svc.read_block("clone-b", 1)
    elapsed = time.monotonic() - t0
    t.join(10)

    assert not t.is_alive()
    assert result["lba"] == 165
    assert elapsed < 1.5


def test_check_reads_consistent_state_under_concurrent_writes(svc):
    """检查期间该盘的写入被序列化：报告必然对应某个完整盘状态。"""
    load_sample(svc, "clone-a")
    svc.disk_delay = 0.05
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            try:
                svc.overwrite_block("clone-a", 10 + (i % 10), b"w" * BLOCK_SIZE)
                i += 1
            except StorageError as exc:  # pragma: no cover
                errors.append(exc)

    w = threading.Thread(target=writer)
    w.start()
    try:
        for _ in range(5):
            r = svc.gpt_check("clone-a")
            assert r["status"] == "ok"  # 数据区写入不影响 GPT 结论
            assert r["table_digest"]
    finally:
        stop.set()
        w.join(10)
    assert not errors
