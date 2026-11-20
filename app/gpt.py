"""GPT（GUID Partition Table）检查与样盘生成。

演练约定（与需求一致）：

- 逻辑扇区固定 512 字节；一个 4 KiB 块 = 8 个扇区。
- 每份分区表最多 128 项，每项 128 字节（表项数组最多 32 个扇区）。
- 主表：头在 LBA 1，表项数组从 LBA 2 开始；
  备份表：头在最后一个 LBA，表项数组紧邻其前。

``inspect_gpt(read_sectors, disk_sectors)`` 只通过回调读取它需要的扇区，
调用方（``BlockService``）保证这些读取落在同一把盘级锁内，即主备表来自
同一盘状态；同时只读取表所在的块，而不是整盘。
"""

from __future__ import annotations

import hashlib
import json
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional

SECTOR_SIZE = 512
SECTORS_PER_BLOCK = 8  # 4096 // 512
MAX_ENTRIES = 128
ENTRY_SIZE = 128
GPT_SIGNATURE = b"EFI PART"
GPT_REVISION = 0x00010000
HEADER_SIZE = 92
DIGEST_VERSION = "gpt-v1"

# 常见分区类型 GUID（用于样盘与展示）
GUID_EFI_SYSTEM = uuid.UUID("c12a7328-f81f-11d2-ba4b-00a0c93ec93b")
GUID_LINUX_FS = uuid.UUID("0fc63daf-8483-4772-8e79-3d69d8477de4")
GUID_LINUX_HOME = uuid.UUID("933ac7e1-2eb4-4f13-b844-0e14e2aef915")

# 检查结论
STATUS_OK = "ok"                # 主备均有效且布局一致
STATUS_DEGRADED = "degraded"    # 仅一份有效，采用该表
STATUS_INVALID = "invalid"      # 两份都无效
STATUS_INCONSISTENT = "inconsistent"  # 两份都有效但布局不一致


def crc32(data: bytes) -> int:
    """GPT 使用的 CRC32（与 zlib 相同的多项式）。"""
    return zlib.crc32(data) & 0xFFFFFFFF


def guid_str(raw: bytes) -> str:
    """GPT 混合端序的 16 字节 GUID -> 标准文本形式。"""
    return str(uuid.UUID(bytes_le=raw))


# ---------------------------------------------------------------------- 解析
@dataclass
class GptHeader:
    signature: bytes
    revision: int
    header_size: int
    header_crc: int
    current_lba: int
    backup_lba: int
    first_usable_lba: int
    last_usable_lba: int
    disk_guid: bytes
    entries_lba: int
    num_entries: int
    entry_size: int
    array_crc: int

    @classmethod
    def parse(cls, sector: bytes) -> "GptHeader":
        (revision, header_size, header_crc, _reserved) = struct.unpack_from("<IIII", sector, 8)
        (current, backup, first_usable, last_usable) = struct.unpack_from("<QQQQ", sector, 24)
        (entries_lba, num_entries, entry_size, array_crc) = struct.unpack_from("<QIII", sector, 72)
        return cls(
            signature=sector[0:8],
            revision=revision,
            header_size=header_size,
            header_crc=header_crc,
            current_lba=current,
            backup_lba=backup,
            first_usable_lba=first_usable,
            last_usable_lba=last_usable,
            disk_guid=sector[56:72],
            entries_lba=entries_lba,
            num_entries=num_entries,
            entry_size=entry_size,
            array_crc=array_crc,
        )

    def to_dict(self, header_crc_ok: Optional[bool], array_crc_ok: Optional[bool]) -> dict:
        return {
            "current_lba": self.current_lba,
            "backup_lba": self.backup_lba,
            "first_usable_lba": self.first_usable_lba,
            "last_usable_lba": self.last_usable_lba,
            "disk_guid": guid_str(self.disk_guid),
            "entries_lba": self.entries_lba,
            "num_entries": self.num_entries,
            "entry_size": self.entry_size,
            "revision": f"0x{self.revision:08x}",
            "header_size": self.header_size,
            "header_crc_ok": header_crc_ok,
            "array_crc_ok": array_crc_ok,
        }


def parse_entry(raw: bytes, slot: int) -> Optional[dict]:
    """解析一个 128 字节表项；未使用的项（类型 GUID 全零）返回 None。"""
    type_guid = raw[0:16]
    if type_guid == b"\x00" * 16:
        return None
    first_lba, last_lba, attrs = struct.unpack_from("<QQQ", raw, 32)
    name = raw[56:128].decode("utf-16-le", errors="replace").rstrip("\x00")
    return {
        "no": slot + 1,  # 分区编号 = 表项槽位号（从 1 起）
        "type_guid": guid_str(type_guid),
        "unique_guid": guid_str(raw[16:32]),
        "first_lba": first_lba,
        "last_lba": last_lba,
        "sectors": last_lba - first_lba + 1 if last_lba >= first_lba else 0,
        "attrs": f"0x{attrs:016x}",
        "name": name,
    }


# ---------------------------------------------------------------------- 报告
@dataclass
class TableReport:
    """一份 GPT（主或备）的体检结果。"""

    label: str
    header_lba: int
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    header: Optional[GptHeader] = None
    header_crc_ok: Optional[bool] = None
    array_crc_ok: Optional[bool] = None
    entries: list[dict] = field(default_factory=list)
    raw_array: Optional[bytes] = None  # 内部使用：原始表项数组（用于一致性/指纹）

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "header_lba": self.header_lba,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "header": self.header.to_dict(self.header_crc_ok, self.array_crc_ok)
            if self.header is not None else None,
            "entries": list(self.entries),
        }


@dataclass
class GptReport:
    disk_sectors: int
    pmbr: dict
    primary: TableReport
    backup: TableReport
    status: str
    used_table: Optional[str]  # "primary" | "backup" | None
    message: str
    consistency_problems: list[str]
    partitions: list[dict]
    table_digest: Optional[str]

    def to_dict(self) -> dict:
        return {
            "sector_size": SECTOR_SIZE,
            "max_entries": MAX_ENTRIES,
            "entry_size": ENTRY_SIZE,
            "disk_sectors": self.disk_sectors,
            "pmbr": self.pmbr,
            "primary": self.primary.to_dict(),
            "backup": self.backup.to_dict(),
            "layout_consistent": not self.consistency_problems,
            "consistency_problems": list(self.consistency_problems),
            "status": self.status,
            "used_table": self.used_table,
            "message": self.message,
            "table_digest": self.table_digest,
            "partitions": list(self.partitions),
        }


def _report_digest(report: GptReport) -> str:
    """页面所见检查结论的指纹。

    覆盖整份报告（两份表的有效性、错误、头字段、表项、PMBR、采用哪份表）：
    盘上任何会改变页面显示的变化都会使指纹失效，读取方据此要求重新检查。
    """
    canonical = report.to_dict()
    canonical.pop("table_digest", None)
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(DIGEST_VERSION.encode("ascii") + b"|" + blob).hexdigest()


# ---------------------------------------------------------------------- 检查
def _check_pmbr(mbr: bytes) -> dict:
    problems = []
    if mbr[510:512] != b"\x55\xaa":
        problems.append("缺少 0x55AA 引导签名")
    entry0 = mbr[446:462]
    if entry0[4] != 0xEE:
        problems.append("缺少 0xEE 保护性分区项")
    else:
        start_lba, size_lba = struct.unpack_from("<II", entry0, 8)
        if start_lba != 1:
            problems.append(f"保护性分区项起始 LBA 为 {start_lba}，应为 1")
        if size_lba == 0:
            problems.append("保护性分区项大小为 0")
    return {"ok": not problems, "problems": problems}


def _inspect_table(
    read_sectors: Callable[[int, int], bytes],
    disk_sectors: int,
    is_primary: bool,
) -> TableReport:
    label = "主表" if is_primary else "备份表"
    header_lba = 1 if is_primary else disk_sectors - 1
    t = TableReport(label=label, header_lba=header_lba)

    raw = read_sectors(header_lba, 1)
    hdr = GptHeader.parse(raw)
    t.header = hdr

    # 1) 签名与头大小：不具备继续解析的条件时直接结束
    if hdr.signature != GPT_SIGNATURE:
        t.errors.append(f"头签名无效：LBA {header_lba} 处未找到 'EFI PART'")
        return t
    if not (HEADER_SIZE <= hdr.header_size <= SECTOR_SIZE):
        t.errors.append(f"头大小 {hdr.header_size} 无效（应在 {HEADER_SIZE}..{SECTOR_SIZE} 之间）")
        return t

    # 2) 头部 CRC（CRC 字段清零后对前 header_size 字节计算）
    hdr_bytes = bytearray(raw[: hdr.header_size])
    hdr_bytes[16:20] = b"\x00\x00\x00\x00"
    t.header_crc_ok = crc32(bytes(hdr_bytes)) == hdr.header_crc
    if not t.header_crc_ok:
        t.errors.append("头部 CRC32 校验失败")

    if hdr.revision != GPT_REVISION:
        t.warnings.append(f"修订号 0x{hdr.revision:08x}，常见值为 0x00010000")

    # 3) 表的位置：头位置、互为备份指针、表项数组位置
    if hdr.current_lba != header_lba:
        t.errors.append(f"当前 LBA={hdr.current_lba}，应为 {header_lba}（表头位置不符）")
    expected_backup = (disk_sectors - 1) if is_primary else 1
    if hdr.backup_lba != expected_backup:
        t.errors.append(f"备份 LBA={hdr.backup_lba}，应为 {expected_backup}")

    entry_params_ok = True
    if hdr.entry_size != ENTRY_SIZE:
        t.errors.append(f"表项大小 {hdr.entry_size} 字节，应为 {ENTRY_SIZE}")
        entry_params_ok = False
    if not (1 <= hdr.num_entries <= MAX_ENTRIES):
        t.errors.append(f"表项数 {hdr.num_entries} 超出范围 1..{MAX_ENTRIES}")
        entry_params_ok = False

    entry_sectors = 0
    if entry_params_ok:
        entry_sectors = -(-hdr.num_entries * hdr.entry_size // SECTOR_SIZE)  # 向上取整
        expected_entries_lba = 2 if is_primary else disk_sectors - 1 - entry_sectors
        if hdr.entries_lba != expected_entries_lba:
            t.errors.append(
                f"表项数组位于 LBA {hdr.entries_lba}，应为 {expected_entries_lba}（表位置不符）"
            )

    # 4) 表项数组：位置在盘内才读取，并核对数组 CRC
    array_crc_ok = None
    if entry_params_ok and 0 <= hdr.entries_lba and hdr.entries_lba + entry_sectors <= disk_sectors:
        t.raw_array = read_sectors(hdr.entries_lba, entry_sectors)[: hdr.num_entries * hdr.entry_size]
        array_crc_ok = crc32(t.raw_array) == hdr.array_crc
        if not array_crc_ok:
            t.errors.append("表项数组 CRC32 校验失败")
    elif entry_params_ok:
        t.errors.append(f"表项数组 [{hdr.entries_lba}, {hdr.entries_lba + entry_sectors - 1}] 越出磁盘")
    t.array_crc_ok = array_crc_ok

    # 5) 可用范围与分区边界
    if entry_params_ok:
        min_first_usable = 2 + entry_sectors
        max_last_usable = disk_sectors - 2 - entry_sectors
        if hdr.first_usable_lba < min_first_usable:
            t.errors.append(
                f"首个可用 LBA {hdr.first_usable_lba} 与主表项数组重叠（应 ≥ {min_first_usable}）"
            )
        if hdr.last_usable_lba > max_last_usable:
            t.errors.append(
                f"末个可用 LBA {hdr.last_usable_lba} 与备份表项数组重叠（应 ≤ {max_last_usable}）"
            )
        if hdr.first_usable_lba > hdr.last_usable_lba:
            t.errors.append(
                f"可用范围为空：[{hdr.first_usable_lba}, {hdr.last_usable_lba}]"
            )

    # 数组 CRC 通过才相信表项内容，逐项检查边界与重叠
    if t.raw_array is not None and array_crc_ok:
        for slot in range(hdr.num_entries):
            entry = parse_entry(t.raw_array[slot * ENTRY_SIZE:(slot + 1) * ENTRY_SIZE], slot)
            if entry is not None:
                t.entries.append(entry)
        if hdr.first_usable_lba <= hdr.last_usable_lba:
            seen: list[dict] = []
            for p in t.entries:
                if p["first_lba"] > p["last_lba"]:
                    t.errors.append(
                        f"分区 {p['no']} 起始 LBA {p['first_lba']} 大于结束 LBA {p['last_lba']}"
                    )
                    continue
                if p["first_lba"] < hdr.first_usable_lba or p["last_lba"] > hdr.last_usable_lba:
                    t.errors.append(
                        f"分区 {p['no']} [{p['first_lba']}, {p['last_lba']}] 越出可用边界 "
                        f"[{hdr.first_usable_lba}, {hdr.last_usable_lba}]"
                    )
                for q in seen:
                    if p["first_lba"] <= q["last_lba"] and q["first_lba"] <= p["last_lba"]:
                        t.errors.append(f"分区 {p['no']} 与分区 {q['no']} 重叠")
                seen.append(p)

    t.valid = not t.errors
    return t


def _consistency_problems(primary: TableReport, backup: TableReport) -> list[str]:
    """主备均有效时核对两者描述的布局是否一致。"""
    problems: list[str] = []
    ph, bh = primary.header, backup.header
    assert ph is not None and bh is not None
    checks = [
        (ph.disk_guid == bh.disk_guid, "磁盘 GUID"),
        (ph.first_usable_lba == bh.first_usable_lba, "首个可用 LBA"),
        (ph.last_usable_lba == bh.last_usable_lba, "末个可用 LBA"),
        (ph.num_entries == bh.num_entries, "表项数"),
        (ph.entry_size == bh.entry_size, "表项大小"),
    ]
    for same, name in checks:
        if not same:
            problems.append(f"{name} 不一致")
    if primary.raw_array != backup.raw_array:
        problems.append("表项数组内容不一致")
    return problems


def inspect_gpt(
    read_sectors: Callable[[int, int], bytes],
    disk_sectors: int,
) -> GptReport:
    """检查整盘 GPT。``read_sectors(lba, count)`` 由调用方提供并保证一致性。"""
    pmbr = _check_pmbr(read_sectors(0, 1))
    primary = _inspect_table(read_sectors, disk_sectors, is_primary=True)
    backup = _inspect_table(read_sectors, disk_sectors, is_primary=False)

    consistency: list[str] = []
    if primary.valid and backup.valid:
        consistency = _consistency_problems(primary, backup)
        if consistency:
            status, used = STATUS_INCONSISTENT, None
            message = "主、备表均有效但布局不一致（" + "；".join(consistency) + "），已停用分区读取。"
        else:
            status, used = STATUS_OK, "primary"
            message = "主、备 GPT 均有效且布局一致，采用主表。"
    elif primary.valid:
        status, used = STATUS_DEGRADED, "primary"
        message = "备份表失效（" + "；".join(backup.errors) + "），采用主表。"
    elif backup.valid:
        status, used = STATUS_DEGRADED, "backup"
        message = "主表失效（" + "；".join(primary.errors) + "），采用备份表。"
    else:
        status, used = STATUS_INVALID, None
        message = (
            "主表失效（" + "；".join(primary.errors) + "）；"
            "备份表失效（" + "；".join(backup.errors) + "）。"
            "无可用分区表，已停用分区读取。"
        )

    used_report = primary if used == "primary" else backup if used == "backup" else None
    partitions = list(used_report.entries) if used_report is not None else []

    report = GptReport(
        disk_sectors=disk_sectors,
        pmbr=pmbr,
        primary=primary,
        backup=backup,
        status=status,
        used_table=used,
        message=message,
        consistency_problems=consistency,
        partitions=partitions,
        table_digest=None,
    )
    if used_report is not None:
        report.table_digest = _report_digest(report)
    return report


# ---------------------------------------------------------------------- 样盘
def build_header(
    *,
    current_lba: int,
    backup_lba: int,
    first_usable_lba: int,
    last_usable_lba: int,
    disk_guid: bytes,
    entries_lba: int,
    num_entries: int,
    entry_size: int,
    array_crc: int,
) -> bytes:
    """构造一个 512 字节的 GPT 头（自动计算头部 CRC）。"""
    buf = bytearray(SECTOR_SIZE)
    buf[0:8] = GPT_SIGNATURE
    struct.pack_into("<IIII", buf, 8, GPT_REVISION, HEADER_SIZE, 0, 0)
    struct.pack_into("<QQQQ", buf, 24, current_lba, backup_lba, first_usable_lba, last_usable_lba)
    buf[56:72] = disk_guid
    struct.pack_into("<QIII", buf, 72, entries_lba, num_entries, entry_size, array_crc)
    struct.pack_into("<I", buf, 16, crc32(bytes(buf[:HEADER_SIZE])))
    return bytes(buf)


def build_entry(
    type_guid: uuid.UUID,
    unique_guid: uuid.UUID,
    first_lba: int,
    last_lba: int,
    name: str,
    attrs: int = 0,
) -> bytes:
    """构造一个 128 字节的分区表项。"""
    e = bytearray(ENTRY_SIZE)
    e[0:16] = type_guid.bytes_le
    e[16:32] = unique_guid.bytes_le
    struct.pack_into("<QQQ", e, 32, first_lba, last_lba, attrs)
    name_bytes = name.encode("utf-16-le")[:72]
    e[56:56 + len(name_bytes)] = name_bytes
    return bytes(e)


def _choose_entry_sectors(disk_sectors: int, min_usable: int = 16) -> int:
    """为样盘挑选表项数组扇区数：优先满配 128 项（32 扇区），盘小则递减。"""
    es = MAX_ENTRIES * ENTRY_SIZE // SECTOR_SIZE  # 32
    while es > 1 and 3 + 2 * es + min_usable > disk_sectors:
        es -= 1
    if 3 + 2 * es > disk_sectors:
        raise ValueError(f"磁盘太小，放不下 GPT：仅 {disk_sectors} 个扇区")
    return es


def build_sample_gpt_image(block_count: int) -> bytes:
    """生成一张 GPT 样盘的完整字节映像（block_count × 4 KiB）。

    布局：保护性 MBR + 主备 GPT，3 个分区（EFI System / Linux root /
    Linux data），每个分区首扇区写入可识别的标识文本。
    """
    disk_sectors = block_count * SECTORS_PER_BLOCK
    entry_sectors = _choose_entry_sectors(disk_sectors)
    num_entries = entry_sectors * (SECTOR_SIZE // ENTRY_SIZE)
    first_usable = 2 + entry_sectors
    last_usable = disk_sectors - 2 - entry_sectors
    usable = last_usable - first_usable + 1
    third = usable // 3

    disk_guid = hashlib.sha256(b"gpt-sample-disk").digest()[:16]
    parts = [
        (GUID_EFI_SYSTEM, "EFI System", first_usable, first_usable + third - 1),
        (GUID_LINUX_FS, "Linux root", first_usable + third, first_usable + 2 * third - 1),
        (GUID_LINUX_HOME, "Linux data", first_usable + 2 * third, last_usable),
    ]

    img = bytearray(block_count * 4096)

    # LBA 0：保护性 MBR
    mbr = bytearray(SECTOR_SIZE)
    mbr[446 + 1:446 + 4] = b"\x00\x02\x00"      # 起始 CHS
    mbr[446 + 4] = 0xEE                          # 分区类型：GPT 保护
    mbr[446 + 5:446 + 8] = b"\xff\xff\xff"      # 结束 CHS
    struct.pack_into("<I", mbr, 446 + 8, 1)      # 起始 LBA
    struct.pack_into("<I", mbr, 446 + 12, min(disk_sectors - 1, 0xFFFFFFFF))
    mbr[510:512] = b"\x55\xaa"
    img[0:SECTOR_SIZE] = mbr

    # 表项数组（主备内容相同）
    array = bytearray(num_entries * ENTRY_SIZE)
    for i, (type_guid, name, first, last) in enumerate(parts):
        unique = uuid.UUID(bytes=hashlib.sha256(f"gpt-sample-part-{i + 1}".encode()).digest()[:16])
        array[i * ENTRY_SIZE:(i + 1) * ENTRY_SIZE] = build_entry(type_guid, unique, first, last, name)
    array_crc = crc32(bytes(array))

    # 主表：头 @ LBA 1，表项 @ LBA 2..
    img[1 * SECTOR_SIZE:2 * SECTOR_SIZE] = build_header(
        current_lba=1, backup_lba=disk_sectors - 1,
        first_usable_lba=first_usable, last_usable_lba=last_usable,
        disk_guid=disk_guid, entries_lba=2,
        num_entries=num_entries, entry_size=ENTRY_SIZE, array_crc=array_crc,
    )
    img[2 * SECTOR_SIZE:2 * SECTOR_SIZE + len(array)] = array

    # 备份表：表项紧邻备份头之前，头 @ 最后 LBA
    backup_entries_lba = disk_sectors - 1 - entry_sectors
    img[backup_entries_lba * SECTOR_SIZE:backup_entries_lba * SECTOR_SIZE + len(array)] = array
    img[(disk_sectors - 1) * SECTOR_SIZE:disk_sectors * SECTOR_SIZE] = build_header(
        current_lba=disk_sectors - 1, backup_lba=1,
        first_usable_lba=first_usable, last_usable_lba=last_usable,
        disk_guid=disk_guid, entries_lba=backup_entries_lba,
        num_entries=num_entries, entry_size=ENTRY_SIZE, array_crc=array_crc,
    )

    # 每个分区首扇区写入标识文本，便于在页面上辨认
    for i, (_t, name, first, last) in enumerate(parts):
        marker = (
            f"SAMPLE GPT DISK | partition {i + 1} ({name}) | "
            f"LBA {first}..{last} | sector 0 of {last - first + 1}\n"
        ).encode("ascii")
        sector = (marker + b"." * SECTOR_SIZE)[:SECTOR_SIZE]
        img[first * SECTOR_SIZE:(first + 1) * SECTOR_SIZE] = sector

    return bytes(img)


def build_sample_gpt_blocks(block_count: int) -> list[bytes]:
    """把样盘映像切成 4 KiB 块，可逐块经原块写接口装入演练卷。"""
    img = build_sample_gpt_image(block_count)
    return [img[i * 4096:(i + 1) * 4096] for i in range(block_count)]
