#!/usr/bin/env python3
"""把 GPT 样盘装入演练卷 —— 只使用原块写接口。

样盘内容（保护性 MBR + 主/备 GPT + 3 个分区）由 ``app.gpt`` 生成，
本脚本逐块调用 ``PUT /api/clones/{name}/blocks/{no}`` 写入目标克隆盘，
与网页上「装入 GPT 样盘」按钮走的是同一个接口。

用法（先启动服务，例如 docker compose up）::

    python tools/load_sample_gpt.py --disk clone-a
    python tools/load_sample_gpt.py --server http://localhost:8000 --disk clone-b
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 允许从仓库任意目录运行：把仓库根目录加入 import 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import gpt  # noqa: E402


def api(server: str, method: str, path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        server.rstrip("/") + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{method} {path} 失败：HTTP {exc.code} {detail}")


def main() -> None:
    ap = argparse.ArgumentParser(description="经原块写接口把 GPT 样盘装入演练卷")
    ap.add_argument("--server", default="http://localhost:8000", help="演练服务地址")
    ap.add_argument("--disk", default="clone-a", choices=["clone-a", "clone-b"],
                    help="目标克隆盘")
    args = ap.parse_args()

    info = api(args.server, "GET", "/api/clones")
    block_count = info["block_count"]
    blocks = gpt.build_sample_gpt_blocks(block_count)
    print(f"目标盘 {args.disk}：{block_count} 块 × 4 KiB，"
          f"样盘含保护性 MBR + 主/备 GPT + 3 个分区")

    for no, data in enumerate(blocks):
        api(args.server, "PUT", f"/api/clones/{args.disk}/blocks/{no}",
            {"data_hex": data.hex()})
        print(f"\r已写入块 {no + 1}/{len(blocks)}", end="", flush=True)
    print()

    report = api(args.server, "GET", f"/api/clones/{args.disk}/gpt")
    print(f"GPT 检查：{report['message']}")
    for p in report["partitions"]:
        print(f"  分区 {p['no']}: LBA {p['first_lba']}..{p['last_lba']} "
              f"({p['sectors']} 扇区) {p['name']}")
    if report["status"] not in ("ok", "degraded"):
        raise SystemExit("样盘装入后 GPT 检查未通过，请排查")


if __name__ == "__main__":
    main()
