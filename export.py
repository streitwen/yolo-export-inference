#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把装甲板关键点模型 best.pt 导出为 ONNX / TorchScript。

用法：
    python export.py --weights models/best.pt --formats onnx torchscript

导出完成后会：
    1) 打印每种格式的文件大小；
    2) 用 onnx.checker 校验 ONNX 计算图是否合法（仅 ONNX）；
    3) 把导出信息写入 results/export_report.json，方便写 README。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 装甲板模型格式导出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", "-w", default="models/best.pt", help="原始 .pt 权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="导出时的输入尺寸，必须与推理时一致")
    parser.add_argument("--formats", nargs="+", default=["onnx", "torchscript"],
                        choices=["onnx", "torchscript"], help="要导出的目标格式")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--device", default="cpu", help="导出使用的设备，CPU 即可")
    parser.add_argument("--report", default="results/export_report.json", help="导出报告保存路径")
    return parser.parse_args()


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def export_one(weights: Path, fmt: str, imgsz: int, opset: int, device: str) -> Path:
    """调用 Ultralytics 完成单种格式的导出，返回导出文件路径。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kwargs = dict(format=fmt, imgsz=imgsz, device=device)
    if fmt == "onnx":
        # dynamic=False：固定 1x3x640x640 输入，批量维度也不留空，便于推理端做静态优化
        # simplify=False：不依赖 onnxslim，少一个可能装不上的可选依赖
        kwargs.update(dynamic=False, simplify=False, opset=opset)
    elif fmt == "torchscript":
        kwargs.update(optimize=False)

    print(f"\n[导出] {fmt} <- {weights.name} (imgsz={imgsz}) ...")
    out = model.export(**kwargs)
    return Path(out)


def check_onnx(path: Path) -> dict:
    """校验 ONNX 图并读取输入输出张量形状，返回摘要。"""
    import onnx

    onnx.checker.check_model(str(path))
    model = onnx.load(str(path))

    def dims(value_info) -> list:
        shape = []
        for d in value_info.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value else d.dim_param or "?")
        return shape

    return {
        "checker": "passed",
        "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in model.opset_import],
        "inputs": [{"name": i.name, "shape": dims(i)} for i in model.graph.input],
        "outputs": [{"name": o.name, "shape": dims(o)} for o in model.graph.output],
    }


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).expanduser()
    if not weights.is_file():
        print(f"[错误] 找不到原始权重: {weights}")
        return 2

    report = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_weights": str(weights),
        "imgsz": args.imgsz,
        "formats": {},
    }

    for fmt in args.formats:
        t0 = time.perf_counter()
        out = export_one(weights, fmt, args.imgsz, args.opset, args.device)
        cost = time.perf_counter() - t0
        if not out.is_file():
            print(f"[错误] {fmt} 导出失败：未生成文件 {out}")
            return 2

        info = {
            "path": str(out),
            "size_bytes": out.stat().st_size,
            "size_mib": round(out.stat().st_size / 1024 / 1024, 2),
            "sha256": sha256(out),
            "export_seconds": round(cost, 1),
        }
        if fmt == "onnx":
            info["graph"] = check_onnx(out)
        report["formats"][fmt] = info
        print(f"[完成] {fmt}: {out}  ({info['size_mib']} MiB, 用时 {cost:.1f}s)")

    report["source"] = {
        "path": str(weights),
        "size_bytes": weights.stat().st_size,
        "size_mib": round(weights.stat().st_size / 1024 / 1024, 2),
        "sha256": sha256(weights),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[信息] 导出报告: {report_path}")

    print("\n================ 导出汇总 ================")
    print(f"{'格式':<12}{'大小 (MiB)':>12}{'耗时 (s)':>12}")
    print(f"{'pt':<12}{report['source']['size_mib']:>12.2f}{'-':>12}")
    for fmt, info in report["formats"].items():
        print(f"{fmt:<12}{info['size_mib']:>12.2f}{info['export_seconds']:>12.1f}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
