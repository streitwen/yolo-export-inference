#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在同一段视频、同一组参数下对比 .pt / .onnx / .torchscript。

测量方法（对应作业里"预热后多次测量取平均"的要求）：
  1. 从视频中等间隔抽取若干帧，先在内存里缓存成 JPEG 数据（几十 MB，
     而不是几 GB 的原始帧），每一轮再解码成 numpy 图像，保证每次喂给模型的输入完全相同；
  2. 每个模型先预热若干帧，预热结果不计入统计；
  3. 重复多轮，且每轮轮换模型顺序，让 CPU 温度和后台负载对三个模型的影响尽量均衡；
  4. 计时只包住"预处理 + 推理 + 后处理"，不含视频解码和写盘；
  5. 以 PT 结果为基准，按类别 + 框 IoU 匹配，统计检测差异和关键点偏差。

端到端 FPS（含 4K 解码、绘图、写盘）请看 inference.py 的输出，两类指标分开报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from armor_core import auto_style, draw_detections, match_detections, put_text
from inference import (OnnxRuntimePredictor, UltralyticsPredictor, model_tag, read_names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PT / ONNX / TorchScript 同条件性能对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+",
                   default=["models/best.pt", "models/best.onnx", "models/best.torchscript"],
                   help="参与对比的模型，第一个作为检测结果基准")
    p.add_argument("--source", "-s", default="data/robomaster.mp4", help="测试视频")
    p.add_argument("--frames", type=int, default=60, help="等间隔抽取的测试帧数")
    p.add_argument("--repeat", type=int, default=3, help="重复测量轮数")
    p.add_argument("--warmup", type=int, default=8, help="每个模型的预热帧数（不计入统计）")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-det", type=int, default=30)
    p.add_argument("--device", default="cpu", help="Ultralytics 后端设备")
    p.add_argument("--threads", type=int, default=12,
                   help="两边统一的线程数，0 = 各自默认（默认 12，本机实测最优）")
    p.add_argument("--out-dir", default="results", help="结果目录")
    p.add_argument("--compare-frames", type=int, default=3, help="生成几张左右对比图")
    return p.parse_args()


def load_samples(path: Path, indices: list[int]) -> list[tuple[int, bytes]]:
    """顺序扫描视频，把命中的帧编码成 JPEG 缓存在内存里（每帧约 1 MB）。"""
    want = set(indices)
    cap = cv2.VideoCapture(str(path))
    samples: list[tuple[int, bytes]] = []
    cur = 0
    while want and cap.isOpened():
        if not cap.grab():
            break
        if cur in want:
            ok, frame = cap.retrieve()
            if ok:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ok:
                    samples.append((cur, buf.tobytes()))
                want.discard(cur)
        cur += 1
    cap.release()
    return sorted(samples, key=lambda s: s[0])


def decode_sample(buf: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


def build(models: list[Path], names: dict[int, str], args):
    out = []
    for path in models:
        if path.suffix.lower() == ".onnx":
            pred = OnnxRuntimePredictor(path, names, args.imgsz, args.conf, args.iou,
                                        args.max_det, threads=args.threads)
        else:
            pred = UltralyticsPredictor(path, names, args.imgsz, args.conf, args.iou,
                                        args.max_det, device=args.device)
        out.append(pred)
    return out


def percentiles(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
    }


def kpt_distance(a: list[dict], b: list[dict], iou_thr: float = 0.5) -> list[float]:
    """返回匹配目标之间 4 个关键点的平均欧氏距离（原始像素）。"""
    if not a or not b:
        return []
    out = []
    boxes_b = np.array([d["box"] for d in b], dtype=float)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    used = set()
    for det_a in a:
        best, best_iou = -1, 0.0
        ba = np.array(det_a["box"], dtype=float)
        area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
        for j, det_b in enumerate(b):
            if j in used or det_b["cls"] != det_a["cls"]:
                continue
            bb = boxes_b[j]
            xx1, yy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
            xx2, yy2 = min(ba[2], bb[2]), min(ba[3], bb[3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            iou_val = inter / max(area_a + area_b[j] - inter, 1e-9)
            if iou_val > best_iou:
                best_iou, best = iou_val, j
        if best >= 0 and best_iou >= iou_thr and det_a.get("kxy") and b[best].get("kxy"):
            pa = np.array(det_a["kxy"], dtype=float)
            pb = np.array(b[best]["kxy"], dtype=float)
            if pa.shape == pb.shape:
                out.append(float(np.linalg.norm(pa - pb, axis=1).mean()))
                used.add(best)
    return out


def side_by_side(frame, dets_list: list[list[dict]], labels: list[str], style) -> np.ndarray:
    """以所有检测框的并集为中心裁剪放大，再左右拼接，保证关键点看得清。"""
    h, w = frame.shape[:2]
    boxes = [d["box"] for dets in dets_list for d in dets]
    if boxes:
        arr = np.array(boxes, dtype=float)
        x1, y1 = arr[:, 0].min(), arr[:, 1].min()
        x2, y2 = arr[:, 2].max(), arr[:, 3].max()
        margin = 0.35 * max(x2 - x1, y2 - y1, 64)
    else:
        x1, y1, x2, y2 = w * 0.25, h * 0.25, w * 0.75, h * 0.75
        margin = 0.0
    x1, y1 = max(0, int(x1 - margin)), max(0, int(y1 - margin))
    x2, y2 = min(w, int(x2 + margin)), min(h, int(y2 + margin))
    # 裁剪区域太扁时补成接近正方形，避免拼出来一条细长图
    bw, bh = x2 - x1, y2 - y1
    if bw / max(bh, 1) > 1.6:
        pad = int((bw - bh) / 2) if bw > bh else 0
        y1, y2 = max(0, y1 - pad), min(h, y2 + pad)
    elif bh / max(bw, 1) > 1.6:
        pad = int((bh - bw) / 2)
        x1, x2 = max(0, x1 - pad), min(w, x2 + pad)

    tiles = []
    for dets, label in zip(dets_list, labels):
        canvas = frame[y1:y2, x1:x2].copy()
        dets = [{**d, "box": [d["box"][0] - x1, d["box"][1] - y1,
                              d["box"][2] - x1, d["box"][3] - y1],
                 "kxy": [[p[0] - x1, p[1] - y1] for p in (d.get("kxy") or [])]}
                for d in dets]
        draw_detections(canvas, dets, style)
        hdr = max(0.7, canvas.shape[1] / 1800)
        put_text(canvas, f"{label} | {len(dets)} target(s)", (14, int(14 + hdr * 24)),
                 hdr, max(1, round(hdr * 2)))
        tiles.append(canvas)
    return np.hstack(tiles)


def main() -> int:
    args = parse_args()
    src = Path(args.source)
    models = [Path(m) for m in args.models if Path(m).is_file()]
    for m in args.models:
        if not Path(m).is_file():
            print(f"[警告] 跳过不存在的模型: {m}")
    if not src.is_file() or not models:
        print("[错误] 视频或模型文件不存在")
        return 2

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 两种后端使用相同线程数，避免"线程更多就跑得更快"污染对比
    if args.threads > 0:
        import torch
        torch.set_num_threads(args.threads)

    n_sample = min(args.frames, total)
    indices = np.linspace(0, total - 1, n_sample).round().astype(int).tolist()
    print(f"[信息] 视频 {src.name}: {width}x{height}, {total} 帧, {fps:.2f} FPS")
    print(f"[信息] 抽取 {n_sample} 帧作为测试集，重复 {args.repeat} 轮，预热 {args.warmup} 帧")
    print(f"[信息] 参数: conf={args.conf} iou={args.iou} imgsz={args.imgsz} "
          f"max_det={args.max_det} threads={args.threads}")

    samples = load_samples(src, indices)
    print(f"[信息] 已缓存测试帧: {len(samples)} 帧，"
          f"共 {sum(len(b) for _, b in samples) / 1024 / 1024:.1f} MiB（JPEG）")

    names = read_names(models[0])
    predictors = build(models, names, args)
    labels = [model_tag(m) for m in models]
    style = auto_style(width, height)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latencies: dict[str, list[float]] = {label: [] for label in labels}
    internal: dict[str, dict[str, list[float]]] = {
        label: {"preprocess": [], "inference": [], "postprocess": []} for label in labels}
    dets_by_model: dict[str, dict[int, list[dict]]] = {label: {} for label in labels}
    per_frame_rows = []

    for pred, label in zip(predictors, labels):
        print(f"\n[预热] {label}: {pred.backend}")
        for idx, buf in samples[:args.warmup]:
            pred.predict(decode_sample(buf))

    # 每轮轮换顺序，减少"总是第一个跑"带来的系统偏差
    for r in range(args.repeat):
        order = list(range(len(models)))
        order = order[r % len(order):] + order[:r % len(order)]
        for m_idx in order:
            pred, label = predictors[m_idx], labels[m_idx]
            round_lat = []
            for idx, buf in samples:
                frame = decode_sample(buf)
                t0 = time.perf_counter()
                dets, timings = pred.predict(frame)
                dt = (time.perf_counter() - t0) * 1000
                latencies[label].append(dt)
                round_lat.append(dt)
                for key in internal[label]:
                    if timings.get(key):
                        internal[label][key].append(float(timings[key]))
                if r == 0:
                    dets_by_model[label][idx] = dets
                    per_frame_rows.append({"model": label, "frame": idx,
                                           "latency_ms": round(dt, 2), "detections": len(dets)})
            print(f"  第 {r + 1}/{args.repeat} 轮 {label:<12} 平均 {np.mean(round_lat):6.2f} ms/帧")

    reference_dets = dets_by_model[labels[0]]
    report = {"video": str(src), "frames_total": total, "sample_frames": len(samples),
              "fps": round(fps, 2), "resolution": [width, height],
              "repeat": args.repeat, "warmup": args.warmup,
              "params": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                         "max_det": args.max_det, "threads": args.threads},
              "models": {}}

    for path, pred, label in zip(models, predictors, labels):
        dets_map = dets_by_model[label]
        class_counter: dict[str, int] = {}
        for dets in dets_map.values():
            for det in dets:
                class_counter[det["name"]] = class_counter.get(det["name"], 0) + 1

        matched, kpt_dists, ref_total = 0, [], 0
        for idx, dets in dets_map.items():
            ref = reference_dets.get(idx, [])
            matched += match_detections(ref, dets)
            ref_total += len(ref)
            kpt_dists.extend(kpt_distance(ref, dets))

        report["models"][label] = {
            "path": str(path),
            "format": path.suffix.lower().lstrip("."),
            "backend": pred.backend,
            "size_bytes": path.stat().st_size,
            "size_mib": round(path.stat().st_size / 1024 / 1024, 2),
            "latency_ms": percentiles(latencies[label]),
            "backend_internal_ms": {k: round(float(np.mean(v)), 2)
                                    for k, v in internal[label].items() if v},
            "detections": int(sum(len(v) for v in dets_map.values())),
            "frames_with_detection": int(sum(1 for v in dets_map.values() if v)),
            "per_class": dict(sorted(class_counter.items(), key=lambda kv: -kv[1])),
            "matched_with_reference": matched,
            "reference_total": ref_total,
            "match_rate": round(matched / ref_total, 4) if ref_total else None,
            "keypoint_mean_abs_diff_px": round(float(np.mean(kpt_dists)), 2) if kpt_dists else None,
        }

    # 选几个有检测的代表帧生成左右对比图
    hits = [i for i in indices if reference_dets.get(i)]
    picks = ([hits[0], hits[len(hits) // 2], hits[-1]] if hits else [])[:args.compare_frames]
    by_idx = {idx: buf for idx, buf in samples}
    for idx in picks:
        frame = decode_sample(by_idx[idx])
        dets_list = [dets_by_model[label][idx] for label in labels]
        canvas = side_by_side(frame, dets_list, labels, style)
        scale = min(1.0, 2400 / canvas.shape[1], 900 / canvas.shape[0])
        if scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out = out_dir / f"compare_frame{idx:05d}.jpg"
        cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"[信息] 对比图: {out}")
        for label, dets in zip(labels, dets_list):
            single = frame.copy()
            draw_detections(single, dets, style)
            cv2.imwrite(str(out_dir / f"sample_{label}_frame{idx:05d}.jpg"), single,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    (out_dir / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "benchmark_frames.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "frame", "latency_ms", "detections"])
        writer.writeheader()
        writer.writerows(per_frame_rows)

    rows = [
        ("模型文件大小 (MiB)", lambda s: f"{s['size_mib']:.2f}"),
        ("平均耗时 (ms/帧)", lambda s: f"{s['latency_ms']['mean']:.2f}"),
        ("中位数耗时 (ms/帧)", lambda s: f"{s['latency_ms']['median']:.2f}"),
        ("P95 耗时 (ms/帧)", lambda s: f"{s['latency_ms']['p95']:.2f}"),
        ("检测目标数", lambda s: str(s["detections"])),
        ("含检测帧数", lambda s: str(s["frames_with_detection"])),
        ("与 PT 匹配率", lambda s: f"{s['match_rate'] * 100:.2f}%"),
        ("关键点平均偏差 (px)", lambda s: str(s["keypoint_mean_abs_diff_px"])),
    ]
    lines = ["| 对比项 | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for title, fn in rows:
        lines.append(f"| {title} | " +
                     " | ".join(fn(report["models"][label]) for label in labels) + " |")
    (out_dir / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n================ 对比汇总 ================")
    print("\n".join(lines))
    print(f"\n[信息] 详细结果: {out_dir / 'benchmark.json'} / benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
