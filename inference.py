#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装甲板关键点推理：同一份代码支持 .pt / .onnx / .torchscript。

三种格式共用 armor_core.py 中的预处理与画图逻辑，因此输出可以直接对比。
.onnx 默认使用 ONNX Runtime 原生推理（而不是让 Ultralytics 包装），
这样对比"PyTorch 推理"和"ONNX Runtime 推理"才有意义。

示例：
    # PyTorch
    python inference.py --model models/best.pt --source data/robomaster.mp4
    # ONNX Runtime
    python inference.py --model models/best.onnx --source data/robomaster.mp4
    # TorchScript
    python inference.py --model models/best.torchscript --source data/robomaster.mp4
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from armor_core import (auto_style, decode_pose_output, draw_detections, preprocess,
                        scale_detections)

DEFAULT_NAMES = {0: "B1", 1: "B2", 2: "B3", 3: "R1", 4: "R2", 5: "R3"}
IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO 装甲板四关键点推理（PT / ONNX / TorchScript）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("输入 / 输出")
    g.add_argument("--model", "--weights", "-m", default="models/best.onnx",
                   help="模型路径，按扩展名自动识别 .pt / .onnx / .torchscript")
    g.add_argument("--source", "-s", default="data/robomaster.mp4", help="输入视频或图片")
    g.add_argument("--output", "-o", default="",
                   help="输出路径（默认 results/<文件名>_<格式>_pred.mp4）")
    g.add_argument("--engine", default="auto", choices=["auto", "onnxruntime", "ultralytics"],
                   help="推理引擎：.onnx 默认 onnxruntime，其余走 ultralytics")
    g.add_argument("--out-scale", type=float, default=1.0, help="输出缩放比例，0.5 = 4K 输出 1080p")
    g.add_argument("--codec", default="", help="视频编码器，默认 mp4v/avc1/XVID 依次尝试")

    g = p.add_argument_group("推理参数")
    g.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    g.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    g.add_argument("--imgsz", type=int, default=640, help="输入尺寸，必须与导出时一致")
    g.add_argument("--max-det", type=int, default=30, help="每帧最多保留的目标数")
    g.add_argument("--device", default="cpu", help="ultralytics 后端使用的设备，cpu / 0")
    g.add_argument("--threads", type=int, default=0,
                   help="ONNX Runtime CPU 线程数，0 = 由运行时自动决定")

    g = p.add_argument_group("额外输出")
    g.add_argument("--save-csv", action="store_true", help="导出逐帧检测明细 CSV")
    g.add_argument("--save-frames", type=int, default=0, help="保存代表性截图数量")
    g.add_argument("--frame-dir", default="results/frames", help="截图目录")
    g.add_argument("--frame-gap", type=int, default=45, help="两张截图的最小帧间隔")
    g.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧，调试用")
    g.add_argument("--no-connect", action="store_true", help="不画关键点连线")
    g.add_argument("--quiet", action="store_true", help="不打印进度")
    return p.parse_args()


def model_tag(path: Path) -> str:
    return {"pt": "pt", "onnx": "onnx", "torchscript": "torchscript"}.get(path.suffix.lower().lstrip("."), "model")


def read_names(path: Path) -> dict[int, str]:
    """读取类别名：PT/TorchScript 用 Ultralytics，ONNX 读文件元数据。"""
    if path.suffix.lower() == ".onnx":
        try:
            import onnx
            model = onnx.load(str(path), load_external_data=False)
            for prop in model.metadata_props:
                if prop.key == "names":
                    value = ast.literal_eval(prop.value)
                    return {int(k): str(v) for k, v in value.items()}
        except Exception:
            pass
        return dict(DEFAULT_NAMES)

    from ultralytics import YOLO
    model = YOLO(str(path), task="pose")
    return {int(k): str(v) for k, v in model.names.items()}


class OnnxRuntimePredictor:
    """ONNX Runtime 原生推理：自己完成前处理、解码、NMS。"""

    def __init__(self, path: Path, names: dict[int, str], imgsz: int, conf: float,
                 iou: float, max_det: int, threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads > 0:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.names, self.imgsz = names, imgsz
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.kpt_shape = (4, 2)
        self.task = "pose"
        self.backend = f"ONNXRuntime {ort.__version__}"

    def predict(self, frame: np.ndarray):
        t0 = time.perf_counter()
        blob, meta = preprocess(frame, self.imgsz)
        t1 = time.perf_counter()
        pred = self.session.run(None, {self.input_name: blob})[0]
        t2 = time.perf_counter()
        dets = decode_pose_output(pred, self.names, self.conf, self.iou, self.max_det,
                                  self.kpt_shape)
        dets = scale_detections(dets, meta)
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000,
                      "postprocess": (t3 - t2) * 1000}


class UltralyticsPredictor:
    """PT / TorchScript 走 Ultralytics 自带后端（PT 为 PyTorch，TorchScript 为 libtorch）。"""

    def __init__(self, path: Path, names: dict[int, str], imgsz: int, conf: float,
                 iou: float, max_det: int, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(path), task="pose")
        self.names, self.imgsz = names, imgsz
        self.conf, self.iou, self.max_det, self.device = conf, iou, max_det, device
        self.kpt_shape = tuple(getattr(self.model.model, "kpt_shape", (4, 2)))
        self.task = getattr(self.model, "task", "pose")
        self.backend = f"Ultralytics {getattr(self.model, 'ckpt_path', '')}".strip()

    def predict(self, frame: np.ndarray):
        t0 = time.perf_counter()
        result = self.model.predict(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            max_det=self.max_det, device=self.device, verbose=False, save=False,
        )[0]
        t1 = time.perf_counter()

        dets = []
        boxes = result.boxes
        kpts = getattr(result, "keypoints", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            kxy = kpts.xy.cpu().numpy() if (kpts is not None and kpts.xy is not None) else None
            for i in range(len(xyxy)):
                dets.append({
                    "box": [float(v) for v in xyxy[i]],
                    "conf": float(confs[i]),
                    "cls": int(clss[i]),
                    "name": self.names.get(int(clss[i]), str(int(clss[i]))),
                    "kxy": None if kxy is None else [[float(x), float(y)] for x, y in kxy[i]],
                    "kconf": None,
                })
        speed = dict(result.speed or {})
        return dets, {"preprocess": speed.get("preprocess", 0.0),
                      "inference": speed.get("inference", 0.0),
                      "postprocess": speed.get("postprocess", 0.0),
                      "total": (t1 - t0) * 1000}


def build_predictor(args, path: Path, names: dict[int, str]):
    engine = args.engine
    if engine == "auto":
        engine = "onnxruntime" if path.suffix.lower() == ".onnx" else "ultralytics"
    common = dict(names=names, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                  max_det=args.max_det)
    if engine == "onnxruntime":
        if path.suffix.lower() != ".onnx":
            raise SystemExit("--engine onnxruntime 只能用于 .onnx 模型")
        return OnnxRuntimePredictor(path, threads=args.threads, **common)
    return UltralyticsPredictor(path, device=args.device, **common)


def fourcc(code: str) -> int:
    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*code)
    return cv2.VideoWriter.fourcc(*code)


def open_writer(path: Path, codecs, fps: float, size):
    for code in codecs:
        writer = cv2.VideoWriter(str(path), fourcc(code), fps, size)
        if writer.isOpened():
            return writer, code
        writer.release()
    return None, None


def write_csv(path: Path, rows: list[dict], nkpt: int) -> None:
    header = ["frame", "time_sec", "cls_id", "class_name", "conf", "x1", "y1", "x2", "y2"]
    for i in range(nkpt):
        header += [f"kp{i + 1}_x", f"kp{i + 1}_y"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            line = [row["frame"], f"{row['t']:.3f}", row["cls"], row["name"], f"{row['conf']:.4f}"]
            line += [f"{v:.1f}" for v in row["box"]]
            for i in range(nkpt):
                if row["kxy"] and i < len(row["kxy"]):
                    line += [f"{row['kxy'][i][0]:.1f}", f"{row['kxy'][i][1]:.1f}"]
                else:
                    line += ["", ""]
            writer.writerow(line)


def run_image(args, predictor, src: Path, out_path: Path, style) -> int:
    frame = cv2.imread(str(src))
    if frame is None:
        print(f"[错误] 无法读取图片: {src}", file=sys.stderr)
        return 2
    style = auto_style(frame.shape[1], frame.shape[0])
    dets, timings = predictor.predict(frame)
    draw_detections(frame, dets, style, connect=not args.no_connect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[信息] 检测到 {len(dets)} 个目标，结果图片: {out_path}")
    print(f"[信息] 单帧耗时: 推理 {timings['inference']:.1f} ms，合计 {sum(timings.values()):.1f} ms")
    return 0


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    src = Path(args.source).expanduser()
    if not model_path.is_file():
        print(f"[错误] 找不到模型: {model_path}", file=sys.stderr)
        return 2
    if not src.is_file():
        print(f"[错误] 找不到输入: {src}", file=sys.stderr)
        return 2

    names = read_names(model_path)
    predictor = build_predictor(args, model_path, names)
    tag = model_tag(model_path)
    print(f"[信息] 模型: {model_path.name} ({tag}) | 后端: {predictor.backend}")
    print(f"[信息] 类别: {names} | 关键点: {predictor.kpt_shape}")
    print(f"[信息] 参数: conf={args.conf} iou={args.iou} imgsz={args.imgsz} max_det={args.max_det}")

    is_image = src.suffix.lower() in IMAGE_SUFFIX
    default_name = f"{src.stem}_{tag}_pred" + (".jpg" if is_image else ".mp4")
    out_path = Path(args.output) if args.output else Path("results") / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_image:
        return run_image(args, predictor, src, out_path, None)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {src}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        ok, probe = cap.read()
        if not ok:
            print("[错误] 视频中没有可读帧", file=sys.stderr)
            return 2
        height, width = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_size = (width, height)
    if abs(args.out_scale - 1.0) > 1e-6:
        out_size = (max(2, int(width * args.out_scale) // 2 * 2),
                    max(2, int(height * args.out_scale) // 2 * 2))
    codecs = [args.codec] if args.codec else ["mp4v", "avc1", "XVID"]
    writer, used = open_writer(out_path, codecs, fps, out_size)
    if writer is None:
        print(f"[错误] 无法创建输出视频（尝试过 {codecs}），可用 --codec 指定", file=sys.stderr)
        return 2

    print(f"[信息] 输入视频: {src.name} | {width}x{height} | {fps:.2f} FPS | 约 {total} 帧")
    print(f"[信息] 输出视频: {out_path}（{used}, {out_size[0]}x{out_size[1]}）")

    style = auto_style(width, height)
    rows, class_count, class_conf = [], Counter(), defaultdict(float)
    frames_with_det, n_frames = 0, 0
    lat = defaultdict(float)
    snaps, last_snap = [], -(10 ** 9)

    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n_frames >= args.max_frames):
            break
        n_frames += 1

        dets, timings = predictor.predict(frame)
        for k, v in timings.items():
            lat[k] += float(v)

        if dets:
            frames_with_det += 1
            draw_detections(frame, dets, style, connect=not args.no_connect)
            for det in dets:
                class_count[det["name"]] += 1
                class_conf[det["name"]] += det["conf"]
                rows.append({"frame": n_frames - 1, "t": (n_frames - 1) / fps,
                             "cls": det["cls"], "name": det["name"], "conf": det["conf"],
                             "box": det["box"], "kxy": det["kxy"]})
            if args.save_frames > 0 and len(snaps) < args.save_frames and \
                    (n_frames - 1) - last_snap >= args.frame_gap:
                snaps.append(n_frames - 1)
                last_snap = n_frames - 1
                Path(args.frame_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.frame_dir) / f"{src.stem}_{tag}_frame{n_frames - 1:05d}.jpg"),
                            frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        writer.write(frame if out_size == (width, height)
                     else cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA))
        if not args.quiet and (n_frames % 30 == 0 or (total and n_frames == total)):
            print(f"\r[进度] {n_frames}/{total or '?'} 帧 | 含检测帧 {frames_with_det}",
                  end="", flush=True)

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t_start
    if not args.quiet:
        print()

    if args.save_csv:
        csv_path = out_path.parent / f"{out_path.stem}_detections.csv"
        write_csv(csv_path, rows, predictor.kpt_shape[0])
        print(f"[信息] 检测明细: {csv_path}（{len(rows)} 条）")

    n_det = sum(class_count.values())
    stats = {
        "model": str(model_path), "format": tag, "backend": predictor.backend,
        "frames": n_frames, "frames_with_detection": frames_with_det,
        "detections": n_det,
        "per_class": {k: {"count": v, "avg_conf": round(class_conf[k] / v, 4)}
                      for k, v in class_count.most_common()},
        "avg_ms": {k: round(v / max(1, n_frames), 2) for k, v in lat.items()},
        "end_to_end_fps": round(n_frames / elapsed, 2) if elapsed > 0 else None,
        "output": str(out_path),
        "params": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                   "max_det": args.max_det},
    }
    stats_path = out_path.parent / f"{out_path.stem}_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ 推理汇总 ================")
    print(f"模型 / 后端      : {model_path.name} / {predictor.backend}")
    print(f"处理帧数         : {n_frames}")
    print(f"含检测帧数       : {frames_with_det} ({frames_with_det / max(1, n_frames) * 100:.1f}%)")
    print(f"目标总数         : {n_det}")
    for name, cnt in class_count.most_common():
        print(f"    {name:<6} {cnt:>5} 个   平均置信度 {class_conf[name] / cnt:.3f}")
    for key in ("preprocess", "inference", "postprocess"):
        if lat.get(key):
            print(f"平均 {key:<11}: {lat[key] / max(1, n_frames):7.2f} ms/帧")
    print(f"端到端帧率       : {n_frames / max(elapsed, 1e-9):.1f} FPS（含解码/绘图/写盘）")
    print(f"结果视频         : {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"统计 JSON        : {stats_path}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
