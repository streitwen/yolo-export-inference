#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-time Hikrobot MVS camera experiments and YOLO armor detection.

Typical Ubuntu commands:
    python camera_cli.py list
    python camera_cli.py preview --serial YOUR_SERIAL
    python camera_cli.py experiment --serial YOUR_SERIAL --scene still
    python camera_cli.py experiment --serial YOUR_SERIAL --scene moving
    python camera_cli.py yolo --serial YOUR_SERIAL --model models/best.onnx
    python camera_cli.py fault-test --serial YOUR_SERIAL --mode timeout
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from armor_core import auto_style, draw_detections, put_text
from inference import OnnxRuntimePredictor, UltralyticsPredictor, read_names
from mvs_camera import CameraConfig, CameraError, MvsCamera, MvsError
from mvs_camera.metrics import (
    CpuMonitor,
    FpsMeter,
    RuntimeStats,
    image_metrics,
    target_metrics,
)
from mvs_camera.sdk import PixelFormat, enumerate_usb_cameras

DEFAULT_OUT = Path("results/industrial_camera")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def operating_system_name() -> str:
    try:
        release = platform.freedesktop_os_release()
        return release.get("PRETTY_NAME") or platform.platform()
    except (AttributeError, OSError):
        return platform.platform()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, default=_json_default) + "\n")


def fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def open_writer(path: Path, fps: float, size: tuple[int, int], code: str = "mp4v"):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc(code), fps, size)
    if not writer.isOpened():
        writer.release()
        raise CameraError(f"无法创建输出视频: {path} (codec={code})")
    return writer


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("相机")
    group.add_argument("--serial", default="", help="相机序列号；留空使用 --index")
    group.add_argument("--index", type=int, default=0, help="多台相机时的序号")
    group.add_argument("--exposure-us", type=float, default=1000.0, help="曝光时间（μs）")
    group.add_argument("--gain-db", type=float, default=0.0, help="增益（dB）")
    group.add_argument(
        "--pixel-format",
        choices=sorted(PixelFormat.VALUES),
        default="BayerRG8",
        help="请求的相机输出格式",
    )
    group.add_argument(
        "--roi",
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        type=int,
        help="ROI：OffsetX OffsetY Width Height；不写则保持相机当前全幅",
    )
    group.add_argument("--frame-rate", type=float, default=None, help="目标帧率；不写则不修改")
    group.add_argument("--timeout-ms", type=int, default=1000, help="单帧等待超时")
    group.add_argument("--latest-image", action="store_true", default=True)
    group.add_argument(
        "--library",
        default="",
        help="MvCameraControl 动态库路径；也可设环境变量 MVS_SDK_LIBRARY",
    )


def config_from_args(args: argparse.Namespace) -> CameraConfig:
    roi = tuple(args.roi) if getattr(args, "roi", None) else None
    return CameraConfig(
        serial=getattr(args, "serial", "") or None,
        index=getattr(args, "index", 0),
        exposure_us=float(args.exposure_us),
        gain_db=float(args.gain_db),
        pixel_format=args.pixel_format,
        roi=roi,
        frame_rate=getattr(args, "frame_rate", None),
        timeout_ms=int(args.timeout_ms),
        latest_images_only=bool(getattr(args, "latest_image", True)),
    )


def cmd_list(args: argparse.Namespace) -> int:
    devices, sdk = enumerate_usb_cameras(args.library or None)
    try:
        if not devices:
            print("未发现 USB3 Vision 相机。")
            return 1
        print("序号 | 型号 | 序列号 | 固件")
        for device in devices:
            print(
                f"{device.index:>3} | {device.model or '-':<18} | "
                f"{device.serial or '-':<18} | {device.device_version or '-'}"
            )
        return 0
    finally:
        sdk.finalize()


def _window_ok(no_display: bool) -> bool:
    if no_display:
        return False
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _wait_window(window: str, delay_ms: int = 1) -> bool:
    key = cv2.waitKey(delay_ms) & 0xFF
    return key not in (27, ord("q"), ord("Q"))


def cmd_preview(args: argparse.Namespace) -> int:
    camera = MvsCamera(config_from_args(args), library=args.library or None)
    show_ui = _window_ok(args.no_display)
    try:
        camera.open()
        camera.start()
        print(json.dumps(camera.camera_info, ensure_ascii=False, indent=2))
        fps_meter = FpsMeter()
        frame_count = 0
        started = time.perf_counter()
        while True:
            frame = camera.grab_frame()
            if frame is None:
                print("[警告] 获取一帧超时，继续等待")
                continue
            frame_count += 1
            fps = fps_meter.tick()
            display = frame.image_bgr
            put_text(
                display,
                f"{fps:5.1f} FPS | {frame.actual_pixel_format} | "
                f"exp={frame.exposure_us:.0f}us gain={frame.gain_db:.1f}dB",
                (16, 34),
                0.8,
                2,
            )
            if show_ui:
                cv2.imshow("MVS camera - q/Esc to exit", display)
                if not _wait_window("MVS camera - q/Esc to exit"):
                    break
            if args.duration and time.perf_counter() - started >= args.duration:
                break
            if args.max_frames and frame_count >= args.max_frames:
                break
        return 0
    finally:
        camera.close()
        if show_ui:
            cv2.destroyAllWindows()
        print("[清理] " + json.dumps(camera.resource_log, ensure_ascii=False))


def _mean_image_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    numeric_keys = [
        "mean_gray",
        "std_gray",
        "saturated_pixel_ratio",
        "dark_pixel_ratio",
        "dark_noise_std",
        "laplacian_variance",
    ]
    result: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(s[key]) for s in samples if s.get(key) is not None]
        result[key] = round(float(np.mean(values)), 6) if values else None
    result["samples"] = len(samples)
    result["saturation_threshold"] = samples[0].get("saturation_threshold", 250)
    return result


def capture_condition(
    *,
    base_config: CameraConfig,
    label: str,
    out_dir: Path,
    duration_s: float,
    warmup_frames: int,
    video_fps: float,
    library: str | None,
    show_ui: bool,
    model_predictor=None,
    model_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Capture one fixed camera condition and record measurements."""
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = MvsCamera(base_config, library=library)
    stats = RuntimeStats()
    image_samples: list[dict[str, Any]] = []
    detection_history: list[dict[str, Any]] = []
    writer = None
    first_raw_path = out_dir / f"{label}_raw.png"
    first_bgr_path = out_dir / f"{label}_bgr.jpg"
    video_path = out_dir / f"{label}.mp4"
    result: dict[str, Any] = {"label": label, "config": asdict(base_config)}
    try:
        camera.open()
        camera.start()
        for _ in range(max(0, warmup_frames)):
            if camera.grab_frame() is None:
                print(f"[{label}] 预热超时")

        stats.started_at = time.perf_counter()
        fps_meter = FpsMeter()
        cpu = CpuMonitor()
        started = time.perf_counter()
        accepted = 0
        timeouts = 0
        while time.perf_counter() - started < duration_s:
            frame = camera.grab_frame()
            if frame is None:
                timeouts += 1
                if timeouts >= 10:
                    raise CameraError(f"{label}: 连续或累计超时过多 ({timeouts})")
                continue

            now = time.perf_counter()
            fps = fps_meter.tick(frame.received_at)
            stats.add(
                fps=fps,
                conversion_ms=frame.conversion_ms,
                cpu=cpu.sample() if accepted % 5 == 0 else {},
            )
            accepted += 1

            if accepted == 1:
                if not cv2.imwrite(str(first_raw_path), frame.raw_image):
                    raise CameraError(f"无法写入原图: {first_raw_path}")
                if not cv2.imwrite(
                    str(first_bgr_path),
                    frame.image_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                ):
                    raise CameraError(f"无法写入 BGR 图: {first_bgr_path}")
                writer = open_writer(
                    video_path,
                    video_fps,
                    (frame.width, frame.height),
                )
            if accepted % 5 == 0:
                image_samples.append(image_metrics(frame.image_bgr))

            display = frame.image_bgr.copy()
            put_text(
                display,
                f"{label} | {fps:5.1f} FPS | {frame.exposure_us:.0f}us "
                f"{frame.gain_db:.1f}dB",
                (16, 34),
                0.75,
                2,
            )
            if model_predictor is not None:
                dets, timings = model_predictor.predict(frame.image_bgr)
                draw_detections(display, dets, auto_style(frame.width, frame.height))
                detection_history.append(
                    {
                        "frame": frame.frame_number,
                        "timestamp_s": round(now - started, 4),
                        "target_count": len(dets),
                        "average_confidence": round(
                            float(np.mean([d["conf"] for d in dets])), 6
                        )
                        if dets
                        else None,
                        "classes": [d["name"] for d in dets],
                        "timings_ms": timings,
                    }
                )
            if writer is not None:
                writer.write(display)
            if show_ui:
                cv2.imshow(label, display)
                if not _wait_window(label):
                    break

        result.update(
            {
                "camera": camera.camera_info,
                "runtime": stats.summary(
                    camera.actual_pixel_format, base_config.pixel_format
                ),
                "image_quality_mean": _mean_image_metrics(image_samples),
                "timeouts": timeouts,
                "raw_image": str(first_raw_path),
                "bgr_image": str(first_bgr_path),
                "video": str(video_path),
            }
        )
        if model_predictor is not None:
            detected_frames = sum(
                1 for item in detection_history if item["target_count"] > 0
            )
            all_dets = sum(item["target_count"] for item in detection_history)
            result["yolo"] = {
                "processed_frames": len(detection_history),
                "frames_with_detection": detected_frames,
                "detection_frame_ratio": round(
                    detected_frames / max(len(detection_history), 1), 6
                ),
                "targets_total": all_dets,
                "mean_targets_per_frame": round(
                    all_dets / max(len(detection_history), 1), 4
                ),
                "mean_inference_ms": round(
                    float(
                        np.mean(
                            [
                                item["timings_ms"].get("inference", 0.0)
                                for item in detection_history
                            ]
                        )
                    ),
                    3,
                )
                if detection_history
                else None,
                "history": detection_history,
            }
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if writer is not None:
            writer.release()
        camera.close()
        result["resource_log"] = camera.resource_log
        result["warnings"] = camera.warnings
        write_json(out_dir / f"{label}.json", result)
        print(f"[完成] {label}: {stats.frame_count} 帧 -> {out_dir}")


def cmd_capture(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    result = capture_condition(
        base_config=config_from_args(args),
        label=args.name,
        out_dir=out_dir,
        duration_s=args.duration,
        warmup_frames=args.warmup,
        video_fps=args.video_fps,
        library=args.library or None,
        show_ui=_window_ok(args.no_display),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def _experiment_specs(args: argparse.Namespace) -> list[tuple[str, CameraConfig]]:
    base = config_from_args(args)
    specs = []
    for exposure in args.exposures:
        specs.append(
            (
                f"exposure_{int(exposure)}us_{args.scene}_gain_0db",
                replace(
                    base,
                    exposure_us=float(exposure),
                    gain_db=0.0,
                    pixel_format="BayerRG8",
                ),
            )
        )
    for gain in args.gains:
        specs.append(
            (
                f"gain_{int(gain)}db_static_exposure_1000us",
                replace(
                    base,
                    exposure_us=1000.0,
                    gain_db=float(gain),
                    pixel_format="BayerRG8",
                ),
            )
        )
    for pixel_format in args.pixel_formats:
        specs.append(
            (
                f"format_{pixel_format.lower()}_1000us_0db",
                replace(
                    base,
                    exposure_us=1000.0,
                    gain_db=0.0,
                    pixel_format=pixel_format,
                ),
            )
        )
    return specs


def cmd_experiment(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, config in _experiment_specs(args):
        print(f"\n=== {label} ===")
        result = capture_condition(
            base_config=config,
            label=label,
            out_dir=out_dir,
            duration_s=args.duration,
            warmup_frames=args.warmup,
            video_fps=args.video_fps,
            library=args.library or None,
            show_ui=_window_ok(args.no_display),
        )
        rows.append(result)
    summary = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scene": args.scene,
        "operating_system": operating_system_name(),
        "platform": platform.platform(),
        "python": sys.version,
        "opencv": cv2.__version__,
        "conditions": rows,
        "note": "数据由真实相机采集；分析文字需根据现场图像和数值填写。",
    }
    write_json(out_dir / "experiment_summary.json", summary)
    write_experiment_csv(out_dir / "experiment_summary.csv", rows)
    print(f"\n[完成] 实验汇总: {out_dir / 'experiment_summary.json'}")
    return 0


def write_experiment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "pixel_format",
        "exposure_us",
        "gain_db",
        "frames",
        "processed_fps",
        "conversion_ms_mean",
        "process_cpu_percent_mean",
        "mean_gray",
        "saturated_pixel_ratio",
        "dark_noise_std",
        "laplacian_variance",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            runtime = row.get("runtime", {})
            quality = row.get("image_quality_mean", {})
            writer.writerow(
                {
                    "label": row.get("label", ""),
                    "pixel_format": runtime.get("actual_camera_pixel_format", ""),
                    "exposure_us": row.get("config", {}).get("exposure_us", ""),
                    "gain_db": row.get("config", {}).get("gain_db", ""),
                    "frames": runtime.get("frames", ""),
                    "processed_fps": runtime.get("processed_fps", ""),
                    "conversion_ms_mean": runtime.get("conversion_ms_mean", ""),
                    "process_cpu_percent_mean": runtime.get(
                        "process_cpu_percent_mean", ""
                    ),
                    "mean_gray": quality.get("mean_gray", ""),
                    "saturated_pixel_ratio": quality.get(
                        "saturated_pixel_ratio", ""
                    ),
                    "dark_noise_std": quality.get("dark_noise_std", ""),
                    "laplacian_variance": quality.get("laplacian_variance", ""),
                }
            )


def build_predictor(args, model_path: Path):
    names = read_names(model_path)
    if model_path.suffix.lower() == ".onnx":
        predictor = OnnxRuntimePredictor(
            model_path,
            names=names,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            threads=args.threads,
        )
    else:
        predictor = UltralyticsPredictor(
            model_path,
            names=names,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
        )
    return predictor, names


def draw_centers(frame: np.ndarray, dets: list[dict], style: dict) -> None:
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        cv2.drawMarker(
            frame,
            (cx, cy),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            max(8, style["radius"] * 3),
            max(1, style["lw"] // 2),
            cv2.LINE_AA,
        )


def stream_yolo(
    *,
    args: argparse.Namespace,
    config: CameraConfig,
    out_dir: Path,
    label: str,
    predictor,
    names: dict[int, str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{label}.mp4"
    first_frame_path = out_dir / f"{label}_first_frame.jpg"
    first_detection_path = out_dir / f"{label}_first_detection.jpg"
    csv_path = out_dir / f"{label}_detections.csv"
    report_path = out_dir / f"{label}_report.json"
    camera = MvsCamera(config, library=args.library or None)
    writer = None
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_handle)
    header = [
        "frame",
        "timestamp_s",
        "class_id",
        "class_name",
        "confidence",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    for index in range(4):
        header.extend([f"kp{index + 1}_x", f"kp{index + 1}_y"])
    csv_writer.writerow(header)
    show_ui = _window_ok(args.no_display)
    frame_count = 0
    detected_frames = 0
    total_targets = 0
    saved_first_frame = False
    saved_first_detection = False
    stats = RuntimeStats()
    latencies_ms: list[float] = []
    confidences: list[float] = []
    class_counts: dict[str, int] = {}
    report: dict[str, Any] = {
        "label": label,
        "model": str(args.model),
        "backend": predictor.backend,
        "classes": names,
        "operating_system": operating_system_name(),
        "python": sys.version,
        "opencv": cv2.__version__,
        "camera_config": asdict(config),
    }
    try:
        camera.open()
        camera.start()
        for _ in range(max(0, args.warmup)):
            camera.grab_frame()
        start = time.perf_counter()
        stats.started_at = start
        fps_meter = FpsMeter()
        cpu = CpuMonitor()

        while True:
            if args.max_frames and frame_count >= args.max_frames:
                break
            if args.duration and time.perf_counter() - start >= args.duration:
                break
            frame = camera.grab_frame()
            if frame is None:
                print("[警告] 相机取帧超时，继续运行")
                continue
            frame_count += 1
            fps = fps_meter.tick(frame.received_at)
            dets, timings = predictor.predict(frame.image_bgr)
            latency_ms = (time.perf_counter() - frame.received_at) * 1000.0
            latencies_ms.append(latency_ms)
            stats.add(
                fps=fps,
                conversion_ms=frame.conversion_ms,
                cpu=cpu.sample() if frame_count % 5 == 0 else {},
            )

            timestamp_s = (time.perf_counter() - start)
            if dets:
                detected_frames += 1
                total_targets += len(dets)
            for det in dets:
                confidences.append(float(det["conf"]))
                class_counts[det["name"]] = class_counts.get(det["name"], 0) + 1
            for det in dets:
                x1, y1, x2, y2 = det["box"]
                row = [
                    frame.frame_number,
                    f"{timestamp_s:.4f}",
                    det["cls"],
                    det["name"],
                    f"{det['conf']:.6f}",
                    f"{(x1 + x2) / 2:.2f}",
                    f"{(y1 + y2) / 2:.2f}",
                    f"{x1:.2f}",
                    f"{y1:.2f}",
                    f"{x2:.2f}",
                    f"{y2:.2f}",
                ]
                for index in range(4):
                    point = (det.get("kxy") or [])[index]
                    row.extend([f"{point[0]:.2f}", f"{point[1]:.2f}"])
                csv_writer.writerow(row)

            style = auto_style(frame.width, frame.height)
            draw_detections(frame.image_bgr, dets, style)
            draw_centers(frame.image_bgr, dets, style)
            status = (
                f"{fps:5.1f} FPS | targets={len(dets)} | "
                f"latency={latency_ms:.1f}ms | {frame.exposure_us:.0f}us "
                f"{frame.gain_db:.1f}dB"
            )
            put_text(frame.image_bgr, status, (16, 34), style["fscale"], style["tth"])

            if not saved_first_frame:
                saved_first_frame = bool(
                    cv2.imwrite(
                        str(first_frame_path),
                        frame.image_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                )
            if dets and not saved_first_detection:
                saved_first_detection = bool(
                    cv2.imwrite(
                        str(first_detection_path),
                        frame.image_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                )
            if writer is None:
                writer = open_writer(
                    video_path,
                    args.video_fps,
                    (frame.width, frame.height),
                )
            writer.write(frame.image_bgr)
            if show_ui:
                cv2.imshow(
                    f"YOLO armor inference - {label}",
                    frame.image_bgr,
                )
                if not _wait_window(f"YOLO armor inference - {label}"):
                    break

        report.update(
            {
                "camera": camera.camera_info,
                "runtime": stats.summary(
                    camera.actual_pixel_format, config.pixel_format
                ),
                "frames": frame_count,
                "frames_with_detection": detected_frames,
                "detection_frame_ratio": round(
                    detected_frames / max(frame_count, 1), 6
                ),
                "targets_total": total_targets,
                "class_counts": class_counts,
                "average_confidence": round(float(np.mean(confidences)), 6)
                if confidences
                else None,
                "processing_latency_ms_mean": round(float(np.mean(latencies_ms)), 3)
                if latencies_ms
                else None,
                "processing_latency_ms_p95": round(
                    float(np.percentile(latencies_ms, 95)), 3
                )
                if latencies_ms
                else None,
                "first_frame_image": str(first_frame_path)
                if saved_first_frame
                else None,
                "first_detection_image": str(first_detection_path)
                if saved_first_detection
                else None,
                "video": str(video_path),
                "csv": str(csv_path),
            }
        )
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        csv_handle.close()
        if writer is not None:
            writer.release()
        camera.close()
        report["resource_log"] = camera.resource_log
        report["warnings"] = camera.warnings
        write_json(report_path, report)
        if show_ui:
            cv2.destroyAllWindows()
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


def add_yolo_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("YOLO 推理")
    group.add_argument("--model", default="models/best.onnx", help=".pt/.onnx/.torchscript")
    group.add_argument("--conf", type=float, default=0.25)
    group.add_argument("--iou", type=float, default=0.7)
    group.add_argument("--imgsz", type=int, default=640)
    group.add_argument("--max-det", type=int, default=30)
    group.add_argument("--threads", type=int, default=0, help="ONNX Runtime CPU 线程数")
    group.add_argument("--device", default="cpu", help="Ultralytics 设备")


def cmd_yolo(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"找不到模型: {model_path}")
    predictor, names = build_predictor(args, model_path)
    print(f"[模型] {model_path} | 后端 {predictor.backend} | 类别 {names}")
    out_dir = Path(args.out_dir)
    label = args.name
    report = stream_yolo(
        args=args,
        config=config_from_args(args),
        out_dir=out_dir,
        label=label,
        predictor=predictor,
        names=names,
    )
    print(f"[完成] 结果视频: {report['video']}")
    return 0


def cmd_yolo_impact(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"找不到模型: {model_path}")
    predictor, names = build_predictor(args, model_path)
    base = config_from_args(args)
    out_dir = Path(args.out_dir) / "yolo_impact"
    conditions = [
        (
            f"exposure_{int(value)}us_gain_0db",
            replace(base, exposure_us=float(value), gain_db=0.0),
        )
        for value in args.exposures
    ]
    conditions.extend(
        (
            f"gain_{int(value)}db_exposure_1000us",
            replace(base, exposure_us=1000.0, gain_db=float(value)),
        )
        for value in args.gains
    )
    # 1000 us / 0 dB appears in both groups; only run once.
    unique = {}
    for label, config in conditions:
        unique[(config.exposure_us, config.gain_db)] = (label, config)

    reports = []
    for label, config in unique.values():
        print(f"\n=== YOLO 影响实验: {label} ===")
        reports.append(
            stream_yolo(
                args=args,
                config=config,
                out_dir=out_dir,
                label=label,
                predictor=predictor,
                names=names,
            )
        )
    write_json(
        out_dir / "yolo_impact_summary.json",
        {
            "model": str(model_path),
            "backend": predictor.backend,
            "conditions": reports,
            "note": "请结合真实画面判断漏检、误检、类别错误和关键点偏差。",
        },
    )
    print(f"[完成] 汇总: {out_dir / 'yolo_impact_summary.json'}")
    return 0


def write_fault_log(
    path: Path,
    mode: str,
    error: str,
    camera: MvsCamera,
    fault_observed: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def successful_steps(*names: str) -> int:
        return sum(
            1
            for item in camera.resource_log
            if item.get("ok") is True and item.get("step") in names
        )

    successful_gets = successful_steps("GetImageBuffer")
    successful_frees = successful_steps(
        "FreeImageBuffer",
        "CleanupFreeImageBuffer",
    )
    data = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "expected_fault_observed": fault_observed,
        "expected_error_observed": error,
        "resource_log": camera.resource_log,
        "warnings": camera.warnings,
        "released": {
            "leased_buffer_clear": (
                camera.leased is None and successful_frees >= successful_gets
            ),
            "grabbing_clear": (
                camera.grabbing is False
                and successful_steps("StopGrabbing")
                >= successful_steps("StartGrabbing")
            ),
            "opened_clear": (
                camera.opened is False
                and successful_steps("CloseDevice")
                >= successful_steps("OpenDevice")
            ),
            "handle_clear": (
                camera.handle is None
                and successful_steps("DestroyHandle")
                >= successful_steps("CreateHandle")
            ),
        },
        "sdk_finalized": camera._owns_sdk_init is False,
        "successful_get_image_buffer_calls": successful_gets,
        "successful_free_image_buffer_calls": successful_frees,
    }
    if extra:
        data.update(extra)
    write_json(path, data)
    print(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))
    return data


def cmd_fault_test(args: argparse.Namespace) -> int:
    camera = MvsCamera(config_from_args(args), library=args.library or None)
    out = Path(args.output)
    error_text = ""
    fault_observed = False
    extra: dict[str, Any] = {}
    try:
        camera.open()
        if args.mode == "timeout":
            # A real SDK timeout without waiting for an unplugged cable: keep
            # grabbing enabled but do not issue a software trigger.
            camera.sdk.set_enum_string(camera.handle, "TriggerMode", "On")
            try:
                camera.sdk.set_enum_string(camera.handle, "TriggerSource", "Software")
            except MvsError as exc:
                camera._warn(f"TriggerSource 无法设为 Software: {exc}")
            camera.start()
            frame = camera.grab_frame(timeout_ms=args.fault_timeout_ms)
            if frame is not None:
                error_text = "未观察到超时：相机的触发配置可能未生效"
            else:
                fault_observed = True
                error_text = f"MV_E_NODATA timeout after {args.fault_timeout_ms} ms"
        elif args.mode == "bad-format":
            camera.start()
            for _ in range(max(1, args.warmup)):
                if camera.grab_frame() is not None:
                    break
            try:
                # Inject an unknown pixel type after a real frame was acquired.
                # The camera layer must reject it and still return the buffer.
                camera.grab_frame(force_pixel_type=0xDEADBEEF)
            except CameraError as exc:
                fault_observed = True
                error_text = str(exc)
            extra["injection"] = (
                "A real frame was acquired, then enPixelType was forced to 0xDEADBEEF "
                "inside the application. This tests error handling and buffer release."
            )
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        extra["stage"] = "camera_open_or_test_execution"
    finally:
        camera.close()
    data = write_fault_log(
        out,
        args.mode,
        error_text,
        camera,
        fault_observed,
        extra,
    )
    if not fault_observed:
        print(
            f"[错误] 没有观察到预期的 {args.mode} 异常；"
            "该 JSON 不能作为成功的异常处理记录",
            file=sys.stderr,
        )
        return 3
    if not all(data["released"].values()):
        print("[错误] 清理不完整", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="海康 MVS 工业相机实验与 YOLO 装甲板实时识别",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="枚举 USB3 相机")
    list_parser.add_argument("--library", default="")
    list_parser.set_defaults(func=cmd_list)

    preview = sub.add_parser("preview", help="实时预览并显示 FPS/参数")
    add_camera_args(preview)
    preview.add_argument("--duration", type=float, default=0, help="运行秒数，0=直到按 q")
    preview.add_argument("--max-frames", type=int, default=0)
    preview.add_argument("--no-display", action="store_true")
    preview.set_defaults(func=cmd_preview)

    capture = sub.add_parser("capture", help="保存单张原图、BGR 图和短视频")
    add_camera_args(capture)
    capture.add_argument("--name", default="capture")
    capture.add_argument("--out-dir", default=str(DEFAULT_OUT / "capture"))
    capture.add_argument("--duration", type=float, default=3.0)
    capture.add_argument("--warmup", type=int, default=5)
    capture.add_argument("--video-fps", type=float, default=30.0)
    capture.add_argument("--no-display", action="store_true")
    capture.set_defaults(func=cmd_capture)

    experiment = sub.add_parser(
        "experiment", help="执行曝光、增益和 BayerRG8/BGR8 实验"
    )
    add_camera_args(experiment)
    experiment.add_argument("--scene", choices=["still", "moving"], required=True)
    experiment.add_argument("--exposures", nargs="+", type=float, default=[500, 1000, 3000])
    experiment.add_argument("--gains", nargs="+", type=float, default=[0, 6, 12])
    experiment.add_argument(
        "--pixel-formats", nargs="+", choices=["BayerRG8", "BGR8"], default=["BayerRG8", "BGR8"]
    )
    experiment.add_argument("--duration", type=float, default=3.0, help="每个条件采集秒数")
    experiment.add_argument("--warmup", type=int, default=5)
    experiment.add_argument("--video-fps", type=float, default=30.0)
    experiment.add_argument("--out-dir", default=str(DEFAULT_OUT / "camera_experiments"))
    experiment.add_argument("--no-display", action="store_true")
    experiment.set_defaults(func=cmd_experiment)

    yolo = sub.add_parser("yolo", help="工业相机实时 YOLO 推理并录制结果视频")
    add_camera_args(yolo)
    add_yolo_args(yolo)
    yolo.add_argument("--name", default="yolo_live")
    yolo.add_argument("--out-dir", default=str(DEFAULT_OUT / "yolo"))
    yolo.add_argument("--duration", type=float, default=20.0, help="运行秒数，0=直到按 q")
    yolo.add_argument("--max-frames", type=int, default=0)
    yolo.add_argument("--warmup", type=int, default=3)
    yolo.add_argument("--video-fps", type=float, default=30.0)
    yolo.add_argument("--no-display", action="store_true")
    yolo.set_defaults(func=cmd_yolo)

    impact = sub.add_parser(
        "yolo-impact", help="自动比较不同曝光和增益下的 YOLO 检测结果"
    )
    add_camera_args(impact)
    add_yolo_args(impact)
    impact.add_argument("--exposures", nargs="+", type=float, default=[500, 1000, 3000])
    impact.add_argument("--gains", nargs="+", type=float, default=[0, 6, 12])
    impact.add_argument("--out-dir", default=str(DEFAULT_OUT))
    impact.add_argument("--duration", type=float, default=3.0)
    impact.add_argument("--max-frames", type=int, default=0)
    impact.add_argument("--warmup", type=int, default=3)
    impact.add_argument("--video-fps", type=float, default=30.0)
    impact.add_argument("--no-display", action="store_true")
    impact.set_defaults(func=cmd_yolo_impact)

    fault = sub.add_parser("fault-test", help="测试取流超时或不支持格式后的资源释放")
    add_camera_args(fault)
    fault.add_argument("--mode", choices=["timeout", "bad-format"], required=True)
    fault.add_argument("--warmup", type=int, default=3)
    fault.add_argument("--fault-timeout-ms", type=int, default=100)
    fault.add_argument(
        "--output",
        default="",
        help="日志 JSON；默认 results/industrial_camera/logs/fault_<mode>.json",
    )
    fault.set_defaults(func=cmd_fault_test)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "fault-test" and not args.output:
        args.output = str(DEFAULT_OUT / "logs" / f"fault_{args.mode}.json")
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[中断] 用户按下 Ctrl+C")
        return 130
    except (CameraError, MvsError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
