from __future__ import annotations

import ctypes
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from camera_cli import build_parser, write_experiment_csv, write_fault_log
from mvs_camera.camera import CameraConfig, CameraError, MvsCamera
from mvs_camera.metrics import FpsMeter, image_metrics, target_metrics
from mvs_camera.sdk import (
    MV_OK,
    MvFrameOut,
    MvFrameOutInfoEx,
    MvPixelConvertParamEx,
    MvsError,
    MvsSdk,
    PixelFormat,
    format_sdk_version,
)


class FakeSdk:
    check = staticmethod(MvsSdk.check)

    def __init__(self, frame: MvFrameOut):
        self.frame = frame
        self.free_calls = 0

    def get_frame(self, _handle, _timeout):
        return MV_OK, self.frame

    def free_frame(self, _handle, _frame):
        self.free_calls += 1

    def mv_convert_pixel(self, _handle, params_pointer):
        params = ctypes.cast(
            params_pointer, ctypes.POINTER(MvPixelConvertParamEx)
        ).contents
        ctypes.memset(params.pDstBuffer, 77, params.nDstBufferSize)
        params.nDstLen = params.nDstBufferSize
        return MV_OK


class CameraCoreTests(unittest.TestCase):
    def test_sdk_constants_and_struct_layout(self):
        self.assertEqual(PixelFormat.BayerRG8, 0x01080009)
        self.assertEqual(PixelFormat.BGR8, 0x02180015)
        self.assertEqual(format_sdk_version(0x04050003), "4.5.0.3")
        self.assertEqual(ctypes.sizeof(MvFrameOutInfoEx), 256)
        self.assertEqual(ctypes.sizeof(MvFrameOut), 328)

    def test_bgr_frame_is_copied_and_returned(self):
        pixels = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        storage = (ctypes.c_ubyte * len(pixels)).from_buffer_copy(pixels)
        frame = MvFrameOut()
        frame.pBufAddr = ctypes.cast(storage, ctypes.c_void_p)
        frame.stFrameInfo.nWidth = 2
        frame.stFrameInfo.nHeight = 2
        frame.stFrameInfo.enPixelType = PixelFormat.BGR8
        frame.stFrameInfo.nFrameLen = len(pixels)
        frame.stFrameInfo.nFrameNum = 7
        fake = FakeSdk(frame)
        camera = MvsCamera(CameraConfig(pixel_format="BGR8"), sdk=fake)
        camera.handle = ctypes.c_void_p(1)
        camera.opened = True
        camera.grabbing = True

        result = camera.grab_frame()

        self.assertIsNotNone(result)
        self.assertEqual(result.image_bgr.shape, (2, 2, 3))
        self.assertEqual(result.image_bgr[1, 1, 2], 12)
        self.assertEqual(fake.free_calls, 1)
        self.assertIsNone(camera.leased)

    def test_unknown_format_still_frees_sdk_buffer(self):
        storage = (ctypes.c_ubyte * 16)()
        frame = MvFrameOut()
        frame.pBufAddr = ctypes.cast(storage, ctypes.c_void_p)
        frame.stFrameInfo.nWidth = 2
        frame.stFrameInfo.nHeight = 2
        frame.stFrameInfo.nFrameLen = 4
        fake = FakeSdk(frame)
        camera = MvsCamera(CameraConfig(pixel_format="BayerRG8"), sdk=fake)
        camera.handle = ctypes.c_void_p(1)
        camera.opened = True
        camera.grabbing = True

        with self.assertRaises(CameraError):
            camera.grab_frame(force_pixel_type=0xDEADBEEF)

        self.assertEqual(fake.free_calls, 1)
        self.assertIsNone(camera.leased)
        self.assertTrue(
            any(item["step"] == "FreeImageBuffer" for item in camera.resource_log)
        )

    def test_bayer_conversion_returns_bgr_and_records_time(self):
        storage = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        frame = MvFrameOut()
        frame.pBufAddr = ctypes.cast(storage, ctypes.c_void_p)
        frame.stFrameInfo.nWidth = 2
        frame.stFrameInfo.nHeight = 2
        frame.stFrameInfo.enPixelType = PixelFormat.BayerRG8
        frame.stFrameInfo.nFrameLen = 4
        fake = FakeSdk(frame)
        camera = MvsCamera(CameraConfig(pixel_format="BayerRG8"), sdk=fake)
        camera.handle = ctypes.c_void_p(1)
        camera.opened = True
        camera.grabbing = True

        result = camera.grab_frame()

        self.assertEqual(result.raw_image.shape, (2, 2))
        self.assertEqual(result.image_bgr.shape, (2, 2, 3))
        self.assertEqual(int(result.image_bgr[0, 0, 0]), 77)
        self.assertGreaterEqual(result.conversion_ms, 0.0)
        self.assertEqual(fake.free_calls, 1)

    def test_bayer_conversion_falls_back_to_opencv(self):
        class FailingConvertSdk(FakeSdk):
            def mv_convert_pixel(self, _handle, _params_pointer):
                raise MvsError("MV_CC_ConvertPixelTypeEx", 0x80000500)

        storage = (ctypes.c_ubyte * 4)(10, 20, 30, 40)
        frame = MvFrameOut()
        frame.pBufAddr = ctypes.cast(storage, ctypes.c_void_p)
        frame.stFrameInfo.nWidth = 2
        frame.stFrameInfo.nHeight = 2
        frame.stFrameInfo.enPixelType = PixelFormat.BayerRG8
        frame.stFrameInfo.nFrameLen = 4
        fake = FailingConvertSdk(frame)
        camera = MvsCamera(CameraConfig(pixel_format="BayerRG8"), sdk=fake)
        camera.handle = ctypes.c_void_p(1)
        camera.opened = True
        camera.grabbing = True

        result = camera.grab_frame()

        self.assertIsNotNone(result)
        self.assertEqual(result.image_bgr.shape, (2, 2, 3))
        self.assertTrue(any("OpenCV" in item for item in camera.warnings))
        self.assertEqual(fake.free_calls, 1)

    def test_zero_pixel_type_is_recovered_from_matching_camera_format(self):
        storage = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        frame = MvFrameOut()
        frame.pBufAddr = ctypes.cast(storage, ctypes.c_void_p)
        frame.stFrameInfo.nWidth = 2
        frame.stFrameInfo.nHeight = 2
        frame.stFrameInfo.enPixelType = 0
        frame.stFrameInfo.nFrameLen = 4
        fake = FakeSdk(frame)
        camera = MvsCamera(CameraConfig(pixel_format="BayerRG8"), sdk=fake)
        camera.handle = ctypes.c_void_p(1)
        camera.opened = True
        camera.grabbing = True

        result = camera.grab_frame()

        self.assertEqual(result.actual_pixel_format, "BayerRG8")
        self.assertEqual(result.image_bgr.shape, (2, 2, 3))
        self.assertTrue(any("enPixelType" in item for item in camera.warnings))
        self.assertEqual(fake.free_calls, 1)

    def test_invalid_mvs_metadata_uses_camera_format_and_exact_length(self):
        storage = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        frame = MvFrameOut()
        frame.pBufAddr = ctypes.cast(storage, ctypes.c_void_p)
        frame.stFrameInfo.nWidth = 2
        frame.stFrameInfo.nHeight = 2
        frame.stFrameInfo.enPixelType = 0
        frame.stFrameInfo.nFrameLen = 0xB0A03F1E
        fake = FakeSdk(frame)
        camera = MvsCamera(CameraConfig(pixel_format="BayerRG8"), sdk=fake)
        camera.handle = ctypes.c_void_p(1)
        camera.opened = True
        camera.grabbing = True

        result = camera.grab_frame()

        self.assertIsNotNone(result)
        self.assertEqual(result.actual_pixel_format, "BayerRG8")
        self.assertEqual(result.frame_len, 4)
        self.assertEqual(result.image_bgr.shape, (2, 2, 3))
        self.assertTrue(any("nFrameLen" in item for item in camera.warnings))
        self.assertEqual(fake.free_calls, 1)

    def test_known_pixel_type_with_obviously_invalid_length_is_recovered(self):
        storage = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        frame = MvFrameOut()
        frame.pBufAddr = ctypes.cast(storage, ctypes.c_void_p)
        frame.stFrameInfo.nWidth = 2
        frame.stFrameInfo.nHeight = 2
        frame.stFrameInfo.enPixelType = PixelFormat.BayerRG8
        frame.stFrameInfo.nFrameLen = 0xB0A03F1E
        fake = FakeSdk(frame)
        camera = MvsCamera(CameraConfig(pixel_format="BayerRG8"), sdk=fake)
        camera.handle = ctypes.c_void_p(1)
        camera.opened = True
        camera.grabbing = True

        result = camera.grab_frame()

        self.assertIsNotNone(result)
        self.assertEqual(result.frame_len, 4)
        self.assertEqual(result.image_bgr.shape, (2, 2, 3))
        self.assertEqual(fake.free_calls, 1)

    def test_image_metrics_and_target_centers(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:, :] = 100
        image[:2, :] = 255
        metrics = image_metrics(image)
        self.assertGreater(metrics["saturated_pixel_ratio"], 0)
        self.assertAlmostEqual(metrics["mean_gray"], 115.5, places=1)

        dets = [
            {
                "box": [2.0, 4.0, 12.0, 14.0],
                "conf": 0.8,
                "name": "B3",
                "cls": 2,
            }
        ]
        targets = target_metrics(dets, image.shape)
        self.assertEqual(targets["centers"][0]["center_x"], 7.0)
        self.assertEqual(targets["centers"][0]["center_y"], 9.0)

    def test_fps_meter(self):
        meter = FpsMeter(window=10)
        for stamp in (0.0, 0.01, 0.02, 0.03):
            meter.tick(stamp)
        self.assertAlmostEqual(meter.value, 100.0, places=4)

    def test_cli_parses_expected_experiment_values(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "experiment",
                "--scene",
                "still",
                "--exposures",
                "500",
                "1000",
                "3000",
                "--gains",
                "0",
                "6",
                "12",
            ]
        )
        self.assertEqual(args.exposures, [500.0, 1000.0, 3000.0])
        self.assertEqual(args.gains, [0.0, 6.0, 12.0])

    def test_experiment_csv_writes_required_columns(self):
        path = Path("results/_test_experiment.csv")
        try:
            write_experiment_csv(
                path,
                [
                    {
                        "label": "x",
                        "config": {"exposure_us": 1000, "gain_db": 0},
                        "runtime": {
                            "actual_camera_pixel_format": "BayerRG8",
                            "frames": 10,
                            "processed_fps": 50.0,
                        },
                        "image_quality_mean": {
                            "mean_gray": 100.0,
                            "saturated_pixel_ratio": 0.01,
                        },
                    }
                ],
            )
            text = path.read_text(encoding="utf-8-sig")
            self.assertIn("saturated_pixel_ratio", text)
            self.assertIn("processed_fps", text)
        finally:
            path.unlink(missing_ok=True)

    def test_fault_log_checks_get_and_free_balance(self):
        camera = SimpleNamespace(
            resource_log=[
                {
                    "step": "CreateHandle",
                    "ok": True,
                },
                {
                    "step": "OpenDevice",
                    "ok": True,
                },
                {
                    "step": "StartGrabbing",
                    "ok": True,
                },
                {
                    "step": "GetImageBuffer",
                    "ok": True,
                },
                {
                    "step": "StopGrabbing",
                    "ok": True,
                },
                {
                    "step": "CloseDevice",
                    "ok": True,
                },
                {
                    "step": "DestroyHandle",
                    "ok": True,
                },
                {
                    "step": "FreeImageBuffer",
                    "ok": False,
                },
            ],
            warnings=[],
            leased=None,
            grabbing=False,
            opened=False,
            handle=None,
            _owns_sdk_init=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fault.json"
            with contextlib.redirect_stdout(io.StringIO()):
                data = write_fault_log(
                    path,
                    "bad-format",
                    "expected",
                    camera,
                    True,
                )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertFalse(data["released"]["leased_buffer_clear"])
        self.assertTrue(saved["expected_fault_observed"])
        self.assertEqual(data["successful_get_image_buffer_calls"], 1)
        self.assertEqual(data["successful_free_image_buffer_calls"], 0)


if __name__ == "__main__":
    unittest.main()
