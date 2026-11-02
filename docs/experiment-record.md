# 工业相机实验记录

> 本文件是模板。所有 `TBD` 必须由实机测试数据替换。请保留失败样本和原始日志。

## 1. 设备与软件

| 项目 | 实测值 |
|---|---|
| 相机型号 | TBD |
| 相机序列号 | TBD |
| 固件版本 | TBD |
| Ubuntu 版本 | TBD |
| CPU / 内存 | TBD |
| Python | TBD |
| MVS 版本 | TBD |
| MVS 动态库路径 | TBD |
| OpenCV | TBD |
| ONNX Runtime / PyTorch | TBD |
| 镜头型号 / 焦距 | TBD |
| 工作距离 | TBD |
| 光源 | TBD |
| 光圈 / 对焦 | TBD |

## 2. 相机公共参数

| 参数 | 值 |
|---|---|
| ROI（OffsetX/Y/Width/Height） | TBD |
| 触发模式 | Off（自由运行） |
| ExposureAuto | Off |
| GainAuto | Off |
| 目标帧率 | TBD |
| 相机实际输出格式 | TBD |
| 程序转换后的格式 | BayerRG8 时为 BGR8 |
| 丢帧/超时数量 | TBD |

## 3. 曝光实验

固定机位、光源、光圈、增益 0 dB。静止与运动分别拍摄。

### 3.1 静止装甲板

| 曝光 μs | 平均灰度 | 饱和像素比例 | 实际 FPS | 程序 CPU % | 画面亮度描述 |
|---:|---:|---:|---:|---:|---|
| 500 | TBD | TBD | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD | TBD | TBD |
| 3000 | TBD | TBD | TBD | TBD | TBD |

### 3.2 运动装甲板

| 曝光 μs | 拖影描述 | 清晰度代理值 | 漏检/误检 | 代表图片 |
|---:|---|---:|---|---|
| 500 | TBD | TBD | TBD | `TBD.jpg` |
| 1000 | TBD | TBD | TBD | `TBD.jpg` |
| 3000 | TBD | TBD | TBD | `TBD.jpg` |

分析：曝光增加为什么改变亮度：TBD。

分析：曝光增加为什么加重运动模糊：TBD。

## 4. 增益实验

固定曝光 1000 μs、机位和光源，装甲板静止。饱和像素按灰度 `>=250` 统计。

| 增益 dB | 平均灰度 | 饱和像素比例 | 暗部噪声标准差 | 过曝描述 |
|---:|---:|---:|---:|---|
| 0 | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD |
| 12 | TBD | TBD | TBD | TBD |

分析：增益提高后画面为何变亮，以及为什么可能降低识别质量：TBD。

## 5. 图像格式与性能

固定 1000 μs、0 dB、相同 ROI。

| 请求格式 | 相机实际格式 | 转换流程 | 平均转换 ms | P95 转换 ms | 实际 FPS | 程序 CPU % |
|---|---|---|---:|---:|---:|---:|
| BayerRG8 | BayerRG8 | `MV_CC_ConvertPixelTypeEx -> BGR8` | TBD | TBD | TBD | TBD |
| BGR8 | TBD | 无转换 / 若失败请说明 | TBD | TBD | TBD | TBD |

分析：格式大小、USB 带宽、主机拷贝和 Bayer 插值的性能差异：TBD。

明确回答：本实验中的“相机的原生输出格式”是 TBD；“程序送给 OpenCV 的格式”是 TBD。

## 6. YOLO 实时识别

| 项目 | 实测值 |
|---|---|
| 模型文件 | TBD |
| 模型格式 | TBD |
| 推理后端 | TBD |
| 推理设备 | CPU / GPU：TBD |
| conf / iou / imgsz | TBD |
| 相机实际 FPS | TBD |
| 从图像接收完成到结果显示的 FPS | TBD |
| 平均推理耗时 | TBD |
| 检测视频 | TBD |
| 结果 CSV | TBD |

识别正常样例：

| 时间/帧号 | 类别 | 置信度 | 关键点是否贴合 | 图片 |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | `TBD.jpg` |

识别异常样例：

| 时间/帧号 | 现象 | 可能原因 | 原始参数 | 图片 |
|---|---|---|---|---|
| TBD | 漏检/误检/类别错误/关键点偏移 | TBD | 曝光 TBD，增益 TBD | `TBD.jpg` |

## 7. 曝光和增益对识别的影响

| 曝光/增益 | 检测目标数 | 含检测帧比例 | 平均置信度 | 主要问题 |
|---|---:|---:|---:|---|
| 500 μs / 0 dB | TBD | TBD | TBD | TBD |
| 1000 μs / 0 dB | TBD | TBD | TBD | TBD |
| 3000 μs / 0 dB | TBD | TBD | TBD | TBD |
| 1000 μs / 6 dB | TBD | TBD | TBD | TBD |
| 1000 μs / 12 dB | TBD | TBD | TBD | TBD |

## 8. 程序退出与错误处理

### 8.1 取流超时

- 日志：`results/industrial_camera/logs/fault_timeout.json`
- 是否确认触发预期异常（`expected_fault_observed`）：TBD
- 观察到的错误码：TBD
- 是否停止取流：TBD
- 是否关闭设备/销毁句柄/Finalize：TBD
- `released` 四项是否全部为 `true`：TBD
- 成功 Get/Free 次数：TBD

### 8.2 不支持的图像格式

- 日志：`results/industrial_camera/logs/fault_bad-format.json`
- 注入类型：`0xDEADBEEF`
- 是否确认触发预期异常（`expected_fault_observed`）：TBD
- 是否拒绝把未知字节解释成图像：TBD
- 是否仍调用 `MV_CC_FreeImageBuffer`：TBD
- `released` 四项是否全部为 `true`：TBD
- 成功 Get/Free 次数：TBD

## 9. Mat 内存观察

运行命令：

```bash
cmake -S camera_examples -B build-memory -DBUILD_MVS_CAPTURE=OFF
cmake --build build-memory -j2
./build-memory/mat_memory
```

记录输出：TBD。

解释：`cv::Mat alias = view` 与 `clone()` 的内存所有权差异：TBD。

## 10. 结论

本次最佳相机参数：TBD。

主要限制：TBD。

如果重新实验，会优先修改：TBD。
