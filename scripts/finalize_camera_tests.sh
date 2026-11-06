#!/usr/bin/env bash
set -u

repo_root="${1:-$HOME/yolo-export-inference}"
cd "$repo_root" || {
    echo "找不到仓库目录: $repo_root" >&2
    exit 1
}

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "找不到 .venv，请先进入仓库并创建/激活 Python 虚拟环境" >&2
    exit 1
fi

source .venv/bin/activate
export MVS_SDK_LIBRARY=/opt/MVS/lib/64/libMvCameraControl.so
mkdir -p results/industrial_camera/logs

echo "=== 1/4 Python 离线测试 ==="
python -m unittest discover -s tests -v \
    2>&1 | tee results/industrial_camera/logs/unittest_terminal.txt

echo "=== 2/4 取流超时与资源释放 ==="
python camera_cli.py fault-test \
    --index 0 \
    --mode timeout \
    2>&1 | tee results/industrial_camera/logs/fault_timeout_terminal.txt

echo "=== 3/4 未知格式与资源释放 ==="
python camera_cli.py fault-test \
    --index 0 \
    --mode bad-format \
    2>&1 | tee results/industrial_camera/logs/fault_bad-format_terminal.txt

echo "=== 4/4 cv::Mat 内存示例 ==="
cmake -S camera_examples -B build-memory -DBUILD_MVS_CAPTURE=OFF \
    2>&1 | tee results/industrial_camera/logs/mat_memory_build.txt
cmake --build build-memory -j2 \
    2>&1 | tee -a results/industrial_camera/logs/mat_memory_build.txt
./build-memory/mat_memory \
    2>&1 | tee results/industrial_camera/logs/mat_memory_output.txt

echo "=== 打包结果 ==="
python - <<'PY'
from datetime import datetime
from pathlib import Path
import zipfile

repo = Path.cwd()
source = repo / "results" / "industrial_camera"
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path.home() / f"industrial_camera_results_{stamp}.zip"

with zipfile.ZipFile(
    out,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
    allowZip64=True,
) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(repo))

print(f"ZIP: {out}")
print(f"ZIP大小: {out.stat().st_size / 1024 / 1024:.2f} MiB")
PY

echo
echo "完成。请把上面的 ZIP 通过 QQ 传到 Windows，再发给 Codex。"
