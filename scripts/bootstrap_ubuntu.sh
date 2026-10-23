#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[环境] $(date -Iseconds)"
python3 --version
if command -v free >/dev/null; then free -h; fi
if command -v df >/dev/null; then df -h .; fi
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || [[ -x .venv/bin/python ]]
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    mkdir -p outputs
    python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py','outputs/get-pip.py')
PY
    .venv/bin/python outputs/get-pip.py
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import numpy, matplotlib, pytest; print("numpy", numpy.__version__, "matplotlib", matplotlib.__version__, "pytest", pytest.__version__)'
echo '[完成] 已安装 .venv，请运行 bash scripts/smoke_test.sh；上传完整位移TXT后运行 bash run_v01.sh'
