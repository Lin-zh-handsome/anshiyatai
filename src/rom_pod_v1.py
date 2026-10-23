#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容入口；核心实现位于pod_engine.py。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pod_engine import *
if __name__ == '__main__':
    sys.exit(main())
