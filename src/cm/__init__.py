"""Contour_Maker — 超声低回声目标分割对比实验.

对比两种方案:
  A) YOLO26n 检测 + Otsu 自适应粗糙轮廓 + contour-refiner 精细化 (本方案)
  B) YOLO26n-seg 端到端实例分割 (baseline)
"""

__version__ = "0.1.0"
