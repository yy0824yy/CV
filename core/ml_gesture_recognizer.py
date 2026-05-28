"""
机器学习手势识别器。

训练脚本会把最佳模型保存为 models/gesture_rf.joblib。该模块负责在
实时系统中加载模型，并复用与训练阶段一致的 21 点特征工程：
    1) 减去 wrist 坐标
    2) 按手掌尺度归一化
    3) 展平成 63 维向量
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.gesture_recognizer import GestureResult
from core.hand_detector import HandLandmarks


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_project_path(path: str) -> str:
    """把相对路径解析到项目根目录。"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def landmarks_to_feature_vector(landmarks_norm: np.ndarray) -> np.ndarray:
    """单只手 landmarks -> (63,) 特征，与训练脚本保持一致。"""
    lm = np.asarray(landmarks_norm, dtype=np.float32)
    if lm.shape != (21, 3):
        raise ValueError(f"手部 landmarks 形状应为 (21, 3)，实际为 {lm.shape}")
    centered = lm - lm[0:1, :]
    palm = float(np.linalg.norm(centered, axis=1).max())
    if palm < 1e-6:
        palm = 1.0
    return (centered / palm).reshape(-1)


def landmarks_to_features_batch(landmarks: np.ndarray) -> np.ndarray:
    """批量 landmarks -> (N, 63) 特征。"""
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.ndim != 3 or lm.shape[1:] != (21, 3):
        raise ValueError(f"landmarks 形状应为 (N, 21, 3)，实际为 {lm.shape}")
    return np.stack([landmarks_to_feature_vector(x) for x in lm], axis=0)


@dataclass
class MLModelInfo:
    model_name: str = ""
    classes: List[str] = None
    train_samples: int = 0
    feature_version: str = ""


class MLGestureRecognizer:
    """基于 sklearn/joblib 模型的手势识别器。"""

    def __init__(self, model_path: str):
        self.model_path = resolve_project_path(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"未找到手势模型: {self.model_path}。请先运行 "
                "python experiments/train_gesture_classifier.py"
            )
        import joblib

        payload = joblib.load(self.model_path)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
            self.info = MLModelInfo(
                model_name=str(payload.get("model_name", "")),
                classes=list(payload.get("classes", [])),
                train_samples=int(payload.get("train_samples", 0)),
                feature_version=str(payload.get("feature_version", "")),
            )
        else:
            self.model = payload
            self.info = MLModelInfo()

    def recognize(self, hand: HandLandmarks) -> GestureResult:
        x = landmarks_to_feature_vector(hand.landmarks_norm).reshape(1, -1)
        name = str(self.model.predict(x)[0])
        score = 1.0
        extra: Dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            best = int(np.argmax(proba))
            score = float(proba[best])
            extra["ml_confidence"] = score
        return GestureResult(
            name=name,
            score=score,
            finger_states=[],
            extra=extra,
        )

    def recognize_all(self, hands: List[HandLandmarks]) -> List[GestureResult]:
        return [self.recognize(h) for h in hands]
