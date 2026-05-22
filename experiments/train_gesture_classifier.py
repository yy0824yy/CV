"""
手势识别：规则方法 vs 机器学习分类器对比实验。

读取 data/outputs/labeled/gesture_*.csv（必须是新版，包含 21 个 landmark 列）
对比四种方法的整体准确率：
    1. Rule-based  —— 现有规则识别器（直接读 CSV 中的 left_gesture/right_gesture）
    2. RandomForest
    3. MLP（多层感知机）
    4. SVM（RBF 核）

特征工程：
    - 减去 wrist（point 0），让坐标相对于手腕
    - 除以 palm_size（手腕到最远关键点的距离），让特征尺度不变
    - flatten 成 63 维向量

输出：
    - 终端打印每个方法的准确率 + macro-F1
    - data/outputs/figures/fig_classifier_comparison.png  四方法柱状图
    - data/outputs/figures/fig_classifier_<name>_cm.png    每个 ML 模型的混淆矩阵

用法：
    python experiments/train_gesture_classifier.py
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

# 为了加噪后能重跑规则识别器
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.gesture_recognizer import GestureRecognizer
from core.hand_detector import HandLandmarks

# 中文字体（沿用 analyze.py 风格）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")


LABELED_DIR = os.path.join("data", "outputs", "labeled")
FIG_DIR = os.path.join("data", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# 1. 加载数据
# ============================================================
def load_gesture_data() -> pd.DataFrame:
    """合并所有 gesture_*.csv，过滤掉无手部检测的帧。"""
    pattern = os.path.join(LABELED_DIR, "gesture_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未找到 {pattern}，请先运行 collect_labeled.py 采集数据")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["__source"] = os.path.basename(fp)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 合并 {len(files)} 个文件，共 {len(df)} 行")

    # 必须包含 landmark 列
    if "hx_0" not in df.columns:
        raise ValueError(
            "CSV 中没有 hx_0 等 landmark 列。"
            "请用最新版 collect_labeled.py 重新采集数据。"
        )

    # 过滤掉没有手部 landmark 的行
    before = len(df)
    df = df.dropna(subset=["hx_0"])
    df = df[df["hx_0"] != ""].reset_index(drop=True)
    print(f"[INFO] 过滤无手部检测帧后剩 {len(df)} / {before} 行")
    return df


# ============================================================
# 2. 特征工程
# ============================================================
def load_raw_landmarks(df: pd.DataFrame) -> np.ndarray:
    """从 DataFrame 读出原始 landmarks，返回 (N, 21, 3) 张量。"""
    xs = df[[f"hx_{i}" for i in range(21)]].astype(float).to_numpy()
    ys = df[[f"hy_{i}" for i in range(21)]].astype(float).to_numpy()
    zs = df[[f"hz_{i}" for i in range(21)]].astype(float).to_numpy()
    return np.stack([xs, ys, zs], axis=-1)  # (N, 21, 3)


def landmarks_to_features(lm: np.ndarray) -> np.ndarray:
    """
    把 (N, 21, 3) landmarks 转成 (N, 63) 特征。
    1) 减 wrist  2) 除 palm_size
    """
    wrist = lm[:, 0:1, :]
    centered = lm - wrist
    palm = np.linalg.norm(centered, axis=2).max(axis=1, keepdims=True)
    palm = np.where(palm < 1e-6, 1.0, palm)
    norm = centered / palm[:, :, None]
    return norm.reshape(len(lm), -1)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """向后兼容接口：返回 (N, 63) 特征。"""
    return landmarks_to_features(load_raw_landmarks(df))


def get_rule_predictions(df: pd.DataFrame) -> np.ndarray:
    """从 CSV 中读出规则识别器的预测结果（无噪声）。"""
    preds = []
    for _, row in df.iterrows():
        side = row.get("hand_side", "")
        if side == "Right":
            preds.append(str(row.get("right_gesture", "") or "Unknown"))
        elif side == "Left":
            preds.append(str(row.get("left_gesture", "") or "Unknown"))
        else:
            preds.append("Unknown")
    return np.array(preds)


def rule_predict_from_landmarks(
    lm: np.ndarray, hand_sides: list, recognizer: GestureRecognizer
) -> np.ndarray:
    """给定 (N, 21, 3) landmarks，重跑规则识别器返回预测。用于加噪实验。"""
    preds = []
    H, W = 480, 640  # 虚假像素尺寸（规则仅用 norm 坐标）
    for i in range(len(lm)):
        norm = lm[i].astype(np.float32)
        px = np.zeros((21, 2), dtype=np.int32)
        px[:, 0] = np.clip(norm[:, 0] * W, 0, W - 1)
        px[:, 1] = np.clip(norm[:, 1] * H, 0, H - 1)
        side = hand_sides[i] if hand_sides[i] in ("Left", "Right") else "Right"
        h = HandLandmarks(handedness=side, score=1.0,
                          landmarks_norm=norm, landmarks_px=px)
        preds.append(recognizer.recognize(h).name)
    return np.array(preds)


# ============================================================
# 3. 训练 + 评估
# ============================================================
def evaluate_model(name: str, model, X: np.ndarray, y: np.ndarray,
                   cv: StratifiedKFold) -> dict:
    """5-fold cross_val_predict，得到所有样本的 OOF 预测，再统计。"""
    print(f"  [训练] {name} ...")
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    print(f"    {name}: 准确率 = {acc:.4f}  macro-F1 = {f1:.4f}")
    return {"name": name, "acc": acc, "f1": f1, "y_pred": y_pred}


# ============================================================
# 4. 画图
# ============================================================
def plot_comparison(results: list, n_samples: int):
    """对比柱状图：四种方法的准确率。"""
    names = [r["name"] for r in results]
    accs = [r["acc"] for r in results]
    f1s = [r["f1"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label="准确率",
                   color="#4C9BD9", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, f1s, w, label="Macro-F1",
                   color="#F0A04B", edgecolor="black", linewidth=0.5)

    for b, v in zip(bars1, accs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for b, v in zip(bars2, f1s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("分数")
    ax.set_title(f"手势识别：规则方法 vs 机器学习分类器  (N={n_samples})")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_classifier_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OUT] {out}")


def plot_confusion(name: str, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"{name} 混淆矩阵")
    # 数字标注
    thr = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black",
                    fontsize=10)
    plt.tight_layout()
    safe = (name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", ""))
    out = os.path.join(FIG_DIR, f"fig_classifier_{safe}_cm.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OUT] {out}")


# ============================================================
# 5. 噪声鲁棒性实验
# ============================================================
def run_noise_robustness(lm_raw: np.ndarray, y: np.ndarray,
                        hand_sides: list, models: dict,
                        sigmas=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
                        n_repeats: int = 3, seed: int = 42):
    """
    在不同高斯噪声幅度下评估所有方法准确率。
    - ML：用干净训练，测试集加噪，重复 n_repeats 次取均值
    - Rule：直接对加噪 landmarks 重跑规则识别器
    """
    print("\n[实验] 噪声鲁棒性 ...")
    rng = np.random.default_rng(seed)
    rule_rec = GestureRecognizer()

    # 记录结果：{model_name: [mean_acc_at_sigma]}
    results = {name: [] for name in ["Rule-based"] + list(models.keys())}

    # ML：一次性拆分（同 sigma 复用同一个训练集）
    X_clean = landmarks_to_features(lm_raw)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_clean, y, np.arange(len(y)),
        test_size=0.3, random_state=seed, stratify=y,
    )
    fitted = {}
    for name, model in models.items():
        m = clone(model)
        m.fit(X_train, y_train)
        fitted[name] = m

    for sigma in sigmas:
        accs_ml = {name: [] for name in models.keys()}
        accs_rule = []
        for rep in range(n_repeats):
            # 在 lm_raw 上加噪（只加在测试 indices）
            noise = rng.normal(0, sigma, lm_raw.shape) if sigma > 0 else 0
            lm_noisy = lm_raw + noise

            # ML：重新计算测试集特征
            X_test_noisy = landmarks_to_features(lm_noisy[idx_test])
            for name, m in fitted.items():
                accs_ml[name].append(accuracy_score(y_test, m.predict(X_test_noisy)))

            # Rule：全量重跑（在同一个测试集上评估以保证对齐）
            sides_test = [hand_sides[i] for i in idx_test]
            rule_pred = rule_predict_from_landmarks(
                lm_noisy[idx_test], sides_test, rule_rec)
            accs_rule.append(accuracy_score(y_test, rule_pred))

        results["Rule-based"].append(float(np.mean(accs_rule)))
        for name in models.keys():
            results[name].append(float(np.mean(accs_ml[name])))
        msg = f"  sigma={sigma:.3f}:  Rule={results['Rule-based'][-1]:.3f}"
        for name in models.keys():
            msg += f"  {name}={results[name][-1]:.3f}"
        print(msg)

    # 画图
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"Rule-based": "#888888", "RandomForest": "#4C9BD9",
              "MLP": "#F0A04B", "SVM (RBF)": "#7BB661"}
    markers = {"Rule-based": "o", "RandomForest": "s",
               "MLP": "^", "SVM (RBF)": "D"}
    for name, accs in results.items():
        ax.plot(sigmas, accs, marker=markers.get(name, "o"),
                label=name, linewidth=2, markersize=7,
                color=colors.get(name, None))
    ax.set_xlabel("高斯噪声标准差  σ（归一化坐标）")
    ax.set_ylabel("准确率")
    ax.set_title("噪声鲁棒性：手部 landmarks 加噪后不同方法的识别准确率")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_classifier_noise_robustness.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OUT] {out}")
    return results


# ============================================================
# 6. 学习曲线实验
# ============================================================
def run_learning_curve(X: np.ndarray, lm_raw: np.ndarray, y: np.ndarray,
                       models: dict,
                       per_class_sizes=(5, 10, 20, 40, 80),
                       n_repeats: int = 3, seed: int = 42,
                       noise_sigma: float = 0.02):
    """
    逐步增加每类训练样本数，记录 ML 模型准确率。
    测试集会加一点高斯噪声（sigma=0.02）以避免全 100% 饫和。
    """
    print(f"\n[实验] 学习曲线（测试集 σ={noise_sigma}） ...")
    rng = np.random.default_rng(seed)
    classes = sorted(set(y))
    n_per_class = min(int(np.sum(y == c)) for c in classes)

    results = {name: {"mean": [], "std": []} for name in models.keys()}
    sizes_actual = [s for s in per_class_sizes if s + 10 <= n_per_class]
    if not sizes_actual:
        sizes_actual = [max(1, n_per_class // 4)]

    for size in sizes_actual:
        accs_at_size = {name: [] for name in models.keys()}
        for rep in range(n_repeats):
            train_idx = []
            test_idx = []
            for c in classes:
                idx_c = np.where(y == c)[0].copy()
                rng.shuffle(idx_c)
                train_idx.extend(idx_c[:size].tolist())
                test_idx.extend(idx_c[size:].tolist())
            if len(test_idx) == 0:
                continue
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            # 测试集加噪（在 raw landmarks 上加，重新抽取特征）
            lm_test_noisy = lm_raw[test_idx] + rng.normal(
                0, noise_sigma, lm_raw[test_idx].shape)
            X_te = landmarks_to_features(lm_test_noisy)
            y_te = y[test_idx]
            for name, model in models.items():
                m = clone(model)
                m.fit(X_tr, y_tr)
                accs_at_size[name].append(accuracy_score(y_te, m.predict(X_te)))
        for name in models.keys():
            arr = np.array(accs_at_size[name]) if accs_at_size[name] else np.array([0.0])
            results[name]["mean"].append(float(arr.mean()))
            results[name]["std"].append(float(arr.std()))
        msg = f"  N={size:3d}/类: "
        for name in models.keys():
            msg += f" {name}={results[name]['mean'][-1]:.3f}\u00b1{results[name]['std'][-1]:.3f}"
        print(msg)

    # 画图
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"RandomForest": "#4C9BD9", "MLP": "#F0A04B", "SVM (RBF)": "#7BB661"}
    markers = {"RandomForest": "s", "MLP": "^", "SVM (RBF)": "D"}
    for name, r in results.items():
        means = np.array(r["mean"])
        stds = np.array(r["std"])
        ax.plot(sizes_actual, means, marker=markers.get(name, "o"),
                label=name, linewidth=2, markersize=7,
                color=colors.get(name, None))
        ax.fill_between(sizes_actual, means - stds, means + stds,
                        alpha=0.15, color=colors.get(name, None))
    ax.set_xlabel("每类训练样本数")
    ax.set_ylabel("测试准确率")
    ax.set_title(f"学习曲线：不同训练样本量下的准确率 (测试加噪 σ={noise_sigma}, {n_repeats} 次重复)")
    # 根据实际数据范围自适应 y 轴
    all_means = [m for r in results.values() for m in r["mean"]]
    all_stds = [s for r in results.values() for s in r["std"]]
    if all_means:
        ymin = max(0.0, min(all_means) - max(all_stds) - 0.05)
    else:
        ymin = 0.0
    ax.set_ylim(ymin, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_classifier_learning_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OUT] {out}")
    return results


# ============================================================
# 7. 主流程
# ============================================================
def main():
    df = load_gesture_data()
    lm_raw = load_raw_landmarks(df)
    X = landmarks_to_features(lm_raw)
    y = df["gt_label"].astype(str).to_numpy()
    hand_sides = df["hand_side"].astype(str).tolist()
    labels = sorted(set(y))
    print(f"[INFO] 特征维度: {X.shape}，类别数: {len(labels)} -> {labels}")

    # 规则方法基线（从 CSV 中已记录的预测列读取）
    rule_pred = get_rule_predictions(df)
    rule_acc = accuracy_score(y, rule_pred)
    rule_f1 = f1_score(y, rule_pred, average="macro")
    print(f"\n[基线] Rule-based: 准确率 = {rule_acc:.4f}  macro-F1 = {rule_f1:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n[INFO] 开始 5-fold 交叉验证 ...")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=400,
            activation="relu", random_state=42)),
    ])
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=10, gamma="scale", kernel="rbf", random_state=42)),
    ])

    results = [
        {"name": "Rule-based", "acc": rule_acc, "f1": rule_f1, "y_pred": rule_pred},
        evaluate_model("RandomForest", rf, X, y, cv),
        evaluate_model("MLP", mlp, X, y, cv),
        evaluate_model("SVM (RBF)", svm, X, y, cv),
    ]

    # 详细分类报告（只打印 ML 中最好的那个）
    print("\n[INFO] 各 ML 模型的详细分类报告：")
    for r in results[1:]:
        print(f"\n--- {r['name']} ---")
        print(classification_report(y, r["y_pred"], digits=3, zero_division=0))

    # 画图
    print("\n[INFO] 输出图表：")
    plot_comparison(results, n_samples=len(y))
    for r in results[1:]:
        plot_confusion(r["name"], y, r["y_pred"], labels)

    # 总结
    print("\n[完成] 结果汇总：")
    print(f"  {'方法':<14} {'准确率':>10} {'Macro-F1':>10}")
    print(f"  {'-'*36}")
    for r in results:
        print(f"  {r['name']:<14} {r['acc']*100:>9.2f}% {r['f1']:>10.4f}")

    # ============================================================
    # 进阶实验：鲁棒性 + 学习曲线
    # ============================================================
    ml_models = {"RandomForest": rf, "MLP": mlp, "SVM (RBF)": svm}
    run_noise_robustness(lm_raw, y, hand_sides, ml_models,
                         sigmas=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
                         n_repeats=3, seed=42)
    run_learning_curve(X, lm_raw, y, ml_models,
                       per_class_sizes=(2, 5, 10, 20, 40, 60),
                       n_repeats=5, seed=42, noise_sigma=0.02)


if __name__ == "__main__":
    main()
