"""
实验数据分析与图表生成。

支持两类输入 CSV：
    1) 普通运行日志：data/outputs/logs/*.csv（无标签，UI 自动记录）
    2) 带标签实验数据：data/outputs/labeled/*.csv（collect_labeled.py 输出）

输出 PNG 图表到 data/outputs/figures/：
    fig_fps_histogram.png            FPS 分布直方图
    fig_distance_histogram.png       人体距离分布
    fig_angle_smoothness.png         角度平滑前后方差对比
    fig_gesture_accuracy_bar.png     [需手势标签] 各手势准确率
    fig_gesture_confusion.png        [需手势标签] 手势混淆矩阵
    fig_action_accuracy_bar.png      [需动作标签] 各动作准确率
    fig_action_confusion.png         [需动作标签] 动作混淆矩阵
    fig_distance_vs_accuracy.png     [需距离标签] 距离对识别的影响

用法：
    python experiments/analyze.py
        [--logs PATH ...]      指定 CSV 文件；不指定则自动扫描
        [--no-logs]            跳过普通日志（仅分析带标签数据）
        [--no-labeled]         跳过带标签数据
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无头后端，脚本无需 GUI 也能运行
import matplotlib.pyplot as plt

# 中文字体（Windows 自带）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 项目路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOGS_DIR = os.path.join(ROOT, "data", "outputs", "logs")
LABELED_DIR = os.path.join(ROOT, "data", "outputs", "labeled")
FIG_DIR = os.path.join(ROOT, "data", "outputs", "figures")


GESTURE_LABELS = [
    "Number_1", "Number_2", "Number_3",
    "Open_Palm", "Fist", "OK", "Thumbs_Up",
]
ACTION_LABELS = [
    "Standing",
    "Raise_Left_Hand", "Raise_Right_Hand", "Raise_Both_Hands",
    "Lean_Left", "Lean_Right",
    "Squat", "Bend_Over",
]


# ============================================================
# 工具
# ============================================================
def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def load_csvs(paths: List[str], label: str = "") -> pd.DataFrame:
    """读多个 CSV 并合并；缺列自动忽略。"""
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__src"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] 读取失败 {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] {label} 合并 {len(paths)} 个文件，共 {len(df)} 行")
    return df


def save_fig(fig, name: str):
    out = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [OUT] {out}")
    return out


# ============================================================
# 图 1: FPS 直方图
# ============================================================
def plot_fps_histogram(df: pd.DataFrame):
    if df.empty or "fps" not in df.columns:
        return
    fps = pd.to_numeric(df["fps"], errors="coerce").dropna()
    fps = fps[fps > 0]   # 排除 0
    if len(fps) < 5:
        print("[SKIP] FPS 数据不足，跳过 fps_histogram")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fps, bins=30, color="#4FC3F7", edgecolor="#01579B")
    mean = float(fps.mean())
    median = float(fps.median())
    ax.axvline(mean, color="red", linestyle="--", linewidth=2,
               label=f"均值 {mean:.1f}")
    ax.axvline(median, color="orange", linestyle=":", linewidth=2,
               label=f"中位数 {median:.1f}")
    ax.set_xlabel("FPS")
    ax.set_ylabel("帧数")
    ax.set_title(f"系统实时帧率分布 (N={len(fps)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "fig_fps_histogram.png")


# ============================================================
# 图 2: 距离分布直方图
# ============================================================
def plot_distance_histogram(df: pd.DataFrame):
    if df.empty or "person_distance_m" not in df.columns:
        return
    d = pd.to_numeric(df["person_distance_m"], errors="coerce").dropna()
    d = d[(d > 0) & (d < 5)]
    if len(d) < 5:
        print("[SKIP] 距离数据不足，跳过 distance_histogram")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=30, color="#81C784", edgecolor="#1B5E20")
    mean = float(d.mean())
    ax.axvline(mean, color="red", linestyle="--", linewidth=2,
               label=f"均值 {mean:.2f} m")
    ax.set_xlabel("人体距离 (m)")
    ax.set_ylabel("帧数")
    ax.set_title(f"人体到摄像头距离分布 (N={len(d)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "fig_distance_histogram.png")


# ============================================================
# 图 3: 角度平滑稳定性（用 left_elbow 作为代表）
# ============================================================
def plot_angle_smoothness(df: pd.DataFrame):
    if df.empty or "left_elbow" not in df.columns:
        return
    e = pd.to_numeric(df["left_elbow"], errors="coerce").dropna().values
    e = e[(e > 0) & (e < 180)]
    if len(e) < 50:
        print("[SKIP] 角度数据不足，跳过 angle_smoothness")
        return
    # 截取最长的连续 500 帧（避免跨 session 拼接产生的跳变）
    if len(e) > 500:
        e = e[:500]

    # 不同 alpha 的 EMA 对比
    def ema(x, alpha):
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    sm_03 = ema(e, 0.3)
    sm_05 = ema(e, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(e, color="#EF5350", alpha=0.6, linewidth=1,
                 label=f"raw (var={e.var():.1f})")
    axes[0].plot(sm_05, color="#FB8C00", alpha=0.85, linewidth=1.5,
                 label=f"EMA α=0.5 (var={sm_05.var():.1f})")
    axes[0].plot(sm_03, color="#43A047", linewidth=2,
                 label=f"EMA α=0.3 (var={sm_03.var():.1f})")
    axes[0].set_xlabel("帧序号")
    axes[0].set_ylabel("左肘角度 (°)")
    axes[0].set_title("肘关节角度时间序列：平滑前后对比")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    var_data = [e.var(), sm_05.var(), sm_03.var()]
    var_labels = ["raw", "EMA α=0.5", "EMA α=0.3"]
    var_colors = ["#EF5350", "#FB8C00", "#43A047"]
    bars = axes[1].bar(var_labels, var_data, color=var_colors,
                       edgecolor="#444", linewidth=1)
    for b, v in zip(bars, var_data):
        axes[1].text(b.get_x() + b.get_width() / 2, v,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("方差")
    axes[1].set_title("方差对比（越小越稳定）")
    axes[1].grid(True, alpha=0.3, axis="y")

    save_fig(fig, "fig_angle_smoothness.png")


# ============================================================
# 工具：手势预测取值
# ============================================================
def _pick_predicted_gesture(row) -> str:
    """带标签数据下选择预测值。
    若任一只手预测等于 gt_label，认为预测对了；否则取 right 优先，再取 left。
    """
    lg = str(row.get("left_gesture", "") or "")
    rg = str(row.get("right_gesture", "") or "")
    gt = str(row.get("gt_label", "") or "")
    if gt and (lg == gt or rg == gt):
        return gt
    return rg if rg else lg


# ============================================================
# 工具：混淆矩阵
# ============================================================
def _plot_confusion(gt: List[str], pred: List[str], labels: List[str],
                    title: str, out_name: str):
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((n, n), dtype=int)
    for g, p in zip(gt, pred):
        if g in idx:
            j = idx.get(p, None)
            if j is not None:
                mat[idx[g], j] += 1

    fig, ax = plt.subplots(figsize=(7, 6.2))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)
    vmax = mat.max() if mat.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            v = int(mat[i, j])
            if v > 0:
                color = "white" if v > vmax * 0.5 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, out_name)


# ============================================================
# 通用：准确率柱状图
# ============================================================
def _plot_accuracy_bar(sub: pd.DataFrame, labels: List[str], pred_col: str,
                       title_prefix: str, color: str, out_name: str):
    acc_data = []
    for lbl in labels:
        rows = sub[sub["gt_label"] == lbl]
        if rows.empty:
            continue
        correct = int((rows[pred_col].astype(str) == lbl).sum())
        total = len(rows)
        acc_data.append((lbl, correct / total, total, correct))

    if not acc_data:
        return None

    names = [x[0] for x in acc_data]
    accs = [x[1] for x in acc_data]
    totals = [x[2] for x in acc_data]

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(names) + 2), 4.5))
    bars = ax.bar(names, accs, color=color, edgecolor="#222", linewidth=1)
    for b, a, t in zip(bars, accs, totals):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02,
                f"{a*100:.1f}%\n(n={t})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("准确率")
    overall = float(sub[pred_col].astype(str).eq(sub["gt_label"]).mean())
    ax.set_title(f"{title_prefix}（总体 {overall*100:.1f}%, N={len(sub)}）")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=20)
    save_fig(fig, out_name)
    return overall


# ============================================================
# 图 4+5: 手势准确率 + 混淆矩阵
# ============================================================
def plot_gesture_metrics(df: pd.DataFrame):
    if df.empty or "gt_label" not in df.columns:
        return
    sub = df[df["gt_label"].astype(str).isin(GESTURE_LABELS)].copy()
    if sub.empty:
        print("[SKIP] 未找到手势标签数据")
        return
    sub["pred"] = sub.apply(_pick_predicted_gesture, axis=1)
    overall = _plot_accuracy_bar(
        sub, GESTURE_LABELS, "pred",
        title_prefix="手势识别准确率",
        color="#FFB74D",
        out_name="fig_gesture_accuracy_bar.png",
    )
    if overall is None:
        return
    _plot_confusion(
        sub["gt_label"].astype(str).tolist(),
        sub["pred"].astype(str).tolist(),
        GESTURE_LABELS, "手势识别混淆矩阵",
        out_name="fig_gesture_confusion.png",
    )


# ============================================================
# 图 6+7: 动作准确率 + 混淆矩阵
# ============================================================
def plot_action_metrics(df: pd.DataFrame):
    if df.empty or "gt_label" not in df.columns or "action" not in df.columns:
        return
    sub = df[df["gt_label"].astype(str).isin(ACTION_LABELS)].copy()
    if sub.empty:
        print("[SKIP] 未找到动作标签数据")
        return
    sub["pred"] = sub["action"].astype(str)
    overall = _plot_accuracy_bar(
        sub, ACTION_LABELS, "pred",
        title_prefix="动作识别准确率",
        color="#9575CD",
        out_name="fig_action_accuracy_bar.png",
    )
    if overall is None:
        return
    _plot_confusion(
        sub["gt_label"].astype(str).tolist(),
        sub["pred"].astype(str).tolist(),
        ACTION_LABELS, "动作识别混淆矩阵",
        out_name="fig_action_confusion.png",
    )


# ============================================================
# 图 8: 距离对识别的影响
# ============================================================
def plot_distance_vs_accuracy(df: pd.DataFrame):
    if df.empty or "gt_distance_m" not in df.columns:
        return
    sub = df.copy()
    sub["gt_distance_m"] = pd.to_numeric(sub["gt_distance_m"], errors="coerce")
    sub = sub.dropna(subset=["gt_distance_m"])
    if sub.empty:
        print("[SKIP] 未找到距离实验数据")
        return

    # 检测成功 = 测出的人体距离与 gt 之差 < 0.5m，且 action 非空
    sub["measured"] = pd.to_numeric(
        sub.get("person_distance_m", np.nan), errors="coerce"
    )
    sub["ok_detect"] = (
        sub["measured"].notna()
        & (np.abs(sub["measured"] - sub["gt_distance_m"]) < 0.5)
        & sub["action"].astype(str).str.len().gt(0)
    )

    grp = (sub.groupby(sub["gt_distance_m"].round(2))
              .agg(success_rate=("ok_detect", "mean"),
                   count=("ok_detect", "size"),
                   measured_mean=("measured", "mean"))
              .reset_index())
    if grp.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左：成功率
    axes[0].plot(grp["gt_distance_m"], grp["success_rate"], "o-",
                 color="#26A69A", linewidth=2, markersize=8)
    for _, row in grp.iterrows():
        axes[0].text(row["gt_distance_m"], row["success_rate"] + 0.04,
                     f"{row['success_rate']*100:.0f}%\n(n={int(row['count'])})",
                     ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.20)
    axes[0].set_xlabel("目标距离 (m)")
    axes[0].set_ylabel("人体检测成功率")
    axes[0].set_title("距离对识别的影响")
    axes[0].grid(True, alpha=0.3)

    # 右：测量值 vs 真值
    axes[1].plot([grp["gt_distance_m"].min(), grp["gt_distance_m"].max()],
                 [grp["gt_distance_m"].min(), grp["gt_distance_m"].max()],
                 "k--", alpha=0.4, label="理想 y=x")
    axes[1].plot(grp["gt_distance_m"], grp["measured_mean"], "s-",
                 color="#EF6C00", linewidth=2, markersize=8, label="实测均值")
    for _, row in grp.iterrows():
        if pd.notna(row["measured_mean"]):
            axes[1].text(row["gt_distance_m"], row["measured_mean"] + 0.05,
                         f"{row['measured_mean']:.2f}",
                         ha="center", fontsize=9)
    axes[1].set_xlabel("目标距离 (m)")
    axes[1].set_ylabel("测量距离 (m)")
    axes[1].set_title("距离测量准确性")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_fig(fig, "fig_distance_vs_accuracy.png")


# ============================================================
# 主流程
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="*", default=None,
                   help="指定 CSV 文件路径（不指定则自动扫描）")
    p.add_argument("--no-logs", action="store_true",
                   help="跳过普通运行日志")
    p.add_argument("--no-labeled", action="store_true",
                   help="跳过带标签实验数据")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_fig_dir()

    # ---------- 收集 CSV ----------
    log_paths: List[str] = []
    labeled_paths: List[str] = []
    if args.logs:
        # 用户指定的，按路径里是否含 'labeled' 自动分类
        for p in args.logs:
            if not os.path.isfile(p):
                print(f"[WARN] 文件不存在: {p}")
                continue
            if "labeled" in p.replace("\\", "/").lower():
                labeled_paths.append(p)
            else:
                log_paths.append(p)
    else:
        if not args.no_logs:
            log_paths = sorted(glob.glob(os.path.join(LOGS_DIR, "*.csv")))
        if not args.no_labeled:
            labeled_paths = sorted(glob.glob(os.path.join(LABELED_DIR, "*.csv")))

    print(f"[INFO] 普通日志 {len(log_paths)} 个 | 带标签 {len(labeled_paths)} 个")
    print(f"[INFO] 输出目录: {FIG_DIR}")

    if not log_paths and not labeled_paths:
        print("\n[ERROR] 没找到任何 CSV！请先：")
        print(f"  - 跑 main.py（自动写入 {LOGS_DIR}）")
        print(f"  - 或跑 experiments/collect_labeled.py（写入 {LABELED_DIR}）")
        return

    df_logs = load_csvs(log_paths, label="普通日志")
    df_labeled = load_csvs(labeled_paths, label="带标签")
    df_all = pd.concat([df_logs, df_labeled], ignore_index=True)
    if df_all.empty:
        print("[ERROR] CSV 内容为空")
        return

    # ---------- 通用图（用全部数据） ----------
    print("\n--- 通用图（不需要标签） ---")
    plot_fps_histogram(df_all)
    plot_distance_histogram(df_all)
    plot_angle_smoothness(df_all)

    # ---------- 带标签图 ----------
    if not df_labeled.empty:
        print("\n--- 带标签图 ---")
        plot_gesture_metrics(df_labeled)
        plot_action_metrics(df_labeled)
        plot_distance_vs_accuracy(df_labeled)
    else:
        print("\n[INFO] 无带标签数据，跳过准确率/混淆矩阵/距离实验图")
        print("       做完 experiments/collect_labeled.py 后重新跑本脚本即可")

    print(f"\n[完成] 全部图保存在 {FIG_DIR}")


if __name__ == "__main__":
    main()
