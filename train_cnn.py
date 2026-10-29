"""1:1 Paper-Style CNN with Anti-Overfitting & Realistic Generalization."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = [
    "膝部拉伸-上山行走",
    "膝部拉伸-正常行走",
    "膝部拉伸-下山行走",
    "呼吸湿度-上山急促呼吸",
    "呼吸湿度-下山状态",
    "呼吸湿度-正常行走",
]
LABELS = list(range(len(CLASS_NAMES)))

DATA_MAPPING = [
    {"label": 0, "sheet": "上山行走", "signal_col": 1, "record_id": "uphill_walk"},
    {"label": 1, "sheet": "正常行走", "signal_col": 1, "record_id": "normal_walk_exp1"},
    {"label": 1, "sheet": "正常行走", "signal_col": 2, "record_id": "normal_walk_exp2"},
    {"label": 2, "sheet": "下山行走", "signal_col": 1, "record_id": "downhill_walk"},
    {"label": 3, "sheet": "上山呼吸", "signal_col": 1, "record_id": "uphill_breath"},
    {"label": 4, "sheet": "下山呼吸", "signal_col": 1, "record_id": "downhill_breath"},
    {"label": 5, "sheet": "正常呼吸", "signal_col": 1, "record_id": "normal_breath"},
]


@dataclass
class Config:
    window_length: int = 64
    stride: int = 48           # 增大步长，大幅降低窗口重叠，杜绝死记硬背
    trend_window: int = 9
    batch_size: int = 32
    epochs: int = 25           # 25轮，与参考论文完全同步
    learning_rate: float = 8e-4
    weight_decay: float = 2e-3 # 适度强化权重衰减，稳定曲线
    seed: int = 2024


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_matplotlib() -> None:
    for font_path in [
        Path(os.environ.get("WINDIR", r"C:\\Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", r"C:\\Windows")) / "Fonts" / "simhei.ttf",
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_records(xlsx_path: Path) -> List[dict]:
    workbook = pd.ExcelFile(xlsx_path)
    records = []
    for item in DATA_MAPPING:
        frame = pd.read_excel(xlsx_path, sheet_name=item["sheet"])
        records.append({
            **item,
            "time": frame.iloc[:, 0].copy(),
            "signal": frame.iloc[:, item["signal_col"]].copy(),
        })
    return records


def clean_record(record: dict) -> dict:
    time = pd.to_numeric(record["time"], errors="coerce").to_numpy(dtype=float)
    signal = pd.to_numeric(record["signal"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(signal)
    first, last = int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1])
    time, signal = time[first : last + 1], signal[first : last + 1]
    order = np.argsort(time, kind="stable")
    return {**record, "segments": [(time[order], signal[order])]}


def make_features(time: np.ndarray, signal: np.ndarray, trend_window: int) -> np.ndarray:
    dt = np.diff(time)
    positive = dt[(dt > 0) & np.isfinite(dt)]
    default_dt = float(np.median(positive)) if len(positive) else 1.0
    safe_time = time.copy()
    for i in range(1, len(safe_time)):
        if not np.isfinite(safe_time[i] - safe_time[i - 1]) or safe_time[i] <= safe_time[i - 1]:
            safe_time[i] = safe_time[i - 1] + default_dt
            
    derivative = np.gradient(signal, safe_time) if len(signal) > 2 else np.diff(signal, prepend=signal[0]) / default_dt
    trend = pd.Series(signal).rolling(trend_window, center=True, min_periods=1).mean().to_numpy()
    return np.column_stack([signal, derivative, trend]).astype(np.float32)


def make_windows_split(records: Sequence[dict], config: Config):
    all_x = []
    all_y = []
    
    for rec in records:
        for time, signal in rec["segments"]:
            features = make_features(time, signal, config.trend_window)
            for start in range(0, len(features) - config.window_length + 1, config.stride):
                window = features[start : start + config.window_length]
                all_x.append(window)
                all_y.append(rec["label"])
                
    X = np.asarray(all_x, dtype=np.float32)
    Y = np.asarray(all_y, dtype=np.int64)
    
    # 75% 训练集, 25% 测试集
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=config.seed, stratify=Y
    )
    
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    return {"train": x_train, "test": x_test}, {"train": y_train, "test": y_test}


class MultiChannelCNN(nn.Module):
    def __init__(self, channels: int = 3, classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout1d(0.15),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout1d(0.20),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), 
            nn.Dropout(0.35),
            nn.Linear(128, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(np.transpose(x, (0, 2, 1)))
    tensor_y = torch.from_numpy(y)
    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=shuffle)


def train_and_eval(x, y, config: Config, device):
    model = MultiChannelCNN().to(device)
    # 核心：引入标签平滑，消除狂妄过拟合，使概率分布平滑真实
    criterion = nn.CrossEntropyLoss(label_smoothing=0.06)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    train_dl = loader(x["train"], y["train"], config.batch_size, True)
    test_dl = loader(x["test"], y["test"], config.batch_size, False)
    
    history = {"test_accuracy": [], "loss": []}
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            # 训练期微弱数据扰动（模拟真实传感器抖动）
            noise = torch.randn_like(xb) * 0.015
            xb = xb + noise
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            
        epoch_loss = total_loss / len(y["train"])
        
        # 测试集评估
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                logits = model(xb.to(device))
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
                trues.extend(yb.numpy().tolist())
        test_acc = accuracy_score(trues, preds)
        
        history["loss"].append(epoch_loss)
        history["test_accuracy"].append(test_acc * 100)
        print(f"Epoch {epoch:02d}/{config.epochs} Loss={epoch_loss:.4f} Test_Acc={test_acc:.2%}")
        
    return model, history, np.asarray(trues), np.asarray(preds)


def plot_paper_curves(history: dict, out_path: Path):
    """1:1 绘制参考论文图(c)的双Y轴曲线图"""
    epochs = np.arange(1, len(history["loss"]) + 1)
    
    fig, ax1 = plt.subplots(figsize=(6, 5))
    color_acc = "#e41a1c"
    color_loss = "#000000"
    
    ax1.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Test accuracy (%)", color=color_acc, fontsize=13, fontweight="bold")
    ax1.plot(epochs, history["test_accuracy"], color=color_acc, marker="s", markersize=4, label="Test accuracy", lw=1.5)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(40, 105)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("LOSS", color=color_loss, fontsize=13, fontweight="bold")
    ax2.plot(epochs, history["loss"], color=color_loss, marker="o", markersize=4, label="Loss", lw=1.5)
    ax2.tick_params(axis="y", labelcolor=color_loss)
    ax2.set_ylim(0, max(history["loss"]) * 1.25)
    
    ax1.set_xticks(np.arange(0, len(epochs) + 1, 4))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_paper_confusion(cm: np.ndarray, out_path: Path, accuracy: float) -> None:
    """1:1 绘制参考论文图(d)风格的混淆矩阵"""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import PercentFormatter

    row_sum = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
    
    cmap = LinearSegmentedColormap.from_list("paper_purple", ["#ffffff", "#dadaeb", "#807dba", "#6a51a3"])
    fig, ax = plt.subplots(figsize=(10, 8.5))
    heatmap = ax.imshow(percentages, cmap=cmap, vmin=0, vmax=100, interpolation="nearest")
    
    ax.set_xticks(LABELS, CLASS_NAMES, rotation=45, ha="right", rotation_mode="anchor", fontsize=11)
    ax.set_yticks(LABELS, CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted Results", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Labels", fontsize=13, fontweight="bold")
    ax.set_title(f"Model Accuracy : {accuracy:.2%}", color="#c62828", fontsize=18, fontweight="bold", pad=20)
    
    for i in LABELS:
        for j in LABELS:
            annotation = f"{percentages[i, j]:.2f}%" if row_sum[i, 0] else "0.00%"
            ax.text(j, i, annotation, ha="center", va="center", fontsize=10.5, fontweight="bold", 
                    color="white" if percentages[i, j] > 50 else "black")
                    
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.045, pad=0.04, ticks=np.arange(0, 101, 20))
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=2))
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    xlsx_path = r"F:\Desktop\实验数据\机器学习\机器学习\机器学习.xlsx"
    out_dir = Path(r"F:\Desktop\实验数据\机器学习\机器学习\results")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(); set_seed(config.seed); configure_matplotlib()
    
    print("Reading data and preparing windows...")
    records = [clean_record(r) for r in load_records(Path(xlsx_path))]
    windows, labels = make_windows_split(records, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model, history, y_true, y_pred = train_and_eval(windows, labels, config, device)
    
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    
    plot_paper_curves(history, out_dir / "paper_style_curves.png")
    plot_paper_confusion(cm, out_dir / "paper_style_confusion.png", acc)
    
    print(f"\n==========================================")
    print(f"🎉 最终测试集总准确率 TEST Accuracy = {acc:.2%}")
    print(f"✅ 双Y轴训练曲线已生成: {out_dir / 'paper_style_curves.png'}")
    print(f"✅ 论文风格混淆矩阵已生成: {out_dir / 'paper_style_confusion.png'}")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()