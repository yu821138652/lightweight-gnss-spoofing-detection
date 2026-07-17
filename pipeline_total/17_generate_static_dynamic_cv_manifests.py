"""在静态多环境 Session-CV 清单中，仅向训练集加入固定的动态 Session。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RECORDING_COLUMNS = ["Environment", "Scenario", "Session"]


def load_dynamic_train_sessions(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, encoding="utf-8-sig")
    required = {*RECORDING_COLUMNS, "split", "is_dynamic"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"动态来源清单缺少字段：{sorted(missing)}")
    dynamic_train = source.loc[
        (source["is_dynamic"].astype(str).str.lower() == "true")
        & (source["split"].astype(str) == "train"),
        RECORDING_COLUMNS,
    ].drop_duplicates()
    if dynamic_train.empty:
        raise ValueError("动态来源清单中没有 split=train 的动态 Session。")
    dynamic_train = dynamic_train.copy()
    dynamic_train["split"] = "train"
    return dynamic_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-cv-dir",
        type=Path,
        default=Path("docs/protocols/static_session_cv_4fold"),
        help="包含 fold_<n>/recording_split_manifest.csv 的静态多环境 CV 目录。",
    )
    parser.add_argument(
        "--dynamic-source-manifest",
        type=Path,
        default=Path("output/tensors_mixed/recording_split_manifest.csv"),
        help="含 is_dynamic 与 split 字段的既有混合场景录制清单。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/protocols/static_dynamic_train_cv_4fold"),
    )
    args = parser.parse_args()
    if not args.static_cv_dir.exists():
        raise FileNotFoundError(args.static_cv_dir)
    if not args.dynamic_source_manifest.exists():
        raise FileNotFoundError(args.dynamic_source_manifest)

    dynamic_train = load_dynamic_train_sessions(args.dynamic_source_manifest)
    fold_paths = sorted(args.static_cv_dir.glob("fold_*/recording_split_manifest.csv"))
    if not fold_paths:
        raise ValueError(f"未在 {args.static_cv_dir} 找到静态 CV fold 清单。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_train.to_csv(args.output_dir / "dynamic_train_sessions.csv", index=False, encoding="utf-8-sig")
    for source_path in fold_paths:
        static_manifest = pd.read_csv(source_path, encoding="utf-8-sig")
        missing = set(RECORDING_COLUMNS + ["split"]).difference(static_manifest.columns)
        if missing:
            raise ValueError(f"{source_path} 缺少字段：{sorted(missing)}")
        static_manifest = static_manifest[RECORDING_COLUMNS + ["split"]].copy()
        combined = pd.concat([static_manifest, dynamic_train], ignore_index=True)
        if combined.duplicated(RECORDING_COLUMNS).any():
            raise ValueError(f"{source_path} 与动态训练 Session 存在重复录制。")
        fold_dir = args.output_dir / source_path.parent.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(fold_dir / "recording_split_manifest.csv", index=False, encoding="utf-8-sig")

    print(
        "已生成静态+动态训练的多环境 CV 清单：\n"
        f"  folds={len(fold_paths)}\n"
        f"  每折额外动态训练 Session={len(dynamic_train)}\n"
        f"  动态清单={args.output_dir / 'dynamic_train_sessions.csv'}\n"
        f"  fold 清单={args.output_dir / 'fold_<n>' / 'recording_split_manifest.csv'}"
    )


if __name__ == "__main__":
    main()
