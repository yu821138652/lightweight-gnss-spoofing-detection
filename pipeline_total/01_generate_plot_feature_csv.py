#!/usr/bin/env python3
"""
generate_plot_features.py - 为论文 Section IV 生成绘图用 CSV

功能:
    1. 批量处理 data_raw 目录下的所有 TXT 文件
    2. 为每个 TXT 生成对应的 *-plot_features.csv
    3. 输出包含 7 个核心特征和标签

用法:
    python scripts/generate_plot_features.py
    
    # 只处理特定场景
    python scripts/generate_plot_features.py --scenario st_L1
    
    # 覆盖已存在的 CSV
    python scripts/generate_plot_features.py --overwrite

作者: AI Assistant
日期: 2026-01-02
"""

import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from the self-contained preprocessing module.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "preprocess", project_root / "pipeline_total" / "04_build_labeled_processed_csv.py"
)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

process_single_file = preprocess.process_single_file
get_spoofing_type_from_path = preprocess.get_spoofing_type_from_path

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# =============================================================================
# CONFIG
# =============================================================================
DATA_RAW_DIR = project_root / "data_raw"
CONFIG_FILE = project_root / "configs" / "preprocessing.yml"

# 论文中使用的设备名映射 (更友好的显示名)
PAPER_DEVICE_MAP = {
    "XiaoMi_MI8": "Xiaomi Mi8",
    "RedMi_K60": "Redmi K60U",
    "Google_Pixel6": "Google Pixel6",
    "HUAWEI_Mate40": "Huawei Mate40",
    "Google_Pixel_Watch1": "Pixel Watch1",
    "Google_Pixel_Watch2": "Pixel Watch2",
}

# 输出列
OUTPUT_COLUMNS = [
    'Environment',
    'Scenario',
    'Session',
    'utcTimeMillis',
    'TimeNanos',
    'TOW',
    'SatelliteID',  # Will be mapped from sv_id
    'SignalID',
    'ConstellationType',
    'Svid',
    'CarrierFrequencyHz',
    'CodeType',
    'SignalBand',
    'FreqBand',
    'Cn0DbHz',
    'Cn0DbHz_dt',
    'Cn0DbHz_std',
    'AgcDb',
    'ReceivedSvTimeUncertaintyNanos',
    'PseudorangeRateUncertaintyMetersPerSecond',
    'AccumulatedDeltaRangeUncertaintyMeters',
    'Label',
    'LabelStatus',
    'DeviceName',
]


def resolve_config_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    return path


def process_single_txt(txt_path, config, data_root, overwrite=False):
    """
    处理单个 TXT 文件，生成对应的 -plot_features.csv
    
    Returns:
        (success: bool, rows: int, output_path: Path or None)
    """
    # 输出路径
    output_path = txt_path.with_suffix('').with_name(txt_path.stem + '-plot_features.csv')
    
    # 检查是否已存在
    if output_path.exists() and not overwrite:
        logging.debug(f"Skipping (exists): {output_path.name}")
        return True, 0, output_path
    
    # 1. 解析、过滤、计算特征并按当前配置打临时标签
    known_types = list(config.get('labeling', {}).get('spoofing_type_to_label', {}).keys())
    spoofing_type = get_spoofing_type_from_path(txt_path, known_types)
    df, _ = process_single_file(txt_path, spoofing_type, config, data_root=data_root)

    if df.empty:
        logging.warning(f"No valid data after parsing/filtering: {txt_path.name}")
        return False, 0, None

    # 2. 重命名列
    df['SatelliteID'] = df['sv_id']
    df['SignalID'] = df['signal_id']

    # 3. 设备名映射 (使用论文友好名称)
    df['DeviceName'] = df['DeviceName'].map(lambda x: PAPER_DEVICE_MAP.get(x, x))
    
    # 4. 选择输出列
    output_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    output_df = df[output_cols].copy()
    
    # 5. 保存 CSV
    output_df.to_csv(output_path, index=False)
    
    return True, len(output_df), output_path


def main():
    parser = argparse.ArgumentParser(description='Generate plot features CSV for Section IV')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Only process specific scenario (e.g., st_L1, dy_L5)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing CSV files')
    parser.add_argument('--config', type=str, default=str(CONFIG_FILE),
                        help='Path to preprocessing config')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root path. Defaults to paths.input_dir in config')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N files, useful for smoke tests')
    args = parser.parse_args()
    
    # 加载配置
    if Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded config from: {args.config}")
    else:
        logging.error(f"Config not found: {args.config}")
        return
    
    # 扫描 TXT 文件
    logging.info("=" * 60)
    logging.info("Generate Plot Features CSV")
    logging.info("=" * 60)
    
    data_root = resolve_config_path(args.data_root or config.get('paths', {}).get('input_dir', DATA_RAW_DIR))
    logging.info(f"Data root: {data_root}")

    file_patterns = config.get('file_patterns', ["gnss_log_*.txt", "log_mimir_*.txt"])
    all_txt_files = []
    seen = set()
    for pattern in file_patterns:
        for txt_path in data_root.rglob(pattern):
            if txt_path in seen:
                continue
            if args.scenario and args.scenario not in txt_path.parts:
                continue
            seen.add(txt_path)
            all_txt_files.append(txt_path)

    all_txt_files = sorted(all_txt_files, key=lambda p: str(p).lower())
    if args.limit is not None:
        all_txt_files = all_txt_files[:args.limit]

    if args.scenario:
        logging.info(f"Scenario filter: {args.scenario}")
    
    logging.info(f"Total: {len(all_txt_files)} TXT files")
    
    if not all_txt_files:
        logging.error("No TXT files found!")
        return
    
    # 处理每个文件
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_rows = 0
    
    for txt_path in tqdm(all_txt_files, desc="Processing"):
        success, rows, output_path = process_single_txt(txt_path, config, data_root, args.overwrite)
        
        if success:
            if rows > 0:
                success_count += 1
                total_rows += rows
            else:
                skip_count += 1
        else:
            fail_count += 1
    
    # 统计
    logging.info("=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  ✅ Generated: {success_count} CSV files")
    logging.info(f"  ⏭️  Skipped:   {skip_count} (already exist)")
    logging.info(f"  ❌ Failed:    {fail_count}")
    logging.info(f"  📊 Total rows: {total_rows:,}")
    
    if fail_count > 0:
        logging.warning("Some files failed to process. Check logs above for details.")
    
    logging.info("=" * 60)
    logging.info("✅ Done! CSV files saved next to original TXT files.")


if __name__ == "__main__":
    main()
