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

# Import from pipeline module (01_preprocess.py)
# Python doesn't allow module names starting with numbers, so we use importlib
import importlib.util
spec = importlib.util.spec_from_file_location("preprocess", project_root / "pipeline" / "01_preprocess.py")
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

parse_gnss_log = preprocess.parse_gnss_log
calculate_derived_features = preprocess.calculate_derived_features
filter_bad_data = preprocess.filter_bad_data
calculate_advanced_features = preprocess.calculate_advanced_features
add_spoofing_labels = preprocess.add_spoofing_labels
get_spoofing_type_from_path = preprocess.get_spoofing_type_from_path
DEFAULT_DEVICE_MAP = preprocess.DEFAULT_DEVICE_MAP
FEATURE_COLS = preprocess.FEATURE_COLS

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
    'TOW',
    'SatelliteID',  # Will be mapped from sv_id
    'FreqBand',
    'Cn0DbHz',
    'Cn0DbHz_dt',
    'Cn0DbHz_std',
    'AgcDb',
    'ReceivedSvTimeUncertaintyNanos',
    'PseudorangeRateUncertaintyMetersPerSecond',
    'AccumulatedDeltaRangeUncertaintyMeters',
    'Label',
    'DeviceName',
]


def process_single_txt(txt_path, config, overwrite=False):
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
    
    # 1. 解析 TXT
    df, device_name = parse_gnss_log(txt_path, DEFAULT_DEVICE_MAP)
    
    if df.empty:
        logging.warning(f"Empty file: {txt_path.name}")
        return False, 0, None
    
    # 2. 计算派生特征
    df = calculate_derived_features(df)
    
    # 3. 过滤低质量数据
    df = filter_bad_data(df)
    
    if df.empty:
        logging.warning(f"No valid data after filtering: {txt_path.name}")
        return False, 0, None
    
    # 4. 计算高级特征
    df = calculate_advanced_features(df)
    
    # 5. 添加欺骗标签
    known_types = list(config.get('labeling', {}).get('spoofing_type_to_label', {}).keys())
    spoofing_type = get_spoofing_type_from_path(txt_path, known_types)
    df = add_spoofing_labels(df, spoofing_type, config)
    
    # 6. 重命名列
    df['SatelliteID'] = df['sv_id']
    
    # 7. 设备名映射 (使用论文友好名称)
    paper_device_name = PAPER_DEVICE_MAP.get(device_name, device_name)
    df['DeviceName'] = paper_device_name
    
    # 8. 选择输出列
    output_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    output_df = df[output_cols].copy()
    
    # 9. 保存 CSV
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
    
    all_txt_files = []
    scenarios = ['st_L1', 'st_L5', 'st_L_15', 'dy_L1', 'dy_L5', 'dy_L_15']
    
    if args.scenario:
        if args.scenario not in scenarios:
            logging.error(f"Unknown scenario: {args.scenario}")
            return
        scenarios = [args.scenario]
    
    for scenario in scenarios:
        scenario_dir = DATA_RAW_DIR / scenario
        if not scenario_dir.exists():
            logging.warning(f"Scenario dir not found: {scenario_dir}")
            continue
        
        # 查找 TXT 文件
        txt_files = list(scenario_dir.rglob("gnss_log_*.txt")) + list(scenario_dir.rglob("log_mimir_*.txt"))
        all_txt_files.extend(txt_files)
        logging.info(f"  {scenario}: {len(txt_files)} TXT files")
    
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
        success, rows, output_path = process_single_txt(txt_path, config, args.overwrite)
        
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
