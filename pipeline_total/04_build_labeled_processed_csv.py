"""
01_preprocess.py - Unified GNSS Data Preprocessing Pipeline

This script consolidates all preprocessing logic into a single self-contained module.
No external dependencies on Addition/gnss_plotter or ConstellationFormer.

Usage:
    # Full pipeline: TXT -> Features -> Labeled CSV
    python pipeline/01_preprocess.py --mode full --config configs/preprocessing.yml
    
    # Only parse TXT files (for debugging)
    python pipeline/01_preprocess.py --mode parse --input data_raw/dy_L1
    
    # Generate CSV only (skip plotting)
    python pipeline/01_preprocess.py --mode csv --config configs/preprocessing.yml
    
    # Plot features for a specific folder (for labeling)
    python pipeline/01_preprocess.py --mode plot --input data_raw/dy_L1/2022.07.08semicircle
"""

import argparse
import logging
import re
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# =============================================================================
# CONSTANTS
# =============================================================================
LIGHT_SPEED = 299792458.0
CONSTELLATION_MAP = {0: 'Un', 1: 'G', 2: 'S', 3: 'R', 4: 'J', 5: 'C', 6: 'E', 7: 'I'}

DEFAULT_DEVICE_MAP = {
    "NOH-AN01": "HUAWEI_Mate40",
    "Google Pixel Watch": "Google_Pixel_Watch1",
    "Google Pixel Watch 2": "Google_Pixel_Watch2",
    "23078RKD5C": "RedMi_K60",
    "MI 8": "XiaoMi_MI8",
    "Pixel 6": "Google_Pixel6",
}

FEATURE_COLS = [
    'Cn0DbHz', 'Cn0DbHz_dt', 'Cn0DbHz_std',
    'AgcDb', 'ReceivedSvTimeUncertaintyNanos',
    'PseudorangeRateUncertaintyMetersPerSecond',
    'AccumulatedDeltaRangeUncertaintyMeters',
]

# =============================================================================
# PARSER MODULE
# =============================================================================
# Column mappings based on number of columns in Raw data line
COLUMN_MAP = {
    38: ["ReadingType", "utcTimeMillis", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", "BiasUncertaintyNanos", "DriftNanosPerSecond", "DriftUncertaintyNanosPerSecond", "HardwareClockDiscontinuityCount", "Svid", "TimeOffsetNanos", "State", "ReceivedSvTimeNanos", "ReceivedSvTimeUncertaintyNanos", "Cn0DbHz", "PseudorangeRateMetersPerSecond", "PseudorangeRateUncertaintyMetersPerSecond", "AccumulatedDeltaRangeState", "AccumulatedDeltaRangeMeters", "AccumulatedDeltaRangeUncertaintyMeters", "CarrierFrequencyHz", "CarrierCycles", "CarrierPhase", "CarrierPhaseUncertainty", "MultipathIndicator", "SnrInDb", "ConstellationType", "AgcDb", "BasebandCn0DbHz", "FullInterSignalBiasNanos", "FullInterSignalBiasUncertaintyNanos", "SatelliteInterSignalBiasNanos", "SatelliteInterSignalBiasUncertaintyNanos", "CodeType", "ChipsetElapsedRealtimeNanos", "IsFullTracking"],
    37: ["ReadingType", "utcTimeMillis", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", "BiasUncertaintyNanos", "DriftNanosPerSecond", "DriftUncertaintyNanosPerSecond", "HardwareClockDiscontinuityCount", "Svid", "TimeOffsetNanos", "State", "ReceivedSvTimeNanos", "ReceivedSvTimeUncertaintyNanos", "Cn0DbHz", "PseudorangeRateMetersPerSecond", "PseudorangeRateUncertaintyMetersPerSecond", "AccumulatedDeltaRangeState", "AccumulatedDeltaRangeMeters", "AccumulatedDeltaRangeUncertaintyMeters", "CarrierFrequencyHz", "CarrierCycles", "CarrierPhase", "CarrierPhaseUncertainty", "MultipathIndicator", "SnrInDb", "ConstellationType", "AgcDb", "BasebandCn0DbHz", "FullInterSignalBiasNanos", "FullInterSignalBiasUncertaintyNanos", "SatelliteInterSignalBiasNanos", "SatelliteInterSignalBiasUncertaintyNanos", "CodeType", "ChipsetElapsedRealtimeNanos"],
    36: ["ReadingType", "utcTimeMillis", "TimeNanos", "LeapSecond", "TimeUncertaintyNanos", "FullBiasNanos", "BiasNanos", "BiasUncertaintyNanos", "DriftNanosPerSecond", "DriftUncertaintyNanosPerSecond", "HardwareClockDiscontinuityCount", "Svid", "TimeOffsetNanos", "State", "ReceivedSvTimeNanos", "ReceivedSvTimeUncertaintyNanos", "Cn0DbHz", "PseudorangeRateMetersPerSecond", "PseudorangeRateUncertaintyMetersPerSecond", "AccumulatedDeltaRangeState", "AccumulatedDeltaRangeMeters", "AccumulatedDeltaRangeUncertaintyMeters", "CarrierFrequencyHz", "CarrierCycles", "CarrierPhase", "CarrierPhaseUncertainty", "MultipathIndicator", "SnrInDb", "ConstellationType", "AgcDb", "BasebandCn0DbHz", "FullInterSignalBiasNanos", "FullInterSignalBiasUncertaintyNanos", "SatelliteInterSignalBiasNanos", "SatelliteInterSignalBiasUncertaintyNanos", "CodeType"]
}

def parse_gnss_log(file_path, device_map=None):
    """
    Parse GNSS log file (TXT format) into DataFrame.
    Uses column count mapping for robustness.
    Returns: (df, device_name)
    """
    if device_map is None:
        device_map = DEFAULT_DEVICE_MAP
    
    file_path = Path(file_path)
    if not file_path.exists():
        return pd.DataFrame(), None
    
    raw_data_lines = []
    device_model = None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('#'):
                    # Extract device model
                    if 'Model:' in line and device_model is None:
                        line_suffix = line.split('Model:')[1].strip()
                        # Match longest key first
                        sorted_keys = sorted(device_map.keys(), key=len, reverse=True)
                        for key in sorted_keys:
                            if key in line_suffix:
                                device_model = key
                                break
                elif line.startswith('Raw,'):
                    # Data line
                    raw_data_lines.append(line.strip().split(','))
    except Exception as e:
        logging.warning(f"Error reading {file_path}: {e}")
        return pd.DataFrame(), None
    
    if not raw_data_lines:
        return pd.DataFrame(), device_map.get(device_model, device_model)
    
    # Determine column mapping based on first line length
    num_columns = len(raw_data_lines[0])
    
    if num_columns not in COLUMN_MAP:
        logging.warning(f"Unknown column count {num_columns} in {file_path.name}")
        return pd.DataFrame(), device_map.get(device_model, device_model)
    
    columns = COLUMN_MAP[num_columns]
    
    # Filter lines with correct column count
    processed_lines = [line for line in raw_data_lines if len(line) == num_columns]
    
    if not processed_lines:
        return pd.DataFrame(), device_map.get(device_model, device_model)
    
    df = pd.DataFrame(processed_lines, columns=columns)
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ['ReadingType', 'CodeType']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing essential values
    df.dropna(subset=['TimeNanos', 'Svid', 'ConstellationType', 'ReceivedSvTimeNanos'], inplace=True)
    
    device_name = device_map.get(device_model, device_model)
    return df, device_name


# =============================================================================
# FEATURE ENGINEERING MODULE
# =============================================================================
def calculate_derived_features(df):
    """Calculate TOW, Pseudorange, FreqBand, sv_id."""
    df['rx_time_sec'] = (df['TimeNanos'] + df['TimeOffsetNanos'] - (df['FullBiasNanos'] + df['BiasNanos'])) * 1e-9
    df['tx_time_sec'] = df['ReceivedSvTimeNanos'] * 1e-9

    df['TOW'] = np.nan
    df['Pseudorange_Calculated'] = np.nan

    # GPS, QZSS, Galileo, SBAS
    is_gps_like = df['ConstellationType'].isin([1, 2, 4, 6])
    df.loc[is_gps_like, 'rx_time_sec_mod'] = df.loc[is_gps_like, 'rx_time_sec'] % 604800
    df.loc[is_gps_like, 'tx_time_sec_mod'] = df.loc[is_gps_like, 'tx_time_sec'] % 604800
    df.loc[is_gps_like, 'TOW'] = np.floor(df.loc[is_gps_like, 'tx_time_sec_mod'] + 0.5)

    # BDS
    is_bds = df['ConstellationType'] == 5
    df.loc[is_bds, 'rx_time_sec_mod'] = (df.loc[is_bds, 'rx_time_sec'] % 604800) - 14
    df.loc[is_bds, 'tx_time_sec_mod'] = df.loc[is_bds, 'tx_time_sec'] % 604800
    df.loc[is_bds, 'TOW'] = np.floor(df.loc[is_bds, 'tx_time_sec_mod'] + 0.5 + 14)
    
    # GLONASS
    is_glonass = df['ConstellationType'] == 3
    df.loc[is_glonass, 'rx_time_sec_mod'] = (df.loc[is_glonass, 'rx_time_sec'] % 86400) + 3 * 3600 - 18
    df.loc[is_glonass, 'tx_time_sec_mod'] = df.loc[is_glonass, 'tx_time_sec'] % 86400
    df.loc[is_glonass, 'TOW'] = np.floor(df.loc[is_glonass, 'tx_time_sec_mod'] + 0.5 + 18)

    df['delta_time'] = df['rx_time_sec_mod'] - df['tx_time_sec_mod']
    df['Pseudorange_Calculated'] = df['delta_time'] * LIGHT_SPEED
    
    # Frequency Band
    df['FreqBand'] = 0
    df.loc[df['CarrierFrequencyHz'] > 1500e6, 'FreqBand'] = 1
    df.loc[df['CarrierFrequencyHz'] < 1300e6, 'FreqBand'] = 5
    
    # Satellite ID
    df['prn'] = df['Svid']
    is_qzss = df['ConstellationType'] == 4
    df.loc[is_qzss, 'prn'] = df.loc[is_qzss, 'prn'] - 192
    df['sv_id'] = df['ConstellationType'].map(CONSTELLATION_MAP).fillna('Un') + df['prn'].astype(int).astype(str).str.zfill(2)
    df.drop(columns='prn', inplace=True, errors='ignore')

    return df


def filter_bad_data(df):
    """Filter low-quality observations."""
    df = df[df['ReceivedSvTimeNanos'] > 1e10].copy()
    df = df[df['ReceivedSvTimeUncertaintyNanos'] <= 500]
    df = df[df['Cn0DbHz'] >= 10]
    df = df[df['PseudorangeRateUncertaintyMetersPerSecond'] <= 20]
    df = df[df['AccumulatedDeltaRangeUncertaintyMeters'] <= 5]
    df = df[~((df['ConstellationType'] == 3) & (df['Svid'] > 35))]
    df = df[df['FreqBand'].isin([1, 5])]
    df.dropna(subset=['TOW'], inplace=True)
    df = df[df['TOW'] > 24 * 3600]
    return df


def calculate_advanced_features(df):
    """Calculate the 7 core features for spoofing detection."""
    WINDOW_SIZE = 5
    df = df.sort_values(by=['sv_id', 'TOW']).copy()
    
    # 1. Cn0DbHz_dt - C/N0 derivative
    df['Cn0DbHz_dt'] = df.groupby('sv_id')['Cn0DbHz'].diff().fillna(0)
    
    # 2. Cn0DbHz_std - Rolling standard deviation of C/N0
    df['Cn0DbHz_std'] = df.groupby('sv_id')['Cn0DbHz'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE, min_periods=2).std()
    ).fillna(0)
    
    # 3-5. Raw uncertainty features (already in data, just ensure they exist)
    # AgcDb, ReceivedSvTimeUncertaintyNanos are parsed from raw
    # PseudorangeRateUncertaintyMetersPerSecond, AccumulatedDeltaRangeUncertaintyMeters
    
    # Ensure uncertainty columns have sensible defaults
    if 'PseudorangeRateUncertaintyMetersPerSecond' not in df.columns:
        df['PseudorangeRateUncertaintyMetersPerSecond'] = 0.0
    if 'AccumulatedDeltaRangeUncertaintyMeters' not in df.columns:
        df['AccumulatedDeltaRangeUncertaintyMeters'] = 0.0
    
    return df


# =============================================================================
# LABELING MODULE
# =============================================================================
def add_spoofing_labels(df, spoofing_type, config):
    """Add spoofing labels based on TOW intervals. Binary classification: 0=authentic, 1=spoofing."""
    labeling_config = config.get('labeling', {})
    label_value = labeling_config.get('spoofing_type_to_label', {}).get(spoofing_type, 0)
    tow_intervals = labeling_config.get('spoofing_tow_intervals', {}).get(spoofing_type, [])
    
    df['Label'] = 0
    
    if spoofing_type != 'normal' and tow_intervals:
        for start_tow, end_tow in tow_intervals:
            time_mask = (df['TOW'] >= start_tow) & (df['TOW'] <= end_tow)
            
            if label_value == 1:  # L1 spoofing
                final_mask = time_mask & (df['FreqBand'] == 1)
            elif label_value == 2:  # L5 spoofing
                final_mask = time_mask & (df['FreqBand'] == 5)
            elif label_value == 3:  # Dual-band
                final_mask = time_mask
            else:
                continue
            
            # Binary: all spoofing types -> 1
            df.loc[final_mask, 'Label'] = 1
    
    return df


def get_spoofing_type_from_path(file_path, known_types):
    """Infer spoofing type from file path."""
    for part in Path(file_path).parts:
        if part in known_types:
            return part
    return "normal"


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def process_single_file(file_path, spoofing_type, config):
    """Full processing pipeline for a single TXT file."""
    df, device_name = parse_gnss_log(file_path, config.get('device_model_map', DEFAULT_DEVICE_MAP))
    
    if df.empty:
        return pd.DataFrame(), device_name
    
    df = calculate_derived_features(df)
    df = filter_bad_data(df)
    
    if df.empty:
        return pd.DataFrame(), device_name
    
    df = calculate_advanced_features(df)
    df = add_spoofing_labels(df, spoofing_type, config)
    df['DeviceName'] = device_name
    df['SpoofingType'] = spoofing_type
    
    return df, device_name


def run_full_pipeline(config):
    """Run the complete preprocessing pipeline."""
    logging.info("=" * 60)
    logging.info("GNSS Preprocessing Pipeline - Full Mode")
    logging.info("=" * 60)
    
    input_dir = Path(config['paths']['input_dir'])
    output_path = Path(config['paths']['output_csv'])
    
    # Find all TXT files
    file_patterns = config.get('file_patterns', ['gnss_log_*.txt', 'log_mimir_*.txt'])
    all_files = []
    for pattern in file_patterns:
        all_files.extend(input_dir.rglob(pattern))
    
    logging.info(f"Found {len(all_files)} TXT files in {input_dir}")
    
    if not all_files:
        logging.error("No files found!")
        return
    
    known_types = list(config.get('labeling', {}).get('spoofing_type_to_label', {}).keys())
    
    all_dfs = []
    for file_path in tqdm(all_files, desc="Processing files"):
        spoofing_type = get_spoofing_type_from_path(file_path, known_types)
        df, device_name = process_single_file(file_path, spoofing_type, config)
        
        if not df.empty:
            df['SourceFile'] = file_path.name
            all_dfs.append(df)
    
    if not all_dfs:
        logging.error("No data processed!")
        return
    
    logging.info("Combining all files...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Select final columns
    final_columns = config.get('final_columns', [
        'TimeNanos', 'TOW', 'sv_id', 'DeviceName', 'Label', 'SpoofingType', 'FreqBand'
    ] + FEATURE_COLS)
    
    available_cols = [c for c in final_columns if c in final_df.columns]
    final_df = final_df[available_cols]
    
    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    logging.info(f"✅ CSV saved to: {output_path}")
    logging.info(f"   Rows: {len(final_df):,}")
    logging.info(f"   Columns: {list(final_df.columns)}")
    
    # Statistics
    if 'SpoofingType' in final_df.columns:
        logging.info("\nSpoofingType distribution:")
        for st, count in final_df['SpoofingType'].value_counts().items():
            logging.info(f"  {st}: {count:,}")
    
    if 'Label' in final_df.columns:
        logging.info("\nLabel distribution:")
        for label, count in final_df['Label'].value_counts().items():
            logging.info(f"  {label}: {count:,}")


def run_plot_mode(input_path, config):
    """Plot features for a specific folder (for labeling)."""
    logging.info(f"Plotting features for: {input_path}")
    
    input_dir = Path(input_path)
    txt_files = list(input_dir.rglob("gnss_log_*.txt")) + list(input_dir.rglob("log_mimir_*.txt"))
    
    if not txt_files:
        logging.warning("No TXT files found!")
        return
    
    # Process all files
    all_dfs = []
    for txt_file in tqdm(txt_files, desc="Parsing"):
        df, device_name = parse_gnss_log(txt_file, config.get('device_model_map', DEFAULT_DEVICE_MAP))
        if not df.empty:
            df = calculate_derived_features(df)
            df = filter_bad_data(df)
            df = calculate_advanced_features(df)
            df['Label'] = 0  # Unknown for labeling
            all_dfs.append(df)
    
    if not all_dfs:
        logging.error("No data!")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Total rows: {len(combined_df)}")
    
    # Plot Cn0DbHz and AgcDb
    for feature in ['Cn0DbHz', 'AgcDb']:
        if feature not in combined_df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for sv_id, group in combined_df.groupby('sv_id'):
            ax.plot(group['TOW'], group[feature], label=sv_id, alpha=0.7, linewidth=0.8)
        
        ax.set_xlabel('TOW (s)')
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} - {input_dir.name}')
        ax.legend(loc='upper right', ncol=4, fontsize=6)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='GNSS Preprocessing Pipeline')
    parser.add_argument('--mode', choices=['full', 'parse', 'csv', 'plot'], default='full',
                        help='Execution mode: full (all steps), parse (debug), csv (generate CSV), plot (visualize)')
    parser.add_argument('--config', type=str, default='configs/preprocessing.yml',
                        help='Path to configuration YAML')
    parser.add_argument('--input', type=str, help='Input directory (for parse/plot modes)')
    parser.add_argument('--output', type=str, help='Output path (overrides config)')
    args = parser.parse_args()
    
    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded config from: {config_path}")
    elif args.mode in ['full', 'csv']:
        logging.error(f"Config file not found: {config_path}")
        return
    
    # Override paths from CLI
    if args.input:
        config.setdefault('paths', {})['input_dir'] = args.input
    if args.output:
        config.setdefault('paths', {})['output_csv'] = args.output
    
    # Execute mode
    if args.mode == 'full' or args.mode == 'csv':
        run_full_pipeline(config)
    elif args.mode == 'plot':
        if not args.input:
            logging.error("--input required for plot mode")
            return
        run_plot_mode(args.input, config)
    elif args.mode == 'parse':
        logging.info("Parse mode: testing individual file parsing")
        if args.input:
            df, device = parse_gnss_log(args.input, config.get('device_model_map'))
            logging.info(f"Parsed {len(df)} rows, device: {device}")
            if not df.empty:
                logging.info(f"Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
