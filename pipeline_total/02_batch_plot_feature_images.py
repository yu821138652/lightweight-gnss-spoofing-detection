"""
批量生成特征时序图
为每个 CSV 文件的每个特征生成独立的时序图

规范:
- X轴: TOW (秒)
- 每颗卫星一条线
- 欺骗区间灰色阴影
- jet colormap
- 每张图下方显示图例
- 数据点间距>10秒不连线
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 150,
})

# 特征列表
FEATURES = [
    ('Cn0DbHz', 'C/N₀ (dB-Hz)'),
    ('Cn0DbHz_dt', 'C/N₀ Change Rate'),
    ('Cn0DbHz_std', 'C/N₀ Std Dev'),
    ('AgcDb', 'AGC (dB)'),
    ('ReceivedSvTimeUncertaintyNanos', 'Time Uncertainty (ns)'),
    ('PseudorangeRateUncertaintyMetersPerSecond', 'PR Rate Uncertainty (m/s)'),
    ('AccumulatedDeltaRangeUncertaintyMeters', 'ADR Uncertainty (m)'),
]

# 全局参数
MAX_GAP_SECONDS = 10

# 欺骗区间
SPOOFING_INTERVALS = {
    'st_L1': [[262228, 262860]],
    'st_L5': [[266310, 267054]],
    'st_L_15': [[258028, 258653]],
    'dy_L1': [[263995, 264050], [264690, 264740], [265015, 265065], 
              [481730, 481940], [482350, 482420], [482650, 482750]],
    'dy_L5': [[269190, 269220], [268960, 268990], [269434, 269470], [483200, 483320]],
    'dy_L_15': [[261285, 261310], [260025, 260050], [260970, 261040]],
}


def plot_with_gap_handling(ax, x, y, color, alpha=0.6, linewidth=0.5, label=None):
    """绘制时序线，超过 MAX_GAP_SECONDS 的间隔断开"""
    if len(x) < 2:
        return
    
    x = np.array(x)
    y = np.array(y)
    
    gaps = np.diff(x) > MAX_GAP_SECONDS
    gap_indices = np.where(gaps)[0] + 1
    segments = np.split(np.arange(len(x)), gap_indices)
    
    for i, seg in enumerate(segments):
        if len(seg) > 1:
            lbl = label if i == 0 else None
            ax.plot(x[seg], y[seg], color=color, alpha=alpha, 
                   linewidth=linewidth, label=lbl)


def create_feature_plot(df, feature_col, feature_label, device_name, 
                        spoofing_intervals, output_path):
    """为单个特征创建时序图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    identity_column = 'SignalID' if 'SignalID' in df.columns else 'SatelliteID'
    identities = sorted(df[identity_column].dropna().unique())
    
    # jet colormap
    cmap = plt.cm.jet
    colors = {identity: cmap(i / max(1, len(identities) - 1))
             for i, identity in enumerate(identities)}

    # Draw each independent signal when available; fall back to satellites for
    # legacy plot-feature CSV files.
    for identity in identities:
        sort_columns = ['TOW'] + (['TimeNanos'] if 'TimeNanos' in df.columns else [])
        identity_data = df[df[identity_column] == identity].sort_values(sort_columns)
        if feature_col in identity_data.columns and len(identity_data) > 0:
            plot_with_gap_handling(
                ax, identity_data['TOW'].values, identity_data[feature_col].values,
                color=colors[identity], alpha=0.7, linewidth=0.6, label=identity
            )
    
    # 标记欺骗区间
    tow_min, tow_max = df['TOW'].min(), df['TOW'].max()
    for interval in spoofing_intervals:
        if interval[0] <= tow_max and interval[1] >= tow_min:
            ax.axvspan(interval[0], interval[1], alpha=0.2, color='gray')
            ax.axvline(interval[0], color='black', linestyle='--', linewidth=0.8)
            ax.axvline(interval[1], color='black', linestyle='--', linewidth=0.8)
    
    ax.set_title(f'{feature_label} on {device_name}')
    ax.set_xlabel('TOW (s)')
    ax.set_ylabel(feature_label)
    
    # 图例放在图下方
    ax.legend(fontsize=6, ncol=10, loc='upper center', 
             bbox_to_anchor=(0.5, -0.1), framealpha=0.9, title='Signal ID / Satellite ID')
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()


def process_scenario(scenario: str, input_base: str = 'data_raw', 
                     output_base: str = 'output_plots'):
    """处理一个场景下的所有 CSV 文件"""
    input_path = Path(input_base) / scenario
    output_path = Path(output_base) / scenario
    
    if not input_path.exists():
        print(f"Scenario not found: {input_path}")
        return
    
    # 获取欺骗区间
    spoof_intervals = SPOOFING_INTERVALS.get(scenario, [])
    
    # 找到所有 CSV 文件
    csv_files = sorted(input_path.rglob("*-plot_features.csv"))
    csv_count_by_parent = {}
    for csv_file in csv_files:
        csv_count_by_parent[csv_file.parent] = csv_count_by_parent.get(csv_file.parent, 0) + 1
    print(f"\n{scenario}: Found {len(csv_files)} CSV files")
    
    total_plots = len(csv_files) * len(FEATURES)
    pbar = tqdm(total=total_plots, desc=f"Plotting {scenario}")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # 获取设备名
            if 'DeviceName' in df.columns:
                device_name = df['DeviceName'].iloc[0] if len(df) > 0 else 'Unknown'
            else:
                device_name = csv_file.parent.name
            
            # 计算相对路径
            rel_path = csv_file.relative_to(input_path)
            out_dir = output_path / rel_path.parent
            # Some playground device folders contain multiple source logs.
            # Keep their plots separate instead of overwriting the same seven
            # feature filenames in the shared device directory.
            if csv_count_by_parent[csv_file.parent] > 1:
                out_dir = out_dir / csv_file.stem
            
            # 为每个特征生成图
            for feat_col, feat_label in FEATURES:
                if feat_col not in df.columns:
                    pbar.update(1)
                    continue
                
                out_file = out_dir / f"{feat_col}.png"
                create_feature_plot(df, feat_col, feat_label, device_name,
                                   spoof_intervals, out_file)
                pbar.update(1)
                
        except Exception as e:
            print(f"\nError processing {csv_file}: {e}")
            pbar.update(len(FEATURES))
    
    pbar.close()
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch plot GNSS feature time-series images.')
    parser.add_argument('--input-base', default='data_raw', help='Directory containing scenario folders.')
    parser.add_argument('--output-base', default='output_plots', help='Directory for generated PNG files.')
    parser.add_argument('--scenario', default=None, help='Only plot one scenario, e.g. st_L1.')
    args = parser.parse_args()

    print("=" * 60)
    print("Batch Feature Time-Series Plotting")
    print("=" * 60)
    print(f"Features: {len(FEATURES)}")
    print(f"Max gap (断线阈值): {MAX_GAP_SECONDS}s")
    print(f"Colormap: jet")
    
    scenarios = ['st_L1', 'st_L5', 'st_L_15', 'dy_L1', 'dy_L5', 'dy_L_15']
    if args.scenario:
        scenarios = [args.scenario]

    for scenario in scenarios:
        process_scenario(scenario, input_base=args.input_base, output_base=args.output_base)
    
    print("\n" + "=" * 60)
    print("All plots generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
