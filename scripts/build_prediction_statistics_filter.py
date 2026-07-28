#!/usr/bin/env python3
"""Build a self-contained browser for aggregated signal prediction statistics.

The source CSV may contain millions of endpoint predictions.  This builder
removes timestamps, paths, and probabilities, then aggregates the data to the
finest review unit requested by the project:

    fold x environment x scenario x session x device x signal_id x outcome

The generated HTML has no external dependencies and can be opened directly
from the local filesystem.  All filtering and regrouping happens on the
aggregated counts embedded in the page.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_EXPERIMENT_ROOT = Path(
    "output/training/"
    "mixed_timeblock_outer_cv4_w5_compact11_tcn16_d10_"
    "state_stratified_interval_all_positive_v1"
)
DEFAULT_INPUT = (
    DEFAULT_EXPERIMENT_ROOT
    / "cv_test_predictions_detailed_signal_tcn_stats_mlp_fusion.csv"
)
DEFAULT_OUTPUT = DEFAULT_EXPERIMENT_ROOT / "prediction_statistics_filter.html"

REQUIRED_COLUMNS = {
    "Fold",
    "ErrorType",
    "Environment",
    "Scenario",
    "Session",
    "DeviceName",
    "signal_id",
    "Label",
    "Prediction",
}
OUTCOME_FROM_LABELS = {
    (0, 0): "TN",
    (0, 1): "FP",
    (1, 0): "FN",
    (1, 1): "TP",
}
OUTCOMES = ("TP", "TN", "FP", "FN")
AGGREGATE_COLUMNS = (
    "Fold",
    "Environment",
    "Motion",
    "SpoofingType",
    "Scenario",
    "Session",
    "DeviceName",
    "Constellation",
    "Satellite",
    "Band",
    "SignalBand",
    "CodeType",
    "signal_id",
    "ErrorType",
)

CONSTELLATION_NAMES = {
    "G": "GPS",
    "C": "BDS",
    "E": "GAL",
    "J": "QZSS",
    "R": "GLO",
    "S": "SBAS",
    "I": "NAVIC",
}
L5_SIGNAL_PREFIXES = (
    "GPS_L5",
    "SBAS_L5",
    "QZSS_L5",
    "NAVIC_L5",
    "GLO_G3",
    "BDS_B2",
    "GAL_E5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_INPUT,
        help="Combined endpoint prediction CSV with a Fold column.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Self-contained HTML output path.",
    )
    parser.add_argument(
        "--title",
        default="GNSS 全量预测统计筛选器",
        help="Page title shown in the generated tool.",
    )
    return parser.parse_args()


def scenario_metadata(scenario: str) -> Tuple[str, str]:
    if scenario.startswith("dy_"):
        motion = "dynamic"
        suffix = scenario[3:]
    elif scenario.startswith("st_"):
        motion = "static"
        suffix = scenario[3:]
    else:
        motion = "unknown"
        suffix = scenario
    spoofing_type = "L1+L5" if suffix == "L_15" else suffix
    return motion, spoofing_type


def signal_metadata(signal_id: str) -> Tuple[str, str, str, str, str]:
    parts = signal_id.split("|")
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"Invalid signal_id: {signal_id!r}")
    satellite, signal_band, code_type = parts
    constellation = CONSTELLATION_NAMES.get(satellite[0], satellite[0])
    band = "L5" if signal_band.startswith(L5_SIGNAL_PREFIXES) else "L1"
    return constellation, satellite, band, signal_band, code_type


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_predictions(
    path: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            scenario = row["Scenario"]
            motion, spoofing_type = scenario_metadata(scenario)
            constellation, satellite, band, signal_band, code_type = signal_metadata(
                row["signal_id"]
            )
            try:
                label = int(row["Label"])
                prediction = int(row["Prediction"])
            except ValueError as exc:
                raise ValueError(f"Invalid label/prediction at CSV line {line_number}") from exc
            expected_outcome = OUTCOME_FROM_LABELS.get((label, prediction))
            if expected_outcome is None or row["ErrorType"] != expected_outcome:
                raise ValueError(
                    f"Outcome mismatch at CSV line {line_number}: "
                    f"label={label}, prediction={prediction}, ErrorType={row['ErrorType']!r}"
                )

            key = (
                row["Fold"],
                row["Environment"],
                motion,
                spoofing_type,
                scenario,
                row["Session"],
                row["DeviceName"],
                constellation,
                satellite,
                band,
                signal_band,
                code_type,
                row["signal_id"],
                expected_outcome,
            )
            counts[key] += 1
            outcome_counts[expected_outcome] += 1
            total += 1

    records: List[Dict[str, object]] = []
    for key, count in sorted(counts.items(), key=lambda item: item[0]):
        record: Dict[str, object] = dict(zip(AGGREGATE_COLUMNS, key))
        record["Count"] = int(count)
        records.append(record)

    metadata: Dict[str, object] = {
        "source": str(path),
        "sourceSha256": sha256(path),
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "totalEndpoints": total,
        "aggregatedRows": len(records),
        "outcomes": {name: int(outcome_counts[name]) for name in OUTCOMES},
        "unit": "one active signal_id at one valid W5 endpoint",
        "labelSemantics": "reviewed_interval_all_positive",
    }
    if sum(metadata["outcomes"].values()) != total:  # type: ignore[union-attr]
        raise AssertionError("Outcome counts do not sum to the endpoint total")
    return records, metadata


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #dbe3ef;
      --blue: #2563eb;
      --blue-soft: #eaf1ff;
      --green: #15803d;
      --green-soft: #eaf8ef;
      --red: #c2413b;
      --red-soft: #fff0ef;
      --amber: #a16207;
      --shadow: 0 8px 24px rgba(15, 23, 42, .07);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
    }
    .app { max-width: 1760px; margin: 0 auto; padding: 24px; }
    .header {
      display: flex; align-items: flex-start; justify-content: space-between;
      gap: 20px; margin-bottom: 18px;
    }
    h1 { margin: 0 0 7px; font-size: 24px; letter-spacing: -.02em; }
    .subtitle { color: var(--muted); font-size: 13px; line-height: 1.65; }
    .status {
      flex: none; padding: 8px 12px; border: 1px solid #bcd0ff;
      color: #1d4ed8; background: var(--blue-soft); border-radius: 999px;
      font-size: 12px; font-weight: 700;
    }
    .panel {
      background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
      box-shadow: var(--shadow); padding: 16px; margin-bottom: 16px;
    }
    .panel-title {
      margin: 0 0 12px; font-size: 15px; display: flex;
      align-items: center; justify-content: space-between; gap: 12px;
    }
    .filters {
      display: grid; grid-template-columns: repeat(6, minmax(145px, 1fr)); gap: 12px;
    }
    label { display: block; color: #475467; font-size: 12px; margin-bottom: 5px; }
    select, input, button {
      width: 100%; min-height: 37px; border: 1px solid #cfd8e6; border-radius: 8px;
      background: #fff; color: var(--ink); font: inherit; font-size: 13px;
    }
    select, input { padding: 7px 9px; }
    select:focus, input:focus { outline: 2px solid #bfd3ff; border-color: var(--blue); }
    .toolbar {
      display: grid; grid-template-columns: minmax(240px, 1.5fr) minmax(210px, 1fr)
        minmax(180px, 1fr) auto auto; gap: 10px; align-items: end; margin-top: 14px;
    }
    button { width: auto; padding: 7px 14px; cursor: pointer; font-weight: 650; }
    button:hover { border-color: #94a3b8; background: #f8fafc; }
    .primary { color: white; border-color: var(--blue); background: var(--blue); }
    .primary:hover { background: #1d4ed8; }
    .kpis {
      display: grid; grid-template-columns: repeat(8, minmax(120px, 1fr)); gap: 10px;
    }
    .kpi { border: 1px solid var(--line); border-radius: 11px; padding: 12px; background: #fbfdff; }
    .kpi-label { color: var(--muted); font-size: 11px; margin-bottom: 6px; }
    .kpi-value { font-weight: 780; font-size: 19px; font-variant-numeric: tabular-nums; }
    .kpi-note { color: var(--muted); font-size: 10px; margin-top: 4px; }
    .outcomes { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .chip { border-radius: 999px; padding: 5px 10px; font-size: 12px; font-weight: 700; }
    .tp, .tn { color: var(--green); background: var(--green-soft); }
    .fp, .fn { color: var(--red); background: var(--red-soft); }
    .table-wrap { border: 1px solid var(--line); border-radius: 10px; overflow: auto; max-height: 65vh; }
    table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 12px; white-space: nowrap; }
    th {
      position: sticky; top: 0; z-index: 2; background: #edf3fb; color: #344054;
      text-align: left; padding: 9px 10px; border-bottom: 1px solid #c9d5e5;
    }
    td { padding: 8px 10px; border-bottom: 1px solid #edf1f6; }
    tbody tr:nth-child(even) { background: #fbfcfe; }
    tbody tr:hover { background: #eef5ff; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    .pager { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 12px; }
    .pager-actions { display: flex; align-items: center; gap: 8px; }
    .pager-actions button { min-width: 78px; }
    .muted { color: var(--muted); font-size: 12px; }
    .empty { padding: 38px; text-align: center; color: var(--muted); }
    @media (max-width: 1200px) {
      .filters { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
      .kpis { grid-template-columns: repeat(4, 1fr); }
      .toolbar { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 700px) {
      .app { padding: 12px; }
      .header { display: block; }
      .status { display: inline-block; margin-top: 10px; }
      .filters, .kpis, .toolbar { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
<main class="app">
  <header class="header">
    <div>
      <h1>__TITLE__</h1>
      <div class="subtitle">
        统计单位：每个有效 W5 endpoint 上的一条 <code>signal_id</code> 观测。<br>
        页面不含时间、文件路径和置信度；所有数字均由锁定四折全量预测预聚合得到。
      </div>
    </div>
    <div class="status" id="status">正在载入…</div>
  </header>

  <section class="panel">
    <h2 class="panel-title"><span>筛选条件</span><span class="muted">每项可选“全部”</span></h2>
    <div class="filters" id="filters"></div>
    <div class="toolbar">
      <div>
        <label for="groupPreset">表格汇总方式</label>
        <select id="groupPreset"></select>
      </div>
      <div>
        <label for="sortMode">排序</label>
        <select id="sortMode">
          <option value="count_desc">样本数：从多到少</option>
          <option value="error_desc">错误率：从高到低</option>
          <option value="far_desc">FAR：从高到低</option>
          <option value="recall_asc">Recall：从低到高</option>
          <option value="name_asc">名称：升序</option>
        </select>
      </div>
      <div>
        <label for="pageSize">每页行数</label>
        <select id="pageSize">
          <option>50</option><option selected>100</option><option>200</option><option>500</option>
        </select>
      </div>
      <button id="reset">清空筛选</button>
      <button class="primary" id="export">导出当前表 CSV</button>
    </div>
  </section>

  <section class="panel">
    <h2 class="panel-title"><span>当前筛选统计</span><span class="muted" id="selectionText"></span></h2>
    <div class="kpis" id="kpis"></div>
    <div class="outcomes" id="outcomes"></div>
  </section>

  <section class="panel">
    <h2 class="panel-title"><span>统计表</span><span class="muted" id="rowSummary"></span></h2>
    <div class="table-wrap"><table id="table"></table></div>
    <div class="pager">
      <div class="muted" id="pageText"></div>
      <div class="pager-actions"><button id="prev">上一页</button><button id="next">下一页</button></div>
    </div>
  </section>
</main>

<script>
const DATA = __DATA__;
const META = __META__;

const LABELS = {
  Fold: "Fold", Environment: "大场景", Motion: "静/动态", SpoofingType: "欺骗类型",
  Scenario: "完整场景", Session: "Session", DeviceName: "设备名称",
  Constellation: "星座", Satellite: "卫星", Band: "L1/L5频段",
  SignalBand: "具体信号频段", CodeType: "码型", signal_id: "signal_id",
  ErrorType: "TP/TN/FP/FN"
};
const VALUE_LABELS = {
  Environment: {new_building: "新主楼", playground: "操场"},
  Motion: {static: "静态", dynamic: "动态", unknown: "未知"}
};
const FILTER_FIELDS = [
  "Environment", "Motion", "SpoofingType", "Scenario", "Session", "DeviceName",
  "Constellation", "Satellite", "Band", "SignalBand", "CodeType", "signal_id",
  "ErrorType", "Fold"
];
const PRESETS = {
  "大场景总览": ["Environment"],
  "大场景 × 静动态": ["Environment", "Motion"],
  "欺骗类型": ["Motion", "SpoofingType"],
  "完整场景": ["Environment", "Scenario"],
  "场景 × 设备": ["Environment", "Scenario", "DeviceName"],
  "Session × 设备": ["Environment", "Scenario", "Session", "DeviceName"],
  "设备 × 频段": ["DeviceName", "Band"],
  "Signal 统计": ["Constellation", "Satellite", "SignalBand", "CodeType", "signal_id"],
  "TP/TN/FP/FN": ["ErrorType"],
  "最细粒度": ["Environment", "Motion", "SpoofingType", "Scenario", "Session", "DeviceName", "signal_id", "ErrorType"]
};
const state = { filters: {}, grouped: [], page: 1 };
const number = new Intl.NumberFormat("zh-CN");
const percent = new Intl.NumberFormat("zh-CN", {style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2});

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}
function displayValue(field, value) { return VALUE_LABELS[field]?.[value] ?? value; }
function ratio(a, b) { return b ? a / b : null; }
function fmtPercent(value) { return value == null || !Number.isFinite(value) ? "—" : percent.format(value); }
function outcomeMetrics(counts) {
  const tp = counts.TP || 0, tn = counts.TN || 0, fp = counts.FP || 0, fn = counts.FN || 0;
  const total = tp + tn + fp + fn;
  const posF1 = ratio(2 * tp, 2 * tp + fp + fn);
  const negF1 = ratio(2 * tn, 2 * tn + fp + fn);
  return {
    total, tp, tn, fp, fn,
    accuracy: ratio(tp + tn, total), errorRate: ratio(fp + fn, total),
    precision: ratio(tp, tp + fp), recall: ratio(tp, tp + fn),
    far: ratio(fp, fp + tn), macroF1: posF1 == null || negF1 == null ? null : (posF1 + negF1) / 2
  };
}
function uniqueValues(field) {
  return [...new Set(DATA.map(row => row[field]))].sort((a, b) => String(a).localeCompare(String(b), "zh-CN", {numeric: true}));
}
function buildControls() {
  const filters = document.getElementById("filters");
  FILTER_FIELDS.forEach(field => {
    const box = document.createElement("div");
    const label = document.createElement("label");
    label.textContent = LABELS[field];
    const select = document.createElement("select");
    select.id = `filter_${field}`;
    select.innerHTML = `<option value="">全部</option>` + uniqueValues(field).map(value => `<option value="${esc(value)}">${esc(displayValue(field, value))}</option>`).join("");
    select.addEventListener("change", () => { state.filters[field] = select.value; state.page = 1; render(); });
    box.append(label, select); filters.appendChild(box);
  });
  const preset = document.getElementById("groupPreset");
  preset.innerHTML = Object.keys(PRESETS).map((name, i) => `<option${i === 4 ? " selected" : ""}>${esc(name)}</option>`).join("");
  preset.addEventListener("change", () => { state.page = 1; render(); });
  document.getElementById("sortMode").addEventListener("change", () => { state.page = 1; render(); });
  document.getElementById("pageSize").addEventListener("change", () => { state.page = 1; renderTable(); });
  document.getElementById("reset").addEventListener("click", () => {
    state.filters = {}; state.page = 1;
    FILTER_FIELDS.forEach(field => document.getElementById(`filter_${field}`).value = "");
    render();
  });
  document.getElementById("prev").addEventListener("click", () => { if (state.page > 1) { state.page--; renderTable(); }});
  document.getElementById("next").addEventListener("click", () => {
    const size = Number(document.getElementById("pageSize").value);
    if (state.page * size < state.grouped.length) { state.page++; renderTable(); }
  });
  document.getElementById("export").addEventListener("click", exportCsv);
}
function filteredRows() {
  return DATA.filter(row => FILTER_FIELDS.every(field => !state.filters[field] || String(row[field]) === state.filters[field]));
}
function summarize(rows) {
  const counts = {TP: 0, TN: 0, FP: 0, FN: 0};
  const sessions = new Set(), devices = new Set(), signals = new Set();
  rows.forEach(row => {
    counts[row.ErrorType] += row.Count;
    sessions.add(`${row.Environment}|${row.Scenario}|${row.Session}`);
    devices.add(row.DeviceName); signals.add(row.signal_id);
  });
  return {...outcomeMetrics(counts), sessionCount: sessions.size, deviceCount: devices.size, signalCount: signals.size};
}
function groupRows(rows, fields) {
  const map = new Map();
  rows.forEach(row => {
    const key = JSON.stringify(fields.map(field => row[field]));
    if (!map.has(key)) map.set(key, {values: fields.map(field => row[field]), counts: {TP:0,TN:0,FP:0,FN:0}});
    map.get(key).counts[row.ErrorType] += row.Count;
  });
  const result = [...map.values()].map(item => ({values: item.values, ...outcomeMetrics(item.counts)}));
  const mode = document.getElementById("sortMode").value;
  result.sort((a, b) => {
    if (mode === "error_desc") return (b.errorRate ?? -1) - (a.errorRate ?? -1) || b.total - a.total;
    if (mode === "far_desc") return (b.far ?? -1) - (a.far ?? -1) || b.total - a.total;
    if (mode === "recall_asc") return (a.recall ?? 2) - (b.recall ?? 2) || b.total - a.total;
    if (mode === "name_asc") return JSON.stringify(a.values).localeCompare(JSON.stringify(b.values), "zh-CN", {numeric:true});
    return b.total - a.total;
  });
  return result;
}
function renderKpis(summary) {
  const cards = [
    ["筛选后样本", number.format(summary.total), `占全部 ${fmtPercent(ratio(summary.total, META.totalEndpoints))}`],
    ["正确率", fmtPercent(summary.accuracy), `${number.format(summary.tp + summary.tn)} 条正确`],
    ["Macro-F1", fmtPercent(summary.macroF1), "按当前筛选重新计算"],
    ["Precision", fmtPercent(summary.precision), "TP / (TP + FP)"],
    ["Recall", fmtPercent(summary.recall), "TP / (TP + FN)"],
    ["FAR", fmtPercent(summary.far), "FP / (FP + TN)"],
    ["Session 数", number.format(summary.sessionCount), `${number.format(summary.deviceCount)} 种设备`],
    ["signal_id 数", number.format(summary.signalCount), "当前筛选下去重"]
  ];
  document.getElementById("kpis").innerHTML = cards.map(card => `<div class="kpi"><div class="kpi-label">${card[0]}</div><div class="kpi-value">${card[1]}</div><div class="kpi-note">${card[2]}</div></div>`).join("");
  document.getElementById("outcomes").innerHTML = ["TP","TN","FP","FN"].map(name => `<span class="chip ${name.toLowerCase()}">${name}：${number.format(summary[name.toLowerCase()])}</span>`).join("");
  const active = FILTER_FIELDS.filter(field => state.filters[field]).map(field => `${LABELS[field]}=${displayValue(field, state.filters[field])}`);
  document.getElementById("selectionText").textContent = active.length ? active.join("；") : "未设置筛选条件";
}
function renderTable() {
  const fields = PRESETS[document.getElementById("groupPreset").value];
  const table = document.getElementById("table");
  const size = Number(document.getElementById("pageSize").value);
  const pages = Math.max(1, Math.ceil(state.grouped.length / size));
  state.page = Math.min(state.page, pages);
  const start = (state.page - 1) * size;
  const rows = state.grouped.slice(start, start + size);
  const metricHeaders = ["样本数", "筛选内占比", "全体占比", "TP", "TN", "FP", "FN", "错误率", "Recall", "FAR"];
  table.innerHTML = `<thead><tr>${fields.map(field => `<th>${LABELS[field]}</th>`).join("")}${metricHeaders.map((name, i) => `<th class="num">${name}</th>`).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  if (!rows.length) {
    body.innerHTML = `<tr><td class="empty" colspan="${fields.length + metricHeaders.length}">没有符合条件的数据</td></tr>`;
  } else {
    body.innerHTML = rows.map(row => `<tr>${row.values.map((value, index) => `<td>${esc(displayValue(fields[index], value))}</td>`).join("")}<td class="num">${number.format(row.total)}</td><td class="num">${fmtPercent(ratio(row.total, currentSummary.total))}</td><td class="num">${fmtPercent(ratio(row.total, META.totalEndpoints))}</td><td class="num">${number.format(row.tp)}</td><td class="num">${number.format(row.tn)}</td><td class="num">${number.format(row.fp)}</td><td class="num">${number.format(row.fn)}</td><td class="num">${fmtPercent(row.errorRate)}</td><td class="num">${fmtPercent(row.recall)}</td><td class="num">${fmtPercent(row.far)}</td></tr>`).join("");
  }
  table.appendChild(body);
  document.getElementById("rowSummary").textContent = `${number.format(state.grouped.length)} 个汇总组合`;
  document.getElementById("pageText").textContent = `第 ${state.page} / ${pages} 页；显示 ${rows.length} 行`;
  document.getElementById("prev").disabled = state.page <= 1;
  document.getElementById("next").disabled = state.page >= pages;
}
let currentSummary = null;
function render() {
  const rows = filteredRows();
  currentSummary = summarize(rows);
  const fields = PRESETS[document.getElementById("groupPreset").value];
  state.grouped = groupRows(rows, fields);
  renderKpis(currentSummary); renderTable();
}
function csvCell(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
function exportCsv() {
  const fields = PRESETS[document.getElementById("groupPreset").value];
  const headers = [...fields.map(field => LABELS[field]), "样本数", "筛选内占比", "全体占比", "TP", "TN", "FP", "FN", "错误率", "Precision", "Recall", "FAR", "Macro-F1"];
  const lines = [headers.map(csvCell).join(",")];
  state.grouped.forEach(row => {
    const values = [...row.values, row.total, ratio(row.total, currentSummary.total), ratio(row.total, META.totalEndpoints), row.tp, row.tn, row.fp, row.fn, row.errorRate, row.precision, row.recall, row.far, row.macroF1];
    lines.push(values.map(value => csvCell(value == null ? "" : value)).join(","));
  });
  const blob = new Blob(["\ufeff" + lines.join("\r\n")], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a"); link.href = url;
  link.download = `prediction_statistics_${new Date().toISOString().slice(0,10)}.csv`;
  document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
}

buildControls();
document.getElementById("status").textContent = `${number.format(META.totalEndpoints)} 条 endpoint → ${number.format(META.aggregatedRows)} 条聚合记录`;
render();
</script>
</body>
</html>
'''


def build_html(
    records: Sequence[Dict[str, object]], metadata: Dict[str, object], title: str
) -> str:
    data_json = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    metadata_json = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__DATA__", data_json)
        .replace("__META__", metadata_json)
    )


def main() -> None:
    args = parse_args()
    records, metadata = aggregate_predictions(args.predictions)
    html = build_html(records, metadata, args.title)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output_html),
                "bytes": args.output_html.stat().st_size,
                **metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
