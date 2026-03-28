"""Generate a Chinese daily DOCX progress report from the current day's log entries."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


CURRENT_LOG_PATH = Path("Current.md")
PROCESSED_MANIFEST_PATH = Path("data") / "processed" / "galaxyppg_hr_w10_s2_median_manifest.json"
BASELINE_PLOT_DIR = Path("experiments") / "baseline_results" / "2026-03-11" / "plots"


@dataclass(slots=True)
class ProcessedArtifacts:
    manifest_path: str
    windows_path: str
    labels_path: str
    num_windows: int
    num_train_windows: int
    num_test_windows: int
    num_participants: int
    train_participants: list[str]
    test_participants: list[str]


@dataclass(slots=True)
class FeatureArtifacts:
    model_name: str
    num_windows: int | None
    embedding_dim: int | None
    target_sampling_hz: int | None
    features_path: str | None
    metadata_path: str | None
    manifest_path: str | None
    checkpoint_path: str | None
    external_repo_root: str | None


@dataclass(slots=True)
class RegressionMetrics:
    model_name: str
    regressor: str
    num_windows: int | None
    num_valid_predictions: int | None
    coverage: float | None
    mae: float | None
    rmse: float | None
    embedding_dim: int | None
    best_params: dict[str, object] | None
    train_participants: list[str]
    test_participants: list[str]
    metrics_path: str | None
    predictions_path: str | None
    estimator_path: str | None


@dataclass(slots=True)
class DailyReportData:
    report_date: str
    log_titles: list[str]
    processed_artifacts: ProcessedArtifacts
    baseline_plot_dir: str
    baseline_plots: list[str]
    pulseppg_features: FeatureArtifacts
    papagei_features: FeatureArtifacts
    pulseppg_ridge: RegressionMetrics
    papagei_ridge: RegressionMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat(), help="Report date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output DOCX path. Default: reports/<M-D>/当前进展报告_<date>.docx",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_daily_log_titles(log_path: Path, report_date: str) -> list[str]:
    heading_pattern = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})(?:\s+(.*))?$")
    titles: list[str] = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        match = heading_pattern.match(raw_line.strip())
        if not match or match.group(1) != report_date:
            continue
        suffix = (match.group(2) or "").strip()
        titles.append(suffix if suffix else "当日更新")
    if not titles:
        raise RuntimeError(f"No log entries found in {log_path.as_posix()} for {report_date}.")
    return titles


def load_processed_artifacts(manifest_path: Path) -> ProcessedArtifacts:
    payload = load_json(manifest_path)
    return ProcessedArtifacts(
        manifest_path=manifest_path.as_posix(),
        windows_path=str(payload["windows_path"]),
        labels_path=str(payload["labels_path"]),
        num_windows=int(payload["num_windows"]),
        num_train_windows=int(payload["num_train_windows"]),
        num_test_windows=int(payload["num_test_windows"]),
        num_participants=int(payload["num_participants"]),
        train_participants=list(payload["train_participants"]),
        test_participants=list(payload["test_participants"]),
    )


def load_feature_artifacts(manifest_path: Path, model_name: str) -> FeatureArtifacts:
    if not manifest_path.exists():
        return FeatureArtifacts(model_name, None, None, None, None, None, None, None, None)
    payload = load_json(manifest_path)
    return FeatureArtifacts(
        model_name=model_name,
        num_windows=payload.get("num_windows"),
        embedding_dim=payload.get("embedding_dim"),
        target_sampling_hz=payload.get("target_sampling_hz"),
        features_path=payload.get("features_path"),
        metadata_path=payload.get("metadata_path"),
        manifest_path=payload.get("manifest_path"),
        checkpoint_path=payload.get("checkpoint_path"),
        external_repo_root=payload.get("external_repo_root"),
    )


def load_regression_metrics(metrics_path: Path, model_name: str, regressor: str) -> RegressionMetrics:
    predictions_path = metrics_path.with_name(f"{model_name}_{regressor}_predictions.csv")
    estimator_path = metrics_path.with_name(f"{model_name}_{regressor}_estimator.joblib")
    if not metrics_path.exists():
        return RegressionMetrics(model_name, regressor, None, None, None, None, None, None, None, [], [], None, None, None)
    payload = load_json(metrics_path)
    return RegressionMetrics(
        model_name=model_name,
        regressor=regressor,
        num_windows=payload.get("num_windows"),
        num_valid_predictions=payload.get("num_valid_predictions"),
        coverage=payload.get("coverage"),
        mae=payload.get("mae"),
        rmse=payload.get("rmse"),
        embedding_dim=payload.get("embedding_dim"),
        best_params=payload.get("best_params"),
        train_participants=list(payload.get("train_participants", [])),
        test_participants=list(payload.get("test_participants", [])),
        metrics_path=metrics_path.as_posix(),
        predictions_path=predictions_path.as_posix() if predictions_path.exists() else None,
        estimator_path=estimator_path.as_posix() if estimator_path.exists() else None,
    )


def collect_report_data(report_date: str) -> DailyReportData:
    pulse_dir = Path("experiments") / "pulseppg_results" / report_date
    papagei_dir = Path("experiments") / "papagei_results" / report_date
    baseline_plots = sorted(path.name for path in BASELINE_PLOT_DIR.glob("*.png")) if BASELINE_PLOT_DIR.exists() else []
    return DailyReportData(
        report_date=report_date,
        log_titles=extract_daily_log_titles(CURRENT_LOG_PATH, report_date),
        processed_artifacts=load_processed_artifacts(PROCESSED_MANIFEST_PATH),
        baseline_plot_dir=BASELINE_PLOT_DIR.as_posix(),
        baseline_plots=baseline_plots,
        pulseppg_features=load_feature_artifacts(pulse_dir / "full" / "pulseppg_manifest.json", "pulseppg"),
        papagei_features=load_feature_artifacts(papagei_dir / "full" / "papagei_manifest.json", "papagei"),
        pulseppg_ridge=load_regression_metrics(pulse_dir / "regression_ridge" / "pulseppg_ridge_metrics.json", "pulseppg", "ridge"),
        papagei_ridge=load_regression_metrics(papagei_dir / "regression_ridge" / "papagei_ridge_metrics.json", "papagei", "ridge"),
    )


def build_report_blocks(data: DailyReportData, output_path: Path) -> list[dict[str, str]]:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    train_participants = "、".join(data.processed_artifacts.train_participants)
    test_participants = "、".join(data.processed_artifacts.test_participants)
    log_scope = "；".join(data.log_titles)
    baseline_plot_names = "、".join(data.baseline_plots) if data.baseline_plots else "当日未检测到图文件"
    pulse_best_params = format_params(data.pulseppg_ridge.best_params)
    papagei_best_params = format_params(data.papagei_ridge.best_params)

    return [
        {"text": "GalaxyPPG 项目当日进展报告", "style": "title"},
        {"text": f"报告日期：{data.report_date}", "style": "normal"},
        {"text": f"生成时间：{current_time}", "style": "normal"},
        {"text": f"输出位置：{output_path.as_posix()}", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "一、当日报告范围与步骤", "style": "heading"},
        {"text": f"本报告只汇总 `Current.md` 中 {data.report_date} 当日记录的进展，不回溯历史日期内容。", "style": "normal"},
        {"text": f"当日日志范围包括：{log_scope}。", "style": "normal"},
        {"text": "当日推进步骤可以分成两条主线：", "style": "normal"},
        {"text": "1. 围绕现有 GalaxyPPG 窗口数据完成处理后缓存导出与基线图表补齐，确保后续实验不再依赖重复全量重建。", "style": "normal"},
        {"text": "2. 按 `ppg-needtodo.docx` 的主线接入 PulsePPG 与 PaPaGei，完成 embedding 导出，并在 participant 级别划分上跑出首轮 Ridge 回归结果。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "二、当日已实现的流程", "style": "heading"},
        {"text": "1. 复查当前窗口结构与标签结构，确认列表型序列字段不能直接安全落成普通 CSV。", "style": "normal"},
        {"text": "2. 实现处理后数据缓存：窗口保存为 `jsonl.gz`，标签保存为 `csv`，运行配置和路径保存为 `manifest.json`。", "style": "normal"},
        {"text": "3. 补充 `src/data/export_processed.py` 导出入口，直接从真实数据构建默认 `10s / 2s / median` 数据集并保存到 `data/processed/`。", "style": "normal"},
        {"text": "4. 实现 `src/baseline/plot_baseline_results.py`，对既有 baseline 结果补出 `Prediction vs Ground Truth`、`Error Distribution`、`Bland-Altman` 三类图。", "style": "normal"},
        {"text": "5. 重新用 UTF-8 阅读中文项目说明，确认当日主线应转向 foundation model，而不是继续扩展 baseline。", "style": "normal"},
        {"text": "6. 接入 `external/pulseppg` 与 `external/papagei-foundation-model`，安装 `torch`，补齐官方 checkpoint。", "style": "normal"},
        {"text": "7. 在 `src/models/` 中实现从处理后窗口到模型输入的适配流程，包括重采样、带通、z-score、checkpoint 加载与 embedding 导出。", "style": "normal"},
        {"text": "8. 在 `src/regression/train_regressor.py` 中实现基于保存特征文件的 participant 分组训练与验证流程，并完成首轮 `Ridge` 评估。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "三、当日遇到的问题、解决方案与相关代码", "style": "heading"},
        {"text": "问题 1：处理后窗口包含 `ppg_values`、时间戳、参考 HR 和 RR 列表，直接落成 CSV 容易损坏结构。", "style": "normal"},
        {"text": "解决：新增 `src/data/cache.py` 与 `src/data/export_processed.py`，采用 `jsonl.gz + csv + manifest.json` 的三件套保存格式。", "style": "normal"},
        {"text": "相关代码：`src/data/cache.py`、`src/data/export_processed.py`、`src/data/windowing.py`。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "问题 2：以 Polar HR 为参考时，RR 列在窗口里逻辑上为空，导出后容易出现脏的空值列表。", "style": "normal"},
        {"text": "解决：在窗口导出过程中清理空 RR 值，确保 HR 参考窗口保存的是干净空列表而不是混杂的空值占位。", "style": "normal"},
        {"text": "相关代码：`src/data/windowing.py`、`src/data/cache.py`。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "问题 3：本地仓库原先只有 `PulsePPG` 和 `PaPaGei` 的占位文件，没有真实源码、权重和推理环境。", "style": "normal"},
        {"text": "解决：拉取官方仓库到 `external/`，安装 `torch`，并把官方 checkpoint 下载到本地路径后再接入项目适配层。", "style": "normal"},
        {"text": "相关代码：`src/models/common.py`、`src/models/pulseppg_feature.py`、`src/models/papagei_feature.py`、`requirements.txt`。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "问题 4：PaPaGei 官方 checkpoint 在 CPU-only 环境下会尝试按 CUDA 设备反序列化，导致加载失败。", "style": "normal"},
        {"text": "解决：在项目适配代码里使用 `torch.load(..., map_location=...)` 并移除可能存在的 `module.` 前缀，使其可以在当前机器上稳定加载。", "style": "normal"},
        {"text": "相关代码：`src/models/papagei_feature.py`。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "问题 5：PulsePPG 公开网络定义的归一化层与当前单通道输入不匹配，会持续输出 `InstanceNorm1d` warning。", "style": "normal"},
        {"text": "解决：在适配层里替换为单通道 instance norm 后再加载 checkpoint，避免无意义 warning 干扰当日日志。", "style": "normal"},
        {"text": "相关代码：`src/models/pulseppg_feature.py`。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "问题 6：Windows 环境下 `GridSearchCV(n_jobs=-1)` 在 grouped CV 阶段触发进程管道权限错误。", "style": "normal"},
        {"text": "解决：把 grouped CV 搜索固定为 `n_jobs=1`，优先保证 participant 分组验证正确且可稳定执行。", "style": "normal"},
        {"text": "相关代码：`src/regression/train_regressor.py`。", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "四、当日获得的进展与对应文件", "style": "heading"},
        {"text": "1. 处理后缓存已经真正落盘，可作为后续所有模型实验的统一输入。", "style": "normal"},
        {"text": f"   - 总窗口数：{data.processed_artifacts.num_windows}", "style": "normal"},
        {"text": f"   - 训练窗口数：{data.processed_artifacts.num_train_windows}", "style": "normal"},
        {"text": f"   - 测试窗口数：{data.processed_artifacts.num_test_windows}", "style": "normal"},
        {"text": f"   - 参与者数：{data.processed_artifacts.num_participants}", "style": "normal"},
        {"text": f"   - 训练参与者：{train_participants}", "style": "normal"},
        {"text": f"   - 测试参与者：{test_participants}", "style": "normal"},
        {"text": f"   - manifest：`{data.processed_artifacts.manifest_path}`", "style": "normal"},
        {"text": f"   - windows：`{data.processed_artifacts.windows_path}`", "style": "normal"},
        {"text": f"   - labels：`{data.processed_artifacts.labels_path}`", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "2. 基线阶段的可视化已在当日补齐。", "style": "normal"},
        {"text": "   - 图文件已追加到现有 baseline 结果目录下的 `plots/` 子目录。", "style": "normal"},
        {"text": f"   - 图文件数量：{len(data.baseline_plots)}", "style": "normal"},
        {"text": f"   - 图文件：{baseline_plot_names}", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "3. PulsePPG 已完成本地接入与全量 embedding 导出。", "style": "normal"},
        {"text": f"   - 外部仓库：`{safe_text(data.pulseppg_features.external_repo_root)}`", "style": "normal"},
        {"text": f"   - checkpoint：`{safe_text(data.pulseppg_features.checkpoint_path)}`", "style": "normal"},
        {"text": f"   - 目标采样率：{safe_text(data.pulseppg_features.target_sampling_hz)} Hz", "style": "normal"},
        {"text": f"   - embedding 维度：{safe_text(data.pulseppg_features.embedding_dim)}", "style": "normal"},
        {"text": f"   - 导出窗口数：{safe_text(data.pulseppg_features.num_windows)}", "style": "normal"},
        {"text": f"   - features：`{safe_text(data.pulseppg_features.features_path)}`", "style": "normal"},
        {"text": f"   - metadata：`{safe_text(data.pulseppg_features.metadata_path)}`", "style": "normal"},
        {"text": f"   - manifest：`{safe_text(data.pulseppg_features.manifest_path)}`", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "4. PaPaGei 已完成本地接入与全量 embedding 导出。", "style": "normal"},
        {"text": f"   - 外部仓库：`{safe_text(data.papagei_features.external_repo_root)}`", "style": "normal"},
        {"text": f"   - checkpoint：`{safe_text(data.papagei_features.checkpoint_path)}`", "style": "normal"},
        {"text": f"   - 目标采样率：{safe_text(data.papagei_features.target_sampling_hz)} Hz", "style": "normal"},
        {"text": f"   - embedding 维度：{safe_text(data.papagei_features.embedding_dim)}", "style": "normal"},
        {"text": f"   - 导出窗口数：{safe_text(data.papagei_features.num_windows)}", "style": "normal"},
        {"text": f"   - features：`{safe_text(data.papagei_features.features_path)}`", "style": "normal"},
        {"text": f"   - metadata：`{safe_text(data.papagei_features.metadata_path)}`", "style": "normal"},
        {"text": f"   - manifest：`{safe_text(data.papagei_features.manifest_path)}`", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "5. 首轮下游回归结果已经落盘。", "style": "normal"},
        {"text": f"   - PulsePPG + Ridge：MAE={format_metric(data.pulseppg_ridge.mae)}，RMSE={format_metric(data.pulseppg_ridge.rmse)}，coverage={format_metric(data.pulseppg_ridge.coverage)}，最优参数={pulse_best_params}", "style": "normal"},
        {"text": f"     指标文件：`{safe_text(data.pulseppg_ridge.metrics_path)}`", "style": "normal"},
        {"text": f"     预测文件：`{safe_text(data.pulseppg_ridge.predictions_path)}`", "style": "normal"},
        {"text": f"     模型文件：`{safe_text(data.pulseppg_ridge.estimator_path)}`", "style": "normal"},
        {"text": f"   - PaPaGei + Ridge：MAE={format_metric(data.papagei_ridge.mae)}，RMSE={format_metric(data.papagei_ridge.rmse)}，coverage={format_metric(data.papagei_ridge.coverage)}，最优参数={papagei_best_params}", "style": "normal"},
        {"text": f"     指标文件：`{safe_text(data.papagei_ridge.metrics_path)}`", "style": "normal"},
        {"text": f"     预测文件：`{safe_text(data.papagei_ridge.predictions_path)}`", "style": "normal"},
        {"text": f"     模型文件：`{safe_text(data.papagei_ridge.estimator_path)}`", "style": "normal"},
        {"text": "", "style": "normal"},
        {"text": "五、当日结论与后续展望", "style": "heading"},
        {"text": "1. 当天已经把项目主线从“只看 baseline”推进到“foundation model 真正接入并跑出首轮结果”的阶段。", "style": "normal"},
        {"text": "2. 处理后缓存、baseline 图表、PulsePPG / PaPaGei embedding、Ridge 回归结果现在都已经形成可复用的本地文件资产。", "style": "normal"},
        {"text": "3. 下一步应继续补齐 `Linear Regression`、`Random Forest`、`Gradient Boosting`，并给 foundation model 预测结果生成三类图表。", "style": "normal"},
        {"text": "4. 在回归器齐备后，需要按 participant、session、HR 区间分析误差模式，判断为什么当前 `PulsePPG` 改善了 RMSE 而 `PaPaGei` 仍偏弱。", "style": "normal"},
        {"text": "5. 后续日报仍应只依据当日 `Current.md` 日志与当日产物生成，避免把旧日期内容再次写入新报告。", "style": "normal"},
    ]


def safe_text(value: object | None) -> str:
    if value in (None, ""):
        return "未记录"
    return str(value)


def format_metric(value: float | None) -> str:
    if value is None:
        return "未生成"
    return f"{value:.6f}"


def format_params(params: dict[str, object] | None) -> str:
    if not params:
        return "未记录"
    return json.dumps(params, ensure_ascii=False)


def render_document_xml(blocks: list[dict[str, str]]) -> str:
    paragraphs = [render_paragraph(block["text"], block["style"]) for block in blocks]
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(paragraphs)
        + (
            "<w:sectPr>"
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="708" w:footer="708" w:gutter="0"/>'
            "</w:sectPr>"
        )
        + "</w:body></w:document>"
    )


def render_paragraph(text: str, style: str = "normal") -> str:
    if text == "":
        return "<w:p/>"
    align = "center" if style == "title" else "left"
    size = "32" if style == "title" else ("26" if style == "heading" else "22")
    bold = style in {"title", "heading"}
    escaped_text = escape(text)
    paragraph_props = f'<w:pPr><w:jc w:val="{align}"/></w:pPr>'
    run_props = f"<w:rPr>{'<w:b/>' if bold else ''}<w:sz w:val=\"{size}\"/><w:szCs w:val=\"{size}\"/></w:rPr>"
    return f'<w:p>{paragraph_props}<w:r>{run_props}<w:t xml:space="preserve">{escaped_text}</w:t></w:r></w:p>'


def write_docx(output_path: Path, document_xml: str) -> None:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""
    core_props = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:dcmitype="http://purl.org/dc/dcmitype/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>GalaxyPPG 项目当日进展报告</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>
"""
    app_props = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as docx_file:
        docx_file.writestr("[Content_Types].xml", content_types)
        docx_file.writestr("_rels/.rels", package_rels)
        docx_file.writestr("word/document.xml", document_xml)
        docx_file.writestr("word/_rels/document.xml.rels", document_rels)
        docx_file.writestr("docProps/core.xml", core_props)
        docx_file.writestr("docProps/app.xml", app_props)


def default_output_path(report_date: str) -> Path:
    year, month, day = report_date.split("-")
    return Path("reports") / f"{int(month)}-{int(day)}" / f"当前进展报告_{year}-{month}-{day}.docx"


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.date)
    data = collect_report_data(args.date)
    blocks = build_report_blocks(data, output_path)
    document_xml = render_document_xml(blocks)
    write_docx(output_path, document_xml)
    print(output_path)


if __name__ == "__main__":
    main()
