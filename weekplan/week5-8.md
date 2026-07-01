# Week 5-8 项目进度汇报

## 一、项目概况

本项目围绕腕部 PPG 心率估计展开，目标是在 corrected GalaxyPPG 流程基础上，构建可复现、可扩展、可用于论文撰写的实验包。项目当前已经形成较完整的工程结构：

- `src/data/`：数据读取、统一 schema、窗口切分、标签生成与外部数据集导出。
- `src/baseline/`：peak detection 与 spectral HR 等传统心率估计 baseline。
- `src/models/`：PulsePPG、PaPaGei foundation model embedding 提取。
- `src/regression/`：基于 embedding 的线性、Ridge、Random Forest 等下游回归。
- `src/utils/`：Week 2-4 artifact 构建、可复现包整理、Week 7 统计检验、Week 8 final freeze。
- `configs/`：固定 participant split、实验模式和 submission protocol。
- `experiments/`：Week 2/3/4/7/8 实验结果、表格、预测、统计和冻结包。
- `data/raw/` 与 `data/processed/`：原始数据放置说明、processed manifest 与窗口数据。

Week 5-8 的主要工作是从 GalaxyPPG 单数据集实验，推进到外部数据集接入、统计检验和最终可复现包冻结。当前阶段已完成代码、文档、artifact 结构和 smoke test；由于真实 PPG-DaLiA / WildPPG wrist 原始数据未包含在仓库中，外部数据集真实数值结果尚未生成。

## 二、总体完成情况

本阶段完成了三类工作。

第一，补齐 Week 2-4 的可复现 artifact 结构。Week 2、Week 3、Week 4 的实验目录已统一增加固定文件名，包括 `predictions.csv`、`metrics.json`、`run_config.json`、`run_log.json`，并生成可复现包 manifest，便于审计和重跑。

第二，完成 Week 5-6 的外部数据集接入代码。新增 PPG-DaLiA 和 WildPPG wrist loader/export，使外部数据可以映射到与 GalaxyPPG 一致的处理规则：10 秒窗口、2 秒 stride、ECG/R-peak 或 IBI-derived instantaneous HR、窗口内 median HR 标签。

第三，完成 Week 7-8 的统计检验和最终冻结包。新增 participant-level 统计工具，比较 learned router 与每个 participant 的 best single expert，并输出 bootstrap CI、paired t-test 和 Wilcoxon signed-rank test；随后将 Week 2/3/4/7 的关键结果复制到最终冻结目录 `experiments/final_frozen_results_2026-06-29/`。

## 三、Week 2-4 可复现包补齐

虽然本汇报覆盖 Week 5-8，但 Week 5-8 的重要前置工作是补齐 Week 2-4 的复现结构。当前已新增工具：

```text
src/utils/package_reproducibility_artifacts.py
```

该工具完成以下工作：

- 为 Week 2/3/4 结果目录补齐标准化文件命名。
- 生成 Week 2 run manifest 与 embedding manifest。
- 检查 PulsePPG / PaPaGei embedding manifest 使用 corrected IBI-based manifest，而不是旧 HR-based manifest。
- 汇总 reproducibility package 索引。

已生成的关键文件包括：

```text
experiments/reproducibility_manifest.json
experiments/reproducibility_manifest.md
experiments/week2_galaxyppg_corrected_2026-05-01/run_manifest.csv
experiments/week2_galaxyppg_corrected_2026-05-01/embedding_manifest.csv
```

从 `embedding_manifest.csv` 看，PulsePPG 与 PaPaGei embedding 均指向：

```text
data/processed/galaxyppg_ibi_w10_s2_beat_interval_instant_hr_median_manifest.json
```

这说明当前复现包已经切换到 corrected IBI-based label manifest，避免继续沿用早期 HR-based manifest。

## 四、Week 4 Router Artifact 扩展

Week 4 router 已从单纯结果表扩展为更完整的可审计 artifact 包。新增或补齐的内容包括：

- motion/quality feature CSV
- router predictions
- router metrics
- fold assignments
- trained router fold model files
- router model manifest

关键生成位置如下：

```text
experiments/week4_galaxyppg_lightweight_router_2026-05-13/features/week4_motion_quality_features.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/predictions/week4_routed_predictions.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/metrics/routing_summary.json
experiments/week4_galaxyppg_lightweight_router_2026-05-13/tables/router_fold_assignments.csv
experiments/week4_galaxyppg_lightweight_router_2026-05-13/models/*.joblib
experiments/week4_galaxyppg_lightweight_router_2026-05-13/models/router_model_manifest.json
```

当前 Week 4 的主要结果为：best single foundation expert MAE 为 6.8182 bpm；motion-plus-quality soft gate MAE 为 6.4001 bpm；motion-plus-quality hard gate MAE 为 6.4426 bpm。hard gate 的 tail robustness 更好，p95 absolute error 为 25.5632 bpm，catastrophic error rate 为 0.0750。

这些结果支持继续保留 peak-based classical expert + PulsePPG foundation expert 的轻量 router 方案，并将 motion/quality feature 作为主要 routing 信号。

## 五、Week 5：PPG-DaLiA 接入

Week 5 的目标是将 PPG-DaLiA 引入当前统一流程，作为 GalaxyPPG 之外的外部验证数据集。

已新增：

```text
src/data/ppgdalia_loader.py
src/data/export_ppgdalia.py
data/raw/PPG-DaLiA/README.md
```

当前实现支持：

- wrist BVP
- wrist ACC
- chest ECG / rpeaks
- participant-level 数据读取
- 转换为项目统一 canonical schema
- 10 秒窗口、2 秒 stride
- 基于 ECG/R-peak IBI-derived instantaneous HR 的 median HR label

README 中已补充 Week 5 复现命令。核心导出命令为：

```bash
python -m src.data.export_ppgdalia --dataset-root data/raw/PPG-DaLiA --output-root data/processed
```

导出后预期生成：

```text
data/processed/ppg_dalia_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json
```

随后可沿用现有 baseline、embedding extraction 和 regressor 训练流程，对 PPG-DaLiA 执行 within-dataset baseline、PulsePPG/PaPaGei embedding 与 Ridge probe。

当前限制是：仓库中没有真实 PPG-DaLiA 原始数据，因此真实 PPG-DaLiA 数值结果尚未生成。已用合成数据完成 export + baseline smoke test，验证代码路径可运行。

## 六、Week 6：WildPPG Wrist 接入

Week 6 的目标是将 WildPPG wrist 数据纳入统一外部验证流程。考虑到 WildPPG wrist 的公开数据包结构可能因来源、镜像或子集不同而变化，本阶段采用 manifest-driven loader，避免在代码中硬编码未知目录结构和列名。

已新增：

```text
src/data/wildppg_loader.py
src/data/export_wildppg_wrist.py
data/raw/WildPPG-wrist/README.md
```

WildPPG wrist 通过以下 manifest 配置数据文件路径和列名：

```text
data/raw/WildPPG-wrist/wildppg_wrist_manifest.csv
```

当前实现支持：

- wrist PPG
- wrist ACC
- ECG/RR 或可转换为 beat interval 的参考标签
- 统一窗口与标签规则
- 与 GalaxyPPG、PPG-DaLiA 一致的 processed manifest 输出

核心导出命令为：

```bash
python -m src.data.export_wildppg_wrist --dataset-root data/raw/WildPPG-wrist --output-root data/processed
```

导出后预期生成：

```text
data/processed/wildppg_wrist_ecg_w10_s2_beat_interval_instant_hr_median_manifest.json
```

当前限制同 Week 5：真实 WildPPG wrist 原始数据未包含在仓库中，因此尚未生成真实外部验证数值。已用合成数据完成 export + baseline smoke test。

## 七、Week 7：Participant-Level 统计检验

Week 7 新增 participant-level 统计工具：

```text
src/utils/build_week7_statistics.py
```

该工具比较 Week 4 routed system 与每个 participant 的 best single expert。统计以 participant-level aggregate 为单位，而不是把 window-level samples 当作独立样本，避免夸大显著性。

已生成：

```text
experiments/week7_final_statistics/participant_level_router_comparison.csv
experiments/week7_final_statistics/paired_significance_tests.csv
experiments/week7_final_statistics/paired_significance_tests.json
experiments/week7_final_statistics/week7_statistics.md
```

当前统计对象为 Week 4 `motion_quality/hard_gate` router，共 5 个 participants。结果中正 delta 表示 router 优于该 participant 的 best single expert。

主要结果如下：

- MAE：mean delta 为 -0.0903 bpm，bootstrap 95% CI 为 [-0.4910, 0.3103]，paired t-test p=0.7277，Wilcoxon p=1.0000。
- P95 absolute error：mean delta 为 -0.6740 bpm，bootstrap 95% CI 为 [-2.8397, 1.3364]，paired t-test p=0.5962，Wilcoxon p=0.8125。
- Catastrophic error rate：mean delta 为 -0.0003，bootstrap 95% CI 为 [-0.0102, 0.0078]，paired t-test p=0.9580，Wilcoxon p=1.0000。

解释上需要谨慎：Week 4 router 在 window-level overall MAE 上优于 best single foundation expert，但 participant-level paired statistics 相对“每个 participant 的 best single expert”未显示显著改善。该结果有助于论文中更准确地区分 overall routing gain 与 participant-level statistical evidence。

## 八、Week 8：最终冻结包

Week 8 新增 final results freeze 工具：

```text
src/utils/freeze_final_results.py
```

已生成最终冻结目录：

```text
experiments/final_frozen_results_2026-06-29/
```

冻结包包含：

- Week 2 tables / metrics / predictions
- Week 3 oracle-routing artifacts
- Week 4 router features / predictions / metrics / models
- Week 7 statistics
- `final_frozen_manifest.json`
- frozen package `README.md`

`final_frozen_manifest.json` 记录了每一类 artifact 的来源路径、目标路径和复制状态。当前 Week 2、Week 3、Week 4、Week 7 的核心 artifact 均已成功复制到冻结目录，可作为后续论文撰写和复现审查的固定版本。

## 九、README 与兼容性修正

主 README 已补充以下内容：

- Week 2/3/4 复现命令
- Week 5 PPG-DaLiA 导出、baseline、embedding 和 probe 命令
- Week 6 WildPPG wrist 导出和后续复用命令
- Week 7 participant-level statistics 命令
- Week 8 final freeze 命令

同时完成了若干兼容性修正：

- `src/data/cache.py` 支持非 GalaxyPPG dataset artifact name。
- `src/baseline/run_baseline.py` 不再对外部 processed manifest 强套 GalaxyPPG split。
- `src/regression/train_regressor.py` 不再对外部 feature manifest 默认套 GalaxyPPG split。
- `.gitignore` 新增外部 raw data 忽略规则，避免误提交 PPG-DaLiA / WildPPG wrist 原始数据。

这些修正确保 GalaxyPPG、PPG-DaLiA 和 WildPPG wrist 可以共享统一 pipeline，同时不把 GalaxyPPG 的固定 split 假设错误迁移到外部数据集。

## 十、验证情况

已通过以下基础验证：

```bash
python -m compileall src
python -m src.data.export_ppgdalia --help
python -m src.data.export_wildppg_wrist --help
python -m src.utils.build_week7_statistics --help
python -m src.utils.freeze_final_results --help
```

并已用合成数据完成 smoke test：

- PPG-DaLiA export + baseline
- WildPPG wrist export + baseline
- Week 7 statistics
- Week 8 final freeze

## 十一、当前限制与后续工作

当前主要限制是外部数据集真实原始数据未在仓库中，因此 Week 5 和 Week 6 只完成了 loader、export、README、命令链路和 smoke test，尚未产出 PPG-DaLiA / WildPPG wrist 的真实实验数值。

后续应优先完成：

- 放置真实 PPG-DaLiA 原始数据并生成 processed manifest。
- 放置真实 WildPPG wrist 原始数据与 `wildppg_wrist_manifest.csv`，生成 processed manifest。
- 对两个外部数据集运行 baseline、PulsePPG/PaPaGei embedding、regressor 和 router 迁移验证。
- 将外部验证结果纳入最终冻结包的下一版本。
- 基于 Week 7 统计结论，在论文中谨慎表述 router 的收益边界。

## 十二、阶段结论

Week 5-8 阶段的核心产出是把项目从单次实验结果推进为可复现、可扩展、可审计的实验工程。当前已经完成外部数据接入接口、Week 2-4 可复现包补齐、Week 4 router artifact 扩展、Week 7 participant-level 统计检验，以及 Week 8 final frozen package。项目已经具备后续接入真实 PPG-DaLiA / WildPPG wrist 数据并补充外部验证结果的工程基础。
