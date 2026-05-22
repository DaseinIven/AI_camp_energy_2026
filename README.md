# AI Camp Energy 2026 — 蒙西电网节点电价预测与储能策略优化

本项目是 **Datawhale × 蒙西电网 AI 训练营** 的参赛基线方案，目标为：基于蒙西电网边界条件（负荷、新能源出力、联络线等），预测实时节点电价 A，并据此制定储能充放电策略，最大化每日收益。

## 任务概述

- **第一阶段（回归）**：利用 15 分钟粒度的边界条件预测值，预测未来每 15 分钟的节点电价 A
- **第二阶段（优化）**：给定全天 96 个时间点的预测价格，找出每日最优的 2 小时充电窗口 + 2 小时放电窗口，最大化价差收益

## 数据说明

```
data/
├── train/
│   ├── mengxi_boundary_anon_filtered.csv   # 训练特征（实际值 + 预测值，约 35K 行）
│   └── mengxi_node_price_selected.csv      # 训练标签（节点电价 A）
└── test/
    └── test_in_feature_ori.csv             # 测试特征（仅预测值，约 5.7K 行）
```

| 特征 | 说明 |
|------|------|
| 系统负荷 | 全网用电负荷 |
| 风光总加 | 风电 + 光伏总出力 |
| 联络线 | 与外网的交换功率 |
| 风电 / 光伏 | 新能源分项出力 |
| 水电 | 水电出力 |
| 非市场化机组 | 非市场化机组出力 |

数据粒度为 **15 分钟/点**（每天 96 点），训练数据覆盖 2025 年全年。

## 项目结构

```
├── sklearn_baseline.py          # 基线模型（GBR + 暴力搜索策略）
├── pyproject.toml               # 项目配置与依赖声明
├── CHANGELOG.md                 # 版本变更记录
├── output/                      # 预测结果与策略输出（已 gitignore）
└── data/                        # 数据文件（Git LFS 管理）
```

## 快速开始

```bash
# 安装依赖
uv pip install pandas numpy scikit-learn

# 运行基线
python sklearn_baseline.py
```

每次运行会在 `output/` 下生成带时间戳的结果文件：
- `sklearn_baseline_output_<timestamp>.csv` — 预测电价
- `output_<timestamp>.csv` — 最终充放电策略（提交文件）

## 基线方案

| 项目 | 说明 |
|------|------|
| 模型 | GradientBoostingRegressor（200 棵树，lr=0.05，max_depth=6） |
| 特征 | 7 个边界条件预测值 + hour / minute / dayofweek / month |
| 验证集 | 时间序列切分，后 20% |
| 策略优化 | 暴力搜索每日最优充放电窗口 |

## 版本记录

详见图表 [CHANGELOG.md](CHANGELOG.md)。

## License

Apache-2.0
