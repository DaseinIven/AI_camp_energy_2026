"""
特征探索分析脚本：相关性热图、分布图、散点图等
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 非交互后端，直接保存图片
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')

plot_dir = os.path.join(current_dir, 'output', 'eda_plots')
os.makedirs(plot_dir, exist_ok=True)

# ==================== 加载数据 ====================
print("加载数据...")
df_feat = pd.read_csv(train_feature_path)
df_label = pd.read_csv(train_label_path)
df = pd.merge(df_feat, df_label, on='times', how='inner')
df['times'] = pd.to_datetime(df['times'])

# 添加时间特征
df['hour'] = df['times'].dt.hour
df['minute'] = df['times'].dt.minute
df['dayofweek'] = df['times'].dt.dayofweek
df['month'] = df['times'].dt.month

target = 'A'
feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
all_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 相关性热图 ====================
print("[1/5] 相关性热图...")
corr_cols = all_features + [target]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=16, pad=15)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, '01_correlation_heatmap.png'), dpi=150)
plt.close(fig)

# ==================== 2. 各特征与目标 A 的相关性柱状图 ====================
print("[2/5] 特征-目标相关性排名...")
target_corr = corr[target].drop(target).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#d9534f' if v < 0 else '#5cb85c' for v in target_corr.values]
ax.barh(range(len(target_corr)), target_corr.values, color=colors)
ax.set_yticks(range(len(target_corr)))
ax.set_yticklabels(target_corr.index)
ax.set_xlabel('Pearson Correlation with A')
ax.set_title('Feature vs Target (A) Correlation', fontsize=14)
for i, v in enumerate(target_corr.values):
    ax.text(v + 0.01 * np.sign(v), i, f'{v:.3f}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, '02_target_correlation.png'), dpi=150)
plt.close(fig)

# ==================== 3. 特征分布直方图 ====================
print("[3/5] 特征分布...")
n_cols = 4
n_rows = int(np.ceil(len(all_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()
for i, col in enumerate(all_features):
    axes[i].hist(df[col].values, bins=50, color='#5b9bd5', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel('')
# 隐藏多余子图
for j in range(len(all_features), len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Feature Distributions', fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, '03_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ==================== 4. 各特征 vs A 的散点图 ====================
print("[4/5] 特征 vs A 散点图...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()
for i, col in enumerate(all_features):
    # 采样以提高绘图速度
    sample = df.sample(min(5000, len(df)), random_state=42)
    axes[i].scatter(sample[col].values, sample[target].values,
                    s=1, alpha=0.3, color='#5b9bd5')
    axes[i].set_xlabel(col, fontsize=9)
    axes[i].set_ylabel(target, fontsize=9)
for j in range(len(all_features), len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Feature vs Target (A) Scatter Plots', fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, '04_feature_vs_target_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ==================== 5. 目标变量 A 的时间模式 ====================
print("[5/5] 目标变量 A 的时间模式...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 按小时的平均价格
hourly = df.groupby('hour')[target].agg(['mean', 'std']).reset_index()
axes[0, 0].fill_between(hourly['hour'], hourly['mean'] - hourly['std'],
                         hourly['mean'] + hourly['std'], alpha=0.2, color='#5b9bd5')
axes[0, 0].plot(hourly['hour'], hourly['mean'], 'o-', color='#2b5797', linewidth=2, markersize=5)
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('A (mean ± std)')
axes[0, 0].set_title('Average Price by Hour', fontsize=12)

# 按月份的平均价格
monthly = df.groupby('month')[target].agg(['mean', 'std']).reset_index()
axes[0, 1].fill_between(monthly['month'], monthly['mean'] - monthly['std'],
                         monthly['mean'] + monthly['std'], alpha=0.2, color='#ed7d31')
axes[0, 1].plot(monthly['month'], monthly['mean'], 's-', color='#c55a11', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('A (mean ± std)')
axes[0, 1].set_title('Average Price by Month', fontsize=12)
axes[0, 1].set_xticks(range(1, 13))

# 按星期几的平均价格
dow = df.groupby('dayofweek')[target].agg(['mean', 'std']).reset_index()
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1, 0].fill_between(dow['dayofweek'], dow['mean'] - dow['std'],
                         dow['mean'] + dow['std'], alpha=0.2, color='#70ad47')
axes[1, 0].plot(dow['dayofweek'], dow['mean'], 'D-', color='#375623', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('A (mean ± std)')
axes[1, 0].set_title('Average Price by Day of Week', fontsize=12)
axes[1, 0].set_xticks(range(7))
axes[1, 0].set_xticklabels(dow_labels)

# 价格分布
axes[1, 1].hist(df[target].values, bins=100, color='#bf8f00', edgecolor='white', alpha=0.85)
axes[1, 1].axvline(df[target].median(), color='red', linestyle='--', linewidth=1.5, label=f'Median: {df[target].median():.2f}')
axes[1, 1].axvline(df[target].mean(), color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {df[target].mean():.2f}')
axes[1, 1].set_xlabel(target)
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Price Distribution (A)', fontsize=12)
axes[1, 1].legend(fontsize=9)

fig.suptitle('Price A — Temporal Patterns', fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, '05_price_temporal_patterns.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'\n图表已保存到: {plot_dir}/')
print('生成完成: 01_correlation_heatmap.png')
print('          02_target_correlation.png')
print('          03_feature_distributions.png')
print('          04_feature_vs_target_scatter.png')
print('          05_price_temporal_patterns.png')
