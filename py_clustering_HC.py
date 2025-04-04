import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 1️=读取数据 ==========
file_path = r"U:\Py_code\py_clustering\data_transformer_one_10_08\transformer.xlsx"
output_dir = r"U:\Py_code\py_clustering\py_clusterfing_HC\results"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(file_path)
df = df.dropna()
df['数据时间'] = pd.to_datetime(df['数据时间'])

# ========== 分组 ==========
grouped_dict = {timestamp: group.copy() for timestamp, group in df.groupby('数据时间')}
print(f"共有 {len(grouped_dict)} 组数据")

# ========== 层次聚类，每组分成3类 ==========
point_cluster_history = {}

for timestamp, df_selected in grouped_dict.items():
    X = df_selected[['A相电压', 'A相电流']].values

    if len(X) < 3:
        print(f"跳过 {timestamp}，数据点不足")
        continue

    try:
        model = AgglomerativeClustering(n_clusters=3, linkage='ward')  # 或 'average', 'complete'
        cluster_labels = model.fit_predict(X)
    except Exception as e:
        print(f"时间 {timestamp} 聚类失败：{e}")
        continue

    for idx, (row, cluster) in enumerate(zip(df_selected.iterrows(), cluster_labels)):
        point_id = row[1]['测量点号']
        if point_id not in point_cluster_history:
            point_cluster_history[point_id] = []
        point_cluster_history[point_id].append(cluster)

print("所有时间点的层次聚类已完成")

# ========== 投票统计最终类别 ==========
final_clusters = {}
for point_id, cluster_list in point_cluster_history.items():
    most_common_cluster = Counter(cluster_list).most_common(1)[0][0]
    final_clusters[point_id] = most_common_cluster

df_final_clusters = pd.DataFrame(final_clusters.items(), columns=['测量点号', '最终类别'])
print(df_final_clusters.head())

# ========== 可视化聚类结果 ==========
df_plot = df[['测量点号', 'A相电压', 'A相电流']].drop_duplicates()
df_plot = df_plot.merge(df_final_clusters, on='测量点号')

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_plot, x='A相电压', y='A相电流', hue='最终类别', palette='Set2', style='最终类别')
plt.xlabel('A相电压')
plt.ylabel('A相电流')
plt.title('最终测量点聚类结果（层次聚类 Hierarchical, 基于 480 组）台区一全部数据')

plot_path = os.path.join(output_dir, "final_cluster_plot_hierarchical.png")
plt.savefig(plot_path)
plt.show()
print(f"聚类图已保存：{plot_path}")

# ========== =保存 Excel 聚类结果 ==========
output_file = os.path.join(output_dir, "final_cluster_results_HC.xlsx")
df_final_clusters.to_excel(output_file, index=False)
print(f"最终聚类结果已保存：{output_file}")
