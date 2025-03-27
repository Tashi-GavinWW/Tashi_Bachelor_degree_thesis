import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ===== 读取数据 =======

file_path = r"U:\Py_code\py_clustering\data_transformer_one_10_08\transformer_01_10_08.xlsx"
output_dir = r"U:\Py_code\py_clustering\py_clustering_Spectial_Clustering\results"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(file_path)
df = df.dropna()
df['数据时间'] = pd.to_datetime(df['数据时间'])

# 对数据进行分组

grouped_dict = {timestamp: group.copy() for timestamp, group in df.groupby('数据时间')}
print(f"共有 {len(grouped_dict)} 组数据")

# ========= 遍历分组进行谱聚类 =========
point_cluster_history = {}

for timestamp, df_selected in grouped_dict.items():
    X = df_selected[['A相电压', 'A相电流']].values

    if len(X) < 5:
        print(f"跳过 {timestamp}，数据点不足")
        continue

    try:
        # 使用谱聚类 Spectral Clustering（3类）
        sc = SpectralClustering(n_clusters=3, affinity='rbf', assign_labels='kmeans', random_state=0)
        cluster_labels = sc.fit_predict(X)
    except Exception as e:
        print(f" 时间 {timestamp} 聚类失败：{e}")
        continue

    for idx, (row, cluster) in enumerate(zip(df_selected.iterrows(), cluster_labels)):
        point_id = row[1]['测量点号']

        if point_id not in point_cluster_history:
            point_cluster_history[point_id] = []

        point_cluster_history[point_id].append(cluster)

print("所有时间点的谱聚类已完成")

# 统计类别
final_clusters = {}
for point_id, cluster_list in point_cluster_history.items():
    most_common_cluster = Counter(cluster_list).most_common(1)[0][0]
    final_clusters[point_id] = most_common_cluster

df_final_clusters = pd.DataFrame(final_clusters.items(), columns=['测量点号', '最终类别'])
print(df_final_clusters.head())

# ====== 可视化与储存 =======
df_plot = df[['测量点号', 'A相电压', 'A相电流']].drop_duplicates()
df_plot = df_plot.merge(df_final_clusters, on='测量点号')

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'        # 黑体 
matplotlib.rcParams['axes.unicode_minus'] = False    # 让负号正常显示


plt.figure(figsize=(8,6))
sns.scatterplot(data=df_plot, x='A相电压', y='A相电流', hue='最终类别', palette='Set2', style='最终类别')
plt.xlabel('A相电压')
plt.ylabel('A相电流')
plt.title('最终测量点聚类结果（谱聚类，基于 96 组）')

plot_path = os.path.join(output_dir, "final_cluster_plot_spectral.png")
plt.savefig(plot_path)
plt.show()
print(f"聚类图已保存：{plot_path}")

output_file = os.path.join(output_dir, "final_cluster_results_spectral.xlsx")
df_final_clusters.to_excel(output_file, index=False)
print(f"最终聚类结果已保存：{output_file}")