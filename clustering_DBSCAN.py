import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ======= 读取 & 处理数据 ======
# 文件读取路径
file_path = r"U:\Py_code\py_clustering\data_transformer_one_10_08\transformer_01_10_08.xlsx"

# 设定存储路径
output_dir = r"U:\Py_code\py_clustering_Kmeans\results"
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

df = pd.read_excel(file_path)

# 去除缺失值
df = df.dropna()

# 梳理数据格式
df['数据时间'] = pd.to_datetime(df["数据时间"])

# 按照时间进行分组
grouped_dict = {timestamp: group.copy() for timestamp, group in df.groupby('数据时间')}
print(f"共有 {len(grouped_dict)} 组数据")  # 确保是 96 组

# 存储每个测量点号的聚类结果
point_cluster_history = {}

# ======== 对96组数据进行DBSCAN =======
# 遍历每个时间点，执行 DBSCAN
for timestamp, df_selected in grouped_dict.items():
    X = df_selected[['A相电压', 'A相电流']].values

    if len(X) < 5:
        print(f"跳过 {timestamp}，数据点不足")
        continue

    dbscan = DBSCAN(eps=1.0, min_samples=3)
    cluster_labels = dbscan.fit_predict(X)

    for idx, (row, cluster) in enumerate(zip(df_selected.iterrows(), cluster_labels)):
        point_id = row[1]['测量点号']

        if point_id not in point_cluster_history:
            point_cluster_history[point_id] = []

        point_cluster_history[point_id].append(cluster)

# ====== 统计每个点的最终类别 ==========
final_clusters = {}

for point_id, cluster_list in point_cluster_history.items():
    # 跳过噪声点 -1
    cleaned = [c for c in cluster_list if c != -1]
    if cleaned:
        most_common_cluster = Counter(cleaned).most_common(1)[0][0]
    else:
        most_common_cluster = -1  # 全部是噪声
    final_clusters[point_id] = most_common_cluster

df_final_clusters = pd.DataFrame(final_clusters.items(), columns=['测量点号', '最终类别'])
print(df_final_clusters.head())


# ========= 可视化 =======
df_plot = df[['测量点号', 'A相电压', 'A相电流']].drop_duplicates()
df_plot = df_plot.merge(df_final_clusters, on='测量点号')

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_plot, x='A相电压', y='A相电流', hue='最终类别', palette='Set2', style='最终类别')
plt.xlabel('A相电压')
plt.ylabel('A相电流')
plt.title('最终测量点聚类结果（DBSCAN, 基于 96 组）')

plot_path = os.path.join(output_dir, "final_cluster_plot_dbscan.png")
plt.savefig(plot_path)
plt.show()
print(f"聚类图已保存：{plot_path}")

output_file = os.path.join(output_dir, "final_cluster_results_dbscan.xlsx")
df_final_clusters.to_excel(output_file, index=False)
print(f"最终聚类结果已保存：{output_file}")
