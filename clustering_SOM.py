import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from minisom import MiniSom

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ==== 读取数据 =========
file_path = r"U:\Py_code\py_clustering\data_transformer_one_10_08\transformer_01_10_08.xlsx"
output_dir = r"U:\Py_code\py_clustering\py_clustering_SOM\results"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(file_path)
df = df.dropna()
df['数据时间'] = pd.to_datetime(df['数据时间'])

# ===== 分组 ======
grouped_dict = {timestamp: group.copy() for timestamp, group in df.groupby('数据时间')}
print(f"共有 {len(grouped_dict)} 组数据")

# ===== 使用SOM+KMeans聚类 ======
point_cluster_history = {}

som_rows, som_cols = 3, 3       # SOM 网格尺寸（建议3×3）
final_cluster_num = 3           # 最终目标聚为3类

for timestamp, df_selected in grouped_dict.items():
    X = df_selected[['A相电压', 'A相电流']].values

    if len(X) < 5:
        print(f"跳过 {timestamp}，数据点不足")
        continue

    try:
        # 数据归一化
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 初始化 SOM
        som = MiniSom(som_rows, som_cols, input_len=2, sigma=1.0, learning_rate=0.5)
        som.random_weights_init(X_scaled)
        som.train_random(X_scaled, num_iteration=100)

        # 找出每个样本对应的神经元位置
        bmus = np.array([som.winner(x) for x in X_scaled])
        bmu_1d = [r * som_cols + c for r, c in bmus]  # 变成 1D 索引

        # KMeans 聚神经元，分成最终的3类
        bmu_centers = np.unique(bmu_1d).reshape(-1, 1)
        km = KMeans(n_clusters=final_cluster_num, random_state=0)
        km.fit(bmu_centers)
        cluster_map = dict(zip(bmu_centers.flatten(), km.labels_))
        cluster_labels = [cluster_map[b] for b in bmu_1d]

        # 记录聚类结果
        for idx, (row, cluster) in enumerate(zip(df_selected.iterrows(), cluster_labels)):
            point_id = row[1]['测量点号']
            if point_id not in point_cluster_history:
                point_cluster_history[point_id] = []
            point_cluster_history[point_id].append(cluster)

    except Exception as e:
        print(f"时间 {timestamp} SOM 聚类失败：{e}")
        continue

print(" 所有时间点的 SOM 聚类已完成")

# ====== 统计最终类别 =========
final_clusters = {}
for point_id, cluster_list in point_cluster_history.items():
    most_common_cluster = Counter(cluster_list).most_common(1)[0][0]
    final_clusters[point_id] = most_common_cluster

df_final_clusters = pd.DataFrame(final_clusters.items(), columns=['测量点号', '最终类别'])
print(df_final_clusters.head())

# ======== 可视化与储存 ========
df_plot = df[['测量点号', 'A相电压', 'A相电流']].drop_duplicates()
df_plot = df_plot.merge(df_final_clusters, on='测量点号')

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_plot, x='A相电压', y='A相电流', hue='最终类别', palette='Set2', style='最终类别')
plt.xlabel('A相电压')
plt.ylabel('A相电流')
plt.title('最终测量点聚类结果（SOM，自组织网络，96组）')

plot_path = os.path.join(output_dir, "final_cluster_plot_som.png")
plt.savefig(plot_path)
plt.show()
print(f"聚类图已保存：{plot_path}")

output_file = os.path.join(output_dir, "final_cluster_results_som.xlsx")
df_final_clusters.to_excel(output_file, index=False)
print(f" 最终聚类结果已保存：{output_file}")