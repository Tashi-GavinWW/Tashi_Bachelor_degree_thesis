import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# ==========  读取 & 预处理数据 ==========
file_path = r"U:\Py_code\py_clustering\data_transformer_one_10_08\transformer_01_10_08.xlsx"

# 设定存储路径
output_dir = r"U:\Py_code\py_clustering\results"
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

df = pd.read_excel(file_path)

# 去除缺失值
df = df.dropna()

# 确保时间列是 datetime 格式
df['数据时间'] = pd.to_datetime(df['数据时间'])

# 按照时间进行分组
grouped_dict = {timestamp: group.copy() for timestamp, group in df.groupby('数据时间')}
print(f"共有 {len(grouped_dict)} 组数据")  # 确保是 96 组

# 存储每个测量点号的聚类结果
point_cluster_history = {}


# ==========  遍历 96 组数据进行 K-Means（K=3） ==========
for timestamp, df_selected in grouped_dict.items():
    X = df_selected[['A相电压', 'A相电流']].values  # 提取特征
    
    if len(X) < 3:  # 确保数据点足够进行聚类
        print(f"跳过 {timestamp}，数据点不足")
        continue
    
    # K-Means 聚类（K=3）
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # 遍历测量点号 & 聚类标签
    for idx, (row, cluster) in enumerate(zip(df_selected.iterrows(), cluster_labels)):
        point_id = row[1]['测量点号']  # 获取测量点号

        if point_id not in point_cluster_history:
            point_cluster_history[point_id] = []

        point_cluster_history[point_id].append(cluster)

print("所有时间点的聚类已完成！")


# ========== 统计每个测量点号的最终类别 ==========
final_clusters = {}

# 计算每个测量点号最常出现的类别
for point_id, cluster_list in point_cluster_history.items():
    most_common_cluster = Counter(cluster_list).most_common(1)[0][0]  # 找到最常见的类别
    final_clusters[point_id] = most_common_cluster

# 转换为 DataFrame
df_final_clusters = pd.DataFrame(final_clusters.items(), columns=['测量点号', '最终类别'])
print(df_final_clusters.head())  # 查看最终结果


# ==========  可视化最终聚类结果（仅最终结果） ==========
# 仅选取唯一的测量点号、电压和电流
df_plot = df[['测量点号', 'A相电压', 'A相电流']].drop_duplicates()
df_plot = df_plot.merge(df_final_clusters, on='测量点号')

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'        # 黑体 
matplotlib.rcParams['axes.unicode_minus'] = False    # 让负号正常显示

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_plot, x='A相电压', y='A相电流', hue='最终类别', palette='viridis')
plt.xlabel('A相电压')
plt.ylabel('A相电流')
plt.title('最终测量点聚类结果（基于 96 组数据）')

# 存储可视化图片
plot_path = os.path.join(output_dir, "final_cluster_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"聚类可视化结果已保存至: {plot_path}")

# ========== 存储最终聚类结果 ==========
output_file = os.path.join(output_dir, "final_cluster_results.xlsx")
df_final_clusters.to_excel(output_file, index=False)
print(f"最终聚类结果已保存至: {output_file}")
