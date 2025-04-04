import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 基准文件路径
base_file = r'U:\Py_code\py_clustering\台区1标答.xlsx'

# 需要比较的聚类结果文件列表
compare_files = [
    r'U:\Py_code\py_clustering\py_clustering_DBSCAN\results\processed_final_result_dbscan.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_DPC\results\processed_final_result_dpc.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_KMeans\results\processed_final_result_kmeans.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_Mean_Shift\results\processed_final_result_meanshift.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_SOM\results\processed_final_result_som.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_Spectial_Clustering\results\processed_final_result_spectral.xlsx',
    r'U:\Py_code\py_clustering\py_clusterfing_HC\results\processed_final_result_hc.xlsx',
]

# 读取基准表格
df_base = pd.read_excel(base_file)

# 定义相似度计算函数
def column_similarity_and_weight(col1, col2):
    set1 = set(col1.dropna().unique())
    set2 = set(col2.dropna().unique())
    weight = len(set1 | set2)
    if not set1 and not set2:
        return 1.0, 1  # 两边都空，视为相似度1，权重1
    similarity = len(set1 & set2) / weight if weight else 1.0
    return similarity, weight

# 存储最终结果向量
result_vectors = []

print("聚类方法\t\t分支01\t分支02\t分支03\t加权相似度")
print("-------------------------------------------------------------")

# 遍历每个聚类结果文件
for file_path in compare_files:
    df_compare = pd.read_excel(file_path)
    filename = os.path.basename(file_path)

    similarities = []
    weights = []

    common_columns = df_base.columns.intersection(df_compare.columns)
    for col in common_columns:
        sim, w = column_similarity_and_weight(df_base[col], df_compare[col])
        similarities.append(round(sim, 4))
        weights.append(w)

    # 加权平均计算总体相似度
    weighted_total = sum(sim * w for sim, w in zip(similarities, weights))
    total_weight = sum(weights)
    overall_similarity = round(weighted_total / total_weight, 4) if total_weight else 1.0

    vector = similarities + [overall_similarity]
    result_vectors.append((filename, vector))

    # 打印结果
    sim_str = "\t".join([f"{v:.2f}" for v in vector])
    print(f"{filename:<30}\t{sim_str}")

# 提取算法名（去掉文件前缀）
method_names = [os.path.splitext(name)[0].replace("processed_final_result_", "") for name, vec in result_vectors]
similarity_scores = [vec[-1] for _, vec in result_vectors]  # 提取加权相似度

# 创建柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(method_names, similarity_scores, color='skyblue')

# 给每个柱子标上数值
for bar, score in zip(bars, similarity_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{score:.2f}", ha='center', fontsize=10)

plt.title("各聚类算法与标答的加权相似度比较", fontsize=14)
plt.xlabel("聚类算法")
plt.ylabel("加权相似度")
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图像（可选）
output_plot_path = r'U:\Py_code\py_clustering\相似度对比图.png'
plt.savefig(output_plot_path)
print(f"\n图像已保存到：{output_plot_path}")

# 显示图像
plt.show()