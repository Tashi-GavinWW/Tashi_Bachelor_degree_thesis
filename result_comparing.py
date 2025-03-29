import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


def compare_tables(df1, df2):
    # 取公共列
    common_cols = df1.columns.intersection(df2.columns)
    df1_common = df1[common_cols].reset_index(drop=True)
    df2_common = df2[common_cols].reset_index(drop=True)

    # 对齐行数
    max_rows = max(len(df1_common), len(df2_common))
    df1_common = df1_common.reindex(range(max_rows))
    df2_common = df2_common.reindex(range(max_rows))

    # 填空
    df1_vals = df1_common.fillna("").values
    df2_vals = df2_common.fillna("").values

    # 比较
    comparison = df1_vals == df2_vals
    total = comparison.size
    match = np.sum(comparison)
    similarity = match / total

    # 记录差异
    diff_indices = np.where(comparison == False)
    diffs = []
    for row, col in zip(*diff_indices):
        colname = common_cols[col]
        val1 = df1_common.iloc[row, col]
        val2 = df2_common.iloc[row, col]
        diffs.append((row + 1, colname, val1, val2))

    return similarity, diffs, common_cols, df1_common, df2_common

# 设置文件路径
base_file = r'U:\Py_code\py_clustering\台区1标答.xlsx'
compare_files = [
    r'U:\Py_code\py_clustering\py_clustering_DBSCAN\results\processed_final_result_dbscan.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_DPC\results\processed_final_result_dpc.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_KMeans\results\processed_final_result_kmeans.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_Mean_Shift\results\processed_final_result_meanshift.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_SOM\results\processed_final_result_som.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_Spectial_Clustering\results\processed_final_result_spectral.xlsx',
    r'U:\Py_code\py_clustering\py_clustering_HC\results\processed_final_result_hc.xlsx',
]

# 读取基准表
df_base = pd.read_excel(base_file)

# 对比每个表格
for compare_file in compare_files:
    df_compare = pd.read_excel(compare_file)
    filename = os.path.basename(compare_file)

    print(f"\n 正在对比：{filename}")
    similarity, diffs, cols, df1_out, df2_out = compare_tables(df_base, df_compare)
    print(f"相似度：{similarity:.2%}，不同单元格数量：{len(diffs)}")

    # 打印部分差异
    for row, col, val1, val2 in diffs[:5]:  # 只看前5条
        print(f" 第{row}行 列[{col}]：base = {val1}, compare = {val2}")
    if len(diffs) > 5:
        print(f"... 还有 {len(diffs) - 5} 条差异未展示")

    # 导出差异对比文件
    output_name = f"对比结果_{os.path.splitext(filename)[0]}.xlsx"
    with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
        df1_out.to_excel(writer, sheet_name='基准表', index=False)
        df2_out.to_excel(writer, sheet_name='对比表', index=False)

        # 生成差异高亮表
        highlight = pd.DataFrame("", index=range(df1_out.shape[0]), columns=cols)
        for row, col, _, _ in diffs:
            highlight.at[row - 1, col] = "不同"
        highlight.to_excel(writer, sheet_name='差异位置', index=False)

    print(f"差异对比文件已保存为：{output_name}")

# 横轴方法名称
methods = ["DBSCAN", "DPC", "KMeans", "MeanShift", "SOM", "Spectral Clustering", "HC"]

# 聚类评估指标（请替换为你的真实数据）
scores = [0.7078, 0.6255, 0.6667, 0.6941, 0.4979, 0.9518, 0.4979]  # ✅ 修复了最后一项

# 可视化（柱状图）
plt.figure(figsize=(10, 7))
bars = plt.bar(methods, scores, color='skyblue')

# 每个柱子上标注数值
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01,
             f"{score:.2f}", ha='center', fontsize=10)

plt.title("不同聚类方法的评估指标对比", fontsize=14)
plt.xlabel("聚类算法")
plt.ylabel("相似度 / 准确率（示例）")
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图像
output_dir = r"U:\Py_code\py_clustering"
plot_path = os.path.join(output_dir, "result_comparing.png")
plt.savefig(plot_path)
print(f"图像已保存：{plot_path}")

# 显示图像
plt.show()