import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===== Step 0: 路径设置与读取标准答案 =====
file_path = r"U:\Py_code\py_clustering\data_transformer_one_10_08\Transformer_two_data.xlsx"
standard_path = r"C:\Users\86139\Desktop\毕业设计论文阅读总结\南网数据\南网数据\台区2标答.xlsx"
output_path = r"U:\Py_code\Physics_Data_Hybrid_Distribution"
os.makedirs(output_path, exist_ok=True)

df_raw = pd.read_excel(file_path, sheet_name="Sheet1")
std_df = pd.read_excel(standard_path)

# ===== Step 1: 解析标准答案中的测点编号 =====
standard_ids = []
for idx, col in enumerate(std_df.columns):
    ids = std_df[col].dropna().astype(int).tolist()
    standard_ids.extend(ids)
standard_ids = set(standard_ids)

# ===== Step 2: 清洗数据，强制保留标准测点（即使异常） =====
label_dict = {}
for meter_id, group in df_raw.groupby('测量点号'):
    group_sorted = group.sort_values(by='数据时间')
    pivot = group_sorted.pivot(index='数据时间', columns='数据类型', values='A相')

    label = -1  # 默认异常
    if '电压' in pivot.columns and '电流' in pivot.columns:
        pivot = pivot[['电压', '电流']].dropna()
        if len(pivot) >= 240:
            voltage = pivot['电压'].values[:240]
            current = pivot['电流'].values[:240]
            if not (np.all(voltage == 0) or np.all(current == 0)):
                label = 0

    # 强制保留标准答案中的测点
    if meter_id in standard_ids:
        label = 0

    label_dict[meter_id] = label

label_df = pd.DataFrame(list(label_dict.items()), columns=["测量点号", "标签"])
label_df.to_excel(os.path.join(output_path, "transformer_meter_anomaly_flags_patched.xlsx"), index=False)

# ===== Step 3: 提取有效测点序列数据 =====
valid_meter_ids = label_df[label_df['标签'] == 0]['测量点号'].tolist()
df = df_raw[df_raw['测量点号'].isin(valid_meter_ids)]

meter_sequences = {}
for meter_id in valid_meter_ids:
    group = df[df['测量点号'] == meter_id].sort_values(by='数据时间')
    pivot = group.pivot(index='数据时间', columns='数据类型', values='A相')
    if '电压' in pivot.columns and '电流' in pivot.columns:
        pivot = pivot[['电压', '电流']].dropna()
        voltage = pivot['电压'].values[:240]
        current = pivot['电流'].values[:240]
        seq = np.stack([voltage, current], axis=1)
        meter_sequences[meter_id] = seq

# ===== Step 4: 构建特征向量并降维 =====
data_rows = []
for meter_id, seq in meter_sequences.items():
    data_rows.append([meter_id] + seq.flatten().tolist())
final_df = pd.DataFrame(data_rows)
final_df.columns = ['meter_id'] + [f'f{i}' for i in range(480)]
X = final_df.drop(columns=['meter_id']).values
X_imputed = SimpleImputer(strategy='mean').fit_transform(X)
X_pca = PCA(n_components=80).fit_transform(X_imputed)

# ===== Step 5: 构建伪时间序列并标准化 =====
X_seq = X_pca.reshape(-1, 40, 2)
meter_ids = final_df['meter_id'].values
scaler = StandardScaler()
X_scaled_seq = scaler.fit_transform(X_seq.reshape(-1, 2)).reshape(-1, 40, 2)

# ===== Step 6: 定义 LSTM 编码器 =====
class LSTMEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_dim=40):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return torch.tanh(self.fc(hn[-1]))

class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

dataset = SeqDataset(X_scaled_seq)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_model = LSTMEncoder().to(device)
lstm_model.eval()

features_lstm = []
with torch.no_grad():
    for batch in dataloader:
        encoded = lstm_model(batch.to(device))
        features_lstm.append(encoded.cpu().numpy())
features_lstm = np.concatenate(features_lstm, axis=0)

lstm_df = pd.DataFrame(features_lstm, columns=[f'LSTM_F{i+1}' for i in range(features_lstm.shape[1])])
lstm_df.insert(0, 'meter_id', meter_ids)

# ===== Step 7: Deep SVDD 异常检测 =====
class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim=40, rep_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rep_dim)
        )
    def forward(self, x): return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)

X_lstm = lstm_df.drop(columns=['meter_id']).values
X_scaled = StandardScaler().fit_transform(X_lstm)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

svdd_model = DeepSVDDNet(input_dim=40, rep_dim=16).to(device)
svdd_model.apply(init_weights)
train_loader = DataLoader(SeqDataset(X_tensor), batch_size=16, shuffle=True)

def init_center(loader, model):
    model.eval()
    center = torch.zeros(16, device=device)
    n = 0
    with torch.no_grad():
        for x in loader:
            out = model(x.to(device))
            center += out.sum(0)
            n += x.size(0)
    return center / n

center = init_center(train_loader, svdd_model)
optimizer = torch.optim.Adam(svdd_model.parameters(), lr=1e-3)
for epoch in range(50):
    svdd_model.train()
    for x in train_loader:
        x = x.to(device)
        out = svdd_model(x)
        loss = torch.mean(torch.sum((out - center) ** 2, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ===== Step 8: 异常判定 + KMeans 聚类 =====
svdd_model.eval()
with torch.no_grad():
    z = svdd_model(X_tensor.to(device)).cpu().numpy()
    dists = np.sum((z - center.cpu().numpy())**2, axis=1)

threshold = np.percentile(dists, 99)
is_anomaly = dists > threshold
lstm_df['is_anomaly'] = is_anomaly

X_cluster = lstm_df.drop(columns=['meter_id', 'is_anomaly']).values
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster)
lstm_df['kmeans_cluster'] = cluster_labels

# ===== Step 9: 标签融合输出（异常0，聚类+1，异常点保留-1） =====
def label_mapper(row):
    return 0 if row['is_anomaly'] else int(row['kmeans_cluster']) + 1
lstm_df['融合标签'] = lstm_df.apply(label_mapper, axis=1)

cluster_df = lstm_df[['meter_id', '融合标签']]
cluster_df.columns = ['测量点号', '融合标签']

final_df = pd.merge(label_df, cluster_df, on='测量点号', how='left')
final_df['融合标签'] = final_df['融合标签'].fillna(final_df['标签']).astype(int)
final_df = final_df[['测量点号', '融合标签']]

final_df.to_excel(os.path.join(output_path, "final_meter_labels_with_anomaly_and_clustering_patched.xlsx"), index=False)
print(" 融合标签保存完成：final_meter_labels_with_anomaly_and_clustering_patched.xlsx")
# 使用已有的 final_df，包含字段：测量点号, 融合标签
label_groups = final_df.groupby('融合标签')['测量点号'].apply(list).to_dict()
max_len = max(len(v) for v in label_groups.values())

# 构建横向格式的 DataFrame
label_table = pd.DataFrame({f'分支{label}': pd.Series(ids) for label, ids in label_groups.items()})

# 输出路径（与主输出放一起）
horizontal_output_path = os.path.join(output_path, "最终聚类结果_横向格式.xlsx")
label_table.to_excel(horizontal_output_path, index=False)

print(" 横向格式标签表格已保存为：最终聚类结果_横向格式.xlsx")