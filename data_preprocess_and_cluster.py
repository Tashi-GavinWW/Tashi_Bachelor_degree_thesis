# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_excel('C:\\Users\\zylpa\\Dropbox\\系统院\\研究生科研\\配电网\\南网丝路电工学会'+\
                   '\\官方提供\\揭榜挂帅-拓扑识别-参考台区数据.xls',sheet_name=list(range(5))[::-1])

keys = df[0]['测量点地址']
values = df[0]['测量点号']
pairs = zip(keys, values)
my_dict = dict(pairs)
df_topo = pd.read_excel('C:\\Users\\zylpa\\Dropbox\\系统院\\研究生科研\\配电网\\南网丝路电工学会'+\
                   '\\官方提供\\揭榜挂帅-拓扑识别-参考台区数据.xls',sheet_name='台区拓扑')

ans1 = []
for key in df_topo['分支01']:
    if not np.isnan(key):
        ans1.append(my_dict[key])

ans2 = []
for key in df_topo['分支02']:
    if not np.isnan(key):
        ans2.append(my_dict[key])
        
    
da=[]
for sheetname, data in df.items():
    da.append(data)
alldata = pd.concat(da,axis=0)
# print(alldata)


groups = alldata.groupby('测量点号') #按户拆表
# 将分组后的数据存储在字典中
grouped_dict = {name: group for name, group in groups}

# userIds = grouped_dict[0][grouped_dict[k]['数据类型']=='电压']['测量点号']


# grouped_dict.to_csv('C:\\Users\\zylpa\\Dropbox\\系统院\\研究生科研\\配电网\\南网丝路电工学会'+\
#                    '\\官方提供\\数据.xls')

## 先看电压
# 关于三相电压如何处理： 法1：只比较A相 （因为所给的数据都有不为0的A相）
volt_data = list()
for k in grouped_dict:

    # 创建 Excel 写入器  
    # excel_file = 'C:\\Users\\zylpa\\Dropbox\\系统院\\研究生科研\\配电网\\南网丝路电工学会'+\
    #                '\\官方提供\\用户'+str(k)+'数据.xls'
    # with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:  
    #     grouped_dict[k][grouped_dict[k]['数据类型']=='电压'].to_excel(writer,sheet_name='电压')
    #     grouped_dict[k][grouped_dict[k]['数据类型']=='电流'].to_excel(writer,sheet_name='电流')
   
    volt_data.append((k, grouped_dict[k][grouped_dict[k]['数据类型']=='电压']['A相'].fillna(method='ffill')))

# fill nan

condition = df.isnull()
nan_values = df[condition]
print(nan_values)




# 关于三相电压如何处理： 法2：三相单相分开，另行比较

from sklearn.cluster import KMeans

# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])
X = [e.to_list() for _,e in volt_data if len(e)==480]
X_id = [k for k,e in volt_data if len(e)==480]

# X = np.array(X)

np.argwhere(np.isnan(X))

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
# kmeans.cluster_centers_


ans_cluster0 = [X_id[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 0]
ans_cluster1 = [X_id[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 1]

# 用谱聚类结果类似/几乎一样
# from sklearn.cluster import SpectralClustering
# clustering = SpectralClustering(n_clusters=2,
#         assign_labels='discretize',
#         random_state=0).fit(X)

# clustering.labels_

# ans_cluster0_spec = [X_id[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 0]
# ans_cluster1_spec = [X_id[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 1]




## -------------------------------再看电流 （数据特征更多变点）
