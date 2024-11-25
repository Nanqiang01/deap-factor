import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from deap_factor import DeapFactor
from fitness import fitness_rankic
from primitive_set import set_pset

# 读取数据
data_dir = "data/"
field_list = [
    "adjopen",
    "adjhigh",
    "adjlow",
    "adjclose",
    "preclose",
    "volume",
    "pct_change",
    "vwap",
]
label = ["vwap"]
X = DataLoader(data_dir, field_list)
y = DataLoader(data_dir, label)

# 切分数据集
X_train = None
X_test = None
y_train = None
y_test = None
# 切分X
for field in field_list:
    df = getattr(X, field)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=False
    )
    setattr(X_train, field, df_train)
    setattr(X_test, field, df_test)
# 切分y
df = getattr(y, label[0]).shift(1)  # 为了预测未来的vwap，所以将vwap shift一期
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
setattr(y_train, label[0], df_train)
setattr(y_test, label[0], df_test)
# 删除无用变量
del X, y, df, df_train, df_test
gc.collect()

# 设置算子
pset = set_pset(field_list)

# 初始化deap因子引擎
engine = DeapFactor(pset, fitness_rankic)
engine.run()
