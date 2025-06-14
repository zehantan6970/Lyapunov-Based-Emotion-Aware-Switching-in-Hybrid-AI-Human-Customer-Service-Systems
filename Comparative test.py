import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 读取数据
fixed_df = pd.read_csv("fixed_emotion_dataset_20250526_205726.csv")
test_df = pd.read_csv("Updated_Lyapunov_Dataset_with_Speaker.csv")

# 数据清洗
fixed_df = fixed_df[(fixed_df['speaker'] == 'user')]
fixed_df = fixed_df.dropna(subset=['pleasure', 'arousal', 'dominance'])
fixed_df[['pleasure', 'arousal', 'dominance']] = fixed_df[['pleasure', 'arousal', 'dominance']].apply(pd.to_numeric, errors='coerce')
fixed_df = fixed_df.dropna(subset=['pleasure', 'arousal', 'dominance'])

# 为每个用户拟合 A_i, P_i, c_i
user_models = {}
Q = 0.001 * np.identity(3)

for user_id, group in fixed_df.groupby('user_id'):
    group_sorted = group.sort_values('turn')
    if len(group_sorted) >= 6:
        X = group_sorted[['pleasure', 'arousal', 'dominance']].values[:5].T
        Y = group_sorted[['pleasure', 'arousal', 'dominance']].values[1:6].T
        try:
            A = Y @ X.T @ np.linalg.inv(X @ X.T)
            P = np.linalg.solve(np.eye(9) - np.kron(A.T @ A, np.eye(3)), Q.flatten()).reshape(3, 3)
            V_vals = [X[:, i].T @ P @ X[:, i] for i in range(5)]
            c = np.mean(V_vals)
            user_models[user_id] = {'A': A, 'P': P, 'c': c}
        except np.linalg.LinAlgError:
            continue

# 计算策略指标函数
def compute_metrics(y_true, y_pred, V_vals):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_V = np.mean(V_vals)
    switch_rate = np.mean(y_pred)
    false_switch_rate = np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 0)) / len(y_true)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Avg. V(x_T)": avg_V,
        "切换率": switch_rate,
        "误切换率": false_switch_rate
    }

# 构造对比数据
oracle, strategy_A, strategy_B, strategy_C, strategy_D, V_all = [], [], [], [], [], []

for _, row in test_df.iterrows():
    uid = row['user_id']
    xt = np.array([row['pleasure'], row['arousal'], row['dominance']])
    if uid in user_models:
        model = user_models[uid]
        V_xt = xt.T @ model['P'] @ xt
        V_all.append(V_xt)
        oracle.append(row['should_switch'])
        strategy_A.append(V_xt > model['c'])
        strategy_B.append(V_xt > 3.0)
        strategy_C.append(row['arousal'] > 0.7)
        strategy_D.append(False)

# 汇总结果
results = {
    "Oracle": {
        "Precision": 1.0,
        "Recall": 1.0,
        "F1": 1.0,
        "Avg. V(x_T)": np.mean(V_all),
        "切换率": np.mean(oracle),
        "误切换率": 0.0
    },
    "A_lyapunov": compute_metrics(oracle, strategy_A, V_all),
    "B_fixed": compute_metrics(oracle, strategy_B, V_all),
    "C_threshold": compute_metrics(oracle, strategy_C, V_all),
    "D_none": compute_metrics(oracle, strategy_D, V_all)
}

results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "策略"})
# 添加打印结果到控制台的方式
print("策略对比结果：")
print(results_df.to_string(index=False))
