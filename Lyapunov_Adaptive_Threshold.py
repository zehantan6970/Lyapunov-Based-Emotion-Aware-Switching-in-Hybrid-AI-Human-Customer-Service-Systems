import pandas as pd
import numpy as np

# 加载你的客服系统对话数据集
df = pd.read_csv("客服系统对话数据.csv")

# Step 1: 数据清洗和预处理
df = df[df['speaker'] == 'user'].copy()
df = df.dropna(subset=['pleasure', 'arousal', 'dominance'])
df[['pleasure', 'arousal', 'dominance']] = df[['pleasure', 'arousal', 'dominance']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['pleasure', 'arousal', 'dominance'])

# Step 2: 拟合A矩阵
A_records = []
for user_id, group in df.groupby('user_id'):
    group_sorted = group.sort_values('turn')
    if len(group_sorted) >= 6:
        X = group_sorted[['pleasure', 'arousal', 'dominance']].values[:5].T
        Y = group_sorted[['pleasure', 'arousal', 'dominance']].values[1:6].T
        try:
            A = Y @ X.T @ np.linalg.inv(X @ X.T)
        except np.linalg.LinAlgError:
            continue
        latest_state = group_sorted[['pleasure', 'arousal', 'dominance']].values[5]
        A_records.append((user_id, A, latest_state, group_sorted.iloc[5]))

# Step 3: 计算V_xt并判断是否切换
updated_rows = []
Q = 0.001 * np.identity(3)

V_list = []
P_dict = {}
for user_id, A_i, x_t, row in A_records:
    try:
        P = np.linalg.solve(np.eye(9) - np.kron(A_i.T @ A_i, np.eye(3)), Q.flatten()).reshape(3, 3)
    except np.linalg.LinAlgError:
        continue
    V_xt = x_t.T @ P @ x_t
    V_list.append(V_xt)
    P_dict[user_id] = (P, x_t, row)

threshold = np.mean(V_list)

for user_id, (P, x_t, row) in P_dict.items():
    V_xt = x_t.T @ P @ x_t
    should_switch = V_xt > threshold
    updated_rows.append({
        'user_id': user_id,
        'turn': row['turn'],
        'V_xt': V_xt,
        'threshold': threshold,
        'should_switch': should_switch,
        'personality_type': row['personality_type'],
        'pleasure': x_t[0],
        'arousal': x_t[1],
        'dominance': x_t[2],
        'agent_type': row['agent_type'],
        'text': row['text'],
        'emotion_label': row['emotion_label'],
        'scenario': row.get('scenario', '')
    })

updated_df = pd.DataFrame(updated_rows)
updated_df['speaker'] = updated_df['agent_type'].apply(lambda x: 'user' if x in ['AI', 'Human'] else ('agent' if x == 'Switched' else 'unknown'))

# 调整列顺序
column_order = [
    'user_id', 'personality_type', 'turn', 'agent_type', 'speaker',
    'emotion_label', 'text', 'pleasure', 'arousal', 'dominance',
    'scenario', 'V_xt', 'threshold', 'should_switch'
]
column_order = [col for col in column_order if col in updated_df.columns]
updated_df = updated_df[column_order]

# 保存文件  用于策略 A_lyapunov 的标签计算与判定依据生成 精确计算 V_xt  包含 threshold, should_switch 字段，是 A_lyapunov 策略的核心依据
output_path = 'Lyapunov_A_Threshold_Generated.csv'
updated_df.to_csv(output_path, index=False)


