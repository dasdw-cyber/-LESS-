import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib

try:
    # 尝试使用 PyCharm 兼容后端，或让其自动选择
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ======================= 配置区域 =======================
# 🌟 输入数据路径 (精准对接上一步 27 参生成的最新数据集)
DATA_PATH = r"E:\叶绿素反演-李文娟老师论文\适配LESS的prosail数据\9波段\PROSAIL_LUT_50k_18Params.csv"

# 🌟 模型保存目录 (更新为 27 参存放目录)
MODEL_SAVE_DIR = r"E:\叶绿素反演-李文娟老师论文\适配LESS的model\9波段"

# 目标变量列表 (对应 CSV 中的列名，当前为叶片叶绿素含量)
TARGETS = ['cab','lai','n']

# 每个变量训练的集成模型数量 (论文要求 10 个)
N_ENSEMBLE = 10


# =======================================================

def train_inversion_models():
    # 1. 准备工作
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误: 找不到训练数据文件: {DATA_PATH}")
        print("请先运行 PROSAIL 数据生成脚本。")
        return

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print(f"📥 正在加载数据: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # 检查数据完整性，删除由拟合失败导致的空值行
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"⚠️ 已移除 {initial_len - len(df)} 行含有空值的数据。")

    print(f"📊 有效训练样本数: {len(df)}")

    # 2. 提取特征 (X)
    # 筛选列名包含 '_k' 的列 (9波段 * 3系数 = 27个核系数)
    feature_cols = [c for c in df.columns if '_k' in c]

    # 🌟 修改点：校验预期维度从 18 变为 27
    if len(feature_cols) != 27:
        print(f"⚠️ 警告: 识别到的特征列数量为 {len(feature_cols)}，预期为 27。")
        print(f"列名: {feature_cols}")

    X = df[feature_cols].values

    # 💥💥💥 核心新增 1：提取物理边界并保存为 JSON 💥💥💥
    print("🔍 正在提取 PROSAIL 核系数物理边界...")
    bounds_dict = {}
    for col in feature_cols:
        # 必须转换为原生的 float 类型，否则 json 无法序列化 numpy 格式
        bounds_dict[col] = [float(df[col].min()), float(df[col].max())]

    bounds_path = os.path.join(MODEL_SAVE_DIR, 'kernel_bounds.json')
    with open(bounds_path, 'w') as f:
        json.dump(bounds_dict, f, indent=4)
    print(f"✅ 物理边界已轻量化保存至: {bounds_path}")

    # 3. 数据标准化 (Standardization)
    print("📏 正在进行数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_path = os.path.join(MODEL_SAVE_DIR, 'scaler_inversion.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"💾 标准化器已保存: {scaler_path}\n")

    # 4. 循环训练每个目标变量
    for target in TARGETS:
        if target not in df.columns:
            print(f"⚠️ 跳过 {target}: 列名不存在")
            continue

        print(f"{'=' * 15} 🚀 开始训练 {target.upper()} 的集成模型 (N={N_ENSEMBLE}) {'=' * 15}")
        y = df[target].values

        # 划分训练集和验证集 (80% / 20%)
        # 注意：使用相同的 random_state=42，确保 10 个模型用的训练/验证切分是同一批数据
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        ensemble_predictions = []

        # 💥💥💥 核心新增 2：10 个网络循环训练与持久化 💥💥💥
        for i in range(N_ENSEMBLE):
            print(f"  ⏳ 正在训练 {target.upper()} 模型 {i + 1}/{N_ENSEMBLE} ...", end="")

            # 定义网络结构 (根据论文描述: 1个隐藏层，5个神经元，使用 tanh 激活函数)
            # 引入 random_state=i，确保每个网络的初始权重不同，从而达到集成学习的效果
            model = MLPRegressor(hidden_layer_sizes=(5,),
                                 activation='tanh',
                                 solver='lbfgs',  # 换用 lbfgs，极大地增强小网络的收敛能力
                                 max_iter=3000,   # 增加最大迭代次数，给 lbfgs 足够的收敛空间
                                 early_stopping=False,  # 强制关闭早停，逼迫网络去寻找特征规律
                                 random_state=i)

            # 训练
            model.fit(X_train, y_train)

            # 评估验证集
            y_pred = model.predict(X_val)
            # 简单的物理约束截断 (防止输出负数，Cab含量不可能小于0)
            y_pred = np.maximum(y_pred, 0)

            ensemble_predictions.append(y_pred)

            # 保存单体模型
            model_path = os.path.join(MODEL_SAVE_DIR, f'bp_ann_model_{target}_{i}.pkl')
            joblib.dump(model, model_path)
            print(f" 完成！")

        # 💥💥💥 核心新增 3：计算 10 个模型的中位数预测结果并评估 💥💥💥
        print(f"\n  📈 计算 {target.upper()} 集成模型 (中位数) 的最终性能...")
        ensemble_predictions = np.column_stack(ensemble_predictions)  # 维度: (n_samples, 10)
        median_pred = np.median(ensemble_predictions, axis=1)  # 沿模型维度取中位数

        rmse = np.sqrt(mean_squared_error(y_val, median_pred))
        r2 = r2_score(y_val, median_pred)

        print(f"  🌟 {target.upper()} 最终集成性能指标: RMSE = {rmse:.4f}, R2 = {r2:.4f}\n")

        # 绘制验证图 (1:1 plot，基于中位数)
        plot_validation(y_val, median_pred, target, r2, rmse)


def plot_validation(y_true, y_pred, target_name, r2, rmse):
    """绘制验证散点图"""
    plt.figure(figsize=(6, 6))

    # 绘制散点 (为了性能，如果是大量数据，只画一部分或用 hexbin)
    if len(y_true) > 5000:
        # 更换为高对比度的科学色带 'viridis'。备选方案: 'turbo', 'jet', 'plasma'
        # 如果数据非常集中导致其他区域看不清，可以给 hexbin 加上参数 bins='log'
        hb = plt.hexbin(y_true, y_pred, gridsize=50, cmap='viridis', mincnt=1, edgecolors='none')
        cb = plt.colorbar(hb, label='Point Density (Count)')
    else:
        # 为普通散点增加白色边缘，增强重叠时的层次感
        plt.scatter(y_true, y_pred, alpha=0.7, s=20, c='#1f77b4', edgecolors='white', linewidth=0.5)

    # 1:1 线改成醒目的红色虚线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')

    plt.xlabel(f'True {target_name.upper()}')
    plt.ylabel(f'Ensemble Median {target_name.upper()}')
    plt.title(f'{target_name.upper()} Ensemble Validation (27 Kernel Params)\n$R^2$={r2:.3f}, RMSE={rmse:.3f}')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(MODEL_SAVE_DIR, f'validation_plot_{target_name}_ensemble.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 避免循环训练时产生过多弹窗影响自动化
    print(f"  🖼️ 集成验证图表已保存: {save_path}\n")

if __name__ == "__main__":
    train_inversion_models()