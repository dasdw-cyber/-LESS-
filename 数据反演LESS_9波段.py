import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass
import matplotlib.pyplot as plt

# 忽略计算过程中的某些常见警告
warnings.filterwarnings('ignore')

# === 解决 matplotlib 中文显示问题 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 配置区域 =======================

# 1. 批量数据目录与类型配置 (请根据实际情况确认 9波段 LESS 数据的路径)
BASE_DATA_DIR = r"E:\叶绿素反演-李文娟老师论文\LESS观测数据获取\9波段_2"
CANOPY_TYPES = ["Erectophile", "Planophile", "Spherical"]

MODEL_DIR = r"E:\叶绿素反演-李文娟老师论文\适配LESS的model\9波段"

# 3. 核心联动：第一步 6S 生成的 9 波段多项式系数 CSV
COEFFS_CSV_FILE = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\基于6s的公式系数\9_bands_coefficients.csv"

# 4. 输出结果路径 (所有 9 波段结果将保存在该目录下)
OUTPUT_DIR = r"E:\叶绿素反演-李文娟老师论文\LESS数据反演result\9波段\LESS_Inversion_Results_9Bands"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5. 仪器观测角配置 (LESS 模拟时设置的观测角度)
VZA_FIXED = 45.0
SENSOR_AZIMUTH = 90.0

# ======================= 动态加载 9波段与系数 =======================
print(f"正在读取 6S 动态多项式系数: {COEFFS_CSV_FILE}")
if not os.path.exists(COEFFS_CSV_FILE):
    raise FileNotFoundError("❌ 错误：找不到 9 波段多项式系数文件，请检查路径！")

coeff_df = pd.read_csv(COEFFS_CSV_FILE)
BAND_NAMES = coeff_df['Band'].tolist()
TARGET_BANDS = [int(name.replace('nm', '')) for name in BAND_NAMES]

F_LAMBDA_COEFFS = {}
for _, row in coeff_df.iterrows():
    F_LAMBDA_COEFFS[row['Band']] = [
        row['x^5'], row['x^4'], row['x^3'], row['x^2'], row['x^1'], row['Intercept']
    ]
n_bands = len(TARGET_BANDS)
print(f"✅ 成功对接 6S 系数，波段数量: {n_bands}，包含波段: {TARGET_BANDS}")


# ======================= 核心函数库 =======================

def roujean_k_vol(sza, vza, phi):
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    cos_xi = np.cos(sza_r) * np.cos(vza_r) + np.sin(sza_r) * np.sin(vza_r) * np.cos(phi_r)
    phase = np.arccos(np.clip(cos_xi, -1.0, 1.0))
    term = (np.pi / 2.0 - phase) * np.cos(phase) + np.sin(phase)
    k_vol = (4.0 / (3.0 * np.pi)) * (term / (np.cos(sza_r) + np.cos(vza_r))) - (1.0 / 3.0)
    return k_vol


def roujean_k_geo(sza, vza, phi):
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    tan_s, tan_v = np.tan(sza_r), np.tan(vza_r)
    delta = np.sqrt(np.maximum(0, tan_s ** 2 + tan_v ** 2 - 2.0 * tan_s * tan_v * np.cos(phi_r)))
    term1 = (np.pi - phi_r) * np.cos(phi_r) + np.sin(phi_r)
    k_geo = (1.0 / (2.0 * np.pi)) * term1 * tan_s * tan_v - (1.0 / np.pi) * (tan_s + tan_v + delta)
    return k_geo


def integrate_diffuse_kernel_value():
    sza_range = np.linspace(0, 89, 90)
    phi_range = np.linspace(0, 359, 36)
    k_vol_sum, k_geo_sum, weight_sum = 0, 0, 0
    for sza in sza_range:
        weight_sza = np.sin(np.radians(sza)) * np.cos(np.radians(sza))
        for phi in phi_range:
            k_vol_sum += roujean_k_vol(sza, VZA_FIXED, phi) * weight_sza
            k_geo_sum += roujean_k_geo(sza, VZA_FIXED, phi) * weight_sza
            weight_sum += weight_sza
    return k_vol_sum / weight_sum, k_geo_sum / weight_sum


K_VOL_DIFF, K_GEO_DIFF = integrate_diffuse_kernel_value()


def calculate_dlc_kernel(sza, vza, phi, f_lambda):
    k_vol = roujean_k_vol(sza, vza, phi)
    k_geo = roujean_k_geo(sza, vza, phi)
    return (1 - f_lambda) * k_vol + f_lambda * K_VOL_DIFF, (1 - f_lambda) * k_geo + f_lambda * K_GEO_DIFF


# ======================= 单个数据集处理流程 =======================
def process_single_canopy(file_path, canopy_type):
    print(f"\n[{canopy_type}] 开始处理...")

    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    df = pd.read_csv(file_path)

    # 提取几何与 f_PAR (直接来源于 LESS 模拟)
    df['sza'] = df['Sun_Zenith']
    df['saa'] = df['Sun_Azimuth']
    df['f_par'] = df['f_PAR']

    rel_phi = np.abs(df['saa'] - SENSOR_AZIMUTH)
    df['rel_phi'] = np.where(rel_phi > 180, 360 - rel_phi, rel_phi)

    # 匹配目标波段 (动态基于 6S 系数 CSV 中的波段)
    selected_cols = {}
    for target in TARGET_BANDS:
        col_name = f"Band_{target}nm"
        if col_name in df.columns:
            selected_cols[f"{target}nm"] = col_name
        else:
            print(f"❌ 缺失关键波段列: {col_name}，当前可用列: {df.columns.tolist()[:10]}...")
            return

    # 滑动窗口准备
    df['dt_pseudo'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d_%H-%M-%S', errors='coerce')
    df['date_pd'] = df['dt_pseudo'].dt.date
    unique_dates = df['date_pd'].dropna().unique()

    results = []

    # 🌟 27 参数边界定义
    bounds_27 = [(0, None), (-0.05, None), (-0.05, None)] * n_bands

    print(f"[{canopy_type}] ⏳ 正在进行 {n_bands * 3} 参数全局核特征拟合...")
    for current_date in unique_dates:
        window_mask = (df['date_pd'] >= current_date - pd.Timedelta(days=1)) & \
                      (df['date_pd'] <= current_date + pd.Timedelta(days=1))
        window_df = df[window_mask].copy()

        # 🌟 必须保证样本数 >= 参数量 (27)
        if len(window_df) < n_bands * 3:
            continue

        sza_arr, phi_arr, f_par_arr = window_df['sza'].values, window_df['rel_phi'].values, window_df['f_par'].values
        vza_arr = np.full_like(sza_arr, VZA_FIXED)

        # 提取当前行的实测真值
        ground_truth_cab = window_df['Cab'].mean()
        ground_truth_lai = window_df['LAI'].mean()

        obs_abs_refls = np.zeros((n_bands, len(window_df)))
        k_vol_dlc_all = np.zeros((n_bands, len(window_df)))
        k_geo_dlc_all = np.zeros((n_bands, len(window_df)))

        for i, band_name in enumerate(BAND_NAMES):
            obs_abs_refls[i, :] = window_df[selected_cols[band_name]].values
            coeffs = F_LAMBDA_COEFFS[band_name]
            f_lam_arr = np.clip(np.where(f_par_arr <= 0.9, np.polyval(coeffs, f_par_arr), f_par_arr), 0, 1)
            k_vol_dlc_all[i], k_geo_dlc_all[i] = calculate_dlc_kernel(sza_arr, vza_arr, phi_arr, f_lam_arr)

        spectral_means = np.mean(obs_abs_refls, axis=0)
        obs_rel_refls = obs_abs_refls / (spectral_means + 1e-6)

        # 27 参数代价函数
        def global_cost_func(params):
            params_2d = params.reshape(n_bands, 3)
            X_mod_abs = np.zeros((n_bands, len(window_df)))
            for i in range(n_bands):
                k0, k1, k2 = params_2d[i]
                X_mod_abs[i] = k0 + k1 * k_vol_dlc_all[i] + k2 * k_geo_dlc_all[i]
            mean_X_mod = np.mean(X_mod_abs, axis=0)
            X_mod_rel = X_mod_abs / (mean_X_mod + 1e-6)
            shape_error = np.sum((obs_rel_refls - X_mod_rel) ** 2)
            scale_penalty = 100.0 * np.sum((mean_X_mod - 1) ** 2)
            return shape_error + scale_penalty

        x0 = []
        for i in range(n_bands):
            x0.extend([np.mean(obs_rel_refls[i, :]), 0.05, 0.05])

        res = minimize(global_cost_func, x0=x0, method='SLSQP', bounds=bounds_27)

        if res.success:
            day_coeffs = {'Timestamp': current_date, 'Cab_True': ground_truth_cab, 'LAI_True': ground_truth_lai}
            params_opt = res.x.reshape(n_bands, 3)
            for i, band in enumerate(BAND_NAMES):
                day_coeffs[f'{band}_k0'] = params_opt[i, 0]
                day_coeffs[f'{band}_k1'] = params_opt[i, 1]
                day_coeffs[f'{band}_k2'] = params_opt[i, 2]
            results.append(day_coeffs)

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print(f"[{canopy_type}] ❌ 拟合失败。可能是单窗口数据不足 27 条，无法求解方程。")
        return

    # 边界过滤 (兼容 9波段)
    BOUNDS_JSON_PATH = os.path.join(MODEL_DIR, "kernel_bounds.json")
    if os.path.exists(BOUNDS_JSON_PATH):
        with open(BOUNDS_JSON_PATH, 'r') as f:
            kernel_bounds = json.load(f)
        valid_mask = np.ones(len(res_df), dtype=bool)
        for col, (k_min, k_max) in kernel_bounds.items():
            if col in res_df.columns:
                valid_mask &= (res_df[col] >= k_min) & (res_df[col] <= k_max)
        filtered_count = len(res_df) - valid_mask.sum()
        if filtered_count > 0:
            print(f"[{canopy_type}] ⚠️ 触发 PROSAIL 边界保护，剔除 {filtered_count} 条越界核特征。")
        res_df = res_df[valid_mask].copy()

    # ANN 预测 (27 特征)
    print(f"[{canopy_type}] 🧠 正在执行 ANN 集成预测 (27 核特征输入)...")
    scaler_path = os.path.join(MODEL_DIR, "scaler_inversion.pkl")
    scaler = joblib.load(scaler_path)
    feature_cols = [c for c in res_df.columns if '_k' in c]

    if len(feature_cols) != n_bands * 3:
        print(f"[{canopy_type}] ❌ 警告：特征数量异常，提取到 {len(feature_cols)} 个，需要 {n_bands * 3} 个！")
        return

    X_scaled = scaler.transform(res_df[feature_cols].values)

    for target in ['lai', 'cab']:
        predictions = []
        for i in range(10):
            p = os.path.join(MODEL_DIR, f"bp_ann_model_{target}_{i}.pkl")
            if os.path.exists(p):
                predictions.append(np.maximum(joblib.load(p).predict(X_scaled), 0))
        if predictions:
            res_df[f'pred_{target}'] = np.median(np.column_stack(predictions), axis=1)

    # 结果保存
    final_csv_path = os.path.join(OUTPUT_DIR, f"LESS_Inversion_{canopy_type}_9Bands.csv")
    res_df.to_csv(final_csv_path, index=False)
    print(f"[{canopy_type}] ✅ 数据表已保存至: {final_csv_path}")

    # 绘制拟合散点图
    if 'pred_cab' in res_df.columns:
        print(f"[{canopy_type}] 📊 正在生成拟合散点图...")
        y_true = res_df['Cab_True']
        y_pred = res_df['pred_cab']

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        plt.figure(figsize=(7, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', c='royalblue', label='反演样本点(9波段)')

        min_val = min(y_true.min(), y_pred.min()) * 0.9
        max_val = max(y_true.max(), y_pred.max()) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 对角线')

        stats_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
        plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                       fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.xlabel('实测 Cab (μg/cm², LESS真值)', fontsize=12)
        plt.ylabel('反演 Cab (μg/cm², 27参数ANN预测)', fontsize=12)
        plt.title(f'LESS模拟数据 ({canopy_type}) 叶绿素反演拟合图 - 9波段', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Cab_Fitting_{canopy_type}_9Bands.png"), dpi=300)
        plt.close()


# ======================= 主入口 =======================
def main():
    print("🚀 开始批量处理 9 波段 LESS 模拟数据...")
    for canopy in CANOPY_TYPES:
        file_path = os.path.join(BASE_DATA_DIR, canopy, "result", f"{canopy}_全数据汇总.csv")
        process_single_canopy(file_path, canopy)

    print("\n🎉 所有叶倾角类型数据处理完毕！")


if __name__ == "__main__":
    main()