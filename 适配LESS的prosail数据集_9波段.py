import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

# 忽略优化过程中的警告，保持控制台整洁
warnings.filterwarnings("ignore")

# ======================= 1. 配置区域 =======================
# 输出 LUT 查找表路径
SAVE_LUT_PATH = r"E:\叶绿素反演-李文娟老师论文\适配LESS的prosail数据\9波段\PROSAIL_LUT_50k_18Params.csv"

# 基础文件路径（请确保路径正确）
GEOMETRY_FILE = r"E:\叶绿素反演-李文娟老师论文\基于物理公式获取计算散射分数的模型的数据\step1_simulated_weather_data_spitters.csv"
COEFFS_CSV_FILE = r"E:\叶绿素反演-李文娟老师论文\新采用的9波段\基于6s的公式系数\9_bands_coefficients.csv"
PROSAIL_CSV_FILE = r"E:\叶绿素反演\李文娟老师论文复现\PROSAIL-DATA\其他\dataSpec_P5.csv"

# 本地化机理：引入三条南京白马基地实测土壤光谱库
CUSTOM_SOIL_FILES = [

    r"E:\叶绿素反演-李文娟老师论文\LESS观测数据获取\LESS土壤光谱\LESS_soil.xlsx",
]

N_CASES = 50000  # LUT 规模（10万条）
VZA_FIXED = 45.0  # 观测天顶角
SENSOR_AZIMUTH = 90.0  # 传感器方位角

# ======================= 2. 初始化环境 =======================
print("正在读取波段与 6S 多项式系数...")
coeff_df = pd.read_csv(COEFFS_CSV_FILE)
BAND_NAMES = coeff_df['Band'].tolist()
NUM_BANDS = len(BAND_NAMES)  # 🌟 动态获取波段数量 (这里是 6)

VISIONPOINT_BANDS = [int(name.replace('nm', '')) for name in BAND_NAMES]
BAND_INDICES = [w - 440 for w in VISIONPOINT_BANDS]

F_LAMBDA_COEFFS = {}
for _, row in coeff_df.iterrows():
    F_LAMBDA_COEFFS[row['Band']] = [
        row['x^5'], row['x^4'], row['x^3'], row['x^2'], row['x^1'], row['Intercept']
    ]


# ======================= 3. PROSAIL 模型核心类 =======================
class ProsailModel:
    def __init__(self, spec_csv_path, custom_soil_paths=None):
        if not os.path.exists(spec_csv_path): raise FileNotFoundError(f"找不到光谱配置文件: {spec_csv_path}")
        with open(spec_csv_path, 'r') as f:
            lines = f.readlines()
        data = np.array([[float(v) for v in line.strip().split(',')[1:]] for line in lines])
        wavs = data[0, :]
        idx_start, idx_end = np.where(wavs == 440)[0][0], np.where(wavs == 921)[0][0] + 1
        self.spectra = data[:, idx_start:idx_end]
        target_len = idx_end - idx_start  # 目标长度应为 482

        self.custom_soils = []
        if custom_soil_paths:
            for path in custom_soil_paths:
                # 🌟 修改点 1：使用 header=None 读取，避免丢失400nm的第一条数据
                df_soil = pd.read_excel(path, header=None)

                # 🌟 修改点 2：无论数据是一行多列还是一列多行，全部展平为一维数组，并清理空值
                custom_soil_full = pd.to_numeric(df_soil.values.flatten(), errors='coerce')
                custom_soil_full = custom_soil_full[~np.isnan(custom_soil_full)]

                # 如果反射率是百分比(>1.0)，转换为0-1
                if np.max(custom_soil_full) > 1.0:
                    custom_soil_full = custom_soil_full / 100.0

                # 🌟 修改点 3：精准截取 440 - 921 nm
                # 400nm 对应索引 0，则 440nm 对应索引 40
                # 921nm 对应索引 521，切片包含末尾需写 522
                if len(custom_soil_full) >= 2101:  # 确保是 400-2500nm 的完整数据
                    custom_soil_trimmed = custom_soil_full[40:522]
                    self.custom_soils.append(custom_soil_trimmed)
                else:
                    # 兜底：如果数据长度异常，退回原先的默认截取方式
                    if len(custom_soil_full) >= target_len:
                        self.custom_soils.append(custom_soil_full[:target_len])
    def _tav_abs(self, theta, refr):
        refr = np.array(refr);
        thetarad = np.radians(theta)
        if theta == 0.: return 4. * refr / (refr + 1.) ** 2
        refr2 = refr * refr
        ax, bx = (refr + 1.) ** 2 / 2., -(refr2 - 1.) ** 2 / 4.
        b1, b2 = ((np.sin(thetarad) ** 2 - (refr2 + 1.) / 2.) ** 2 + bx) ** 0.5, np.sin(thetarad) ** 2 - (
                    refr2 + 1.) / 2.
        b0 = b1 - b2
        ts = (bx ** 2 / (6. * b0 ** 3) + bx / b0 - b0 / 2.) - (bx ** 2 / (6. * ax ** 3) + bx / ax - ax / 2.)
        tp1, tp2 = -2. * refr2 * (b0 - ax) / (refr2 + 1.) ** 2, -2. * refr2 * (refr2 + 1.) * np.log(b0 / ax) / (
                    refr2 - 1.) ** 2
        tp3, tp4 = refr2 * (1. / b0 - 1. / ax) / 2., 16. * refr2 ** 2 * (refr2 ** 2 + 1.) * np.log(
            (2. * (refr2 + 1.) * b0 - (refr2 - 1.) ** 2) / ((2. * (refr2 + 1.) * ax - (refr2 - 1.) ** 2))) / (
                               (refr2 + 1.) ** 3 * (refr2 - 1.) ** 2)
        tp5 = 16. * refr2 ** 3 * (1. / (2. * (refr2 + 1.) * b0 - ((refr2 - 1.) ** 2)) - 1. / (
                    2. * (refr2 + 1.) * ax - (refr2 - 1.) ** 2)) / (refr2 + 1.) ** 3
        return (ts + tp1 + tp2 + tp3 + tp4 + tp5) / (2. * np.sin(thetarad) ** 2)

    def _prospect_5B(self, N, Cab, Car, Cbrown, Cw, Cm):
        k = (Cab * self.spectra[2] + Car * self.spectra[3] + Cbrown * self.spectra[4] + Cw * self.spectra[5] + Cm *
             self.spectra[6]) / N
        refractive = self.spectra[1]
        tau, xx, yy = np.zeros(k.size), np.zeros(k.size), np.zeros(k.size)

        for i in range(tau.size):
            if k[i] <= 0.0:
                tau[i] = 1
            elif (k[i] > 0.0 and k[i] <= 4.0):
                xx[i] = 0.5 * k[i] - 1.0
                yy[i] = (((((((((((((((-3.60311230482612224e-13 * xx[i] + 3.46348526554087424e-12) * xx[
                    i] - 2.99627399604128973e-11) * xx[i] + 2.57747807106988589e-10) * xx[i] - 2.09330568435488303e-9) *
                                   xx[i] + 1.59501329936987818e-8) * xx[i] - 1.13717900285428895e-7) * xx[
                                     i] + 7.55292885309152956e-7) * xx[i] - 4.64980751480619431e-6) * xx[
                                   i] + 2.63830365675408129e-5) * xx[i] - 1.37089870978830576e-4) * xx[
                                 i] + 6.47686503728103400e-4) * xx[i] - 2.76060141343627983e-3) * xx[
                               i] + 1.05306034687449505e-2) * xx[i] - 3.57191348753631956e-2) * xx[
                             i] + 1.07774527938978692e-1) * xx[i] - 2.96997075145080963e-1
                yy[i] = (yy[i] * xx[i] + 8.64664716763387311e-1) * xx[i] + 7.42047691268006429e-1
                yy[i] = yy[i] - np.log(k[i])
                tau[i] = (1.0 - k[i]) * np.exp(-k[i]) + k[i] ** 2 * yy[i]
            elif (k[i] > 4.0 and k[i] <= 85.0):
                xx[i] = 14.5 / (k[i] + 3.25) - 1.0
                yy[i] = (((((((((((((((-1.62806570868460749e-12 * xx[i] - 8.95400579318284288e-13) * xx[
                    i] - 4.08352702838151578e-12) * xx[i] - 1.45132988248537498e-11) * xx[
                                        i] - 8.35086918940757852e-11) * xx[i] - 2.13638678953766289e-10) * xx[
                                      i] - 1.10302431467069770e-9) * xx[i] - 3.67128915633455484e-9) * xx[
                                    i] - 1.66980544304104726e-8) * xx[i] - 6.11774386401295125e-8) * xx[
                                  i] - 2.70306163610271497e-7) * xx[i] - 1.05565006992891261e-6) * xx[
                                i] - 4.72090467203711484e-6) * xx[i] - 1.95076375089955937e-5) * xx[
                              i] - 9.16450482931221453e-5) * xx[i] - 4.05892130452128677e-4) * xx[
                            i] - 2.14213055000334718e-3
                yy[i] = ((yy[i] * xx[i] - 1.06374875116569657e-2) * xx[i] - 8.50699154984571871e-2) * xx[
                    i] + 9.23755307807784058e-1
                yy[i] = np.exp(-k[i]) * yy[i] / k[i]
                tau[i] = (1.0 - k[i]) * np.exp(-k[i]) + k[i] ** 2 * yy[i]
            else:
                tau[i] = 0

        t1, t2 = self._tav_abs(90., refractive), self._tav_abs(40., refractive)
        x1, x2, x3, x4 = 1 - t1, t1 ** 2 * tau ** 2 * (
                    refractive ** 2 - t1), t1 ** 2 * tau * refractive ** 2, refractive ** 4 - tau ** 2 * (
                                     refractive ** 2 - t1) ** 2
        x5 = t2 / t1
        x6 = x5 * (t1 - 1) + 1 - t2
        r, t = x1 + x2 / x4, x3 / x4
        ra, ta = x5 * r + x6, x5 * t
        delta = (t ** 2 - r ** 2 - 1) ** 2 - 4 * r ** 2
        va = (1 + r ** 2 - t ** 2 + delta ** 0.5) / (2 * r)
        vb = (((1 + r ** 2 - t ** 2 - delta ** 0.5) / (2 * r) * (va - r)) / (
                    va * ((1 + r ** 2 - t ** 2 - delta ** 0.5) / (2 * r) - r))) ** 0.5
        s1 = ra * (va * vb ** (N - 1) - va ** (-1) * vb ** (-(N - 1))) + (ta * t - ra * r) * (
                    vb ** (N - 1) - vb ** (-(N - 1)))
        s3 = va * vb ** (N - 1) - va ** (-1) * vb ** (-(N - 1)) - r * (vb ** (N - 1) - vb ** (-(N - 1)))
        return s1 / s3, (ta * (va - va ** (-1))) / s3

    def _campbell(self, n, ala):
        tx2 = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88.])
        tx1 = np.array([10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88., 90.])
        tl1, tl2 = tx1 * np.arctan(1.) / 45., tx2 * np.arctan(1.) / 45.
        excent = np.exp(-1.6184e-5 * ala ** 3 + 2.1145e-3 * ala ** 2 - 1.2390e-1 * ala + 3.2491)
        x1, x2 = excent / (np.sqrt(1. + excent ** 2 * np.tan(tl1) ** 2)), excent / (
            np.sqrt(1. + excent ** 2 * np.tan(tl2) ** 2))
        if excent == 1.: return np.absolute(np.cos(tl1) - np.cos(tl2)) / np.sum(np.absolute(np.cos(tl1) - np.cos(tl2)))
        alpha = excent / np.sqrt(np.absolute(1. - excent ** 2))
        alpha2, x12, x22 = alpha ** 2, x1 ** 2, x2 ** 2
        if excent > 1:
            alpx1, alpx2 = np.sqrt(alpha2 + x12), np.sqrt(alpha2 + x22)
            freq = np.absolute((x1 * alpx1 + alpha2 * np.log(x1 + alpx1)) - (x2 * alpx2 + alpha2 * np.log(x2 + alpx2)))
        else:
            almx1, almx2 = np.sqrt(alpha2 - x12), np.sqrt(alpha2 - x22)
            freq = np.absolute(
                (x1 * almx1 + alpha2 * np.arcsin(x1 / alpha)) - (x2 * almx2 + alpha2 * np.arcsin(x2 / alpha)))
        return freq / np.sum(freq)

    def _volscatt(self, tts, tto, psi, ttl):
        rd = np.pi / 180.
        costs, costo, sints, sinto = np.cos(rd * tts), np.cos(rd * tto), np.sin(rd * tts), np.sin(rd * tto)
        cospsi, psir, costl, sintl = np.cos(rd * psi), rd * psi, np.cos(rd * ttl), np.sin(rd * ttl)
        cs, co, ss, so = costl * costs, costl * costo, sintl * sints, sintl * sinto
        cosbts, cosbto = (-cs / ss if np.absolute(ss) > 1e-6 else 5.), (-co / so if np.absolute(so) > 1e-6 else 5.)
        bts, ds = (np.arccos(cosbts), ss) if np.absolute(cosbts) < 1. else (np.pi, cs)
        chi_s = 2. / np.pi * ((bts - np.pi * .5) * cs + np.sin(bts) * ss)
        bto, doo = (np.arccos(cosbto), so) if np.absolute(cosbto) < 1. else (np.pi, co) if tto < 90. else (0, -co)
        chi_o = 2. / np.pi * ((bto - np.pi * .5) * co + np.sin(bto) * so)
        btran1, btran2 = np.absolute(bts - bto), np.pi - np.absolute(bts + bto - np.pi)
        if psir <= btran1:
            bt1, bt2, bt3 = psir, btran1, btran2
        else:
            bt1, bt2, bt3 = (btran1, psir, btran2) if psir <= btran2 else (btran1, btran2, psir)
        t1 = 2. * cs * co + ss * so * cospsi
        t2 = np.sin(bt2) * (2. * ds * doo + ss * so * np.cos(bt1) * np.cos(bt3)) if bt2 > 0. else 0.
        denom = 2. * np.pi * np.pi
        return chi_s, chi_o, max(((np.pi - bt2) * t1 + t2) / denom, 0), max((-bt2 * t1 + t2) / denom, 0)

    def _PRO4SAIL(self, rho, tau, lidf, lai, q, tts, tto, psi, rsoil):
        if lai <= 0: return rsoil, rsoil
        litab = np.array([5., 15., 25., 35., 45., 55., 65., 75., 81., 83., 85., 87., 89.])
        rd = np.pi / 180.
        cts, cto, tants, tanto = np.cos(rd * tts), np.cos(rd * tto), np.tan(rd * tts), np.tan(rd * tto)
        dso = np.sqrt(tants ** 2 + tanto ** 2 - 2. * tants * tanto * np.cos(rd * psi))
        ks, ko, bf, sob, sof = 0, 0, 0, 0, 0
        ctl = np.cos(rd * litab)
        for i in range(13):
            chi_s, chi_o, frho, ftau = self._volscatt(tts, tto, psi, litab[i])
            ks += (chi_s / cts) * lidf[i];
            ko += (chi_o / cto) * lidf[i]
            bf += (ctl[i] * ctl[i]) * lidf[i]
            sob += (frho * np.pi / (cts * cto)) * lidf[i];
            sof += (ftau * np.pi / (cts * cto)) * lidf[i]

        sdb, sdf, dob, dof, ddb, ddf = 0.5 * (ks + bf), 0.5 * (ks - bf), 0.5 * (ko + bf), 0.5 * (ko - bf), 0.5 * (
                    1. + bf), 0.5 * (1. - bf)
        sigb, sigf = ddb * rho + ddf * tau, ddf * rho + ddb * tau
        att = 1. - sigf
        m = np.sqrt(np.maximum((att + sigb) * (att - sigb), 0))
        sb, sf, vb, vf = sdb * rho + sdf * tau, sdf * rho + sdb * tau, dob * rho + dof * tau, dof * rho + dob * tau
        w = sob * rho + sof * tau
        e1, e2 = np.exp(-m * lai), np.exp(-2 * m * lai)
        rinf = (att - m) / sigb
        rinf2, re, denom = rinf * rinf, rinf * e1, 1. - rinf ** 2 * e2

        J1ks, J2ks, J1ko, J2ko = (1. - np.exp(-(ks + m) * lai)) / (ks + m), (1. - np.exp(-(ks + m) * lai)) / (ks + m), (
                    1. - np.exp(-(ko + m) * lai)) / (ko + m), (1. - np.exp(-(ko + m) * lai)) / (ko + m)
        Ps, Qs, Pv, Qv = (sf + sb * rinf) * J1ks, (sf * rinf + sb) * J2ks, (vf + vb * rinf) * J1ko, (
                    vf * rinf + vb) * J2ko
        rdd, tdd = rinf * (1. - e2) / denom, (1. - rinf2) * e1 / denom
        tsd, rsd, tdo, rdo = (Ps - re * Qs) / denom, (Qs - re * Ps) / denom, (Pv - re * Qv) / denom, (
                    Qv - re * Pv) / denom
        tss, too = np.exp(-ks * lai), np.exp(-ko * lai)
        g1, g2 = ((1. - np.exp(-(ks + ko) * lai)) / (ks + ko) - J1ks * too) / (ko + m), (
                    (1. - np.exp(-(ks + ko) * lai)) / (ks + ko) - J1ko * tss) / (ks + m)
        rsod = ((vf * rinf + vb) * g1 * (sf + sb * rinf) + (vf + vb * rinf) * g2 * (sf * rinf + sb) - (
                    rdo * Qs + tdo * Ps) * rinf) / (1. - rinf2)

        alf = min((dso / q) * 2. / (ks + ko) if q > 0. else 1e6, 200.)
        tsstoo = tss if alf == 0. else np.exp(-(ko + ks) * lai + lai * np.sqrt(ko * ks) * (1. - np.exp(-alf)) / alf)
        sumint = (1 - tss) / (ks * lai) if alf == 0 else (1 - np.exp(-(ks + ko) * lai)) / (ks + ko)

        rsos = w * lai * sumint
        dn = 1. - rsoil * rdd
        rdot = rdo + tdd * rsoil * (tdo + too) / dn
        rsot = rsos + tsstoo * rsoil + rsod + ((tss + tsd) * tdo + (tsd + tss * rsoil * rdd) * too) * rsoil / dn
        return rsot, rdot

    def run(self, N, Cab, Car, Cbrown, Cw, Cm, LAI, hspot, tts, tto, psi, ala, Bs, soil_idx):
        lidf = self._campbell(13, ala)
        rho, tau = self._prospect_5B(N, Cab, Car, Cbrown, Cw, Cm)
        rsoil0 = np.clip(self.custom_soils[soil_idx] * Bs, 0.0, 1.0)
        return self._PRO4SAIL(rho, tau, lidf, LAI, hspot, tts, tto, psi, rsoil0)


prosail_model = ProsailModel(spec_csv_path=PROSAIL_CSV_FILE, custom_soil_paths=CUSTOM_SOIL_FILES)


# ======================= 4. 核模型与 DLC 积分 =======================
def roujean_k_vol(sza, vza, phi):
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    cos_xi = np.cos(sza_r) * np.cos(vza_r) + np.sin(sza_r) * np.sin(vza_r) * np.cos(phi_r)
    phase = np.arccos(np.clip(cos_xi, -1.0, 1.0))
    term = (np.pi / 2.0 - phase) * np.cos(phase) + np.sin(phase)
    return (4.0 / (3.0 * np.pi)) * (term / (np.cos(sza_r) + np.cos(vza_r))) - (1.0 / 3.0)


def roujean_k_geo(sza, vza, phi):
    sza_r, vza_r, phi_r = np.radians(sza), np.radians(vza), np.radians(phi)
    tan_s, tan_v = np.tan(sza_r), np.tan(vza_r)
    delta = np.sqrt(np.maximum(0, tan_s ** 2 + tan_v ** 2 - 2.0 * tan_s * tan_v * np.cos(phi_r)))
    term1 = (np.pi - phi_r) * np.cos(phi_r) + np.sin(phi_r)
    return (1.0 / (2.0 * np.pi)) * term1 * tan_s * tan_v - (1.0 / np.pi) * (tan_s + tan_v + delta)


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
    k_vol_dir, k_geo_dir = roujean_k_vol(sza, vza, phi), roujean_k_geo(sza, vza, phi)
    return (1 - f_lambda) * k_vol_dir + f_lambda * K_VOL_DIFF, (1 - f_lambda) * k_geo_dir + f_lambda * K_GEO_DIFF


# ======================= 5. LHS 采样 =======================
# ======================= 5. LHS 采样 =======================
def get_weiss_parameters(n_samples):
    sampler = qmc.LatinHypercube(d=8, seed=42)
    lhs_samples = sampler.random(n=n_samples)
    u_lai, u_cab, u_cw, u_hspot = lhs_samples[:, 0], lhs_samples[:, 1], lhs_samples[:, 2], lhs_samples[:, 3]
    u_ala, u_n, u_bs, u_soil = lhs_samples[:, 4], lhs_samples[:, 5], lhs_samples[:, 6], lhs_samples[:, 7]

    # 🌟 修改点 4：因为现在仅使用一个实测土壤光谱库，土壤索引全部固定为 0
    soil_idx = np.zeros(n_samples, dtype=int)
    # 删除了原来根据 u_soil 划分为 1 和 2 的逻辑

    lai = np.clip(-2 * np.log(u_lai * (np.exp(-0.5 * 0.1) - np.exp(-0.5 * 7)) + np.exp(-0.5 * 7)), 0.1, 7.0)
    cab = np.clip(-100 * np.log(u_cab * (np.exp(-0.01 * 65.0) - np.exp(-0.01 * 1.0)) + np.exp(-0.01 * 1.0)), 1.0, 65.0)
    cw = -1 / 50 * np.log(u_cw * (np.exp(-50 * 0.005) - np.exp(-50 * 0.025)) + np.exp(-50 * 0.025))
    hspot = -1 / 3 * np.log(u_hspot * (np.exp(-3 * 0.05) - np.exp(-3 * 1)) + np.exp(-3 * 1))
    ala = np.rad2deg(np.arccos(u_ala * (np.cos(np.deg2rad(20)) - np.cos(np.deg2rad(75))) + np.cos(np.deg2rad(75))))
    n_struct = u_n * (2.5 - 1.0) + 1.0
    bs = u_bs * (1.6 - 0.7) + 0.5

    pure_bg_mask = np.random.rand(n_samples) < 0.03
    lai[pure_bg_mask] = 0.0
    cab[pure_bg_mask] = 0.0
    cw[pure_bg_mask] = 0.0

    return pd.DataFrame({
        "lai": lai, "cab": cab, "cw": cw, "hspot": hspot,
        "ala": ala, "n": n_struct, "bs": bs, "soil_idx": soil_idx
    })

# ======================= 6. LUT 联合求解工作流 =======================
def process_lut_case(args):
    case_id, bio_dict, geo_list = args
    curr_cm, curr_bs, curr_soil_idx = bio_dict['cw'] / 4.0, bio_dict['bs'], int(bio_dict['soil_idx'])
    time_series_data = []

    # 1. 模拟一整天的真实光照观测
    for row in geo_list:
        tts = row.get('cos_theta_s', row.get('tts', 0.0))
        if tts <= 1.0: tts = np.degrees(np.arccos(np.clip(tts, -1.0, 1.0)))
        if tts > 60: continue
        saa = row.get('sun_azimuth', row.get('saa', 180.0))
        psi = np.abs(saa - SENSOR_AZIMUTH)
        if psi > 180: psi = 360 - psi
        f_par = row.get('f_par', 0.2)

        rsot_spec, rdot_spec = prosail_model.run(
            N=bio_dict['n'], Cab=bio_dict['cab'], Car=8, Cbrown=0, Cw=bio_dict['cw'], Cm=curr_cm,
            LAI=bio_dict['lai'], hspot=bio_dict['hspot'], tts=tts, tto=VZA_FIXED,
            psi=psi, ala=bio_dict['ala'], Bs=curr_bs, soil_idx=curr_soil_idx
        )

        rho_dir, rho_hdr = rsot_spec[BAND_INDICES], rdot_spec[BAND_INDICES]
        moment_refls_clean, f_lams_list = [], []

        # 🌟 动态循环 NUM_BANDS
        for i, band in enumerate(BAND_NAMES):
            f_lam = np.clip(np.poly1d(F_LAMBDA_COEFFS[band])(f_par) if f_par <= 0.9 else f_par, 0, 1)
            f_lams_list.append(f_lam)
            refl = (1 - f_lam) * rho_dir[i] + f_lam * rho_hdr[i]
            moment_refls_clean.append(refl)

        # 🌟 修正：随机噪声的维度改成 NUM_BANDS
        noisy_refls = np.clip(
            np.array(moment_refls_clean) * (1 + np.random.normal(0, 0.02, NUM_BANDS)) + np.random.normal(0, 0.01,
                                                                                                         NUM_BANDS),
            0.0, 1.0)
        time_series_data.append({'sza': tts, 'phi': psi, 'f_lams': f_lams_list, 'abs_refls_noisy': noisy_refls})

    T = len(time_series_data)
    if T < 5: return None

    # 2. 准备代价函数矩阵
    X_sza, X_phi = np.array([m['sza'] for m in time_series_data]), np.array([m['phi'] for m in time_series_data])
    X_vza = np.full(T, VZA_FIXED)

    # 🌟 修正：所有矩阵维度使用 NUM_BANDS
    obs_abs_refls, f_lams_matrix = np.zeros((NUM_BANDS, T)), np.zeros((NUM_BANDS, T))
    for i in range(NUM_BANDS):
        obs_abs_refls[i, :] = [m['abs_refls_noisy'][i] for m in time_series_data]
        f_lams_matrix[i, :] = [m['f_lams'][i] for m in time_series_data]

    y_rel_matrix = obs_abs_refls / (np.mean(obs_abs_refls, axis=0) + 1e-6)

    k_vol_dlc_all, k_geo_dlc_all = np.zeros((NUM_BANDS, T)), np.zeros((NUM_BANDS, T))
    for i in range(NUM_BANDS):
        k_vol_dlc_all[i], k_geo_dlc_all[i] = calculate_dlc_kernel(X_sza, X_vza, X_phi, f_lams_matrix[i])

    # 3. 联合优化求解
    def global_cost_func(params):
        params_2d = params.reshape(NUM_BANDS, 3)  # 🌟 动态 Reshape
        X_mod_abs = np.zeros((NUM_BANDS, T))
        for i in range(NUM_BANDS):
            X_mod_abs[i] = params_2d[i, 0] + params_2d[i, 1] * k_vol_dlc_all[i] + params_2d[i, 2] * k_geo_dlc_all[i]
        mean_X_mod = np.mean(X_mod_abs, axis=0)
        X_mod_rel = X_mod_abs / (mean_X_mod + 1e-6)
        return np.sum((y_rel_matrix - X_mod_rel) ** 2) + 100.0 * np.sum((mean_X_mod - 1) ** 2)

    x0 = []
    for i in range(NUM_BANDS): x0.extend([np.mean(y_rel_matrix[i, :]), 0.05, 0.05])
    bounds_dynamic = [(0, None), (-0.05, None), (-0.05, None)] * NUM_BANDS  # 🌟 动态边界

    res = minimize(global_cost_func, x0=x0, method='SLSQP', bounds=bounds_dynamic)

    if res.success:
        params_opt = res.x.reshape(NUM_BANDS, 3)
        result_row = {'lut_id': case_id, 'lai': bio_dict['lai'], 'cab': bio_dict['cab']}
        for i, band in enumerate(BAND_NAMES):
            result_row[f"{band}_k0"] = params_opt[i, 0]
            result_row[f"{band}_k1"] = params_opt[i, 1]
            result_row[f"{band}_k2"] = params_opt[i, 2]
        return result_row
    return None


# ======================= 7. 主程序 =======================
if __name__ == "__main__":
    print(f"🚀 初始化 PROSAIL LUT 生成引擎 (样本数 N={N_CASES})...")

    # 加载气象序列
    daily_geo_df = pd.read_csv(GEOMETRY_FILE)
    daily_geo_df.columns = [str(c).lower() for c in daily_geo_df.columns]

    # 获取模拟一天的数据点（假设你已改成高频了，比如一天 40个点）
    geo_list = daily_geo_df.iloc[:56].to_dict('records')

    # 抽取生物物理参数
    df_bio = get_weiss_parameters(N_CASES)

    print("⏳ 正在组装多进程任务队列...")
    tasks = []
    for i in range(N_CASES):
        tasks.append((i, df_bio.iloc[i].to_dict(), geo_list))

    print(f"⚡ 启动多进程计算 (正演一天光谱 + 联合求解 {NUM_BANDS * 3} 核参数)...")
    with Pool(cpu_count()) as p:
        results = [r for r in tqdm(p.imap(process_lut_case, tasks), total=N_CASES) if r is not None]

    if results:
        final_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(SAVE_LUT_PATH), exist_ok=True)
        final_df.to_csv(SAVE_LUT_PATH, index=False)
        print(f"✅ 完成! {len(final_df)} 条有效的 LUT 特征数据已保存至:\n {SAVE_LUT_PATH}")
    else:
        print("❌ 严重错误：未产生任何有效模拟案例。")