import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取CSV数据文件 (由于没有表头，设置 header=None)
df = pd.read_excel(r"E:\叶绿素反演-李文娟老师论文\LESS观测数据获取\LESS土壤光谱\LESS_soil.xlsx", header=None)

# 2. 提取第一行的反射率数据
reflectance = df.iloc[0].values

# 3. 生成横坐标：波长从 400nm 到 2500nm，点数与数据长度一致 (2101个)
wavelengths = np.linspace(400, 2500, len(reflectance))

# 4. 绘制反射率曲线
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, reflectance, color='#1f77b4', linewidth=1.5, label='Soil Reflectance')

# 5. 设置图表属性
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title('Reflectance Curve (400-2500 nm)', fontsize=14)
plt.xlim(400, 2500)  # 限制横坐标范围为400-2500
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')

# 显示并保存图片
plt.tight_layout()
plt.savefig('reflectance_curve.png', dpi=300)
plt.show()