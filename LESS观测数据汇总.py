# coding: utf-8
import os
import pandas as pd

# ================= 1. 路径配置 =================
# 根目录：你的6波段_2存放数据的顶级文件夹
ROOT_DIR = r"E:\叶绿素反演-李文娟老师论文\LESS观测数据获取\9波段_2"

# 你需要整理的三大叶倾角类型
LEAF_TYPES = ['Erectophile', 'Planophile', 'Spherical']

print(f"🚀 开始执行数据大一统整理程序...")
print(f"📁 目标根目录: {ROOT_DIR}\n")

# ================= 2. 遍历并合并 =================
for leaf_type in LEAF_TYPES:
    type_dir = os.path.join(ROOT_DIR, leaf_type)

    # 检查该类型的文件夹是否存在，不存在则跳过
    if not os.path.exists(type_dir):
        print(f"⚠️ 找不到文件夹跳过: {type_dir}")
        continue

    print(f"🔍 正在扫描分类: {leaf_type} ...")

    # 在当前类型文件夹下，创建 result 存放目录
    result_dir = os.path.join(type_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    all_dfs = []  # 存放当前类型下所有的 DataFrame
    file_count = 0  # 统计成功读取了几个 CSV 文件

    # os.walk 深度遍历当前类型目录下的所有子文件夹
    for root, dirs, files in os.walk(type_dir):
        # 🛡️ 安全机制：跳过 result 文件夹自身，防止把刚合并出来的表又被读进去
        if os.path.basename(root) == "result":
            continue

        for file in files:
            # 仅处理以 .csv 结尾的文件
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                try:
                    # 读取独立的 CSV 数据
                    df = pd.read_csv(file_path)

                    # 确保表不是空的（排除了只有表头或0字节的坏表）
                    if not df.empty:
                        all_dfs.append(df)
                        file_count += 1
                except pd.errors.EmptyDataError:
                    print(f"   ⚠️ 跳过空文件: {file}")
                except Exception as e:
                    print(f"   ❌ 读取失败 {file}: {e}")

    # ================= 3. 大表拼接与输出 =================
    if all_dfs:
        # ignore_index=True 代表重新编排行号 (从 0 到 几万)
        print(f"   🔄 正在将 {file_count} 个碎片表拼接为超级总表...")
        final_df = pd.concat(all_dfs, ignore_index=True)

        # 导出路径
        output_filename = f"{leaf_type}_全数据汇总.csv"
        output_path = os.path.join(result_dir, output_filename)

        # 输出保存
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 【拼接完成】分类 {leaf_type} 共整合 {len(final_df)} 行数据！")
        print(f"   💾 保存至 -> {output_path}\n")
    else:
        print(f"   ⚠️ 在 {leaf_type} 下未找到任何有效的 CSV 结果文件。\n")

print("🎉 所有分类的数据整理全部完成！")