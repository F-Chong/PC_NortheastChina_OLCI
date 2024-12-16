import os
import csv
import numpy as np
from osgeo import gdal

# 设置tif影像路径和输出csv文件路径
tif_dir = r"C:\Users\PC208\Desktop\反演\徐怡沛\年均值\年平均_裁剪后汇总\NPML_年平均"
csv_file = r"C:\Users\PC208\Desktop\反演\徐怡沛\反演年均值csv\NPML_年均值.csv"

# 获取所有tif文件的路径
tif_files = [os.path.join(tif_dir, f) for f in os.listdir(tif_dir) if f.endswith(".tif")]

# 创建csv文件并写入表头
with open(csv_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "mean"])

# 循环读取每个tif文件的均值并写入csv文件
for tif_file in tif_files:
    # 打开tif文件并读取数据
    dataset = gdal.Open(tif_file)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    # 将NaN值替换为0，计算均值并忽略NaN值
    data = np.nan_to_num(data, nan=0)
    mean = np.mean(data[data != 0])

    # 获取文件名并将均值写入csv文件
    filename = os.path.basename(tif_file)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename, mean])

print("All done")
