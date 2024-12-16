import os
import re
import numpy as np
from osgeo import gdal

base_path = r"D:\NPML_tiff_PC-result_Clip\20240118不区分湖泊&年份_裁剪结果"  # 替换为.tif文件的路径
out_path = r"D:\NPML_tiff_PC-result_mean\20240119日均值结果"  # 替换为输出.tif文件的路径

# 更新的正则表达式以匹配新的文件名格式
filename_regex = re.compile(r'(\d+)_(\d{4})_(\d{2})_(\d{2})_.*\.tif')


data_dict = {}

# 遍历base_path下的所有文件
for subdir, _, files in os.walk(base_path):
    for file in files:
        if file.endswith('.tif'):
            match = filename_regex.match(file)
            if not match:
                continue

            lake, _, month, day = match.groups()
            lake_number = int(lake)
            date_key = (lake_number, month, day)  # 修改为包含湖泊编号和日期（不包括年份）

            dataset = gdal.Open(os.path.join(subdir, file))
            if dataset is None:
                continue
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()

            if not np.all(np.isfinite(data)):
                continue
            data = np.nan_to_num(data, nan=0)

            if date_key not in data_dict:
                data_dict[date_key] = {'sum': data, 'count': 1}
            else:
                data_dict[date_key]['sum'] += data
                data_dict[date_key]['count'] += 1

for (lake_number, month, day), values in data_dict.items():
    mean_data = values['sum'] / values['count']
    out_tif = os.path.join(out_path, f"lake_{lake_number}_{month}_{day}_multiyear_daily_mean.tif")

    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(out_tif, mean_data.shape[1], mean_data.shape[0], 1, gdal.GDT_Float32)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(mean_data)
    out_band.FlushCache()

    out_dataset = None

print("处理完成")
