import os
import re
import numpy as np
from osgeo import gdal

base_path = r"D:\NPML_tiff_PC-result_Clip\20240118不区分湖泊&年份_裁剪结果"
out_path = r"D:\NPML_tiff_PC-result_mean\20240128多年月均值_不区分湖泊&年份_保留坐标_忽略0值\所有"

filename_regex = re.compile(r'(\d+)_(\d{4})_(\d{2})_.*\.tif')

data_dict = {}
geo_transforms = {}  # 用于存储每个湖泊第一个数据集的地理转换信息
projections = {}  # 用于存储每个湖泊第一个数据集的投影信息

for subdir, _, files in os.walk(base_path):
    for file in files:
        if file.endswith('.tif'):
            match = filename_regex.match(file)
            if not match:
                continue
            print(file)
            lake, year, month = match.groups()
            lake_number = int(lake)
            month_key = (lake_number, month)

            dataset = gdal.Open(os.path.join(subdir, file))
            if dataset is None:
                continue
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()

            if not np.all(np.isfinite(data)):
                continue
            data = np.nan_to_num(data, nan=0)

            if month_key not in data_dict:
                data_dict[month_key] = {'sum': np.where(data != 0, data, 0), 'count': np.where(data != 0, 1, 0)}
                # 保存第一个数据集的空间信息
                geo_transforms[month_key] = dataset.GetGeoTransform()
                projections[month_key] = dataset.GetProjection()
            else:
                data_dict[month_key]['sum'] += np.where(data != 0, data, 0)
                data_dict[month_key]['count'] += np.where(data != 0, 1, 0)

for (lake_number, month), values in data_dict.items():
    mean_data = np.where(values['count'] != 0, values['sum'] / values['count'], 0)
    out_tif = os.path.join(out_path, f"lake_{lake_number}_0_{month}_month_mean.tif")

    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(out_tif, mean_data.shape[1], mean_data.shape[0], 1, gdal.GDT_Float32)

    # 更新月均值数据集的空间信息
    out_dataset.SetGeoTransform(geo_transforms[(lake_number, month)])
    out_dataset.SetProjection(projections[(lake_number, month)])

    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(mean_data)
    out_band.FlushCache()

    out_dataset = None

print("处理完成")
