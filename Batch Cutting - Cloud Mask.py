import os
from osgeo import gdal

def clip_and_resample_images(base_dir, out_dir, shp_dir):
    # 存储每个湖泊的像元大小和行列数
    lake_pixel_sizes = {}
    lake_dimensions = {}

    # 遍历base_dir下的所有文件
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".tif"):
            # 从文件名中提取湖泊编号和其他信息
            parts = file_name.split('_')
            lake_number = parts[0]
            A_B_info = parts[1]
            year_info = parts[5]
            month_info = parts[6]
            day_info = parts[7]
            data_layer_info = '_'.join(parts[-2:])

            src = os.path.join(base_dir, file_name)
            SavePath = os.path.join(out_dir, lake_number)
            new_file_name = f"{lake_number}_{year_info}_{month_info}_{day_info}_{A_B_info}_{data_layer_info}"
            dst = os.path.join(out_dir, new_file_name)

            if lake_number not in lake_pixel_sizes:
                # 记录该湖泊第一幅影像的分辨率和行列数
                src_ds = gdal.Open(src)
                geo_transform = src_ds.GetGeoTransform()
                x_res, y_res = geo_transform[1], abs(geo_transform[5])  # 获取分辨率
                cols, rows = src_ds.RasterXSize, src_ds.RasterYSize  # 获取行列数
                lake_pixel_sizes[lake_number] = (x_res, y_res)
                lake_dimensions[lake_number] = (cols, rows)
                src_ds = None

            # 执行裁剪和重采样操作
            shapefile_path = os.path.join(shp_dir, lake_number, lake_number + ".shp")
            gdal.Warp(dst, src, format='GTiff', cutlineDSName=shapefile_path, cropToCutline=True,
                      xRes=lake_pixel_sizes[lake_number][0], yRes=lake_pixel_sizes[lake_number][1])
                      #width=lake_dimensions[lake_number][0], height=lake_dimensions[lake_number][1], dstNodata=0)

            print(f"文件 {file_name} 已裁剪并重采样为 {new_file_name}")

    print("所有文件处理完成。")

# 设置文件夹路径
base_dir = r"D:\NPML_tiff_af-cloud\20240301-云掩膜-369波段-不区分湖泊&年份\20240301_tif_不区分年份&湖泊_cloud-masked"  # 云掩膜处理后的文件位置
out_dir = r"D:\NPML_tiff_af-cloud\20240301-云掩膜-369波段-不区分湖年-裁剪后"  # 裁剪重采样后的文件保存位置
shp_dir = r"X:\大工\论文\20231214-反演湖泊shp\NPML_20_分散\20231214-编号重新调整\过程"  # 形状文件位置

# 执行裁剪和重采样
clip_and_resample_images(base_dir, out_dir, shp_dir)
