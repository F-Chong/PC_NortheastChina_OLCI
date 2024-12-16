from xgboost import XGBRegressor as XGBR
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import joblib
from osgeo import gdal
import os
import sys
import time
import pandas as pd



###定义代码运行路径导入相关模型
os.chdir(r"D:\Phycocyanin_PC\Python\分类及反演\尝试\test")##设定代码运行环境的路径(即所调用模型代码的存储路径)
SVC_clf = joblib.load('SVC_Classification_PC_45_model(Single_parameter)_gamma48.m')##模型输入格式为reshape(-1,1)
PC_class1 = SVR()
PC_class1= joblib.load("SVR_Predict_PC_0-45_model(multi_parameter)_gamma90.m")##模型输入格式为reshape(-1,8)
PC_class2 = SVR()
PC_class2= joblib.load("SVR_Predict_PC_45-1300_model(multi_parameter)_gamma2.5.m")##模型输入格式为reshape(-1,7)



#定义相关数据获取及保存位置
FilePath = r"D:\Phycocyanin_PC\Python\分类及反演\尝试\test\tif文件"  ###需要处理的tif图像的位置,只能嵌套一层文件夹目录，例如：所填目标路径/文件夹/a.tif
file_names = os.listdir(FilePath)
SavePath = r"D:\Phycocyanin_PC\Python\分类及反演\尝试\test\反演_test_result"  ###反演后结果tif图像的保存位置，这里不要求自建新文件夹，代码会自动建立

# 定义图像的打开方式
def image_open(img):
    data = gdal.Open(img)
    if data == "None":
        print("图像无法读取")
    return data


# 批量+处理+输出
# 批量过程
start_time = time.time()  # 开始时间
# 循环叠加处理
k = 0
n = 0
for m in file_names:
    k = k + 1
    father_file_path = FilePath + "/" + m
    ##在保存路径中创建新的文件夹
    new_file_path = SavePath + "/" + m + "_PCresult"
    if os.path.exists(new_file_path):  ##目录已经存在则跳过
        continue
    else:
        os.mkdir(new_file_path)  ##否则创建目录

    try:
        son_file_names = os.listdir(father_file_path)
    except NotADirectoryError:
        print("该目录下没有文件夹")
        break
    else:
        if son_file_names == []:
            print("文件夹 " + m + " 为空")
            print("--------------------------------------------------------------------------------------")
            continue
        else:
            print("现在载入文件夹 " + m)
            j = 0
            for i in son_file_names:  # 按照文件个数设置循环次数
                QZ = os.path.splitext(i)[0]  # 前缀提取
                HZ = os.path.splitext(i)[1]  # 后缀提取
                if (HZ == ".tif"):
                    j = j + 1
                    image = father_file_path + "/" + i
                    dat = image
                    img = gdal.Open(dat)
                    print("----------读取影像数据----------")
                    img_cols = img.RasterXSize  # 栅格矩阵的列数(X方向像素数)
                    img_rows = img.RasterYSize  # 栅格矩阵的行数(Y方向像素数)
                    img_bands = img.RasterCount  # 波段数
                    img_geotrans = img.GetGeoTransform()  # 仿射矩阵
                    img_proj = img.GetProjection()  # 地图投影信息


                    mapping_X = pd.DataFrame()
                    for i in range(1, img_bands + 1):
                        band_i = img.GetRasterBand(i)  # 读取波段,参数为波段的索引号(波段索引号从1开始)   img_bands+1
                        band_img_value = band_i.ReadAsArray()  # 读取整幅影像
                        value = band_img_value.flatten()  # 把array从二维数组降到一维，默认是按行的方向降
                        mapping_X[i] = value
                    mapping_X.columns = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', '11', '12', '13',
                                         '14', '15', '16', '17', '18', '19', '20']
                    data = mapping_X.loc[:, ['b3','b4','b5','b6','b7','b8','b9', 'b10', '11', '12','13','14','15','16','17','18','19']]
                    Blue = mapping_X.loc[:,['b4']]
                    data['a'] = mapping_X['13'] / mapping_X['b9']
                    data['b'] = mapping_X['13'] / mapping_X['11']
                    data['c'] = mapping_X['b9'] / mapping_X['b8']
                    data['d'] = mapping_X['14'] - mapping_X['b8']
                    data['e'] = mapping_X['14'] / mapping_X['b8']
                    data['f'] = 0.01 * mapping_X['14'] / mapping_X['b10']
                    data['g'] = (mapping_X['14'] - mapping_X['b4']) / mapping_X['b10']
                    data['h'] = mapping_X['11'] / mapping_X['b10']

                    B1 = data['b4'].values.reshape(-1)
                    B2 = data['b8'].values.reshape(-1)
                    B3 = data['b9'].values.reshape(-1)
                    B4 = data['b10'].values.reshape(-1)
                    B5 = data['11'].values.reshape(-1)
                    B6 = data['12'].values.reshape(-1)
                    B7 = data['13'].values.reshape(-1)
                    B8 = data['14'].values.reshape(-1)
                    B9 = data['17'].values.reshape(-1)
                    B10 = data['19'].values.reshape(-1)
                    B11 = data['a'].values.reshape(-1)
                    B12 = data['b'].values.reshape(-1)
                    B13 = data['c'].values.reshape(-1)
                    B14 = data['d'].values.reshape(-1)
                    B15 = data['e'].values.reshape(-1)
                    B16 = data['f'].values.reshape(-1)
                    B17 = data['g'].values.reshape(-1)
                    B18 = data['h'].values.reshape(-1)

                    b3 = data['b3'].values.reshape(-1)
                    b5 = data['b5'].values.reshape(-1)
                    b6 = data['b6'].values.reshape(-1)
                    b7 = data['b7'].values.reshape(-1)
                    b15 = data['15'].values.reshape(-1)
                    b16 = data['16'].values.reshape(-1)
                    b18 = data['18'].values.reshape(-1)


                    # 云掩膜/将blue波段反射率为0的值的位置设为掩膜
                    cloud_mask = B2 > 0.064   ###参考自师兄大论文
                    B9[cloud_mask] = 2  # 令B9波段云掩膜处的值等于1，方便参与NDWI的计算而不报错，后续会剔除这些像素点，所以不影响
                    # 根据NDWI的计算结果设置水体掩膜

                    # 根据NDWI的计算结果设置水体掩膜
                    # NDWI = (B2 - B9) / (B2 + B9)
                    valid_ndwi_mask = (B2 + B9) != 0  # 确保分母不为零
                    NDWI = np.zeros_like(B2)  # 初始化为零
                    NDWI[valid_ndwi_mask] = (B2[valid_ndwi_mask] - B9[valid_ndwi_mask]) / (
                                B2[valid_ndwi_mask] + B9[valid_ndwi_mask])
                    ##以下两行代码是输出NDWI计算结果, 可以不考虑
                    #ndwi_df = pd.DataFrame({'NDWI': NDWI.flatten()})
                    #ndwi_df.to_excel(r"D:\Phycocyanin_PC\Python\分类及反演\尝试\test\反演_test_result" + "/" + QZ + "_in_process.xlsx", na_rep="Nan", inf_rep="Inf", header=True)
                    water = NDWI > 0
                    water_mask = ~water
                    # 云掩膜和水体掩膜取并集，即就是当一个像素点存在云或不属于水体，就将其赋值参与计算，同理，后面会剔除
                    Mask = water_mask + cloud_mask
                    B1[Mask] = 2
                    B2[Mask] = 1
                    B3[Mask] = 1
                    B4[Mask] = 0
                    B5[Mask] = 1
                    B1[Mask] = 1
                    B2[Mask] = 1
                    B3[Mask] = 1
                    B4[Mask] = 1
                    B5[Mask] = 1
                    B6[Mask] = 1
                    B7[Mask] = 1
                    B8[Mask] = 1
                    B9[Mask] = 1
                    B10[Mask] = 1
                    B11[Mask] = 1
                    B12[Mask] = 1
                    B13[Mask] = 1
                    B14[Mask] = 1
                    B15[Mask] = 1
                    B16[Mask] = 1
                    B17[Mask] = 1
                    B18[Mask] = 1

                    b3[Mask] = 1
                    b5[Mask] = 1
                    b6[Mask] = 1
                    b7[Mask] = 1
                    b15[Mask] = 1
                    b16[Mask] = 1
                    b18[Mask] = 1

                    ##根据FAI的公式计算水华区域的掩膜，并将有水华覆盖区域的部分设置为掩膜部分
                    FAI = pd.DataFrame({'FAI': B9 - B4 + ((B10 - B4) * (865 - 665) / (1016 - 665))})
                    FAI = FAI.values.reshape(-1)
                    FAI_mask = FAI > -0.004
                    ##将FAI的掩膜也与总体的掩膜合并取并集
                    MASK = Mask + FAI_mask

                    ###分类计算
                    Clf_input = pd.DataFrame({'b1': b3, 'b2': B1, 'b3': b5, 'b4': b6, 'b5': b7, 'b6': B2, 'b7': B3,
                                              'b8': B4, 'b9': B5, 'b10': B6, 'b11': B7, 'b12': B8, 'b13': b15,
                                              'b14': b16, 'b15': B9, 'b16': b18, 'b17': B10, 'b11/b7': B11,
                                              'b11/b9': B12, 'b12-b6': B14, 'b12/b6': B15,
                                              '0.01*(b12-b2)': B16, '(b12-b2)/b10': B17, 'b9/b8':B18})
                    Clf_input = Clf_input.values.reshape(-1, 24)

                    Clf_nan_cover = np.isnan(Clf_input)
                    Clf_input[Clf_nan_cover] = 0
                    Clf_input[Clf_input > 100] = 0
                    Clf_input[Clf_input < -100] = 0  ###处理异常值的代码

                    Clf_result = SVC_clf.predict(Clf_input)  ## Clf_result的shape为-1
                    ###制作三个类别的像素提取掩膜
                    Class1_mask = Clf_result == 1
                    Class2_mask = Clf_result == 2

                    ###不分类别，将整幅图像分别按照1，2，3三种分类反演方式计算结果
                    Class1_input = pd.DataFrame({'b7': B3, "b8": B4,"b9": B5,"b10": B6, "b11/b7": B11})

                    Class1_input = Class1_input.values.reshape(-1, 5)
                    Class1_result = PC_class1.predict(Class1_input).reshape(
                        -1)  ##TP_class1.predict()的输出shape为(-1,1),reshape成-1
                    ############################################################

                    Class2_input = pd.DataFrame({"b12-b6": B14, "b12/b6": B15,"0.01*(b12/b8)": B16,"(b12-b2)/b8": B17,"b9/b8": B18})

                    Class2_input = Class2_input.values.reshape(-1, 5)
                    Class2_result = PC_class2.predict(Class2_input).reshape(
                        -1)  ##TP_class2.predict()的输出shape为(-1,1),reshape成-1
                    ############################################################

                    ###以下几行代码为解决bug测试所用，可以忽略
                    # bug_process = pd.DataFrame({'B1':B1,'B2':B2,'B3':B3,'B4':B4,'B5':B5,
                    #                           'Class1_result':Class1_result,
                    #                           'Class2_result':Class2_result,
                    #                           'Class3_result':Class3_result})
                    # bug_process.to_excel(r"D:\Desktop\代码测试\bug" + "/" + QZ + "_in_process.xlsx",na_rep = "Nan",inf_rep = "Inf",header =True)

                    ###将每个类别的计算结果中属于该类别之外的像素点的值剔除，赋值为0
                    Class1_result[~Class1_mask] = 0
                    Class2_result[~Class2_mask] = 0

                    ###最终反演结果等于三个类别的结果的加和（只是位置的叠加，不改变数据）
                    PC_Result = Class1_result + Class2_result
                    PC_Result[MASK] = 0
                    # 将负值设置为零
                    PC_Result[PC_Result < 0] = 0

                    ###准备输出结果
                    mapping = PC_Result.reshape(img_rows, img_cols)

                    # 输出数据
                    A_output = gdal.GetDriverByName("GTiff")  ##可选GTiff、ENVI
                    output_0 = A_output.Create(new_file_path + "/" + QZ + "_PC_result.tif", img_cols, img_rows, 1,
                                               gdal.GDT_Float64)

                    output_0.SetProjection(img_proj)
                    output_0.SetGeoTransform(img_geotrans)

                    band1 = output_0.GetRasterBand(1).WriteArray(mapping)
                    output_0 = None
                    print(image + " 已完成，这是 " + m + " 文件夹中的第 " + str(j) + " 个文件")


            n = n + j

    print("文件夹 " + m + " 目录下所有结果已生成,这是第 " + str(k) + " 个文件夹")

print("--------------------------------------------------------------------------------------")
print("所有文件夹已处理完毕！")
end_time = time.time()  # 结束时间
print("文件夹总数为：" + str(k))
print("处理文件个数为：" + str(n))
print("处理时间:%d" % (end_time - start_time))  # 结束时间-开始时间
sys.exit()