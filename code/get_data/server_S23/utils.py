import re
from datetime import datetime
import geojson
import xarray as xr
from io import BytesIO
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from pyproj import CRS
from pyproj import Proj, transform
from pyresample import geometry
from pyresample.kd_tree import resample_nearest
from collections import defaultdict
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from osgeo import gdal, osr, ogr
import geojson
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import lmdb
import pickle
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import rasterio
import numpy as np
import glob
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tqdm import tqdm
import requests
import pandas as pd
import datetime
from urllib.parse import quote
from shapely.geometry import box

def select_s2_by_year(file_list, year=2024):
    out_list = []
    for file in file_list:
        if file.split("L2A_")[1][:4] == str(year):
            out_list.append(file)
    return out_list
    
"比较两个文件的日期是否相同"
def compare_dates(s2, s3):
    """
    比较两个文件的生产日期是否为同一天。
    """
    # 从文件名中提取日期
    date_s3 = s3.split("____")[1][:8]
    date_s2 = s2.split("MSIL2A_")[1][:8]
    if date_s3 == date_s2:
        return True
    else:
        return False


"读取S3的边界范围"
def get_s3_geo(s3):
    with zipfile.ZipFile(s3, 'r') as zip_ref:
        # 获取包含坐标数据的文件
        geo_coords_file = [f for f in zip_ref.namelist() if 'geo_coordinates.nc' in f][0]
        # 获取每个波段的辐射强度数据文件
        radiance_files = [f for f in zip_ref.namelist() if 'radiance.nc' in f]
        with zip_ref.open(geo_coords_file) as geo_file:
            geo_data = xr.open_dataset(BytesIO(geo_file.read()))
            # 提取经纬度，确保它们是二维数组
            latitude = geo_data['latitude'].values
            longitude = geo_data['longitude'].values
        lat_min, lat_max = np.min(latitude), np.max(latitude)
        lon_min, lon_max = np.min(longitude), np.max(longitude)
        bounding_box = geojson.geometry.Polygon([[
            (lon_min, lat_min), (lon_max, lat_min), 
            (lon_max, lat_max), (lon_min, lat_max), 
            (lon_min, lat_min)]])
        # 转换为 OGR Geometry 对象
        footprint_geo_s3 = ogr.CreateGeometryFromJson(str(bounding_box))
    return footprint_geo_s3


"读取S2的边界范围"
def get_s2_geo(s2):
    s = gdal.Open(s2)
    metadata = s.GetMetadata()
    src = gdal.Open(s.GetSubDatasets()[0][0])
    projection = src.GetProjection()
    projection_EPSG = int(s.GetSubDatasets()[0][0].split("EPSG_")[1])
    gt = src.GetGeoTransform()
    lon1 = gt[0]
    lon2 = lon1 + gt[1] * src.RasterXSize
    lat1 = gt[-3]
    lat2 = lat1 + gt[5] * src.RasterYSize

    # 转换坐标
    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    in_srs = osr.SpatialReference()
    in_srs.ImportFromEPSG(projection_EPSG)
    coordinate_transforer = osr.CoordinateTransformation(in_srs, srs4326)

    # 注意，这里是反的，如果不对，可能是和gdal2.x 或者 3.x有关
    lat1, lon1 = coordinate_transforer.TransformPoint(lon1, lat1)[:2]
    lat2, lon2 = coordinate_transforer.TransformPoint(lon2, lat2)[:2]

    geometry = geojson.geometry.Polygon(
        [[(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]])
    footprint_geo_s2 = ogr.CreateGeometryFromJson(str(geometry))
    # print(footprint_geo_s2.Intersection(footprint_geo_s3).GetArea())
    return footprint_geo_s2


"寻找S2有同一天的对"
def match_pairs(input_list):
    grouped_data = defaultdict(list)
    for s2s3_pair in input_list:
        parts = os.path.basename(s2s3_pair[0]).split('_')
        if (parts[1] == "MSIL2A") and (len(parts) >= 6):  # Ensure the file_name has the expected structure
            tile_code = parts[5]  # This is the tile id part
            # Use both tile_code and polygon area as the grouping key
            group_key = tile_code
            grouped_data[group_key].append(s2s3_pair)
    return grouped_data



############################################################
############### 抽取波段，并生城300*300小瓦片 ################
############################################################
"处理Sentinel2数据"
def process_sentinel2_data(zip_file, output_folder, block_size=(300, 300)):
    """
    处理 Sentinel-2 数据，从 ZIP 文件中提取 B02, B03, B04, B8A 波段，
    对 B8A 波段进行下采样，并将所有波段保存为一个多波段的 GeoTIFF 文件。
    
    参数:
        zip_file (str): 输入的 Sentinel-2 数据 ZIP 文件路径。
        output_folder (str): 输出文件夹路径，用于保存最终的 GeoTIFF 文件。
    
    返回:
        str: 最终生成的 GeoTIFF 文件的路径。
    """

    # 打开ZIP文件并读取文件名
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_file_names = zip_ref.namelist()

        # 定义波段关键字
        band_keywords = [["B02", "20m"], ["B03", "20m"], ["B04", "20m"], ["B8A", "20m"]]

        # 用于存储波段文件的路径
        band_files = {}

        # 遍历文件名，自动匹配B02, B03, B04, B8A
        for keyword in band_keywords:
            for file in zip_file_names:
                if keyword[0] in file and keyword[1] in file and file.endswith(".jp2"):
                    band_files[keyword[0]] = file
                    break  # 找到一个匹配项就停止查找

    # 使用GDAL从ZIP文件中读取每个波段
    bands = {}

    example_20m_path = None
    
    for band, file_path in band_files.items():
        # 使用GDAL打开ZIP文件中的波段数据
        band_file_path = '/vsizip/' + zip_file + '/' + file_path 
        if "20m" in band_file_path:
            example_20m_path = band_file_path
        
        ds = gdal.Open(band_file_path)
    
        # 如果打开成功，读取数据
        if ds is not None:
            bands[band] = ds.ReadAsArray()
        else:
            print(f"无法加载 {band} 波段数据")


    # 获取波段数据的空间参考信息
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    # 切割所有波段并保存为300x300的小块
    output_files = []
    height, width = bands["B02"].shape

    # 按300x300切割图像
    block_width, block_height = block_size
    for y in range(0, height, block_height)[:-1]:
        for x in range(0, width, block_width)[:-1]:
            # 计算切割块的尺寸
            x_end = min(x + block_width, width)
            y_end = min(y + block_height, height)

            # 切割并保存每个波段的小块
            block_bands = {}
            for band in bands:
                block_bands[band] = bands[band][y:y_end, x:x_end]

            # 不完整的边缘瓦片剔除
            if block_bands["B02"].shape[1] < block_width or block_bands["B02"].shape[1] < block_height:
                pass
            
            # 创建输出文件路径
            output_file = os.path.join(output_folder, f"tile_{y // block_height}_{x // block_width}.tif")
            output_files.append(output_file)

            # 创建输出数据集
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(output_file, 
                                   block_bands["B02"].shape[1],  # 宽度
                                   block_bands["B02"].shape[0],  # 高度
                                   4,  # 波段数量 (B02, B03, B04, B8A)
                                   gdal.GDT_UInt16)  # 数据类型

            # 设置空间参考信息
            # 左上角坐标 (Top Left X, Top Left Y)
            top_left_x = geo_transform[0] + x * geo_transform[1]
            top_left_y = geo_transform[3] + y * geo_transform[5]

            # 设置空间参考信息
            out_ds.SetGeoTransform([top_left_x, geo_transform[1], geo_transform[2], top_left_y, geo_transform[4], geo_transform[5]])
            out_ds.SetProjection(projection)  # 可根据需要调整投影

            # 写入每个波段
            out_ds.GetRasterBand(1).WriteArray(block_bands["B02"])  # B02 (Blue)
            out_ds.GetRasterBand(2).WriteArray(block_bands["B03"])  # B03 (Green)
            out_ds.GetRasterBand(3).WriteArray(block_bands["B04"])  # B04 (Red)
            out_ds.GetRasterBand(4).WriteArray(block_bands["B8A"])  # B8A (NIR)

            # 关闭输出数据集
            out_ds = None

    return output_files, example_20m_path
    


############################################################
##################### 生成低分辨率的副本 #####################
############################################################
def make_low_res(output_folder, s3_path, example_s2_tif, block_size=(30, 30)):
    """处理所有TIFF文件"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    "在上一级生成s3的tif文件"
    parent_folder = os.path.dirname(output_folder)
    s3_tif = make_s3_tif(s3_path, parent_folder, example_s2_tif)
    make_s3_tile(s3_tif, output_folder, block_size=block_size)
    
"将s3_tif依据某种规则进行剪裁和输出"
def make_s3_tile(s3_tif, output_folder, block_size=(10, 10)):
    src = gdal.Open(s3_tif)
    height = src.RasterYSize
    width = src.RasterXSize

    # 获取源栅格的GeoTransform和投影
    geo_transform = src.GetGeoTransform()
    projection = src.GetProjection()
    output_files = []
    # 按300x300切割图像
    block_width, block_height = block_size
    for y in range(0, height, block_height)[:-1]:
        for x in range(0, width, block_width)[:-1]:
            # 创建输出文件路径
            output_file = os.path.join(output_folder, f"tile_{y // block_height}_{x // block_width}.tif")
            output_files.append(output_file)

            # 创建输出数据集
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(output_file, 
                                   block_width, 
                                   block_height,
                                   src.RasterCount,  # 波段数量 (B02, B03, B04, B8A)
                                   gdal.GDT_UInt16)  # 数据类型

            # 设置空间参考信息
            # 左上角坐标 (Top Left X, Top Left Y)
            top_left_x = geo_transform[0] + x * geo_transform[1]
            top_left_y = geo_transform[3] - y * geo_transform[5]

            # 设置空间参考信息
            out_ds.SetGeoTransform([top_left_x, geo_transform[1], geo_transform[2], top_left_y, geo_transform[4], geo_transform[5]])
            out_ds.SetProjection(projection)  # 可根据需要调整投影

            # 写入每个波段
            for band in range(src.RasterCount):
                array = src.GetRasterBand(band+1).ReadAsArray(xoff=x,yoff=y,win_xsize=block_width,win_ysize=block_height)
                out_ds.GetRasterBand(band+1).WriteArray(array)
            # 关闭输出数据集
            out_ds = None

    return output_files


"从给定的tif中抽取resample库所需的投影"
def generate_projection_info(example_tif):
    dataset = gdal.Open(example_tif)
    
    # 获取数据集的投影信息
    projection = dataset.GetProjection()

    epsg = "32" + projection.split('AUTHORITY["EPSG","32')[1][:3]
    utm_zone = projection.split('PROJCS["WGS 84 / UTM zone ')[1][:3]


    area_id = 'sentinel2_grid'
    description = f'Sentinel-2 Grid (Reprojected to UTM Zone {utm_zone})'
    proj_id = f'utm_{epsg}'
    projection = {'proj': 'utm', 'zone': int(utm_zone[:2]), 'datum': 'WGS84', 'units': 'm'}

    
    return area_id, description, proj_id, projection_info


"将S3依据给定的example_tif重新投影和剪裁"
def make_s3_tif(s3_path, output_folder, example_tif):
    resolution_factor = 8 # Sentinel2和3之间的重采样倍率差距
    
    # Step 1: 创建输出文件路径
    output_tif_path = os.path.join(output_folder, os.path.basename(s3_path).split(".")[0] + ".tif")
    
    # Step 2: 打开 .zip 文件，读取其中的文件
    with zipfile.ZipFile(s3_path, 'r') as zip_ref:
        # 获取包含坐标数据的文件
        geo_coords_file = [f for f in zip_ref.namelist() if 'geo_coordinates.nc' in f][0]
        # 获取每个波段的辐射强度数据文件
        radiance_files = [f for f in zip_ref.namelist() if 'radiance.nc' in f]
        # print(f"Geo coordinates file: {geo_coords_file}")
        # print(f"Radiance files: {radiance_files}")
    
        # Step 3: 读取 geo_coordinates.nc 文件以获取坐标
        with zip_ref.open(geo_coords_file) as geo_file:
            geo_data = xr.open_dataset(BytesIO(geo_file.read()))
            # 提取经纬度，确保它们是二维数组
            latitude = geo_data['latitude'].values
            longitude = geo_data['longitude'].values
    
        # Step 4: 读取辐射强度数据（例如 Oa03_radiance.nc）
        bands = {}
        for radiance_file in radiance_files:
            band_name = os.path.basename(radiance_file).split('_')[0]  # 从文件名中提取波段名，例如 "Oa03"
            # print(band_name)
            # if band_name in ["Oa03", "Oa04", "Oa08", "Oa10"]:
            if band_name in ["Oa04", "Oa06", "Oa08", "Oa17"]:
                with zip_ref.open(radiance_file) as rad_file:
                    rad_data = xr.open_dataset(BytesIO(rad_file.read()), decode_cf=False)
                    bands[band_name] = rad_data.to_dataarray()  # 提取辐射数据
    
        # print(f"Extracted bands: {list(bands.keys())}")

    swath_def = geometry.SwathDefinition(lons=longitude, lats=latitude)
    area_id, description, proj_id, projection, new_width, new_height, area_extent, proj, new_pixel_size_x, new_pixel_size_y = extract_tif_info(example_tif, resolution_factor=resolution_factor)
    area_def = geometry.AreaDefinition(area_id, description, proj_id, projection, new_width, new_height, area_extent)
    
    # Step 9: 重投影并重采样数据
    reprojected_bands = {}
    for band_name, band_data in bands.items():
        # print(f"Resampling band {band_name}")
        
        # 使用最近邻方法进行重采样
        reprojected_band = resample_nearest(swath_def, np.array(band_data), area_def, fill_value=65535, radius_of_influence=500,
                                            # nprocs=35,
                                            epsilon=0.5)
            
        # 存储重投影后的数据
        reprojected_bands[band_name] = reprojected_band
    
    # Step 10: 创建输出 GeoTIFF 文件
    with rasterio.open(
        output_tif_path, 'w', driver='GTiff', 
        height=new_height, width=new_width, count=len(reprojected_bands), 
        dtype='uint16', crs=proj,  # 目标投影坐标系
        transform=from_origin(area_extent[0], area_extent[-1], new_pixel_size_x, new_pixel_size_y)
        
    ) as dst:
        for i, (band_name, reprojected_band) in enumerate(reprojected_bands.items(), 1):
            # print(f"Writing band {band_name} to GeoTIFF")
            dst.write(reprojected_band, i)  # 写入每个波段到 GeoTIFF
            
    return output_tif_path


def extract_tif_info(tif_path, resolution_factor=30):
    """
    从 TIFF 文件中提取信息并生成 AreaDefinition 所需的输入参数。
    
    :param tif_path: 输入的 TIFF 文件路径。
    :param resolution_factor: 目标分辨率与原分辨率的倍数，默认为 30，表示新的分辨率为原始分辨率的 1/30。
    :return: 返回包含 area_id, description, proj_id, projection, width, height, area_extent 的元组。
    """
    # 打开 TIFF 文件
    ds = gdal.Open(tif_path)
    
    if not ds:
        raise ValueError(f"无法打开 TIFF 文件: {tif_path}")
    
    # 获取投影信息
    proj = ds.GetProjection()
    
    # 使用 osr 来解析投影信息
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(proj)
    
    # 提取投影的 EPSG 代码（如果有）
    epsg_code = None
    if spatial_ref.IsProjected():
        epsg_code = spatial_ref.GetAttrValue('AUTHORITY', 1)  # 获取 EPSG 代码
    
    # 获取地理变换信息 (GeoTransform)
    geo_transform = ds.GetGeoTransform()
    
    # 获取原始的分辨率 (像素大小)
    pixel_size_x = geo_transform[1]  # X 方向的像素大小
    pixel_size_y = geo_transform[5]  # Y 方向的像素大小
    
    # 计算新的分辨率（原始分辨率的 1/30）
    new_pixel_size_x = pixel_size_x * resolution_factor
    new_pixel_size_y = pixel_size_y * resolution_factor
    
    # 获取影像的宽度和高度
    width = ds.RasterXSize
    height = ds.RasterYSize
    
    # 计算新的宽度和高度
    new_width = int(width * (pixel_size_x / new_pixel_size_x))
    new_height = int(height * (pixel_size_y / new_pixel_size_y))
    
    # 获取影像的地理范围（左、上、右、下）
    min_x = geo_transform[0]
    max_y = geo_transform[3]
    max_x = min_x + geo_transform[1] * width
    min_y = max_y + geo_transform[5] * height
    
    # 构造 AreaDefinition 的参数
    epsg = "32" + proj.split('AUTHORITY["EPSG","32')[1][:3]
    utm_zone = proj.split('PROJCS["WGS 84 / UTM zone ')[1][:3]
    area_id = 'sentinel2_grid'
    description = f'Sentinel-2 Grid (Reprojected to UTM Zone {utm_zone})'
    proj_id = f'utm_{epsg}'
    projection = {'proj': 'utm', 'zone': int(utm_zone[:2]), 'datum': 'WGS84', 'units': 'm'}
    
    area_extent = (min_x, min_y, max_x, max_y)
    
    # 输出结果
    return area_id, description, proj_id, projection, new_width, new_height, area_extent, proj, new_pixel_size_x, new_pixel_size_y


# 错误自动重新执行
def retry_function(func, *args, **kwargs):
    retry_count = 0
    while True:
        try:
            return func(*args, **kwargs)  # 尝试执行传入的函数
        except Exception as e:
            retry_count += 1  # 出现异常时，增加重试次数
            print(f"Attempt {retry_count} failed: {e}")
            print("Retrying...")  # 打印重试信息
    

# 处理一个tile
def make_tile_file(key, t1s2, t1s3, t2s2, t2s3, outfolder, zip_outfolder):
    try:
        tile_folder = os.path.join(outfolder, key)
        zip_outfile = os.path.join(zip_outfolder, key)
        tile_folder_t1f = os.path.join(tile_folder, "t1f")
        tile_folder_t2f = os.path.join(tile_folder, "t2f")
        tile_folder_t1c = os.path.join(tile_folder, "t1c")
        tile_folder_t2c = os.path.join(tile_folder, "t2c")
    
        os.makedirs(tile_folder_t1f, exist_ok=True)
        os.makedirs(tile_folder_t2f, exist_ok=True)
        os.makedirs(tile_folder_t1c, exist_ok=True)
        os.makedirs(tile_folder_t2c, exist_ok=True)
        
        with open(os.path.join(tile_folder, 'output.txt'), 'w') as file:
            for name in [t1s2, t1s3, t2s2, t2s3]:
                file.write("\n".join(name))  # 每个元素占一行
    
        "切分Sentinel-2"
        output_files, example_20m_path_0 = process_sentinel2_data(t1s2, tile_folder_t1f, block_size=(1024, 1024))
        # print("****************************T1F FINISHED!****************************")
        output_files, example_20m_path_1 = process_sentinel2_data(t2s2, tile_folder_t2f, block_size=(1024, 1024))
        # print("****************************T2F FINISHED!****************************")
    
        make_low_res(tile_folder_t1c, t1s3, example_20m_path_0, block_size=(1024//8, 1024//8))
        # print("****************************T1C FINISHED!****************************")
        make_low_res(tile_folder_t2c, t2s3, example_20m_path_1, block_size=(1024//8, 1024//8))
        # print("****************************T2C FINISHED!****************************")
    except:
        print("fail")
        raise
    return zip_outfile, tile_folder

def parallel_process_make_tile_file(tasks, tile_root_folder, zip_root_folder, max_workers=8):
    """使用线程池并行处理"""
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for key, t1s2, t1s3, t2s2, t2s3 in tasks:
            future = executor.submit(
                make_tile_file, key, t1s2, t1s3, t2s2, t2s3, tile_root_folder, zip_root_folder
            )
            futures.append(future)
        
        # 等待完成并统计
        for future in tqdm(futures, total=len(futures), desc="处理图块"):
            try:
                future.result()
                success_count += 1
            except:
                pass
    
    print(f"处理完成: 成功 {success_count}, 失败 {len(tasks) - success_count}")
    return success_count
    
# 检查是否有0或者nan的S3数据
def check_tile_folder(tile_folder):
    # 找出所有以S3开头、.tif结尾的文件
    tif_files = [
        os.path.join(tile_folder, f)
        for f in os.listdir(tile_folder)
        if f.startswith("S3") and f.endswith(".tif")
    ]

    # 判断数量是否为2
    if len(tif_files) != 2:
        return False

    # 检查每个.tif文件是否符合条件
    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            data = src.read(1)  # 读取第一个波段（通常为单波段数据）

            # 检查是否有NaN或者小于等于0的值
            if np.isnan(data).any() or np.any(data <= 0):
                return False
    return True


"寻找S2和S3配对"


def build_query_url(start_date, end_date, wkt):
    start_str = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59.000Z")

    base_prefix = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter="
    str_in_name = f"contains(Name,'OL_1_EFR')"
    collection = f"Collection/Name eq 'SENTINEL-3'"
    roi = f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
    time_range = f"ContentDate/Start ge {start_str} and ContentDate/Start le {end_str}"
    top_expand = "&$top=1000&$expand=Assets"

    query_url = base_prefix + str_in_name + " and " + collection + " and " + roi + " and " + time_range + top_expand
    return quote(query_url, safe=':/&?=(),\'')

def get_s3_with_geo_from_esa(month, s3_path, year=2024):
    month = int(month)

    min_lon, min_lat = 73.5, 18.0
    max_lon, max_lat = 135.1, 53.6
    search_lim = box(min_lon, min_lat, max_lon, max_lat)

    start_date = datetime.date(year, month, 1)

    if month == 12:
        end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        end_date = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    
    current_start = start_date
    all_products = []
    while current_start <= end_date:
        current_end = min(current_start + datetime.timedelta(days=15 - 1), end_date)
        query_url = build_query_url(current_start, current_end, search_lim.wkt)
        resp = requests.get(query_url, timeout=60)
        resp.raise_for_status()
        data_json = resp.json()
        product = data_json.get("value", [])
        all_products += product
        current_start = current_end + datetime.timedelta(days=1)

    s3_geo = {}
    for product in all_products:
        s3_geo[product["Name"] + ".zip"] = product["GeoFootprint"]


    out_s3_path = []
    for s3 in s3_path:
        if os.path.basename(s3) in s3_geo.keys():
            out_s3_path.append([s3, ogr.CreateGeometryFromJson(str(s3_geo[os.path.basename(s3)]))])

    return out_s3_path


def get_filtered_sentinel23pairs(filtered_sentinel2_cloud_zip_files, sentinel3_zip_files_with_geo, max_workers=None):
    filtered_sentinel23pairs_cloud_zip_files = []
    
    # 使用字典根据日期索引 Sentinel-3 数据
    s3_date_dict = defaultdict(list)
    for s3_file, s3_geo in sentinel3_zip_files_with_geo:
        s3_date = s3_file.split("____")[1][:8]  # 获取S3数据的日期
        s3_date_dict[s3_date].append((s3_file, s3_geo))

    # 使用线程池（避免序列化问题）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 定义处理函数（在线程中可以使用局部函数）
        def process_s2(s2):
            s2_geo = get_s2_geo(s2)
            s2_date = s2.split("MSIL2A_")[1][:8]

            if s2_date in s3_date_dict:
                s3_list = []
                for s3_file, s3_geo in s3_date_dict[s2_date]:
                    if s3_geo.Contains(s2_geo):
                        s3_list.append(s3_file)
                
                if s3_list:
                    return [s2, s3_list]
            return None
        
        # 并行处理
        results = list(tqdm(executor.map(process_s2, filtered_sentinel2_cloud_zip_files), 
                           total=len(filtered_sentinel2_cloud_zip_files), 
                           desc="S2S3配对"))
    
    # 过滤有效结果
    filtered_sentinel23pairs_cloud_zip_files = [result for result in results if result is not None]
    
    print(f"成功配对数量: {len(filtered_sentinel23pairs_cloud_zip_files)}")
    return filtered_sentinel23pairs_cloud_zip_files

    


"Get all file"
def find_zip_files(root_folder):
    zip_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".zip") and "MSIL2A" in file:
                zip_files.append(os.path.join(root, file))
    return zip_files

"Get and fileter the percentage >= 1"
def filter_data_percentage(zip_file):
    try:
        s = gdal.Open(zip_file)
    
        metadata = s.GetMetadata()
        src = gdal.Open(s.GetSubDatasets()[0][0])
        projection = src.GetProjection()
        projection_EPSG = int(s.GetSubDatasets()[0][0].split("EPSG_")[1])
    
        gt = src.GetGeoTransform()
        lon1 = gt[0]
        lon2 = lon1 + gt[1] * src.RasterXSize
        lat1 = gt[-3]
        lat2 = lat1 + gt[5] * src.RasterYSize
    
        # 转换坐标
        srs4326 = osr.SpatialReference()
        srs4326.ImportFromEPSG(4326)
        in_srs = osr.SpatialReference()
        in_srs.ImportFromEPSG(projection_EPSG)
        coordinate_transforer = osr.CoordinateTransformation(in_srs, srs4326)
    
        # 注意，这里是反的，如果不对，可能是和gdal2.x 或者 3.x有关
        lat1, lon1 = coordinate_transforer.TransformPoint(lon1, lat1)[:2]
        lat2, lon2 = coordinate_transforer.TransformPoint(lon2, lat2)[:2]
    
        geometry = geojson.geometry.Polygon(
            [[(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]])
    
        footprint_geo = ogr.CreateGeometryFromJson(str(geometry))
        footprint_with_data_geo = ogr.CreateGeometryFromWkt(metadata['FOOTPRINT'])
    
        # 获得geojson
        footprint = footprint_geo.ExportToJson()
        footprint_with_data = footprint_with_data_geo.ExportToJson()
    
        # 获得数据占比
        data_percentage = footprint_with_data_geo.GetArea() / footprint_geo.GetArea()
        if data_percentage > 1:
            data_percentage = 1
        return data_percentage
    except:
        return 0


def filter_files(zip_files):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(filter_data_percentage, zip_files)
    return [file for file, percentage in zip(zip_files, results) if percentage >= 0.9]
    


"Get and fileter the percentage >= 0.1"
def filter_cloud(zip_file):
    cloud_limit = 0.1 # 1
    try:
        s = gdal.Open(zip_file)
        metadata = s.GetMetadata()
        cloud = metadata['CLOUD_COVERAGE_ASSESSMENT']
        if float(cloud) <= cloud_limit:
            return 1
        else:
            return 0 
    except Exception as e:
        raise(e)
        return 0

def filter_files_clouds(zip_files):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(filter_cloud, zip_files)
    return [file for file, cloud in zip(zip_files, results) if cloud == 1]





"""生成sample.pkl"""
"从S3中看是否有0或者nan值"
def find_all_s3_tifs(root_dir):
    s3_tif_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("S3") and file.endswith(".tif"):
                s3_tif_files.append(os.path.join(root, file))
    return s3_tif_files

"找到有nan或者0的文件夹"
def find_nan_tifs(file_paths):
    nan_files = []
    for file_path in tqdm(file_paths, desc="检查 NaN 和最小值为0"):
        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                has_nan = np.isnan(data).any()
                has_zero_min = np.min(data) == 0
                if has_nan or has_zero_min:
                    nan_files.append(os.path.dirname(file_path))
        except Exception as e:
            nan_files.append(os.path.dirname(file_path))
            print(f"⚠️ 无法读取文件 {file_path}: {e}")
    return nan_files

def filter_full(pkl_path_filtered, pkl_path): #
    with open(pkl_path, "rb") as f:
        all_samples = pickle.load(f)
    
    filtered_samples = []
    for data in tqdm(all_samples):
        good = 1
        for i in [data[1][0], data[1][2]]: # 筛选S2中的nan
            with rasterio.open(os.path.join(data[0], i)) as src:
                a = src.read()
                if np.isnan(np.max(a)):
                    good = 0
        if good:
            filtered_samples.append(data)
        else:
            print(data[0])
            
    with open(pkl_path_filtered, "wb") as f:
        pickle.dump(all_samples, f)

def make_pkl(pkl_path, folder_root):
    # S3中如果有Nan，则整个文件舍弃
    all_samples = []
    bad_num = 0
    for sub_folder in tqdm(glob.glob(os.path.join(folder_root, '*')), desc="解析样本数据进度"):
        try:
            # 假设文件夹内结构如下：每个 zip 解压后的文件夹包含 't1c', 't1f', 't2c', 't2f'
            t1c_dir = [os.path.basename(path) for path in glob.glob(os.path.join(sub_folder, 't1c', '*.tif'))]
            t1f_dir = [os.path.basename(path) for path in glob.glob(os.path.join(sub_folder, 't1f', '*.tif'))]
            t2c_dir = [os.path.basename(path) for path in glob.glob(os.path.join(sub_folder, 't2c', '*.tif'))]
            t2f_dir = [os.path.basename(path) for path in glob.glob(os.path.join(sub_folder, 't2f', '*.tif'))]
            for fname in t1c_dir:
                paths = [f for f in [t1c_dir, t1f_dir, t2c_dir, t2f_dir] if fname in f]
                if len(paths) == 4:
                    items = [sub_folder, ['t1c/' + fname, 't1f/' + fname, 't2c/' + fname, 't2f/' + fname]]
                    # for item in items[1]:
                    #     all_samples.append(items)
                    all_samples.append(items)
        except:
            continue
    
    with open(pkl_path, "wb") as f:
        pickle.dump(all_samples, f)


def has_zero_or_nan(arr):
    """
    判断数组中是否包含0或NaN
    """
    arr = np.array(arr)
    
    # 检查是否有0
    has_zero = np.any(arr == 0)

    # 
    
    # 检查是否有NaN
    has_nan = np.any(np.isnan(arr))
    
    return has_zero or has_nan

def has_bad_value(arr, bad_value):
    arr = np.array(arr)
    has_bad = np.any(arr == bad_value)
    has_nan = np.any(np.isnan(arr))

    return has_bad or has_nan

def enlarge_size(array, factor):
    array = array.transpose([1,2,0])
    if len(array.shape) == 3:
        out = np.zeros(shape=(array.shape[0] * factor, array.shape[1] * factor, array.shape[-1])).astype(array.dtype)
    else:
        out = np.zeros(shape=(array.shape[0] * factor, array.shape[1] * factor)).astype(array.dtype)

    for x in range(array.shape[1]):
        for y in range(array.shape[0]):
            out[y * factor:(y + 1) * factor, x * factor:(x + 1) * factor] = array[y, x]

    return out.transpose([2,0,1])

import blosc
import io

def pack_arrays(arr_list, cname='lz4', clevel=3):
    """压缩 list 的 numpy 数组"""
    buf = io.BytesIO()
    for arr in arr_list:
        np.save(buf, arr, allow_pickle=False)
    raw_bytes = buf.getvalue()
    compressed = blosc.compress(raw_bytes, cname=cname, clevel=clevel)
    return compressed
    
def convert_sentinel_to_lmdb(samples, lmdb_path, map_size=130 * 1024**3, batch_size=1000):
    """
    将Sentinel数据集转换为LMDB格式，每 batch_size 个样本提交一次事务
    
    Args:
        samples: 样本列表，格式 [[base_path, [file1, file2, file3, file4]], ...]
        lmdb_path: LMDB数据库保存路径
        map_size: LMDB数据库最大大小
        batch_size: 每多少个样本 commit 一次
    """


    # 创建输出目录
    os.makedirs(lmdb_path, exist_ok=True)
    
    total_samples = len(samples) * 4
    print(f"开始转换 {len(samples)} 个样本，共 {total_samples} 个子样本到 LMDB...")

    env = lmdb.open(lmdb_path, map_size=map_size)
    sample_count = 0

    # 写入元数据
    with env.begin(write=True) as txn:
        metadata = {
            'total_samples': total_samples,
            'sample_structure': '每个样本包含4个numpy数组: [LR_t1, HR_t1, LR_t2, HR_t2]'
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))

    # 开始批事务写入
    txn = env.begin(write=True)
    for sample_idx, sample in enumerate(tqdm(samples, desc="处理主样本")):
        base_path, file_names = sample

        for i in range(2):  # 行索引 0-1
            for j in range(2):  # 列索引 0-1
                try:
                    arrays = []
                    
                    # 文件1: LR_t1
                    with rasterio.open(os.path.join(base_path, file_names[0])) as src:
                        arr = src.read(window=((i*64, (i+1)*64), (j*64, (j+1)*64))).astype(np.uint16)
                        if has_bad_value(arr, 65535):
                            continue
                        arrays.append(arr)
        
                    # 文件2: HR_t1
                    with rasterio.open(os.path.join(base_path, file_names[1])) as src:
                        arr = src.read(window=((i*512, (i+1)*512), (j*512, (j+1)*512))).astype(np.uint16)
                        if has_bad_value(arr, 0):
                            continue
                        arrays.append(arr)
        
                    # 文件3: LR_t2
                    with rasterio.open(os.path.join(base_path, file_names[2])) as src:
                        arr = src.read(window=((i*64, (i+1)*64), (j*64, (j+1)*64))).astype(np.uint16)
                        if has_bad_value(arr, 65535):
                            continue
                        arrays.append(arr)
        
                    # 文件4: HR_t2
                    with rasterio.open(os.path.join(base_path, file_names[3])) as src:
                        arr = src.read(window=((i*512, (i+1)*512), (j*512, (j+1)*512))).astype(np.uint16)
                        if has_bad_value(arr, 0):
                            continue
                        arrays.append(arr)
        
                    if len(arrays) == 4: 
                        # 创建样本数据
                        sample_data = {
                            'arrays': pack_arrays(arrays),
                            'metadata': {
                                'tile': os.path.basename(base_path),
                                'sub_index': (i, j),  # 改为存储行列索引
                            }
                        }
        
                        key = f"{sample_count:08d}".encode()
                        value = pickle.dumps(sample_data)
                        txn.put(key, value)
        
                        sample_count += 1
        
                        # 每 batch_size 个样本 commit 一次事务
                        if sample_count % batch_size == 0:
                            txn.commit()
                            txn = env.begin(write=True)
        
                except Exception as e:
                    print(f"处理样本 {sample_idx}-({i},{j}) 时出错: {e}")
                    continue

    # 提交最后一个事务
    txn.commit()


    # 重新写入更新后的元数据
    with env.begin(write=True) as txn:
        metadata = {
            'total_samples': sample_count,  # 使用实际成功转换的样本数量
            'original_total_samples': total_samples,  # 保留原始总数用于参考
            'sample_structure': '每个样本包含4个numpy数组: [LR_t1, HR_t1, LR_t2, HR_t2]'
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))
    
    env.sync()
    env.close()
    print(f"转换完成! 成功转换 {sample_count}/{total_samples} 个样本到 {lmdb_path}")





import os
import re
from collections import defaultdict

def group_files_by_month(folder_list):
    """
    从文件夹列表中获取包含OL_1_EFR____的文件路径，并按月份分组
    
    Args:
        folder_list: 包含文件夹路径的列表
    
    Returns:
        dict: 按月份分组的文件路径列表，键为月份(01-12)，值为文件完整路径列表
    """
    # 初始化12个月份的列表
    monthly_files = {f"{i:02d}": [] for i in range(1, 13)}
    
    # 正则表达式匹配文件名中的日期部分
    pattern = r'OL_1_EFR____(\d{4})(\d{2})\d{2}T'
    
    for folder_path in folder_list:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在 - {folder_path}")
            continue
            
        print(f"正在扫描文件夹: {folder_path}")
        
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 检查文件名是否包含目标字符串
            if "OL_1_EFR____" in filename:
                # 获取文件的完整路径
                file_path = os.path.join(folder_path, filename)
                
                # 使用正则表达式提取月份
                match = re.search(pattern, filename)
                if match:
                    year = match.group(1)
                    month = match.group(2)
                    
                    # 验证月份是否有效 (01-12)
                    if month in monthly_files:
                        monthly_files[month].append(file_path)
                        print(f"找到文件 [{month}月]: {file_path}")
                    else:
                        print(f"警告: 无效月份 {month} 在文件 {filename}")
                else:
                    print(f"警告: 无法解析日期在文件 {filename}")
    
    # 打印统计信息
    print("\n=== 文件统计 ===")
    total_files = 0
    for month in sorted(monthly_files.keys()):
        count = len(monthly_files[month])
        total_files += count
        print(f"{month}月: {count} 个文件")
    print(f"总计: {total_files} 个文件")
    
    return monthly_files


#######可视化MGRS##############
import folium
from mgrs import MGRS
import math

def mgrs_to_latlon(mgrs_string):
    """
    将 MGRS 坐标转换为经纬度
    """
    try:
        m = MGRS()
        lat, lon = m.toLatLon(mgrs_string)
        return lat, lon
    except Exception as e:
        print(f"转换 MGRS '{mgrs_string}' 时出错: {e}")
        return None, None

def create_mgrs_tile_map(mgrs_list, output_file="mgrs_tiles_map.html"):
    """
    创建 MGRS tile 的可视化地图
    
    参数:
    mgrs_list: MGRS tile ID 列表
    output_file: 输出HTML文件名
    """
    
    # 计算所有tile的中心点
    lats = []
    lons = []
    valid_tiles = []
    
    for mgrs_id in mgrs_list:
        lat, lon = mgrs_to_latlon(mgrs_id[1:])
        if lat is not None and lon is not None:
            lats.append(lat)
            lons.append(lon)
            valid_tiles.append((mgrs_id, lat, lon))
    
    if not valid_tiles:
        print("没有有效的 MGRS tile 可以显示")
        return
    
    # 计算地图中心点
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # 创建地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # 为每个tile添加标记和边界框
    for mgrs_id, lat, lon in valid_tiles:
        # 添加标记
        folium.Marker(
            [lat, lon],
            popup=f"MGRS: {mgrs_id}",
            tooltip=f"点击查看 {mgrs_id}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        # 添加圆形表示tile区域（近似）
        folium.Circle(
            location=[lat, lon],
            radius=5000,  # 5km半径，可根据需要调整
            popup=f"Tile Area: {mgrs_id}",
            color='red',
            fill=True,
            fillOpacity=0.2
        ).add_to(m)
    
    # 添加标题
    title_html = '''
                 <h3 align="center" style="font-size:20px"><b>MGRS Tiles 可视化</b></h3>
                 <p align="center">总共有 {} 个 tile</p>
                 '''.format(len(valid_tiles))
    m.get_root().html.add_child(folium.Element(title_html))
    
    # 保存地图
    m.save(output_file)
    print(f"地图已保存为: {output_file}")
    print(f"成功可视化了 {len(valid_tiles)} 个 MGRS tile")
    
    return m

def create_advanced_mgrs_map(mgrs_list, output_file="advanced_mgrs_map.html"):
    """
    创建更高级的 MGRS tile 可视化，包含网格和不同颜色
    """
    
    valid_tiles = []
    for mgrs_id in mgrs_list:
        lat, lon = mgrs_to_latlon(mgrs_id)
        if lat is not None and lon is not None:
            valid_tiles.append((mgrs_id, lat, lon))
    
    if not valid_tiles:
        return None
    
    # 计算中心点
    lats = [lat for _, lat, _ in valid_tiles]
    lons = [lon for _, _, lon in valid_tiles]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # 创建地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9
    )
    
    # 使用不同颜色区分tile
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
              'gray', 'black', 'lightgray']
    
    # 为每个tile添加特性
    for i, (mgrs_id, lat, lon) in enumerate(valid_tiles):
        color = colors[i % len(colors)]
        
        # 添加正方形表示tile
        folium.Rectangle(
            bounds=[
                [lat - 0.05, lon - 0.05],  # 西南角
                [lat + 0.05, lon + 0.05]   # 东北角
            ],
            popup=f"""
            <b>MGRS Tile:</b> {mgrs_id}<br>
            <b>经纬度:</b> {lat:.4f}, {lon:.4f}<br>
            <b>序号:</b> {i+1}
            """,
            tooltip=f"Tile: {mgrs_id}",
            color=color,
            fill=True,
            fillOpacity=0.3,
            weight=2
        ).add_to(m)
        
        # 添加中心点标记
        folium.Marker(
            [lat, lon],
            popup=f"Center: {mgrs_id}",
            tooltip=f"中心点: {mgrs_id}",
            icon=folium.Icon(color=color, icon='star')
        ).add_to(m)
    
    # 添加图例
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><strong>MGRS Tiles 图例</strong></p>
    <p>● 彩色方块: Tile 区域</p>
    <p>★ 星星: Tile 中心点</p>
    <p>总数: {}</p>
    </div>
    '''.format(len(valid_tiles))
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m