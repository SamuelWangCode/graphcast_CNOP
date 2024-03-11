import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from CNOP.parameters import start_time_dictionary, sampling_time_step


def adjust_datetime(ds, base_time):
    # 将 xarray 数据集中的时间转换为 pandas Timestamp 对象
    times = pd.to_datetime(ds['time'].values)
    # 计算时间偏移量
    time_offset = times - base_time
    # 如果数据集中存在 'batch' 维度，将 'datetime' 调整为二维数组
    if 'batch' in ds.dims:
        time_offset = np.expand_dims(time_offset, axis=0)  # 添加 'batch' 维度
    # 更新 'datetime' 维度
    new_dims = ('batch', 'time') if 'batch' in ds.dims else ('time',)
    ds = ds.assign_coords(datetime=(new_dims, time_offset))
    return ds


def remove_batch_coord(ds):
    # 检查 'batch' 是否在坐标中，并从坐标中移除
    if 'batch' in ds.coords:
        ds = ds.drop_vars('batch')
    return ds


if __name__ == '__main__':
    # 每个台风采多少时间点
    preprocess_address = '/data2/wxz/developer/graphcast/preprocess/'
    for typhoon_name in start_time_dictionary.keys():
        print(typhoon_name)
        start_time_str = start_time_dictionary[typhoon_name]
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H")
        # 计算基础时间戳
        for step in range(sampling_time_step):
            base_time = (start_time - timedelta(hours=6 * (step + 2)))
            file_list = []
            current = base_time.replace(hour=0)
            while current <= start_time.replace(hour=0):
                file_list.append(preprocess_address + current.strftime("%Y-%m-%d") + ".nc")
                current += timedelta(days=1)
            base_timestamp = base_time.strftime("%Y-%m-%d %H:%M")
            start_timestamp = start_time.strftime("%Y-%m-%d %H:%M")
            base_time = pd.Timestamp(base_timestamp)
            datasets = []
            for file in file_list:
                ds = xr.open_dataset(file)
                ds = adjust_datetime(ds, base_time)
                ds = remove_batch_coord(ds)
                ds = ds.sel(time=slice(base_timestamp, start_timestamp))
                datasets.append(ds)
            # 合并数据集
            combined_ds = xr.concat(datasets, dim='time')
            if 'time' in combined_ds['geopotential_at_surface'].dims:
                combined_ds['geopotential_at_surface'] = combined_ds['geopotential_at_surface'].isel(time=0)
            if 'time' in combined_ds['land_sea_mask'].dims:
                combined_ds['land_sea_mask'] = combined_ds['land_sea_mask'].isel(time=0)
            combined_ds.to_netcdf(f'{typhoon_name}_{step}_sample_origin.nc')
            print(typhoon_name + str(step + 1) + 'over')
