from datetime import datetime, timedelta

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

infile = xr.open_dataset('./preprocess/IBTrACS.WP.v04r00.nc')
# read variables
lat = infile['usa_lat']  # storm center latitude
lon = infile['usa_lon']  # storm center longitude          # 2 = WP - western north Pacific
stormYear = infile['season'].data  # year based on season
sid = infile['sid']
number = infile['number'].data
time2d = infile['iso_time'].data  # time step
name = infile['name'].data
basin = infile['basin'].data
basin1 = basin[:, 0]
tracktype = infile['track_type'].data

start_time_dictionary = {
    'Hinnamnor': '2022-08-30 00',
    'Muifa': '2022-09-11 12',
    'Nanmadol': '2022-09-15 00',
    'Doksuri': '2023-07-24 12',
    'Khanun': '2023-07-30 12',
    'Saola': '2023-08-26 00',
}
end_time_dictionary = {
    'Hinnamnor': '2022-09-06 00',
    'Muifa': '2022-09-15 00',
    'Nanmadol': '2022-09-19 00',
    'Doksuri': '2023-07-28 12',
    'Khanun': '2023-08-10 12',
    'Saola': '2023-09-02 00',
}

for typhoon_name in start_time_dictionary.keys():
    start_time_str = start_time_dictionary[typhoon_name]
    end_time_str = end_time_dictionary[typhoon_name]
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H")
    start_year = start_time.year
    if typhoon_name == 'Doksuri':
        bname = 'DOKSURI:DORA'
    else:
        bname = typhoon_name.upper()
    tcmask = np.where((stormYear == start_year) & (name == f'{bname}'.encode('utf-8')))
    print(tcmask)
    latselect = lat[tcmask][0]
    lonselect = lon[tcmask][0]
    timeselect = time2d[tcmask]
    # 起始和结束时间
    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H')
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H')
    # 将二进制字符串转换为 datetime 对象
    times = [datetime.strptime(t.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for t in timeselect.flatten() if t]
    # 筛选出 0, 6, 12, 18 时刻的数据
    # 筛选出在起始和结束时间之间的数据
    selected_times = [t for t in times if start_time <= t <= end_time]
    selected_times = [t for t in selected_times if t.hour in [0, 6, 12, 18]]
    # 获取相应时刻的纬度和经度值
    selected_lats = [latselect[i] for i, t in enumerate(times) if t in selected_times]
    selected_lons = [lonselect[i] for i, t in enumerate(times) if t in selected_times]
    initial_lon = selected_lons[0]
    initial_lat = selected_lats[0]
    # 读取预报文件
    ds = xr.open_dataset(f"./forecast/predictions_{typhoon_name}.nc")

    lons, lats = [initial_lon], [initial_lat]
    obs_lons, obs_lats = [initial_lon], [initial_lat]
    # 台风移动速度阈值（每小时）
    speed_threshold = 0.5  # 纬度30°以南
    speed_threshold_north = 1.0  # 纬度30°以北

    # 追踪台风路径
    for time_step in ds.time.values:
        # 上一个台风中心位置
        last_lon, last_lat = lons[-1], lats[-1]
        # 台风可能移动的最大距离（根据纬度调整移动速度）
        max_distance = speed_threshold * 6 if last_lat < 30 else speed_threshold_north * 6

        # 在可能的范围内搜索气压最低点
        slp = ds['mean_sea_level_pressure'].sel(time=time_step)[0]
        lat_range = slice(max(last_lat - max_distance, -90), min(last_lat + max_distance, 90))
        lon_range = slice(max(last_lon - max_distance, -180), min(last_lon + max_distance, 180))
        slp_min_area = slp.sel(lat=lat_range, lon=lon_range)

        # 找到slp_min_area内最小值的位置
        slp_min_coords = np.unravel_index(slp_min_area.argmin(), slp_min_area.shape)
        new_lat, new_lon = slp_min_area.lat.values[slp_min_coords[0]], slp_min_area.lon.values[slp_min_coords[1]]

        # 更新台风中心位置
        lons.append(new_lon)
        lats.append(new_lat)

    # 绘制台风路径
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min(lons) - 10, max(lons) + 10, min(lats) - 10, max(lats) + 10], crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 不在顶部显示标签
    gl.right_labels = False  # 不在右侧显示标签
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    ax.plot(lons, lats, marker='o', color='red', transform=ccrs.PlateCarree())
    ax.plot(selected_lons, selected_lats, marker='o', color='black', transform=ccrs.PlateCarree())
    plt.savefig(f'./forecast/path_{typhoon_name}.png')
