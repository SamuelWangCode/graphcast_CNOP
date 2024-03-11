import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

if __name__ == '__main__':
    typhoon_name = 'Hinnamnor'
    ds = xr.open_dataset(f"./Hinnamnor_sampling_0.nc")
    # 截取指定的经纬度范围
    ds = ds.sel(lat=slice(-5, 55), lon=slice(95, 155))
    # 对于每个时间点绘制图像
    for time_index in range(len(ds['time'])):
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([95, 155, -5, 55], crs=ccrs.PlateCarree())

        # 添加地图特征
        ax.coastlines()
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        # 绘制海平面气压
        mslp = ds['mean_sea_level_pressure'].isel(time=time_index).squeeze()
        levels = np.arange(mslp.min(), mslp.max(), 2)
        mslp_contour = ax.contourf(mslp['lon'], mslp['lat'], mslp, levels=levels, cmap='viridis', extend='both')

        # 添加 colorbar
        plt.colorbar(mslp_contour, ax=ax, orientation='horizontal', pad=0.05, aspect=50, label='MSLP (Pa)')

        # 绘制风速风向
        u_wind = ds['u_component_of_wind'].isel(time=time_index).sel(level=850.0).squeeze()
        v_wind = ds['v_component_of_wind'].isel(time=time_index).sel(level=850.0).squeeze()
        subsample = 4
        quiver = ax.quiver(u_wind['lon'][::subsample], u_wind['lat'][::subsample], u_wind[::subsample, ::subsample],
                           v_wind[::subsample, ::subsample], pivot='middle', color='black', scale=500)

        # 添加标题
        plt.title(f'Wind Field and Mean Sea Level Pressure at Time Index {time_index}')

        plt.savefig(f'./{typhoon_name}_{time_index}_diff.png')
