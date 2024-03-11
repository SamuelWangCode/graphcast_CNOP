import cdsapi
import os
from datetime import datetime, timedelta
from netCDF4 import Dataset
import numpy as np
from parameter_download import *


def download_era5_data(date_str, levels, grid, save_dir):
    c = cdsapi.Client()
    if os.path.exists(f'{save_dir}/{date_str}_levels.nc'):
        print('levels file exists')
    else:
        # download the data at all levels
        c.retrieve('reanalysis-era5-complete', {
            'date': date_str,
            'levelist': '/'.join([str(x) for x in levels]),
            'levtype': 'pl',
            'param': '/'.join([str(x) for x in era5_pram_codes_levels]),
            'stream': 'oper',
            'time': '00/to/23/by/6',
            'type': 'an',
            'grid': grid,
            'format': 'netcdf',
        }, f'{save_dir}/{date_str}_levels.nc')
    if os.path.exists(f'{save_dir}/{date_str}_surface.nc'):
        print('surface file exists')
    else:
        # download the data at the surface
        c.retrieve('reanalysis-era5-single-levels', {
            'date': date_str,
            'product_type': 'reanalysis',
            'param': '/'.join([str(x) for x in era5_pram_codes_surface]),
            'time': '00/to/23/by/6',
            'grid': grid,
            'format': 'netcdf',
        }, f'{save_dir}/{date_str}_surface.nc')

    if os.path.exists(f'{save_dir}/{date_str}_precipitation.nc'):
        print('precipitation file exists')
    else:
        # download the precipitation data for the given day
        c.retrieve('reanalysis-era5-single-levels', {
            'date': date_str,
            'product_type': 'reanalysis',
            'param': era5_precipitation_code,
            'time': '00/to/18/by/1',
            'grid': grid,
            'format': 'netcdf',
        }, f'{save_dir}/{date_str}_precipitation.nc')
    if os.path.exists(f'{save_dir}/{date_str}_prev_precipitation.nc'):
        print('prev_precipitation file exists')
    else:
        # download the last 6 hours of the previous day
        date = datetime.strptime(date_str, '%Y-%m-%d')
        prev_date = date - timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        c.retrieve('reanalysis-era5-single-levels', {
            'date': prev_date_str,
            'product_type': 'reanalysis',
            'param': era5_precipitation_code,
            'time': '19/to/23/by/1',
            'grid': grid,
            'format': 'netcdf',
        }, f'{save_dir}/{date_str}_prev_precipitation.nc')

    # calculate the total precipitation for the previous 6 hours
    ds_l = Dataset(f'{save_dir}/{date_str}_levels.nc')
    ds_s = Dataset(f'{save_dir}/{date_str}_surface.nc')
    dsp1 = Dataset(f'{save_dir}/{date_str}_prev_precipitation.nc')
    dsp2 = Dataset(f'{save_dir}/{date_str}_precipitation.nc')
    precip = np.concatenate([dsp1['tp'][:], dsp2['tp'][:]], axis=0)
    _, nlat, nlon = precip.shape
    precip_6hr = np.zeros((4, nlat, nlon))
    for i in range(4):
        precip_6hr[i, :, :] = np.sum(precip[i * 6:(i + 1) * 6, :, :], axis=0)

    # create the new combined dataset
    ds = Dataset(f'{save_dir}/{date_str}.nc', 'w', format='NETCDF4')

    # time dimension
    t_dim = ds.createDimension('time', 4)
    t_var = ds.createVariable('time', 'f4', ('time',))
    t_var.units = 'hours since 1900-01-01 00:00:00'
    t_var[:] = ds_s['time'][:]

    # batch dimension
    b_dim = ds.createDimension('batch', 1)
    b_var = ds.createVariable('batch', 'i4', ('batch',))
    b_var.units = 'batch'
    b_var[:] = [0]

    # datetime dimension
    dt_dim = ds.createDimension('datetime', 4)
    dt_var = ds.createVariable('datetime', 'i4', ('batch', 'datetime',))
    dt_var.units = 'hours'
    dt_var[:] = [[int(x * 6) for x in range(4)]]

    # level dimension
    l_dim = ds.createDimension('level', len(levels))
    l_var = ds.createVariable('level', 'f4', ('level',))
    l_var.units = 'millibars'
    l_var[:] = ds_l['level'][:]

    # latitude and longitude dimensions
    lat_dim = ds.createDimension('lat', nlat)
    lat_var = ds.createVariable('lat', 'f4', ('lat',))
    lat_var.units = 'degrees_north'
    lat_var[:] = np.flip(ds_s['latitude'][:], axis=0)
    lon_dim = ds.createDimension('lon', nlon)
    lon_var = ds.createVariable('lon', 'f4', ('lon',))
    lon_var.units = 'degrees_east'
    lon_var[:] = ds_s['longitude'][:]

    # temperature
    t_var = ds.createVariable('temperature', 'f4', ('batch', 'time', 'level', 'lat', 'lon',))
    t_var.units = 'K'
    t_var.missing_value = -32767
    t_var[:] = np.expand_dims(np.flip(ds_l['t'][:], axis=2), axis=0)

    # u component of wind
    u_var = ds.createVariable('u_component_of_wind', 'f4', ('batch', 'time', 'level', 'lat', 'lon',))
    u_var.units = 'm/s'
    u_var.missing_value = -32767
    u_var[:] = np.expand_dims(np.flip(ds_l['u'][:], axis=2), axis=0)

    # v component of wind
    v_var = ds.createVariable('v_component_of_wind', 'f4', ('batch', 'time', 'level', 'lat', 'lon',))
    v_var.units = 'm/s'
    v_var.missing_value = -32767
    v_var[:] = np.expand_dims(np.flip(ds_l['v'][:], axis=2), axis=0)

    # w vertical velocity
    w_var = ds.createVariable('vertical_velocity', 'f4', ('batch', 'time', 'level', 'lat', 'lon',))
    w_var.units = 'Pa/s'
    w_var.missing_value = -32767
    w_var[:] = np.expand_dims(np.flip(ds_l['w'][:], axis=2), axis=0)

    # specific humidity
    q_var = ds.createVariable('specific_humidity', 'f4', ('batch', 'time', 'level', 'lat', 'lon',))
    q_var.units = 'kg/kg'
    q_var.missing_value = -32767
    q_var[:] = np.expand_dims(np.flip(ds_l['q'][:], axis=2), axis=0)

    # geopotential
    z_var = ds.createVariable('geopotential', 'f4', ('batch', 'time', 'level', 'lat', 'lon',))
    z_var.units = 'm^2/s^2'
    z_var.missing_value = -32767
    z_var[:] = np.expand_dims(np.flip(ds_l['z'][:], axis=2), axis=0)

    # 10m u wind
    u10_var = ds.createVariable('10m_u_component_of_wind', 'f4', ('batch', 'time', 'lat', 'lon',))
    u10_var.units = 'm/s'
    u10_var.missing_value = -32767
    u10_var[:] = np.expand_dims(np.flip(ds_s['u10'][:], axis=1), axis=0)

    # 10m v wind
    v10_var = ds.createVariable('10m_v_component_of_wind', 'f4', ('batch', 'time', 'lat', 'lon',))
    v10_var.units = 'm/s'
    v10_var.missing_value = -32767
    v10_var[:] = np.expand_dims(np.flip(ds_s['v10'][:], axis=1), axis=0)

    # 2m temperature
    t2m_var = ds.createVariable('2m_temperature', 'f4', ('batch', 'time', 'lat', 'lon',))
    t2m_var.units = 'K'
    t2m_var.missing_value = -32767
    t2m_var[:] = np.expand_dims(np.flip(ds_s['t2m'][:], axis=1), axis=0)

    # mean sea level pressure
    msl_var = ds.createVariable('mean_sea_level_pressure', 'f4', ('batch', 'time', 'lat', 'lon',))
    msl_var.units = 'Pa'
    msl_var.missing_value = -32767
    msl_var[:] = np.expand_dims(np.flip(ds_s['msl'][:], axis=1), axis=0)

    # toa solar radiation
    tisr_var = ds.createVariable('toa_incident_solar_radiation', 'f4', ('batch', 'time', 'lat', 'lon',))
    tisr_var.units = 'J/m^2'
    tisr_var.missing_value = -32767
    tisr_var[:] = np.expand_dims(np.flip(ds_s['tisr'][:], axis=1), axis=0)

    # precipitation
    precip_var = ds.createVariable('total_precipitation_6hr', 'f4', ('batch', 'time', 'lat', 'lon',))
    precip_var.units = 'm'
    precip_var.missing_value = -32767
    precip_var[:] = np.expand_dims(np.flip(precip_6hr, axis=1), axis=0)

    # geopotential at the surface
    gh_var = ds.createVariable('geopotential_at_surface', 'f4', ('lat', 'lon',))
    gh_var.units = 'm^2/s^2'
    gh_var.missing_value = -32767
    gh_var[:] = np.flip(ds_s['z'][0, :, :], axis=0)

    # land sea mask
    lsm_var = ds.createVariable('land_sea_mask', 'f4', ('lat', 'lon',))
    lsm_var.units = '1'
    lsm_var.missing_value = -32767
    lsm_var[:] = np.flip(ds_s['lsm'][0, :, :], axis=0)

    ds.close()

    return f'{save_dir}/{date_str}.nc'


def download_era5_data_range(start_date, end_date, levels, grid, save_dir):
    # 转换为 datetime 对象
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    # 遍历日期范围内的每一天
    current = start
    while current <= end:
        date_str = current.strftime('%Y-%m-%d')
        download_era5_data(date_str, operational_model_levels, operational_model_grid, save_dir)
        current += timedelta(days=1)  # 增加一天


# download_era5_data_range('2022-09-09', '2022-09-10', operational_model_levels, operational_model_grid, '.')
# 轩岚诺 2022 08-30 ~ 09-06
# download_era5_data_range('2022-08-27', '2022-08-28', operational_model_levels, operational_model_grid, '.')
# 南玛都 2022 09-15 ~ 09-19
# download_era5_data_range('2022-09-16', '2022-09-19', operational_model_levels, operational_model_grid, '.')
# 杜苏芮 2023 07-24 12时 ~ 07-28 12时
# download_era5_data_range('2023-07-22', '2023-07-23', operational_model_levels, operational_model_grid, '.')
# 卡努 2023 07-30 12时 ~ 08-10 12时
# download_era5_data_range('2023-07-29', '2023-07-29', operational_model_levels, operational_model_grid, '.')
# 苏拉 2023 08-26 ~ 09-02
# download_era5_data_range('2023-08-23', '2023-08-24', operational_model_levels, operational_model_grid, '.')
# 小犬 2023 10-02 ~ 10-09
download_era5_data_range('2023-10-01', '2023-10-09', operational_model_levels, operational_model_grid, '.')