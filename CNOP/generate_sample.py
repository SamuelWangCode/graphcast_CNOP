import xarray

from CNOP.parameters import start_time_dictionary, sampling_time_step, sampling_step_num

for typhoon_name in start_time_dictionary:
    print(typhoon_name)
    i = 0
    for time_step in range(sampling_time_step):
        origin_ds = xarray.open_dataset(f'./{typhoon_name}_{time_step}_predict_origin.nc')
        for num in range(sampling_step_num):
            sample_ds = xarray.open_dataset(
                f'./{typhoon_name}_sample_predictions_{time_step * sampling_step_num + num}.nc')
            diff = sample_ds - origin_ds
            for time_index in range(len(diff['time'])):
                diff_time = diff.isel(time=time_index).sel(lat=slice(0, 50), lon=slice(100, 150))
                diff_time.to_netcdf(f'{typhoon_name}_sampling_result_{i}.nc')
                i = i + 1
