from CNOP.parameters import start_time_dictionary, sampling_time_step, sampling_step_num
from CNOP.functions import generate_perturbation, predict


for typhoon_name in start_time_dictionary.keys():
    print(typhoon_name)
    for step in range(sampling_time_step):
        print('step:' + str(step))
        for num in range(sampling_step_num):
            print('num:' + str(num))
            new_ds = generate_perturbation(f'./{typhoon_name}_{step}_sample_origin.nc')
            for var in new_ds.data_vars:
                # 检查并清除每个变量的 _FillValue 和 missing_value 编码
                if '_FillValue' in new_ds[var].encoding:
                    del new_ds[var].encoding['_FillValue']
                if 'missing_value' in new_ds[var].encoding:
                    del new_ds[var].encoding['missing_value']
            new_ds.to_netcdf(f'./{typhoon_name}_{step}_sample_generate.nc')
            eval_targets, predictions = predict(f'./{typhoon_name}_{step}_sample_generate.nc', step + 1)
            predictions.to_netcdf(f'./{typhoon_name}_sample_predictions_{step * sampling_step_num + num}.nc')
