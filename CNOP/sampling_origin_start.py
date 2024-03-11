from CNOP.functions import predict
from CNOP.parameters import start_time_dictionary

if __name__ == '__main__':
    # 每个台风采多少时间点
    sampling_time_step = 5
    sampling_file_address = '.'
    for typhoon_name in start_time_dictionary.keys():
        print(typhoon_name)
        for step in range(sampling_time_step):
            sample_file_address = sampling_file_address + '/' + typhoon_name + '_' + str(step) + '_sample_origin.nc'
            eval_targets, predictions = predict(sample_file_address, step + 1)
            predictions.to_netcdf(f'./{typhoon_name}_{step}_predict_origin.nc')
