import xarray
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from CNOP.parameters import variables_to_modify


def normalize_and_pca(data, n_components=30):
    # 重新塑形为 (300, 201*201)
    reshaped_data = data.reshape(data.shape[-2] * data.shape[-1], -1)
    # 归一化
    print(reshaped_data.max())
    print(reshaped_data.min())
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(reshaped_data)

    # 应用PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(normalized_data)
    restored_data = scaler.inverse_transform(pca_data)
    print(restored_data.max())
    print(restored_data.min())
    return restored_data


if __name__ == '__main__':
    sampling_num = 300
    sampling_all_data = []
    for i in range(sampling_num):
        all_data = []  # 存储每个变量的所有层
        ds = xarray.open_dataset(f'./Doksuri_sampling_result_{i}.nc')
        for var in variables_to_modify:
            if var in ds.data_vars:
                data = ds[var].values
                if len(data.shape) == 4:  # 三维变量
                    for layer in range(data.shape[1]):
                        all_data.append(data[:, layer, :, :])
                        # 处理后的数据保存或操作
                elif len(data.shape) == 3:  # 二维变量
                    all_data.append(data)
        sampling_all_data.append(all_data)
    sampling_all_data = np.squeeze(np.array(sampling_all_data))
    for i in range(56):
        sample_var_level = sampling_all_data[:,i,:,:]
        processed_data = normalize_and_pca(sample_var_level)
        print(processed_data.shape)
