import numpy as np

eval_step_dictionary = {
    'Hinnamnor': 28,
    'Muifa': 14,
    'Nanmadol': 16,
    'Doksuri': 16,
    'Khanun': 44,
    'Saola': 28,
    'Koinu': 28,
}

start_time_dictionary = {
    'Hinnamnor': '2022-08-30 00',
    'Muifa': '2022-09-11 12',
    'Nanmadol': '2022-09-15 00',
    'Doksuri': '2023-07-24 12',
    'Khanun': '2023-07-30 12',
    'Saola': '2023-08-26 00',
    'Koinu': '2023-10-02 00',
}

end_time_dictionary = {
    'Hinnamnor': '2022-09-06 00',
    'Muifa': '2022-09-15 00',
    'Nanmadol': '2022-09-19 00',
    'Doksuri': '2023-07-28 12',
    'Khanun': '2023-08-10 12',
    'Saola': '2023-09-02 00',
    'Koinu': '2023-10-09 00'
}

sampling_time_step = 5
sampling_step_num = 20

variables_to_modify = [
    "temperature", "u_component_of_wind", "v_component_of_wind",
    "specific_humidity", "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
    "mean_sea_level_pressure"
]

lon_grid = 201
lat_grid = 201
level_grid = 13
batch_size = 1
dim = 30
grid_sum = lon_grid * lat_grid * level_grid
beta_arr = np.arange(0.03, 0.61, 0.03)
