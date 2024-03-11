# parameter codes information at https://apps.ecmwf.int/codes/grib/param-db/
era5_pram_codes_levels = [
    130,  # temperature
    131,  # u component of wind
    132,  # v component of wind
    135,  # w vertical velocity
    133,  # q specific humidity
    129,  # z geopotential
]
era5_pram_codes_surface = [
    165,  # 10u 10m component of wind
    166,  # 10v 10m component of wind
    167,  # 2t temperature
    151,  # msl mean sea level pressure
    212,  # tisr toa solar radiation
    129,  # z geopotential at the surface
    172,  # lsm land sea mask
]
era5_precipitation_code = 228

# prod model
prod_model_levels = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550,
                     600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
prod_model_grid = '0.25/0.25'
prod_checkpoint = 'params/params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz'

# operational model
operational_model_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
operational_model_grid = '0.25/0.25'
operational_checkpoint = 'params/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'

# small models
small_model_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
small_model_grid = '1.0/1.0'
small_checkpoint = 'params/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
