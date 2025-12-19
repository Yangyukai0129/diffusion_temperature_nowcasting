import cdsapi
import os

client = cdsapi.Client()

dataset = 'reanalysis-era5-pressure-levels'

years = [str(y) for y in range(1965, 1984)]  # 1965~2024
variables =["v_component_of_wind"]

for var in variables:
    for year in years:
        print(f"下載 {var} {year}")

        request = {
        'product_type': ['reanalysis'],
        'variable': var,
        'year': year,
        'month': ['06','07','08'],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
        ],
        'pressure_level': ['850'],
        'format': 'netcdf',
        "area": [80, -180, 20, 180]
        }

        
        target = f'download_nc/{var}/{year}.nc'
        client.retrieve(dataset, request, target)