#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Jay Fahlen
#          Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

from herbie import Herbie
import numpy as np
import pandas as pd
import json
import os
import cdsapi
import requests
import xarray as xr
import datetime
import click
import tempfile
import logging

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument(
    'current_wind_speed_csv_filename',
    type=click.Path(
        # The file may not exist yet (we may be creating a new version),
        # so we only require the directory to be present.
        exists=False,
        file_okay=True,
        dir_okay=False,
    ),
    metavar='CSV_PATH',
)
@click.option(
    '--plume-file',
    default='/store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='Full path to the plume GeoJSON file.',
    show_default=True,
)
@click.option(
    '--write-rate',
    default=10,
    type=click.IntRange(min=1),
    help='How often (in plumes) the CSV is flushed to disk. '
         '1 = after every plume, 10 = every ten plumes.',
    show_default=True,
)
@click.option(
    '--plume-list',
    multiple=True,
    type=str,
    help='Explicit list of plume IDs to process. '
         'If supplied, overrides the IDs read from the plume file. '
         'Give one ID per flag, e.g. "--plume-list ID1 --plume-list ID2".',
)
def cli(
    current_wind_speed_csv_filename: str,
    plume_file: str,
    write_rate: int,
    plume_list: tuple,
) -> None:
    """Create or update a wind‑speed CSV for EMIT plume lists.

    The positional argument ``CSV_PATH`` must follow the pattern
    ``/full/path/filename_0000.csv`` – the script will write a new file
    whose numeric suffix is incremented automatically.
    """
    # Convert the Click ``multiple`` tuple into the ``list`` the function expects,
    # or ``None`` if the user didn't pass the flag.
    plume_list_arg = list(plume_list) if plume_list else None

    # Normalise the output path (optional but handy for downstream code).
    csv_path = os.path.abspath(current_wind_speed_csv_filename)

    update_EMIT_plume_list_windspeeds(
        csv_path,
        plume_file=plume_file,
        write_rate=write_rate,
        plume_list=plume_list_arg,
    )

def update_EMIT_plume_list_windspeeds(current_wind_speed_csv_filename = None, 
                                 plume_file = '/store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json',
                                 write_rate = 10,
                                 plume_list = None):
    '''Create or update a windspeed csv file for a list of plumes contained in plume_file. See the comments for compute_wind_values_from_chip
    for details about the csv columns.

    To create a new windspeed csv, provide a full path to current_wind_speed_csv_filename that does not yet exist. 
    It should have the format: /full/path/filename_0000.csv. This code will output a new file with the same name
    but the 0000 will incremented by one.

    To add the windspeed for new plumes into an existing windspeed csv file produced by this function, provide
    the full path to the existing windspeed csv (produced as in the previous paragraph) to current_wind_speed_csv_filename 
    and a new file will be output with the new plumes and the number in the filename will be incremented by 1.

    Parameters
    ----------
    current_wind_speed_csv_filename : string
        Full path to a csv filename with /full/path/filename_0000.csv. See above comments for details
    plume_file : string
        Full path to a geojson file containing the list of plumes to get the windspeed for
    write_rate : integer
        Specifies how often to save the output csv file. 1 means after each plume, 10 is after every 10 plumes.
        Helps to guard against failures and timeouts and stuff like that because, in the event of a failed run,
        the output can be fed back into the function to start again where that last run crashed without having to
        download all the previously collected plumes.
    plume_list : list of strings
        If provided, use this list of plume IDs instead of those found in plume_file. Plume_file must contain 
        these plumes.
    '''
    
    j = json.load(open(plume_file, 'r'))
    all_plume_list = [x['properties']['Plume ID'] for x in j['features']]
    
    if plume_list is None:
        plume_list = all_plume_list

    new_plumes = [x for x in plume_list]

    df_combined = None
    if os.path.exists(current_wind_speed_csv_filename):
        df_current = pd.read_csv(current_wind_speed_csv_filename)
        df_combined = df_current.copy()

        new_plumes = list(set(plume_list) - set(df_current['plume_id']))
        new_plumes.sort(key = lambda x: int(x.split('-')[-1]))
    logging.debug(f'Processing {len(new_plumes)} new plumes...')

    # Create new filename by incrementing the counter of the current_wind_speed_csv_filename by 1
    folder, name = os.path.split(current_wind_speed_csv_filename)
    num = int(os.path.splitext(name)[0].split('_')[-1])
    fname = '_'.join(os.path.splitext(name)[0].split('_')[:-1])
    output_csv_filename = os.path.join(folder, f'{fname}_{num+1:04d}.csv')

    results_list = []
    for i, new_plume in enumerate(new_plumes):
        logging.debug(new_plume)
        ind = all_plume_list.index(new_plume)
        
        fids_str = '_'.join(j['features'][ind]['properties']['fids'])
        if j['features'][ind]['properties']['Origin'] == '':
            d = {'plume_id': new_plume, 'FID': fids_str}
            results_list.append(d)
            continue

        d = get_EMIT_plume_windspeeds(new_plume, current_wind_speed_csv_filename)
        d = d.to_dict(orient='records')[0]

        #lon, lat, _ = json.loads(j['features'][ind]['properties']['Origin'])['coordinates']
        #fids_str = '_'.join(j['features'][ind]['properties']['fids'])
        #fid = j['features'][ind]['properties']['fids'][0] # Just use the first one if there are two

        #if fid[:4].lower() == 'emit':
        #    date = fid[4:8] + '-' + fid[8:10] + '-' + fid[10:12]
        #    frac_time = float(fid[13:15]) + float(fid[15:17])/60 + float(fid[17:19])/3600
        #elif fid[:3].lower() == 'av3':
        #    date = fid[3:7] + '-' + fid[7:9] + '-' + fid[9:11]
        #    frac_time = float(fid[12:14]) + float(fid[14:16])/60 + float(fid[16:18])/3600
        #elif fid[:3].lower() == 'ang':
        #    date = fid[3:7] + '-' + fid[7:9] + '-' + fid[9:11]
        #    frac_time = float(fid[12:14]) + float(fid[14:16])/60 + float(fid[16:18])/3600
        #else:
        #    raise ValueError(f'Instrument type unrecognized from FID: {fid}')

        #with tempfile.TemporaryDirectory() as hrrr_products:
        #    r_HRRR, r_ERA5 = get_w10_reanalysis(lat, lon, date, frac_time, hrrr_products)

        #d = {'plume_id': new_plume,
        #     'FID': fids_str}

        #HRRR_names = ['w10_hrrr_m_per_s', 'w10_dir_hrrr_deg_N_from_E', 'w10_avg_hrrr_m_per_s', 'w10_std_hrrr_m_per_s', 'n_valid_u10_hrrr', 'n_valid_v10_hrrr', 't2m_hrrr_K', 'surface_pressure_hrrr_Pa']
        #[d.update({key:item}) for key, item in zip(HRRR_names, r_HRRR)]

        #era5_names = ['w10_era5_m_per_s', 'w10_dir_era5_deg_N_from_E', 'w10_avg_era5_m_per_s', 'w10_std_era5_m_per_s', 'n_valid_u10_era5', 'n_valid_v10_era5', 't2_era5_K', 'surface_pressure_era5_Pa']
        #[d.update({key:item}) for key, item in zip(era5_names, r_ERA5)]

        results_list.append(d)

        if i % write_rate == write_rate - 1:
            df_new = pd.DataFrame(results_list)
            if df_combined is None:
                df_combined = df_new
            else:
                df_combined = pd.concat([df_combined, df_new], ignore_index = True, sort = False)
            df_combined.to_csv(output_csv_filename, index = False)
            results_list = []

    df_new = pd.DataFrame(results_list)
    if df_combined is None:
        df_combined = df_new
    else:
        df_combined = pd.concat([df_combined, df_new], ignore_index = True, sort = False)
    df_combined.to_csv(output_csv_filename, index = False)

    return df_new

def get_EMIT_plume_windspeeds(plume, input_wind_speed_csv_filename = None):
    ''' Similar to update_EMIT_plume_list_windspeeds, but without the writing, and for single plumes
    Parameters
    ----------
    plume: dict
        A single plume feature dictionary from the EMIT plume geojson
    input : string
        Full path to a csv filename with /full/path/filename_0000.csv. See above comments for details

    Returns
    -------
    dict - dictionary of windspeed results
    bool - indication of whether the windspeed was newly computed (True) or read from input file (False)
    '''
    
    logging.debug(f'Get windspeed for {plume["properties"]["Plume ID"]}')
    if 'Origin' not in plume['properties'] or plume['properties']['Origin'] == '' or len(plume['properties']['Origin']) == 0 :
        return None

    lat = plume['properties']['Origin'][0]['coords'][1]
    lon = plume['properties']['Origin'][0]['coords'][0]
    date, frac_time = get_datetime_from_fid(plume['properties']['fids'][0])

    if os.path.exists(input_wind_speed_csv_filename):
        df_current = pd.read_csv(input_wind_speed_csv_filename)

        # Previously, plume_id was used.  Updated to care about lat/lon/time, which is the real intersection
        match_idx = np.logical_and.reduce((
            df_current['date'] == date,
            df_current['frac_time'] == frac_time,
            np.isclose(df_current['lat'], lat, 1e-5),
            np.isclose(df_current['lon'], lon, 1e-5)
        ))
        #match_idx = df_current['plume_id'] == plume['properties']['Plume ID']
        if np.sum(match_idx) == 1:
            cached_row = df_current[match_idx]
            # If ERA5 windspeed was previously unavailable, fall through to recompute
            if pd.isna(cached_row['w10_era5_m_per_s'].values[0]):
                logging.info(f'Cached windspeed for {plume["properties"]["Plume ID"]} has NaN ERA5, will retry full computation...')
            else:
                return cached_row, False
        elif np.sum(match_idx) > 1:
            logging.warning(f'Multiple matching windspeed entries found for plume {plume["properties"]["Plume ID"]} at lat {lat}, lon {lon}, date {date}, frac_time {frac_time}. Using the first match.')
            return df_current[match_idx].iloc[[0]], False 
        else:
            logging.debug('No matching windspeed entry found, computing new windspeed...')

    fids_str = '_'.join(plume['properties']['fids'])

    with tempfile.TemporaryDirectory() as hrrr_products:
        r_HRRR, r_ERA5 = get_w10_reanalysis(lat, lon, date, frac_time, hrrr_products)

    d = {'plume_id': plume['properties']['Plume ID'],
         'lat': lat,
         'lon': lon,
         'date': date,
         'frac_time': frac_time,
         'FID': fids_str}

    HRRR_names = ['w10_hrrr_m_per_s', 'w10_dir_hrrr_deg_N_from_E', 'w10_avg_hrrr_m_per_s', 'w10_std_hrrr_m_per_s', 'n_valid_u10_hrrr', 'n_valid_v10_hrrr', 't2m_hrrr_K', 'surface_pressure_hrrr_Pa']
    [d.update({key:item}) for key, item in zip(HRRR_names, r_HRRR)]

    era5_names = ['w10_era5_m_per_s', 'w10_dir_era5_deg_N_from_E', 'w10_avg_era5_m_per_s', 'w10_std_era5_m_per_s', 'n_valid_u10_era5', 'n_valid_v10_era5', 't2_era5_K', 'surface_pressure_era5_Pa']
    [d.update({key:item}) for key, item in zip(era5_names, r_ERA5)]

    d = pd.DataFrame([d])

    return d, True


def get_datetime_from_fid(fid):
    if fid[:4].lower() == 'emit':
        date = fid[4:8] + '-' + fid[8:10] + '-' + fid[10:12]
        frac_time = float(fid[13:15]) + float(fid[15:17])/60 + float(fid[17:19])/3600
    elif fid[:3].lower() == 'av3':
        date = fid[3:7] + '-' + fid[7:9] + '-' + fid[9:11]
        frac_time = float(fid[12:14]) + float(fid[14:16])/60 + float(fid[16:18])/3600
    elif fid[:3].lower() == 'ang':
        date = fid[3:7] + '-' + fid[7:9] + '-' + fid[9:11]
        frac_time = float(fid[12:14]) + float(fid[14:16])/60 + float(fid[16:18])/3600
    else:
        raise ValueError(f'Instrument type unrecognized from FID: {fid}')
    return date, np.round(frac_time, 5)


def get_w10_reanalysis(plume_lat, plume_lon, date, frac_time, save_path): 
    """
    Does the interpolation in time.

    plume_lat: latitude [deg N]
    plume_lon: pseudo-origin longitude [deg E]

    date as '%Y-%m-%d', ex. '2023-03-25'
    frac_time as float fractional hour of the day [UTC]
    """
    
    acq_time_pre = int(np.floor(frac_time))
    acq_time_post = int(np.ceil(frac_time))

    # Round the hour down and up
    dt = pd.to_datetime(date) + datetime.timedelta(hours = frac_time)
    dt_rounded_down = dt.replace(second = 0, microsecond = 0, minute = 0)
    dt_rounded_up = (dt + datetime.timedelta(hours = 1)).replace(second = 0, microsecond = 0, minute = 0)

    date_pre = dt_rounded_down.strftime('%Y-%m-%d')
    date_post = dt_rounded_up.strftime('%Y-%m-%d')
    hour_pre = dt_rounded_down.strftime('%H') + ':00:00'
    hour_post = dt_rounded_up.strftime('%H') + ':00:00'

    # Check if source is in HRRR bounds
    if (21.13812300000003 <= plume_lat <= 52.61565330680793) and (225.90452026573686 <= plume_lon%360 <= 299.0828072281622): 
        r_HRRR_pre = herbie_hrrr(plume_lat, plume_lon%360, date_pre, hour_pre, save_path)
        r_HRRR_post = herbie_hrrr(plume_lat, plume_lon%360, date_post, hour_post, save_path)
        r_HRRR_interp = [np.interp(frac_time, [acq_time_pre, acq_time_post], [pre,post]) for pre, post in zip(r_HRRR_pre, r_HRRR_post)]
    else: 
        r_HRRR_interp = [np.nan]*6
    
    # ERA-5 from CDS API
    r_ERA5_pre = get_w10_from_ERA5_Climate_Data_Store(plume_lat, plume_lon, date_pre, hour_pre, save_path)
    r_ERA5_post = get_w10_from_ERA5_Climate_Data_Store(plume_lat, plume_lon, date_post, hour_post, save_path)
    r_ERA5_interp = [np.interp(frac_time, [acq_time_pre, acq_time_post], [pre,post]) for pre, post in zip(r_ERA5_pre, r_ERA5_post)]

    return r_HRRR_interp, r_ERA5_interp

def get_w10_from_ERA5_Climate_Data_Store(plume_lat, plume_lon, date, hour_rounded, save_path):
    '''Get the wind speed info from ERA5 Land data 10m_u_component_of_wind and 10m_v_component_of_wind near the
    plume_lat and plume_lon at the date and hour requested. See compute_wind_values_from_chip for details of the
    return parameters. This uses the Climate Data Stor and requires that you have a .cdsapi file that has a
    valid API key. See here for details: https://cds.climate.copernicus.eu/how-to-api
    '''

    fname_ext = '.grib'
    year, month, day = date.split('-')
    fname = os.path.join(save_path, year + month + day + '_' + hour_rounded.replace(':','_') + fname_ext)
    dl = 0.25 * 3
    
    dataset = "reanalysis-era5-land"
    request = {
       "product_type": ["reanalysis"],
       "variable": [
           "10m_u_component_of_wind",
           "10m_v_component_of_wind",
           "2m_temperature",
           "surface_pressure"
       ],
       "year": [year],
       "month": [month],
       "day": [day],
       "time": [hour_rounded],
       "data_format": "grib",
       "download_format": "unarchived"
    }

    client = cdsapi.Client(quiet=False,debug=False)
    try:
        client.retrieve(dataset, request).download(fname)
    except requests.exceptions.HTTPError as e:
        if 'not available yet' in str(e):
            logging.warning(f'ERA5 data not yet available for {date} {hour_rounded}: {e}')
            return tuple([np.nan] * 8)
        raise

    ds = xr.open_dataset(fname)
    lat, lon = ds['latitude'], ds['longitude'] # Longitude is deg E, 0 to 360

    lat_min = np.argmin(np.abs(lat.values - plume_lat))
    lon_min = np.argmin(np.abs(lon.values - ((plume_lon + 360)%360)))
    x_min, x_max = max(lat_min - 1, 0), min(lat_min + 2, ds['u10'].shape[0])
    y_min, y_max = max(lon_min - 1, 0), min(lon_min + 2, ds['u10'].shape[1])

    u10_data_3x3 = ds['u10'].values[x_min:x_max, y_min:y_max]
    v10_data_3x3 = ds['v10'].values[x_min:x_max, y_min:y_max]
    t2_data_3x3 = ds['t2m'].values[x_min:x_max, y_min:y_max]
    sp_data_3x3 = ds['sp'].values[x_min:x_max, y_min:y_max]
    
    return compute_wind_values_from_chip(u10_data_3x3, v10_data_3x3, lat[x_min:x_max].values, lon[y_min:y_max].values, plume_lat, (plume_lon + 360)%360, 'regular', t2_data_3x3, sp_data_3x3)

def herbie_hrrr(plume_lat, plume_lon, date, hour_rounded, save_path, curr_model = "hrrr"): 
    """
    Get the HRRR windspeed using Herbie
    plume_lat: pseudo-origin latitude [deg]
    plume_lon: pseudo-origin longitude [deg]
    date: str year-month-day e.g., "2024-03-05"
    hour_rounded: e.g., 19:00:00
    curr_model: defined in https://herbie.readthedocs.io/en/latest/index.html
    """

    if curr_model == "hrrr":
        H = Herbie(date + ' ' + hour_rounded, model="hrrr", fxx=0, save_dir = save_path)
        ds_u = H.xarray(":UGRD:10 m")
        ds_v = H.xarray(":VGRD:10 m")
        ds_p = H.xarray(":PRES:surface")
        ds_t = H.xarray(":TMP:2 m above ground")
    else:
        raise NotImplementedError
    
    # Find closest point in HRRR grid to desired lat/lon
    sub_array = [ds_u[var].values for var in ['latitude', 'longitude']]
    abs_diffs = [np.abs(arr - target) for arr, target in zip(sub_array, [plume_lat,plume_lon])]
    total_diff = sum(abs_diffs)
    min_index = np.unravel_index(np.argmin(total_diff), total_diff.shape)
    
    # Extract U10 and V10 windspeed at 3x3 grid around index
    u10_data, v10_data, t2m_data, sp_data = ds_u['u10'], ds_v['v10'], ds_t['t2m'], ds_p['sp']
    x_min, x_max = max(min_index[0] - 1, 0), min(min_index[0] + 2, u10_data.shape[0]) # Latitude varies in the short dimension
    y_min, y_max = max(min_index[1] - 1, 0), min(min_index[1] + 2, u10_data.shape[1]) # Longitude varies in the long dimension

    u10_data_3x3 = u10_data.values[x_min:x_max, y_min:y_max]
    v10_data_3x3 = v10_data.values[x_min:x_max, y_min:y_max]
    t2m_data_3x3 = t2m_data.values[x_min:x_max, y_min:y_max]
    p_data_3x3 = sp_data.values[x_min:x_max, y_min:y_max]
    
    lat = ds_u['latitude'].values[x_min:x_max, y_min:y_max]
    lon = ds_u['longitude'].values[x_min:x_max, y_min:y_max]

    return compute_wind_values_from_chip(u10_data_3x3, v10_data_3x3, lat, lon, plume_lat, plume_lon, 'irregular', t2m_data_3x3, p_data_3x3)

def compute_wind_values_from_chip(u10_data_3x3, v10_data_3x3, lat, lon, plume_lat, plume_lon, regular_or_irregular,
                                  t_data_3x3, sp_data_3x3):
    '''Given a 3x3 u10 and v10 chip with the center pixel chosen nearest the lat/lon of interest, compute:
        w10_mps: magnitude windspeed of the pixel (1,1) in meters per second
        w10_deg_N_from_E: wind direction (blowing towards) in deg North from East in pixel (1,1)
        w10_mps: average magnitude windspeed over 3x3 chip in meters per second
        w10_std: standard deviation of magnitude windspeed over 3x3 chip in meters per second
        n_valid_u10, n_valid_v10: the number of valid (np.isfinite) pixels in the 3x3 chip in both directions
    '''
    t_interp, sp_interp = interpolate_to_lat_lon(t_data_3x3, sp_data_3x3, lat, lon, 
                                                    plume_lat, plume_lon, regular_or_irregular)
    u10_interp, v10_interp = interpolate_to_lat_lon(u10_data_3x3, v10_data_3x3, lat, lon, 
                                                    plume_lat, plume_lon, regular_or_irregular)
    w10_mps = np.hypot(u10_interp, v10_interp)
    #w10_mps = np.hypot(u10_data_3x3[1,1], v10_data_3x3[1,1])

    #w10_deg_N_from_E = np.arctan2(v10_data_3x3[1,1], u10_data_3x3[1,1]) * 180 / np.pi
    w10_deg_N_from_E = np.arctan2(v10_interp, u10_interp) * 180 / np.pi

    n_valid_u10 = np.sum(np.isfinite(u10_data_3x3))
    n_valid_v10 = np.sum(np.isfinite(v10_data_3x3))

    # Standard dev and mean of windspeed calc
    w10_avg = np.nanmean(np.sqrt(u10_data_3x3**2 + v10_data_3x3**2))
    w10_std = np.nanstd(np.sqrt(u10_data_3x3**2 + v10_data_3x3**2))

    return w10_mps, w10_deg_N_from_E, w10_avg, w10_std, n_valid_u10, n_valid_v10, t_interp, sp_interp

def interpolate_to_lat_lon(u, v, lat, lon, plume_lat, plume_lon, regular_or_irregular, show_plots = False):
    '''
    Spatially interpolate to the specific plume_lat and plume_lon. Be careful that lat/lon and 
    plume_lat/plume_lon are the same coordinate system, like 0-360 vs -180 to 180 for example.

    The interpolation is bilinear if regular_or_irregular is "regular" and 
    triangulated somehow if it's "irregular". See the documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html
    The difference is that if lat/lon is on a regular, uniform grid, use "regular", if it's 
    not uniformly spaced, use "irregular".
    '''
    from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

    if regular_or_irregular == 'regular':
        interpolator = RegularGridInterpolator
        f_u = interpolator((lat, lon), u)
        f_v = interpolator((lat, lon), v)

        if show_plots:
            from matplotlib import pyplot as plt
            plt.figure()
            plotlat = np.random.uniform(low = lat.min(), high = lat.max(), size = 1000)
            plotlon = np.random.uniform(low = lon.min(), high = lon.max(), size = 1000)
            plt.scatter(lat, lon, c = f_u((plotlat,plotlon)), cmap = 'viridis', vmin = u.min(), vmax = u.max(), s = 50)
            la = (np.ones_like(lon)[:,np.newaxis] * lat).T
            lo = np.ones_like(lat)[:,np.newaxis] * lon
            plt.scatter(la, lo, c = u, cmap = 'viridis', vmin = u.min(), vmax = u.max(), s = 400)
            plt.show()
    
    elif regular_or_irregular == 'irregular':
        interpolator = LinearNDInterpolator
        f_u = interpolator(np.stack([lat.flatten(), lon.flatten()]).T, u.flatten())
        f_v = interpolator(np.stack([lat.flatten(), lon.flatten()]).T, v.flatten())

        if show_plots:
            from matplotlib import pyplot as plt
            plt.figure()
            plotlat = np.random.uniform(low = lat.min(), high = lat.max(), size = 1000)
            plotlon = np.random.uniform(low = lon.min(), high = lon.max(), size = 1000)
            plt.scatter(plotlat, plotlon, c = f_u((plotlat,plotlon)), cmap = 'viridis', vmin = u.min(), vmax = u.max(), s = 50)
            plt.scatter(lat, lon, c = u, cmap = 'viridis', vmin = u.min(), vmax = u.max(), s = 400)
            plt.show()
        
    else:
        raise ValueError(f'Invalid value {regular_or_irregular} for regular_or_irregular.')

    u_interp = f_u((plume_lat, plume_lon))
    v_interp = f_v((plume_lat, plume_lon))

    return u_interp, v_interp

def windspeed_key_names(primary_windspeed_source, secondary_windspeed_source, BASE='w10', UNITS='m_per_s'):

    BASE   = 'w10'      # fixed wind‑speed variable name
    UNITS  = 'm_per_s'  # fixed unit specifier

    windspeed_primary_key       = f'{BASE}_{primary_windspeed_source}_{UNITS}'
    windspeed_secondary_key     = f'{BASE}_{secondary_windspeed_source}_{UNITS}'
    windspeed_std_primary_key   = f'{BASE}_std_{primary_windspeed_source}_{UNITS}'
    windspeed_std_secondary_key = f'{BASE}_std_{secondary_windspeed_source}_{UNITS}'
    windspeed_key               = f'{BASE}_{primary_windspeed_source}_{secondary_windspeed_source}_{UNITS}'
    windspeed_std_key           = f'{BASE}_std_{primary_windspeed_source}_{secondary_windspeed_source}_{UNITS}'

    return {'primary': windspeed_primary_key,
            'secondary': windspeed_secondary_key,
            'primary_std': windspeed_std_primary_key,
            'secondary_std': windspeed_std_secondary_key,
            'combined': windspeed_key,
            'combined_std': windspeed_std_key}

if __name__ == '__main__':
    cli()