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

import numpy as np
import logging
import json
import os
import pandas as pd
from typing import List
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from quantification import compute_flux, windspeed

EMISSIONS_DELIVERY_INFO = {
                    'Wind Speed (m/s)': 'NA',
                    'Wind Speed Std (m/s)': 'NA',
                    'Wind Speed Source': 'NA',
                    'Emissions Rate Estimate (kg/hr)': 'NA',
                    'Emissions Rate Estimate Uncertainty (kg/hr)': 'NA',
                    'Fetch Length (m)': 'NA',
                }

def compute_Q_and_uncertainty(C_Q, sumC2, windspeed_m_s, windspeed_std_m_s, fetch_m, fetch_unc_fraction):
    """
    Compute the emission rate Q and its uncertainties. Uses standard error propagation of:
    Q = windspeed / fetch * sum_over_pixels(k * concentration_length)

    Parameters
    ----------
    C_Q : ndarray of floats
        Emission rate divided by windspeed [kg/hr / (m/s)]
    sumC2 : ndarray of floats
        Sum of squared concentration uncertainties across all pixels.
    windspeed_m_s : ndarray of floats
        Mean wind speed [m/s]
    windspeed_std_m_s : ndarray of floats
        Standard deviation of wind speed measurements in [m/s]
    fetch_m : ndarray of floats
        Fetch length [m]
    fetch_unc_fraction : ndarray of floats
        Fractional uncertainty in fetch length (e.g., 0.1 for 10 % uncertainty) [unitless]

    Returns
    -------
    Q : ndarray of floats
        Computed emission rate in kg/hr
    sigma_Q : ndarray of floats
        Standard deviation of total emission rate uncertainty in kg/hr
    sigma_C : ndarray of floats [kg/hr]
        Standard deviation of emission rate uncertainty due to concentration.
    sigma_w : ndarray of floats [kg/hr]
        Standard deviation of emission rate uncertainty due to wind speed.
    sigma_f : ndarray of floats [kg/hr]
        Standard deviation of emission rate uncertainty due to fetch length.
    """
    # Compute emission rate Q as conversion factor times wind speed
    Q = C_Q * windspeed_m_s

    # Pre‑compute term (C_Q * fetch_m)^2 for later uncertainty calculations
    C2 = (C_Q * fetch_m)**2

    # Uncertainty in fetch length (absolute value)
    fetch_std = fetch_unc_fraction * fetch_m

    # Squared uncertainty in Q due to concentration length uncertainty in each pixel
    sigma2_C = windspeed_m_s**2 / fetch_m**2 * sumC2

    # Squared uncertainty in Q due to fetch length uncertainty
    sigma2_f = windspeed_m_s**2 / fetch_m**4 * C2 * fetch_std**2

    # Squared uncertainty in Q due to wind speed uncertainty
    sigma2_w = 1 / fetch_m**2 * C2 * (windspeed_std_m_s)**2

    # Total squared uncertainty in Q (sum of independent contributions)
    sigma2_Q = sigma2_C + sigma2_f + sigma2_w

    # Return emission rate and standard deviations (square roots of variance terms)
    return Q, np.sqrt(sigma2_Q), np.sqrt(sigma2_C), np.sqrt(sigma2_w), np.sqrt(sigma2_f)



def single_plume_emissions(feat: dict,
                           poly_plume: dict, 
                           quant_dir: str, 
                           proc_dir: str,
                           delivery_raster_file: str, 
                           delivery_sens_file: str,
                           delivery_uncert_file: str,
                           annotation_file: str,
                           overrule_simple_ime_flag: bool = False,
                           working_windspeed_csv: str = None,
                           ):
    """
    Compute emissions and uncertainties for a single plume.
    Inputs:
    feat: dict
        Feature dictionary containing plume metadata.
    poly_plume: dict
        Polygon geometry of the plume.
    quant_dir: str
        Directory to store quantification outputs.
    proc_dir: str
        Directory to store intermediate processing outputs.
    delivery_raster_file: str
        Path to the delivery raster file, (populated, read only).
    delivery_sens_file: str
        Path to the delivery sensitivity file, (populated, read only).
    delivery_uncert_file: str
        Path to the delivery uncertainty file (populated, read only).
    annotation_file: str
        Path to the annotation file (populated, read only).
    overrule_simple_ime_flag: bool
        If true, ignore "Simple IME Valid" and calc emissions anyway.
    working_windspeed_csv: str
        Path to the working windspeed CSV file (populated, read only).
    

    Returns:
    emissions_info: dict
        Dictionary containing computed emissions and uncertainties.
    windspeed_info: dict
        Dictionary containing windspeed information.
    """
    emissions_info = deepcopy(EMISSIONS_DELIVERY_INFO)
    if feat['properties']['Simple IME Valid'] == 'Yes' or overrule_simple_ime_flag:
        class flux_args:
            def __init__(self):
                self.minppmm = 500
                self.mergedistm = 200
                self.maxppmm = 3000
                self.maxfetchm = 1000
                self.minaream2 = 0

                self.plot_diag = True
                self.verbose = False
                self.fid = poly_plume['properties']['Plume ID']

                self.name_suffix = ''
                self.plot_path = quant_dir

                if 'Origin' not in feat['properties'].keys() or feat['properties']['Origin'] is None or len(feat['properties']['Origin']) == 0:
                    self.lat = None
                    self.lng = None
                else:
                    pso = feat['properties']['Origin'][0]
                    self.lat = pso['coords'][1]
                    self.lng = pso['coords'][0]

                self.cmfimgf = delivery_raster_file
                self.sns_file = delivery_sens_file
                self.unc_file = delivery_uncert_file
 
                mb_coords = poly_plume['geometry']['coordinates']
                if isinstance(mb_coords[0][0], List):
                    mb_coords = mb_coords[0] 
                self.manual_boundary_coordinates_lon_lat = [item for sublist in mb_coords for item in sublist]

                self.mask_mode = 'infer'
                self.q_method = 'mask'

                self.rgbimgf = None
                self.labimgf = None

                #self.log_level = args.loglevel
                self.log_level = 'DEBUG'
                self.logfile = None

            unc_file = None

        lfa = flux_args()
        if lfa.lat is None or lfa.lng is None:
            logging.warning(f'Plume {lfa.fid} missing Origin, skipping plume')
            return emissions_info, None

        with open(os.path.join(proc_dir, f'flux_args_{poly_plume["properties"]["Plume ID"]}.json'),'wb') as pf:
            pf.write(json.dumps(lfa.__dict__, indent=2).encode('utf-8'))


        # Compute Flux
        #quant_res: [plume_complex, C_Q_MASK, C_Q_CC, lng, lat, fetchm, mergedistm, args.minppmm, args.maxppmm, args.minaream2, ps, C2_UNC_MASK]
        flux_status, flux_res = compute_flux.compute_flux(lfa)
        if flux_status != 'success':
            logging.warning(f'Flux calculation failed for plume {lfa.fid} with status {flux_status}, skipping plume')
            return emissions_info, None

        logging.debug(f'Flux results: {flux_status} {flux_res}')
        # Compute Windspeed
        original_log_level = logging.getLogger().level

        logging.getLogger().setLevel(logging.ERROR)
        dfw, dfw_new = windspeed.get_EMIT_plume_windspeeds(feat, working_windspeed_csv)
        logging.getLogger().setLevel(original_log_level)

        wsk = windspeed.windspeed_key_names('hrrr', 'era5', 'w10', 'm_per_s')
        ws = dfw[wsk['primary']].fillna(dfw[wsk['secondary']])
        ws_std = dfw[wsk['primary_std']].fillna(dfw[wsk['secondary_std']])
        ws_source = np.where(dfw[wsk['primary']].notna(), 'hrrr', 'era5')

        dfw = dfw.to_dict(orient='records')[0] 
        if dfw_new is False: # only return to concatenate if new
            dfw = None

        # Compute Emissions
        Q, sigma_Q, sigma_C, sigma_w, sigma_f = \
            compute_Q_and_uncertainty(
                flux_res[1], # C_Q_mask_kg_hr_mpers
                flux_res[11], #sum_C_sqrd
                ws, 
                ws_std, 
                flux_res[5], #fetch
                0.0) #fetch_unc_frac
        emissions_info['Wind Speed (m/s)'] = float(np.round(ws.values[0],4))
        emissions_info['Wind Speed Std (m/s)'] = float(np.round(ws_std.values[0],4))
        emissions_info['Wind Speed Source'] = ws_source[0].upper()
        emissions_info['Emissions Rate Estimate (kg/hr)'] = float(np.round(Q.values[0],4))
        emissions_info['Emissions Rate Estimate Uncertainty (kg/hr)'] = float(np.round(sigma_Q.values[0],4))
        emissions_info['Fetch Length (m)'] = float(np.round(flux_res[5],4))

        imefigf = os.path.join(lfa.plot_path, lfa.fid+'_ql.png')
        add_results_to_image(lfa.fid, emissions_info, poly_plume["properties"]["UTC Time Observed"], lfa.lat, lfa.lng, imefigf)

        for key in emissions_info.keys():
            try:
                if np.isnan(emissions_info[key]):
                    emissions_info[key] = 'NA'
            except TypeError:
                pass


        if dfw is not None:
            for key in dfw.keys():
                try:
                    if pd.isna(dfw[key]):
                        dfw[key] = 'NA'
                except TypeError:
                    pass

        logging.info(f'Populated emissions info: {emissions_info}')
        return emissions_info, dfw
    
    return emissions_info, None

def add_results_to_image(plume_id, emissions_info, date_time_str, lat_float, lng_float, input_image_filename, output_image_filename = None, dpi = 150):
    """
    Add emission estimate results to the plume images created in compute_flux.
    Expected operation is to overwrite the input image by leaving output_image_filename
    as the default None.

    Inputs:
    plume_id: string
        Plume ID, eg: CH4_PlumeComplex-50
    emissions_info: dict
        Contains the emissions rate info corresponding to plume plume_id
    date_time_str: str
        Date and time of data collection
    lat_float, lon_float: float
        Pseudo-origin latitutde and longitude as floats
    input_image_filename: str
        Full filename to the plume image file created by compute_flux.py
    output_image_filename: str
        Full filename to save the updated image. Leave as the default None
        to simply overwrite the input image.
    dpi: float
        DPI for the output image using matplotlib's savefig()
    """
    pm = "\u00B1" # plus minus symbol

    if output_image_filename is None:
        output_image_filename = input_image_filename
    
    IME = emissions_info['Emissions Rate Estimate (kg/hr)'] * emissions_info['Fetch Length (m)'] / emissions_info['Wind Speed (m/s)'] / 3600

    s = f'{plume_id}\n' + \
        f'{date_time_str}\n' + \
        f'({lat_float:.3f}, {lng_float:.3f})\n' + \
        f'Q: {emissions_info["Emissions Rate Estimate (kg/hr)"]:.0f} {pm} {emissions_info["Emissions Rate Estimate Uncertainty (kg/hr)"]:.0f} kg/hr\n' + \
        f'{emissions_info["Wind Speed Source"].upper()}: {emissions_info["Wind Speed (m/s)"]:.1f} {pm} {emissions_info["Wind Speed Std (m/s)"]:.1f} m/s\n' + \
        f'IME: {IME:.0f} kg\n' + \
        f'fetch: {emissions_info["Fetch Length (m)"]:.0f} m'

    if not os.path.exists(input_image_filename):
        raise IOError(f'Input image {input_image_filename} does not exist.')

    img = mpimg.imread(input_image_filename)
    h_px, w_px = img.shape[:2]                # height, width in pixels

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])           # left, bottom, width, height (0‑1)
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.008, 0.99, s, color='black', fontsize=12, ha='left',va='top', transform=ax.transAxes, linespacing=1.5,
            bbox=dict(facecolor='white', alpha=0.5, pad=5, edgecolor='none'))

    fig.savefig(output_image_filename, dpi=dpi)
    plt.close(fig)
