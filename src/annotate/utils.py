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
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov



import numpy as np
import logging
import subprocess
from scipy.ndimage import gaussian_filter
from copy import deepcopy
import datetime
from osgeo import gdal

def get_daac_link(feature, product_version):
    prod_v = product_version.split('V')[-1]
    fid=feature['Scene FIDs'][0]
    cid= feature['Plume ID'].split('-')[-1].zfill(6)
    link = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4PLM.{prod_v}/EMIT_L2B_CH4PLM_{prod_v}_{fid[4:12]}T{fid[13:19]}_{cid}/EMIT_L2B_CH4PLM_{prod_v}_{fid[4:12]}T{fid[13:19]}_{cid}.tif'
    return link


def gauss_blur(img, sigma, preserve_nans=False):
    
    V=img.copy()
    V[np.isnan(img)]=0
    VV=gaussian_filter(V,sigma=sigma)

    W=0*img.copy()+1
    W[np.isnan(img)]=0
    WW=gaussian_filter(W,sigma=sigma)

    img_smooth=VV/WW
    if preserve_nans:
        img_smooth[np.isnan(img)] = np.nan
    
    return img_smooth

def print_and_call(cmd_str):
    logging.debug(f'Calling: {cmd_str}')
    subprocess.call(cmd_str, shell=True)

def compare_raster_data(file1, file2):
    ds1 = gdal.Open(file1)
    ds2 = gdal.Open(file2)

    data1 = ds1.ReadAsArray()
    data2 = ds2.ReadAsArray()

    trans_match = ds1.GetGeoTransform() == ds2.GetGeoTransform()
    proj_match = ds1.GetProjection() == ds2.GetProjection()

    if data1.shape != data2.shape:
        return False

    if np.array_equal(data1, data2) and trans_match and proj_match:
        return True
    else:
        return False
    

def plume_mask_threshold(input: np.array, pm, style='ch4'):

    y_locs = np.where(np.sum(pm > 0, axis=1))[0]
    x_locs = np.where(np.sum(pm > 0, axis=0))[0]

    plume_dat = input[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]].copy()
    plume_dat[pm[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] == 0] = 0
    
    local_output_mask = None
    if style == 'ch4':
        plume_dat = gauss_blur(plume_dat, 3, preserve_nans=False)
        local_output_mask = plume_dat > 50
    else:
        plume_dat = gauss_blur(plume_dat, 5, preserve_nans=False)
        local_output_mask = plume_dat > 10000
    
    output_plume_mask = np.zeros(pm.shape,dtype=bool)
    output_plume_mask[y_locs[0]:y_locs[-1],x_locs[0]:x_locs[-1]] = local_output_mask
    
    return output_plume_mask


def prep_predictor_image(predictor, data, ptype):
    image = data.copy()
    if ptype == 'co2':
        image = (image / 1500).astype(np.uint8)
    else:
        image = (image / 100000).astype(np.uint8)
    image[image > 1] = 1
    image[image < 0] = 0
    image *= 255
    oi = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    oi[...] = image.squeeze()[:,:,np.newaxis]

    predictor.set_image(oi)

def sam_segmentation(predictor, rawspace_coords, manual_mask, n_input_points=20):
    min_x = np.min([x[0] for x in rawspace_coords])
    max_x = np.max([x[0] for x in rawspace_coords])
    min_y = np.min([x[1] for x in rawspace_coords])
    max_y = np.max([x[1] for x in rawspace_coords])
    bbox = np.array([min_x, min_y, max_x, max_y])

    input_labels, input_points = [],[]
    for _n in range(n_input_points):
        pt = [np.random.randint(min_x, max_x), np.random.randint(min_y,max_y)]
        input_labels.append(manual_mask[pt[1],pt[0]])
        input_points.append(pt)
        
    masks, _, _ = predictor.predict(
                                    point_coords=np.array(input_points),
                                    point_labels=np.array(input_labels),
                                    box=bbox,
                                    multimask_output=False,
                                   )

    return masks[0,...]

def build_plume_properties(plume_input_properties, plume_geometry, plume_data, trans, data_version, xsize_m, ysize_m, nodata_value=-9999):

    loc_pp = deepcopy(plume_input_properties)
    loc_pp['geometry'] = plume_geometry

    props = {}
    props['Plume ID'] = loc_pp['Plume ID']
    props['Scene FIDs'] = loc_pp['fids']
    props['Orbit'] = loc_pp['orbit']
    props['DCID'] = loc_pp['dcid']
    props['daac_scenes'] = loc_pp['daac_scenes']
    props["Data Download"] = get_daac_link(props, data_version)

    props['Sector'] = loc_pp['Sector']
    props['Sector Confidence'] = loc_pp['Sector Confidence']
    props['Delineation Mode'] = loc_pp['Delineation Mode']
    props['Origin'] = loc_pp['Origin'] if 'Origin' in loc_pp.keys() else []
    props['Simple IME Valid'] = loc_pp['Simple IME Valid']
    props['Time Created'] = loc_pp['Time Created']



    start_datetime = datetime.datetime.strptime(loc_pp['fids'][0][4:], "%Y%m%dt%H%M%S")
    end_datetime = start_datetime + datetime.timedelta(seconds=1)

    start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_datetime = end_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
    props["UTC Time Observed"] = start_datetime
    props["map_endtime"] = end_datetime

   
    # Set defaults, to be overwritten
    props["Max Plume Concentration (ppm m)"] = "NA"
    props["Latitude of max concentration"] = "NA"
    props["Longitude of max concentration"] = "NA"
    max_loc_x = trans[0] + trans[1] * (plume_data.shape[1] / 2)
    max_loc_y = trans[3] + trans[5] * (plume_data.shape[0] / 2)
    
    inplume_dat = plume_data[plume_data != nodata_value]

    if np.sum(plume_data != nodata_value) > 0:

        maxval = np.nanmax(inplume_dat)
        if len(inplume_dat) >= 5 and maxval >= 200 and np.isnan(maxval) == False:

            rawloc = np.where(plume_data == maxval)
            maxval = np.round(maxval)

            #sum and convert to kg.  conversion:
            # ppm m / 1e6 ppm * x_pixel_size(m)*y_pixel_size(m) 1e3 L / m^3 * 1 mole / 22.4 L * 0.01604 kg / mole
            ime_scaler = (1.0/1e6)* ((np.abs(xsize_m*ysize_m))/1.0) * (1000.0/1.0) * (1.0/22.4)*(0.01604/1.0)

            ime = np.nansum(inplume_dat) * ime_scaler
            ime_p = np.nansum(inplume_dat[inplume_dat > 0]) * ime_scaler

            ime = np.round(ime,2)
            #ime_uncert = np.round(len(inplume_dat) * background * ime_scaler,2)

            max_loc_y = trans[3] + trans[5]*(rawloc[0][0]+0.5)
            max_loc_x = trans[0] + trans[1]*(rawloc[1][0]+0.5)

            props["Max Plume Concentration (ppm m)"] = maxval
            props["Latitude of max concentration"] = np.round(max_loc_y, 5)
            props["Longitude of max concentration"] = np.round(max_loc_x, 5)
        else:
            logging.warning(f'Plume {loc_pp["Plume ID"]} has insufficient data...skipping max concentration values')
    else:
        logging.warning(f'Plume {loc_pp["Plume ID"]} has insufficient data...skipping max concentration values')
    

    # For R1 Review
    if not loc_pp['R1 - Reviewed']:
        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "red", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
        review_status = 'R1 Pending'  
    
    # For R2 Review
    if loc_pp['R1 - Reviewed'] and loc_pp['R1 - VISIONS'] and not loc_pp['R2 - Reviewed']:
        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "green", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
        review_status = 'R1 Approved, R2 Pending'  

    # Accept
    if loc_pp['R1 - Reviewed'] and loc_pp['R1 - VISIONS'] and loc_pp['R2 - Reviewed'] and loc_pp['R2 - VISIONS']:
        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "white", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
        review_status = 'Approved'  
    
    # Reject
    if (loc_pp['R1 - Reviewed'] and not loc_pp['R1 - VISIONS']) or (loc_pp['R2 - Reviewed'] and not loc_pp['R2 - VISIONS']):
        props['style'] = {"maxZoom": 20, "minZoom": 0, "color": "yellow", 'opacity': 1, 'weight': 2, 'fillOpacity': 0}
        review_status = 'Rejected'  

    props['Review Status'] = review_status
    poly_res = {"geometry": loc_pp['geometry'],
               "type": "Feature",
               "properties": props}

    props['style']['radius'] = 10
    point_res = {"geometry": {"coordinates": [max_loc_x, max_loc_y, 0.0], "type": "Point"},
               "properties": props,
               "type": "Feature"}
    
    return poly_res, point_res

