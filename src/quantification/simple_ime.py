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
#          Brian Bue
#          Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import json
import pandas as pd
import numpy as np
import os
import subprocess
import glob
from osgeo import osr, gdal
import argparse
from typing import List, Tuple

import subprocess

def cut_masks(input_plume_list, plume_file, folder_of_mask_tifs, output_folder):
    #plume_file = '/store/jfahlen/EMIT_simple_IME/previous_manual_annotation_oneback_copied20240228.json'
    j = json.load(open(plume_file, 'r'))
    plume_list = [x['properties']['Plume ID'] for x in j['features']]
    #fids_str = ['_'.join(x['properties']['fids']) for x in j['features']]
    #plume_paths = ['/scratch/brodrick/methane/visions_delivery_20240409/', '/scratch/brodrick/methane/visions_delivery/']
    plume_paths = ['/store/jfahlen/coincident_overpasses/av3_October_2023/']
    #mask_tif_filenames = glob.glob(f'{folder_of_mask_tifs}/*_mask.tif')

    for plume_id in input_plume_list:

        # The mask file created by the simple IME code (from Brian Bue)
        mask_tif_filename = folder_of_mask_tifs + '/' + plume_id + '_mask.tif'
        if not os.path.exists(mask_tif_filename):
            continue

        # Get FID corresponding to this plume. Note that this won't work if the plume crosses a scene boundary
        ind = plume_list.index(plume_id)
        fid = j['features'][ind]['properties']['fids'][0]

        # Get path to output of ghg_process.py and parallel_mf.py code
        #cmf_filename = construct_EMIT_path(plume_paths, fid, 'junk')
        cmf_filename = construct_Jays_AV3_path(plume_paths, fid, 'junk')
        cmf_filename = [x for x in cmf_filename if 'SNS' in x][0]

        # Get EPSG code for the mask file
        dataset = gdal.Open(mask_tif_filename)
        srs = dataset.GetProjection()
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(srs)
        epsg_code = spatial_ref.GetAuthorityCode(None)

        # Convert CMF data to UTM using the EPSG code from the mask file
        cmf_utm_filename = '/local/' + cmf_filename.split('/')[-1].split('.')[0] + '_utm.tif'
        cmd = f'gdalwarp -t_srs EPSG:{epsg_code} {cmf_filename} {cmf_utm_filename}'
        subprocess.call(cmd, shell = True)

        # Convert to 0-255 values and apply mask
        #output = f'{output_folder}/{plume_id}_masked'
        #cmd = f'gdal_calc.py -A {cmf_utm_filename} -B {mask_tif_filename} --calc "A*(B/255.0)" --format ENVI --NoDataValue=-9999 --overwrite --outfile {output}'
        output = f'{output_folder}/{plume_id}_masked.tif'
        cmd = f'gdal_calc.py -A {cmf_utm_filename} -B {mask_tif_filename} --calc "A*(B/255.0)" --format GTiff --NoDataValue=-9999 --overwrite --outfile {output}'
        subprocess.call(cmd, shell = True)


def construct_Jays_AV3_path(plume_paths, fids, data_not_found_str):
    n = len(fids) if len(fids) > 1 else len(fids) + 1
    scene_numbers = sorted([x.strip().split('_')[-1] for x in fids*n])
    name = fids[0].split('_')[0] + f'_{scene_numbers[0]}_{scene_numbers[-1]}'

    fid = fids[0]
    full_path, sns_path, unc_path = [], [], []
    for plume_path in plume_paths:
        base = os.path.join(plume_path, fid)
        namebase = os.path.join(plume_path, name)
        files = glob.glob(f'{base}/{fid}*_CH4_ORT.vrt') + glob.glob(f'{base}*/{fid}*_CH4_ORT.tif') + glob.glob(f'{plume_path}/{fid}*_CH4_ort') + glob.glob(f'{namebase}/{name}*_CH4_ORT.vrt')
        sns_files = glob.glob(f'{base}/{fid}*_CH4_SNS_ORT.vrt') + glob.glob(f'{base}*/{fid}*_CH4_SNS_ORT.tif') + glob.glob(f'{plume_path}/{fid}*_CH4_SNS_ort') + glob.glob(f'{namebase}/{name}*_CH4_SNS_ORT.vrt')
        unc_files = glob.glob(f'{base}/{fid}*_CH4_UNC_ORT.vrt') + glob.glob(f'{base}*/{fid}*_CH4_UNC_ORT.tif') + glob.glob(f'{plume_path}/{fid}*_CH4_UNC_ort') + glob.glob(f'{namebase}/{name}*_CH4_UNC_ORT.vrt')
        if len(files) != 1:
            continue
        full_path.append(files[0])
        if len(files) == 1 and len(sns_files) == 1 and len(unc_files) == 1:
            sns_path.append(sns_files[0])
            unc_path.append(unc_files[0])
    
    if full_path == []:
        raise FileNotFoundError(f'Could not find AV3 data')

    return full_path[0], sns_path[0], unc_path[0]

def construct_EnMAP_path(plume_paths, fid, data_not_found_str):

    full_path = []
    for plume_path in plume_paths:
        files = glob.glob(f'{plume_path}/*{fid}*.tif')
        if len(files) != 1:
            continue
        full_path.append(files[0])
    
    if full_path == []:
        return f'{data_not_found_str} in {" ".join([x for x in plume_paths])}',[]
    
    return full_path[0], []


def _find_file(pattern: str) -> str:
    """Helper function to find a file matching the given pattern."""
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    return ''

def construct_EMIT_path(
        plume_paths: List[str],
        fids: List[str],
        data_not_found_str: str
    ) -> Tuple[str, str, str]:

    scene_mf_filenames: List[str] = []
    scene_sens_filenames: List[str] = []
    scene_unc_filenames: List[str] = []
    
    for fid in fids:
        for plume_path in plume_paths:
            base_path = os.path.join(plume_path, fid[4:12], fid, 'ghg', 'ch4')
            
            scene_mf_filename = _find_file(os.path.join(base_path, f'{fid}_*_ghg_ortch4_*_v02.tif'))
            scene_sens_filename = _find_file(os.path.join(base_path, f'{fid}_*_ghg_ortsensch4_*_v02.tif'))
            scene_unc_filename = _find_file(os.path.join(base_path, f'{fid}_*_ghg_ortuncertch4_*_v02.tif'))
            
            # Check if all files are found
            if scene_mf_filename and scene_sens_filename and scene_unc_filename:
                if os.path.exists(scene_mf_filename) and os.path.exists(scene_sens_filename) and os.path.exists(scene_unc_filename):
                    scene_mf_filenames.append(scene_mf_filename)
                    scene_sens_filenames.append(scene_sens_filename)
                    scene_unc_filenames.append(scene_unc_filename)
        
    if not scene_mf_filenames:
        raise FileNotFoundError(f'Could not find MF file')
    if not scene_sens_filenames:
        raise FileNotFoundError(f'Could not find SENS file')
    if not scene_unc_filenames:
        raise FileNotFoundError(f'Could not find UNC file')

    if len(scene_mf_filenames) != len(fids):
        raise ValueError(f'The number of scene_mf_filenames is {len(scene_mf_filenames)} but there were {len(fids)} provided!')
    
    # If there is only one fid in fids, then just return the single files without making VRTs
    if len(fids) == 1 and len(scene_mf_filenames) == 1:
        return scene_mf_filenames[0], scene_sens_filenames[0], scene_unc_filenames[0]
    
    fid_name = '_'.join(fids)
    scene_mf_filename = f'/local/{fid_name}_mf.vrt'
    scene_sens_filename = f'/local/{fid_name}_sens.vrt'
    scene_unc_filename = f'/local/{fid_name}_unc.vrt'
    
    myVRT = gdal.BuildVRT(scene_mf_filename, scene_mf_filenames)
    del myVRT
    myVRT = gdal.BuildVRT(scene_sens_filename, scene_sens_filenames)
    del myVRT
    myVRT = gdal.BuildVRT(scene_unc_filename, scene_unc_filenames)
    del myVRT
    
    return scene_mf_filename, scene_sens_filename, scene_unc_filename


def construct_local_EMIT_path(
        plume_paths: List[str],
        fids: List[str],
        data_not_found_str: str
    ) -> Tuple[str, str, str]:

    scene_mf_filenames: List[str] = []
    scene_sens_filenames: List[str] = []
    scene_unc_filenames: List[str] = []
    
    for fid in fids:
        for plume_path in plume_paths:
            base_path = os.path.join(plume_path, fid)
            
            scene_mf_filename = _find_file(os.path.join(base_path, f'{fid}*_mf_ort'))
            scene_sens_filename = _find_file(os.path.join(base_path, f'{fid}*_sens_ort'))
            scene_unc_filename = _find_file(os.path.join(base_path, f'{fid}*_uncert_ort.tif'))
            
            # Check if all files are found
            if scene_mf_filename and scene_sens_filename and scene_unc_filename:
                if os.path.exists(scene_mf_filename) and os.path.exists(scene_sens_filename) and os.path.exists(scene_unc_filename):
                    scene_mf_filenames.append(scene_mf_filename)
                    scene_sens_filenames.append(scene_sens_filename)
                    scene_unc_filenames.append(scene_unc_filename)
        
    if not scene_mf_filenames:
        raise FileNotFoundError(f'Could not find MF file')
    if not scene_sens_filenames:
        raise FileNotFoundError(f'Could not find SENS file')
    if not scene_unc_filenames:
        raise FileNotFoundError(f'Could not find UNC file')

    if len(scene_mf_filenames) != len(fids):
        raise ValueError(f'The number of scene_mf_filenames is {len(scene_mf_filenames)} but there were {len(fids)} provided!')
    
    # If there is only one fid in fids, then just return the single files without making VRTs
    if len(fids) == 1 and len(scene_mf_filenames) == 1:
        return scene_mf_filenames[0], scene_sens_filenames[0], scene_unc_filenames[0]
    
    fid_name = '_'.join(fids)
    scene_mf_filename = f'/local/{fid_name}_mf.vrt'
    scene_sens_filename = f'/local/{fid_name}_sens.vrt'
    scene_unc_filename = f'/local/{fid_name}_unc.vrt'
    
    myVRT = gdal.BuildVRT(scene_mf_filename, scene_mf_filenames)
    del myVRT
    myVRT = gdal.BuildVRT(scene_sens_filename, scene_sens_filenames)
    del myVRT
    myVRT = gdal.BuildVRT(scene_unc_filename, scene_unc_filenames)
    del myVRT
    
    return scene_mf_filename, scene_sens_filename, scene_unc_filename


#def construct_local_EMIT_path(plume_paths, fid_in, data_not_found_str):
#    fid = fid_in[0]
#    pdb.set_trace()
#    for plume_path in plume_paths:
#        #curr_folder = os.path.join(fid[4:12], 'l2bch4enh')
#        #curr_file = fid + 'ch4_enh.tif'
#        full_path = os.path.join(plume_path, fid)
#
#        files = glob.glob(plume_path + f'/*{fid[4:12]}*_mf_ort')
#        if len(files) != 1:
#            raise IOError(f'Could not find mf file for {fid}')
#        else:
#            mf_filename =  files[0]
#
#        files = glob.glob(plume_path + f'/*{fid[4:12]}*_sens_ort')
#        if len(files) != 1:
#            raise IOError(f'Could not find sens file for {fid}')
#        else:
#            sens_filename =  files[0]
#
#        files = glob.glob(plume_path + f'/*{fid[4:12]}*_uncert_ort')
#        if len(files) != 1:
#            raise IOError(f'Could not find uncert file for {fid}')
#        else:
#            uncert_filename =  files[0]
#        
#        return mf_filename, sens_filename, uncert_filename


def construct_EMIT_DAAC_path(plume_paths, fids, data_not_found_str):
    for fid in fids:
        for plume_path in plume_paths:
            folder_search = os.path.join(plume_path, f'*{fid[4:19]}*')
            folders = glob.glob(folder_search)
            if len(folders) != 1:
                raise ValueError(f'Found {len(folders)} matching folders with {folder_search}')
            folder = folders[0]

            ch4_files = glob.glob(os.path.join(folder, f'*CH4ENH*.tif')) + \
                        glob.glob(os.path.join(folder, f'*CH4ENH*.vrt'))
            sns_files = glob.glob(os.path.join(folder, f'*CH4SENS*.tif')) + \
                        glob.glob(os.path.join(folder, f'*CH4SENS*.vrt'))
            unc_files = glob.glob(os.path.join(folder, f'*CH4UNCERT*.tif')) + \
                        glob.glob(os.path.join(folder, f'*CH4UNCERT*.vrt'))
            if len(ch4_files) != 1:
                continue
            else:
                return ch4_files[0], sns_files[0], unc_files[0]
    
    # Should not get here if we have a valid path
    #raise IOError(f'Data not found in {plume_paths} and fid {fid}was found in construct_EMIT_path')
    return f'Data not found in {[x for x in plume_paths]}',[]

from typing import List, Tuple, Callable

def get_plume_metadata_MMGIS(
        curr_id: int, 
        construct_path_function: Callable[[List[str], List[str], str], Tuple[str, str, str]], 
        data_not_found_str: str, 
        meta_path: str = "/store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json",
        plume_paths: List[str] = ["/scratch/brodrick/methane/visions_delivery/"]
    ) -> Tuple[float, float, str, str, str, str]:
    """
    curr_id [int]: plume ID from VISIONS/MMGIS (e.g., 74 = CH4_PlumeComplex-74)
    meta_path: fpath to json metadata file
    plume_path: fpath to delineated EMIT plume COGs
    """
    with open(meta_path, 'r') as f:
        meta = json.loads(f.read())
    
    plume_ids = [x['properties']['Plume ID'] for x in meta['features']]
    
    try:
        feature = meta['features'][plume_ids.index(curr_id)]
    except:
        raise IndexError(f'Plume ID {curr_id} not found in metadata!')

    if 'Origin' not in feature['properties'] or len(feature['properties']['Origin']) == 0:
        raise ValueError('Missing pseudo-origin data!')
    
    pseudo_o = feature['properties']['Origin'][0]
    plume_lat = pseudo_o['coords'][1]
    plume_lon = pseudo_o['coords'][0]
    fids_name = 'fids' if 'fids' in list(feature['properties'].keys()) else 'Scene FIDs'
    scene_fids = feature['properties'][fids_name]
    scene_fids = scene_fids if isinstance(scene_fids, list) else [scene_fids]
    
    if isinstance(feature['geometry']['coordinates'][0][0], List):
        manual_boundary_coordinates_lon_lat = feature['geometry']['coordinates'][0] 
    elif isinstance(feature['geometry']['coordinates'][0], List):
        manual_boundary_coordinates_lon_lat = feature['geometry']['coordinates']
    else:
        manual_boundary_coordinates_lon_lat = ''
    
    full_path, sns_path, unc_path = construct_path_function(plume_paths, scene_fids, data_not_found_str)
    
    return plume_lat, plume_lon, full_path, sns_path, unc_path, manual_boundary_coordinates_lon_lat

def compute_Q_from_plume_json(
        instrument_name: str,
        plume_complex_ids: List[str],
        meta_path: str,
        plume_paths: List[str],
        plot_path: str,
        data_not_found_str: str = 'Data not found',
        do_sensitivity_correction: bool = True,
        use_manual_plume_boundary: bool = True
    ) -> None:
    '''Function to compute and process plume data from various instruments like AV3, EMIT, and EnMAP.
    This function reads metadata, constructs paths, and runs a compute_flux process for each specified plume.'''

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    for plume_complex_id in plume_complex_ids:
        print(f'Now processing: {plume_complex_id}')

        if instrument_name.lower() == 'emit':
            construct_path_function = construct_EMIT_path
        elif instrument_name.lower() == 'av3':
            construct_path_function = construct_Jays_AV3_path
        elif instrument_name.lower() == 'emit_local':
            construct_path_function = construct_local_EMIT_path
        elif instrument_name.lower() == 'emit_daac':
            construct_path_function = construct_EMIT_DAAC_path
        elif instrument_name.lower() == 'enmap':
            construct_path_function = construct_EnMAP_path

        csv_path = f'{plot_path}/'
        csv_file = f'{plot_path}/simple_IME_estimates.csv'
        if do_sensitivity_correction:
            csv_file = f'{plot_path}/simple_IME_estimates_sns_corrected.csv'

        try:
            plume_lat, plume_lon, full_path, sns_path, unc_path, manual_boundary_coordiantes_lon_lat = \
                get_plume_metadata_MMGIS(plume_complex_id, construct_path_function, data_not_found_str, meta_path = meta_path, plume_paths = plume_paths)
        except Exception as e:
            print(e)
            df = pd.DataFrame({'plume_id':[plume_complex_id], 'status': [f'Failed: {e}']})
            if os.path.exists(csv_file):
                df_old = pd.read_csv(csv_file)
                df = pd.concat([df_old, df], ignore_index = True, sort = False)
                df = df.sort_values('plume_id', key = lambda x: x.str.split('-').str[-1].astype(int))
            df.to_csv(csv_file, index = False)
            continue

        if do_sensitivity_correction:
            name_suffix = '_sns_corrected'
            sns_arg = sns_path
            unc_arg = unc_path
        else:
            name_suffix = ''
            sns_arg, unc_arg = 'None', 'None'

        bound_str = ''
        if use_manual_plume_boundary:
            if manual_boundary_coordiantes_lon_lat != '':
                bound_str = ' '.join([str(item) for sublist in manual_boundary_coordiantes_lon_lat for item in sublist])
            else:
                raise ValueError('Manual plume boundary not found.')
        
        if instrument_name[:3].lower() == 'av3':
            ## AV3 controlled release 
            # Andrew's suggestions for the Diamond Feeders case: AV320240711t170628_003_L2B_GHG_ad03567e_CH4_ORT.tif
            #cmd = f'python3 compute_flux.py --minppmm 400 --mergedistm 10 --maxfetchm 300 --minaream2 10 --cmfimgf {full_path} ' + \
            #    f'--lng {plume_lon} --lat {plume_lat} --plot_diag --verbose --csv_file {csv_file} --fid {plume_complex_id} ' + \
            #    f'--csv_path {csv_path} --plot_path {plot_path} --name_suffix "{name_suffix}" --sns_file "{sns_arg}" ' + \
            #    f'--mask_mode infer --q_method mask'
            #cmd = f'python3 compute_flux.py --minppmm 400 --mergedistm 5 --maxfetchm 100 --minaream2 10 --cmfimgf {args.plume_file} ' + \
            #cmd = f'python3 compute_flux.py --minppmm 200 --mergedistm 100 --maxfetchm 1000 --minaream2 10 --cmfimgf {full_path} ' + \
            #cmd = f'python3 compute_flux.py --minppmm 400 --mergedistm 5 --maxfetchm 100 --minaream2 {5**2 * 3} --cmfimgf {full_path} ' + \
            cmd = f'python3 compute_flux.py --minppmm 400 --mergedistm 5 --maxfetchm 100 --minaream2 0 --cmfimgf {full_path} ' + \
                f'--lng {plume_lon} --lat {plume_lat} --plot_diag --verbose --csv_file {csv_file} --fid {plume_complex_id} ' + \
                f'--csv_path {csv_path} --plot_path {plot_path} --name_suffix "{name_suffix}" --sns_file "{sns_arg}" --unc_file "{unc_arg}" ' + \
                f'--manual_boundary_coordinates_lon_lat {bound_str} --mask_mode infer --q_method mask'

        elif instrument_name.lower().startswith('emit'):
            ## EMIT
            cmd = f'python3 quantification/compute_flux.py --minppmm 500 --mergedistm 200 --maxppmm 3000 --maxfetchm 1000 --minaream2 0 --cmfimgf {full_path} ' + \
                f'--lng {plume_lon} --lat {plume_lat} --plot_diag --verbose --csv_file {csv_file} --fid {plume_complex_id} ' + \
                f'--csv_path {csv_path} --plot_path {plot_path} --name_suffix "{name_suffix}" --sns_file "{sns_arg}" --unc_file "{unc_arg}" ' + \
                f'--manual_boundary_coordinates_lon_lat {bound_str} --mask_mode infer --q_method mask' 

        elif instrument_name.lower().startswith('enmap'):
            ## EnMAP
            cmd = f'python3 compute_flux.py --minppmm 1250 --mergedistm 200 --maxppmm 3000 --maxfetchm 1000 --minaream2 300 --cmfimgf {full_path} ' + \
                f'--lng {plume_lon} --lat {plume_lat} --plot_diag --verbose --csv_file {csv_file} --fid {plume_complex_id} ' + \
                f'--csv_path {csv_path} --plot_path {plot_path} --name_suffix "{name_suffix}" --sns_file "{sns_arg}" ' + \
                f'--mask_mode infer --q_method mask' 

        subprocess.call(cmd, shell = True)
        
def main(args):
    data_folders = [x+'/' for x in getattr(args, 'data_folder')]

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # WRite arguments to a file for reference
    args_dict = vars(args)
    with open(os.path.join(args.output_path, 'command_args.json'), 'w') as f:
        json.dump(args_dict, f, sort_keys=True, indent=4)

    j = json.load(open(args.plume_json_filename, 'r'))
    all_plumes = [x['properties']['Plume ID'] for x in j['features']]
    
    if args.plume_id and args.update_only:
        raise ValueError(f'Cannot specify both update_only and plume_id')

    ############################################################################################################################
    # Use this to update the runs in the simple_IME_estimates.csv file
    if args.update_only:
        df = pd.read_csv(f'{args.output_path}/simple_IME_estimates.csv')
        plumes_to_do = list(set(all_plumes) - set(df['plume_id']))
        plumes_to_do.sort(key = lambda x: int(x.split('-')[-1]))
    else:
        plumes_to_do = all_plumes
    ############################################################################################################################

    if args.plume_id:
        plumes_to_do = args.plume_id

        invalid_ids = [pid for pid in args.plume_id if pid not in all_plumes]
        if invalid_ids:
            raise ValueError(f'Plume IDs {invalid_ids} are not present in {args.plume_json_filename}')
    
    compute_Q_from_plume_json(args.instrument,
                              plumes_to_do,
                              args.plume_json_filename,
                              data_folders,
                              plot_path = args.output_path + '/',
                              do_sensitivity_correction= not args.no_sensitivity_correction,
                              use_manual_plume_boundary= not args.no_manual_plume_boundary)

# All EMIT plumes
# python simpleIME.py
#   --data_folder /store/emit/ops/data/acquisitions 
#   --output_path ADD_PATH
#   --instrument emit 
#   --plume_json_filename /store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json

# Now all in one line for convenience:
# nice -n 19 python simpleIME.py --data_folder /store/emit/ops/data/acquisitions --output_path ADD_PATH --instrument emit --plume_json_filename /store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json

# AiRMAPS
#python simpleIME.py 
#    --data_folder /store/jfahlen/AiRMAPS
#    --instrument av3
#    --plume_json_filename /store/jfahlen/AiRMAPS/plume_list.json
#    --output_path ADD_PATH

# Casa Grande November, 2024
#python simpleIME.py 
#    --data_folder /store/jfahlen/av3_casa_grande_Nov2024/
#    --instrument av3
#    --plume_json_filename /store/jfahlen/av3_casa_grande_Nov2024/plume_list.json
#    --output_path ADD_PATH

# Wyoming September, 2024
#python simpleIME.py 
#    --data_folder /store/jfahlen/av3_wyoming_Sept2024/
#    --instrument av3
#    --plume_json_filename /store/jfahlen/av3_wyoming_Sept2024/plume_list.json
#    --output_path ADD_PATH

# Casa Grande Pitch Experiment January, 2025
#python simpleIME.py 
#    --data_folder /store/jfahlen/casagrande_pitch_20250126/
#    --instrument av3
#    --plume_json_filename /store/jfahlen/casagrande_pitch_20250126/plume_list.json
#    --output_path ADD_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process plumes with Simple IME method")
    parser.add_argument("--output_path", help="Path to the output folder")
    parser.add_argument("--plume_json_filename", help="Path to the JSON file containing the plumes to compute the simple IME")
    parser.add_argument('--instrument', type=str, default='emit', choices=['emit','av3', 'enmap', 'emit_daac', 'emit_local'],
                        help='Instrument name, selects which set of parameters for compute_flux')
    parser.add_argument('--data_folder', '--names-list', action="append", help="Path to a potential data folder, add as many as you like")
    parser.add_argument('--update_only', action='store_true', default=False, help='Use the previous simple_IME_estimates.csv file to only process new plumes in the plume_json_filename')
    parser.add_argument('--no_sensitivity_correction', action='store_true', default=False, help='Turn off sensivity correction in IME')
    parser.add_argument('--no_manual_plume_boundary', action='store_true', default=False, help='Do not use the manual plume boundary to clip the inferred mask')
    parser.add_argument('--plume_id', nargs='+', help='ID of the plume, add as many as you like')
    args = parser.parse_args()
    main(args)