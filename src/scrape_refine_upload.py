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

'''
Major Steps, at each iteration of the loop:

1. Download the latest plume annotations, and intersect them with coverage to add FIDS, Orbits, etc.  ID new plumes during this
2. Find all DCIDs that have new plumes, and mosaic each 
3. Step through each plume in the DCID, cut it out, calculate stats and metadata
4. Prep each plume for delivery, writing output files
5. Merge all plumes into single output metadata, merge all plumes within a DCID to a single COG / tiled dataset

Note that previously, the necessity of the background calculation required an extra iteration through each DCID in 3, which was costly.
Removing the need for the background calcluation eliminates this step, and it has been cut out of the code for speed.
'''


import argparse
import os, filecmp
import numpy as np
import json
import time
import datetime
from shapely.geometry import Polygon
from osgeo import gdal
from rasterio.features import rasterize
import logging
from emit_main.workflow.workflow_manager import WorkflowManager
import pandas as pd
from typing import List
import subprocess
import click
import yaml
import geopandas as gpd

import pv.pv
from annotate import plume_io, filter, utils
from quantification import compute_flux, windspeed, compute_Q_and_uncertainty_utils

gdal.osr.UseExceptions()


def get_sds_cog(fid, enh_version, dtype='ch4', data_value=''):
    date = fid[4:12]
    path=f'/store/emit/ops/data/acquisitions/{date}/{fid.split("_")[0]}/ghg/{dtype}/{fid.split("_")[0]}*_ghg_ort{data_value}{dtype}_b0106_{enh_version}.tif'
    return path
   

@click.command(name="scrape_refine_upload")
@click.argument('key', type=str)
@click.argument('id', type=str)
@click.argument('out_dir', type=str)
@click.argument('data_version', type=str)
@click.option('--enh_data_version', type=str, default='v02')
@click.option('--gtype', type=click.Choice(['ch4','co2']), default='ch4')
@click.option('--database_config', type=str,  default='/store/emit/ops/repos/emit-main/emit_main/config/ops_sds_config.json')
@click.option('--pv_config', type=str, default='TBD')
@click.option('--loglevel', type=str, default='DEBUG', help='logging verbosity')
@click.option('--logfile', type=str, default=None, help='output file to write log to')
@click.option('--continuous', is_flag=True, help='run continuously')
@click.option('--track_coverage_file', default='/store/brodrick/emit/emit-visuals/track_coverage_pub.json', help='Path to track coverage file')
@click.option('--plume_buffer_px', type=int, default=100, help='number of pixels to buffer plume cutouts by')
@click.option('--write_dcid_tifs', is_flag=True, help='write out dcid level tifs for debugging')
@click.option('--n_cores', type=int, default=1, help='number of CPUs to use')
@click.option('--num_dcids', type=int, default=-1, help='number of DCIDs to process, -1 for all')
@click.option('--specific_pid', type=str, default=None, help='Run this and only this plume ID (for debugging)')
@click.option('--sync_results', is_flag=True, help='sync results to remove server')
@click.option('--software_build_version', type=str, default=None, help='overwrite current tag with this software build version')
@click.option('--raw_annotation_override', type=str, default=None, help='ignore the key and id, and use this local file as the raw annotation input')
@click.option('--sync_only', is_flag=True, help='Only sync data')
def main(key: str, id: str, out_dir: str, data_version: str, enh_data_version: str, 
         gtype: str, database_config: str, pv_config:str, loglevel, logfile, continuous, track_coverage_file,
         plume_buffer_px, write_dcid_tifs, n_cores, num_dcids, specific_pid, sync_results, 
         software_build_version, raw_annotation_override, sync_only):

    # make an args like object that holds the parameters
    class Args:
        pass
    args = Args()
    args.key = key
    args.id = id
    args.out_dir = out_dir
    args.data_version = data_version
    args.enh_data_version = enh_data_version
    args.type = gtype
    args.database_config = database_config
    args.pv_config = pv_config
    args.loglevel = loglevel
    args.logfile = logfile
    args.continuous = continuous
    args.track_coverage_file = track_coverage_file
    args.plume_buffer_px = plume_buffer_px
    args.write_dcid_tifs = write_dcid_tifs
    args.n_cores = n_cores
    args.num_dcids = num_dcids
    args.specific_pid = specific_pid
    args.sync_results = sync_results
    args.software_build_version = software_build_version
    args.raw_annotation_override = raw_annotation_override
    args.sync_only = sync_only
    

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    np.random.seed(13)
    max_runs = 1
    if args.continuous:
        max_runs = int(1e15)
    
    if args.raw_annotation_override is not None:
        max_runs = 1
        logging.info('Using raw annotation override file, running single iteration')

    try:
        database = WorkflowManager(config_path=args.database_config).database_manager
    except:
        raise AttributeError('Could not open databse - check args.database_config')

    # Global File names
    fn = Filenames(args, create=True)
    fn.coverage_size = os.path.getsize(args.track_coverage_file)
    fn.annotation_size = None
    coverage = None

    if args.sync_only:
        sync_all(fn)
        return  


    # Loop only serves to rerun same instances over and over
    for run in range(max_runs):
        logging.debug('Loading Data')
        ######## Step 1 ###########

        # Update annotations file load
        if args.raw_annotation_override is None:
            utils.print_and_call(f'curl "https://popo.jpl.nasa.gov/mmgis/API/files/getfile" -H "Authorization:Bearer {args.key}" --data-raw "id={args.id}" > {fn.annotation_file_raw} 2>/dev/null')
        else:
            utils.print_and_call(f'cp {args.raw_annotation_override} {fn.annotation_file_raw}')
        
        if fn.annotation_size is not None and os.path.getsize(fn.annotation_file_raw) == fn.annotation_size:
            time.sleep(10)
            continue
        fn.annotation_size = os.path.getsize(fn.annotation_file_raw)

        manual_annotations = json.load(open(fn.annotation_file_raw,'r'))['body']['geojson']
        for _feat in range(len(manual_annotations['features'])):
            manual_annotations['features'][_feat]['properties']['Plume ID'] = manual_annotations['features'][_feat]['properties'].pop('name')

        manual_annotations_previous = None
        if os.path.isfile(fn.previous_annotation_file):
            manual_annotations_previous = json.load(open(fn.previous_annotation_file,'r'))

        logging.debug('Load pre-existing public-facing outputs (combined)')
        outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84" }, "type": "name"},"features":[],"name":"methane_metadata","type":"FeatureCollection" }
        if os.path.isfile(fn.output_json_internal):
            outdict = json.load(open(fn.output_json_internal,'r'))

        # If a plume didn't make it into the _internal.json, then it wasn't processed yet - remove it from the previous annotations to ensure rerun
        if len(outdict['features']) > 0 and manual_annotations_previous is not None:
            processed_plume_ids = pd.json_normalize(outdict['features'])['properties.Plume ID'].tolist()
            manual_annotations_previous['features'] = [feat for feat in manual_annotations_previous['features'] if feat['properties']['Plume ID'] in processed_plume_ids]

        # Reload coverage file if needed
        if coverage is None or os.path.getsize(args.track_coverage_file) != fn.coverage_size:
            fn.coverage_size = os.path.getsize(args.track_coverage_file)
            coverage = json.load(open(args.track_coverage_file,'r'))

      
        logging.info(f'Total plumes: {len(manual_annotations["features"])}')
        logging.debug('Run Spatial-Temporal Intersection to find FIDs')
        if args.specific_pid is None:
            manual_annotations, new_plumes = filter.add_fids(manual_annotations, coverage, manual_annotations_previous)
        else:
            # This is a special case primarily for development
            logging.info(f'Filtering to only {args.specific_pid}')
            manual_annotations['features'] = [feat for feat in manual_annotations['features'] if feat['properties']['Plume ID'] == args.specific_pid]
            manual_annotations, new_plumes = filter.add_fids(manual_annotations, coverage, None)

        latest_bulk_stats = {}
        latest_bulk_stats['New plumes, post fid filter'] = len(new_plumes)
        manual_annotations_df = pd.json_normalize(manual_annotations['features'])
        most_recent_plume_create = pd.to_datetime(manual_annotations_df['properties.Time Created']).max()
        last_time_range = pd.to_datetime(manual_annotations_df['properties.Time Range End']).dt.tz_localize('UTC').max()


        logging.debug('Pull Summary Stats')
        coverage_df = pd.json_normalize(coverage['features'])
        r0_review_needed = np.sum(pd.to_datetime(coverage_df['properties.end_time']) > last_time_range)

        double_approved_count = np.sum([ np.all([x['properties'][k] for k in ['R1 - Reviewed', 'R1 - VISIONS', 'R2 - Reviewed', 'R2 - VISIONS']]) for x in manual_annotations['features']])
        populated_emission_count = np.sum([ np.all([x['properties'][k] for k in ['R1 - Reviewed', 'R1 - VISIONS', 'R2 - Reviewed', 'R2 - VISIONS']]) and x['properties']['Simple IME Valid'] != 'NA' for x in manual_annotations['features']])
        reporting_emission_count = np.sum([ np.all([x['properties'][k] for k in ['R1 - Reviewed', 'R1 - VISIONS', 'R2 - Reviewed', 'R2 - VISIONS']]) and x['properties']['Simple IME Valid'] == 'Yes' for x in manual_annotations['features']])
        r1_review_count = np.sum([ x['properties']['R1 - Reviewed'] is False for x in manual_annotations['features']])
        r2_review_count = np.sum([ np.all([x['properties'][k] for k in ['R1 - Reviewed', 'R1 - VISIONS']]) and not x['properties']['R2 - Reviewed'] for x in manual_annotations['features']])

        latest_bulk_stats['Plume Complexes Approved for VISIONS'] = double_approved_count
        latest_bulk_stats['Approved Plume Complexes with Simple IME Evaluated'] = populated_emission_count
        latest_bulk_stats['Plume Complexes with Emission Estimate'] = reporting_emission_count
        latest_bulk_stats['R1 Review Deck'] = r1_review_count
        latest_bulk_stats['R2 Review Deck'] = r2_review_count
        latest_bulk_stats['Most Recent R0 Plume Created'] = str(most_recent_plume_create)
        latest_bulk_stats['Most Recent R0 Plume Scene'] = str(last_time_range)
        latest_bulk_stats['Most Recent Scene'] = str(coverage["features"][-1]["properties"]["end_time"])
        latest_bulk_stats['Inferred scenes needing R0 Review'] = r0_review_needed
        for key, val in latest_bulk_stats.items():
            logging.info(f'{key}: {val}')
        json.dump(latest_bulk_stats, open(fn.output_summary_json, 'w'), indent=2, cls=plume_io.SerialEncoder)

        # If there's nothing new, sleep and retry
        if len(new_plumes) == 0:
            time.sleep(10)
            continue

        logging.debug('Querry the EMIT DB to add Orbits and DCIDs')
        manual_annotations, new_plumes     = filter.add_orbits(manual_annotations, new_plumes, database)
        logging.info(f'New plume, post orbit: {len(new_plumes)}')
        del coverage, coverage_df


        ######## Step 2 ###########
        unique_fids = np.unique([sublist for x in new_plumes for sublist in manual_annotations['features'][x]['properties']['fids']])
        unique_orbits = np.unique([manual_annotations['features'][x]['properties']['orbit'] for x in new_plumes]).tolist()
        unique_dcids = np.unique([manual_annotations['features'][x]['properties']['dcid'] for x in new_plumes]).tolist()

        if args.num_dcids > 0:
            unique_dcids = unique_dcids[0:args.num_dcids]

        # Dump out the udpated manual annotations set, so it holds FIDs / orbits for next round
        logging.debug('Dump out the updated manual annotations set, overwrting original')
        plume_io.write_geojson_linebyline(fn.annotation_file, manual_annotations)

        for feat in manual_annotations['features']:
            if 'dcid' not in feat['properties'].keys():
                logging.error('Feature missing DCID: {feat}')

        updated_windspeeds = []
        if args.n_cores > 1:
            import ray

            @ray.remote(num_cpus=1)
            def process_dcid_ray(dcid, manual_annotations, new_plumes, fn, args):
                return process_dcid(dcid, manual_annotations, new_plumes, fn, args)
            
            ray.init(num_cpus=args.n_cores, object_store_memory=10*1024**3, _temp_dir='/local/ray', include_dashboard=False)

            # Steps 3 and 4 in parallel
            jobs = [process_dcid_ray.remote(dcid, manual_annotations, new_plumes, fn, args) for dcid in unique_dcids]
            results =  ray.get(jobs)

            # Step 5
            for res in results:
                updated_plumes_point, updated_plumes_poly, ws_update = res
                outdict = update_features(outdict, updated_plumes_poly, is_point=False, add_imgs=True, fn=fn)
                outdict = update_features(outdict, updated_plumes_point, is_point=True, add_imgs=True, fn=fn)
                updated_windspeeds.extend(ws_update)

            # Final write
            plume_io.write_geojson_linebyline(fn.output_json_internal, outdict)
            plume_io.write_external_geojson(fn.output_json_internal, fn.output_json_external)
            utils.print_and_call(f'cp {fn.annotation_file} {fn.previous_annotation_file}')
            ray.shutdown()

        else:
            # Preserve the ray-free mode, mostly for debugging

            # Now step through each DCID, get the relevant FIDs, mosaic, and cut out each plume
            for dcid in unique_dcids:

                # Step 3 and 4
                updated_plumes_point, updated_plumes_poly, ws_update = process_dcid(dcid, manual_annotations, new_plumes, fn, args)
                updated_windspeeds.extend(ws_update)

                # Step 5
                outdict = update_features(outdict, updated_plumes_poly, is_point=False, add_imgs=True, fn=fn)
                outdict = update_features(outdict, updated_plumes_point, is_point=True, add_imgs=True, fn=fn)

                utils.print_and_call(f'cp {fn.annotation_file} {fn.previous_annotation_file}')
                plume_io.write_geojson_linebyline(fn.output_json_internal, outdict) # Final write
                plume_io.write_external_geojson(fn.output_json_internal, fn.output_json_external)
                export_windspeeds(fn.working_windspeed_csv, updated_windspeeds)
                updated_windspeeds = []
        
        export_windspeeds(fn.working_windspeed_csv, updated_windspeeds)

        fn.daac_sync()

        # Sync
        if args.sync_results:
            sync_all(fn)
    


def sync_all(fn):

    utils.print_and_call(f'scp -q {fn.output_json_internal} ${{USER}}@${{NGIS_DATA_IP}}:/data/emit/mmgis/coverage/')
    utils.print_and_call(f'scp -q {fn.output_json_external} ${{USER}}@${{NGIS_DATA_IP}}:/data/emit/mmgis/coverage/')
    utils.print_and_call(f'/store/shared/rclone/bin/rclone sync {fn.quant_dir}/ {fn.dst_img_dir} --progress --transfers=16 --include="*_ql.png" -q > NUL 2>&1')
    fn.stac_sync()

       


class Filenames:
    def __init__(self, args, create=False):

        self.annotation_file_raw = os.path.join(args.out_dir, "manual_annotation_raw.json") # Straight from MMGIS
        self.annotation_file = os.path.join(args.out_dir, "manual_annotation.json") # MMGIS + FIDs + Orbits 
        self.previous_annotation_file = os.path.join(args.out_dir, "previous_manual_annotation.json") # Last editted version of annotation file

        # Output / intput jsons are identical, internal just includes additional information
        self.output_summary_json = os.path.join(args.out_dir, f'{args.type}_summary.json') # Output (public facing) metadata file
        self.output_json_external = os.path.join(args.out_dir, f'{args.type}_combined_plume_metadata_external.json') # Output (public facing) metadata file
        self.output_json_internal = os.path.join(args.out_dir, f'{args.type}_combined_plume_metadata_internal.json') # Output (internal facing) metadata file
        self.daac_dir = os.path.join(args.out_dir, 'daac') # Ready for DAAC sync
        self.delivery_dir = os.path.join(args.out_dir, 'delivery') # Delivery file directory
        self.quant_dir = os.path.join(args.out_dir, 'quantification') # Quantification working directory
        self.proc_dir = os.path.join(args.out_dir, 'processing') # Processing working directory
        self.pv_dir = os.path.join(args.out_dir, 'plume_vetting') # Plume vetting working directory
        self.working_windspeed_csv = os.path.join(args.out_dir, 'working_windspeed_estimates.csv') # Quantification windspeed working file

        self.dst_plm_cogdir = f'redhat:/data/emit/mmgis/mosaics/plm_cogs/{args.type}'
        self.dst_img_dir = f'redhat:/data/emit/mmgis/mosaics/images/{args.type}_q'

        self.gtype = args.type


        if create:
            os.makedirs(self.delivery_dir, exist_ok=True)
            os.makedirs(self.quant_dir, exist_ok=True)
            os.makedirs(self.proc_dir, exist_ok=True)
            os.makedirs(self.daac_dir, exist_ok=True)
            os.makedirs(self.pv_dir, exist_ok=True)
        
    @staticmethod
    def plume_delivery_basename(outdir, feat):
        return os.path.join(outdir, feat['properties']['Scene FIDs'][0][4:12], feat['properties']['Scene FIDs'][0] + '_' + feat['properties']['Plume ID'])

    @staticmethod
    def plume_working_basename(outdir, feat):
        return os.path.join(outdir, feat['properties']['Plume ID'])
 
    def feature_filenames(self, feat):
        outbase = self.plume_working_basename(self.proc_dir, feat)
        outmask_finepoly_file = outbase + '_finepolygon.json'
        outmask_poly_file = outbase + '_polygon.json'
        outmask_ort_file = outbase + '_mask_ort.tif'
        return outmask_finepoly_file, outmask_poly_file, outmask_ort_file

    def plume_vetting_filenames(self, feat):
        outbase = self.plume_working_basename(self.pv_dir, feat)
        out_inoutplume_file = outbase + '_inoutplume.png'
        out_spectralmatch_file = outbase + '_spectralmatch.png'

        return out_inoutplume_file, out_spectralmatch_file, outbase

    def quantification_filenames(self, poly_plume):
        base = self.plume_working_basename(self.quant_dir, poly_plume)

        raster_file = base + '_unmasked_' + self.gtype.upper() + '.tif'
        unc_file = base + '_unmasked_' + self.gtype.upper() + '_unc.tif'
        sns_file = base + '_unmasked_' + self.gtype.upper() + '_sns.tif'

        return raster_file, unc_file, sns_file
    
    def cmr_filename(self, poly_plume):
        base = self.plume_delivery_basename(self.daac_dir, poly_plume)
        cmr_file = base + f'.cmr.json'
        return cmr_file
    
    def delivery_filenames(self, poly_plume, daac_version=False):
        if daac_version:
            delivery_base = self.plume_delivery_basename(self.daac_dir, poly_plume)
        else:
            delivery_base = self.plume_delivery_basename(self.delivery_dir, poly_plume)

        os.makedirs(os.path.dirname(delivery_base), exist_ok=True)
        delivery_raster_file = delivery_base + '.tif'
        delivery_ql_file = delivery_base + '.png'
        delivery_json_file = delivery_base + '.json'
        delivery_uncert_file = delivery_raster_file.replace(self.gtype.upper(), self.gtype.upper() + '_UNC')
        delivery_sens_file = delivery_raster_file.replace(self.gtype.upper(), self.gtype.upper() + '_SNS')

        return delivery_raster_file, delivery_ql_file, delivery_json_file, delivery_uncert_file, delivery_sens_file
    
    def mmgis_q_img_filename_dict(self, plume):

        basedir = f'Layers/mosaics/images/{self.gtype}_q/'
        img_dicts = [{
                'name': 'Quantification Mask',
                'url': basedir +  plume['properties']['Plume ID'] + '_ql.png',
                'type': 'image'
                },
                {
                'name': 'Quantification Sweep',
                'url': basedir +  plume['properties']['Plume ID'] + '_sweep.pdf',
                'type': 'document'
            }]
        return img_dicts
    
    def stac_sync(self):
        output_plumes = [x for x in json.load(open(self.output_json_external, 'r'))['features'] if x['geometry']['type'] == 'Polygon']
        for plume in output_plumes:
            plm_in_file = self.delivery_filenames(plume, daac_version=True)[0]
            dt = datetime.datetime.strptime(os.path.basename(plm_in_file).split('_')[0], 'emit%Y%m%dt%H%M%S')
            plm_out_file = os.path.join(self.dst_plm_cogdir, dt.strftime('%Y-%m-%dT%M_%H_%S') + '-to-' + (dt + datetime.timedelta(seconds=1)).strftime('%Y-%m-%dT%M_%H_%S'))
            cmd_str = f'/store/shared/rclone/bin/rclone copyto {plm_in_file} {plm_out_file}.tif --progress -q > NUL 2>&1'
            utils.print_and_call(cmd_str)

    def daac_sync(self):

        output_plumes = [x for x in json.load(open(self.output_json_external, 'r'))['features'] if x['geometry']['type'] == 'Polygon']
        for plume in output_plumes:
            delivery_raster, delivery_ql, delivery_json, delivery_uncert, delivery_sens = self.delivery_filenames(plume, daac_version=False)
            daac_raster, daac_ql, daac_json, daac_uncert, daac_sens = self.delivery_filenames(plume, daac_version=True)
            cmr_file = self.cmr_filename(plume)  
            for de, da in zip([delivery_raster, delivery_ql, delivery_json],
                               [daac_raster, daac_ql, daac_json]):

                # desitnation and CMR file (indicating delivery) need to be present before we worry.  Otherwise, just copy
                if os.path.isfile(da) and os.path.isfile(cmr_file): 
                    # Check if file contents are the same - skip geotiff metadata
                    same = False
                    if da.endswith('.json'):
                        if filecmp.cmp(de, da, shallow=False):
                            same = True
                    elif da.endswith('.tif') or da.endswith('.png'):
                        if utils.compare_raster_data(de, da):
                            same = True

                    # If it's there and the same, skip safely
                    if same:
                        continue
                    else:
                        # If it's there and different, raise the alarm
                        logging.warning(f'DAAC sync - file {da} exists and is different from delivery - we cannot delivery this file in the same version')

                else:
                    # If it's not there yet, copy
                    utils.print_and_call(f'cp {de} {da}')
                    continue


def export_windspeeds(working_windspeed_csv, updated_windspeeds):
    if len(updated_windspeeds) == 0:
        return
    updated_windspeeds = [x for x in updated_windspeeds if x is not None]
    if os.path.isfile(working_windspeed_csv):
        wd_df = pd.read_csv(working_windspeed_csv)
        ws_df = pd.concat([wd_df, pd.DataFrame(updated_windspeeds)], ignore_index=True)
    else:
        ws_df = pd.DataFrame(updated_windspeeds)
    ws_df.drop_duplicates(subset=['plume_id'], keep='last', inplace=True)
    ws_df.to_csv(working_windspeed_csv, index=False)






def process_dcid(dcid, manual_annotations, new_plumes, fn, args):

    logging.info(f'Processing DCID {dcid}...')
    #plumes_idx_in_dcid = [x for x in range(len(manual_annotations['features'])) if manual_annotations['features'][x]['properties']['dcid'] == dcid]
    plumes_idx_in_dcid = [x for x in new_plumes if manual_annotations['features'][x]['properties']['dcid'] == dcid]
    fids_in_dcid = np.unique([sublist for x in plumes_idx_in_dcid for sublist in manual_annotations['features'][x]['properties']['fids']])
    logging.info(f'...found {len(plumes_idx_in_dcid)} plumes to update and {len(fids_in_dcid)} FIDs')

    ort_dat_files = [get_sds_cog(fid, args.enh_data_version, dtype=args.type) for fid in fids_in_dcid]
    ort_sens_files = [get_sds_cog(fid, args.enh_data_version, dtype=args.type, data_value='sens') for fid in fids_in_dcid]
    unc_dat_files = [get_sds_cog(fid, args.enh_data_version, dtype=args.type, data_value='uncert') for fid in fids_in_dcid]

    dcid_ort_vrt_file = os.path.join(fn.proc_dir, f'dcid_{dcid}_mf_ort.vrt')
    dcid_ort_unc_vrt_file = os.path.join(fn.proc_dir, f'dcid_{dcid}_unc_ort.vrt')
    dcid_ort_sns_vrt_file = os.path.join(fn.proc_dir, f'dcid_{dcid}_sns_ort.vrt')

    dcid_ort_tif_file = os.path.join(fn.proc_dir, f'dcid_{dcid}_mf_ort.tif')
    dcid_ort_unc_tif_file = os.path.join(fn.proc_dir, f'dcid_{dcid}_unc_ort.tif')
    dcid_ort_sns_tif_file = os.path.join(fn.proc_dir, f'dcid_{dcid}_sns_ort.tif')
    utils.print_and_call(f'gdalbuildvrt {dcid_ort_vrt_file} {" ".join(ort_dat_files)} --quiet')
    if args.write_dcid_tifs:
        utils.print_and_call(f'gdal_translate {dcid_ort_vrt_file} {dcid_ort_tif_file} -co COMPRESS=LZW --quiet')
    else:
        dcid_ort_tif_file = dcid_ort_vrt_file  # just use the VRT directly to save space/time

    ort_ds = gdal.Open(dcid_ort_tif_file)
    trans = ort_ds.GetGeoTransform()

    # Only create sns and unc if needed

    utils.print_and_call(f'gdalbuildvrt {dcid_ort_unc_vrt_file} {" ".join(unc_dat_files)} --quiet')
    utils.print_and_call(f'gdalbuildvrt {dcid_ort_sns_vrt_file} {" ".join(ort_sens_files)} --quiet')
    if args.write_dcid_tifs:
        utils.print_and_call(f'gdal_translate {dcid_ort_unc_vrt_file} {dcid_ort_unc_tif_file} -co COMPRESS=LZW --quiet')
        utils.print_and_call(f'gdal_translate {dcid_ort_sns_vrt_file} {dcid_ort_sns_tif_file} -co COMPRESS=LZW --quiet')
    else:
        dcid_ort_unc_tif_file = dcid_ort_unc_vrt_file
        dcid_ort_sns_tif_file = dcid_ort_sns_vrt_file
    unc_ds = gdal.Open(dcid_ort_unc_tif_file)
    sns_ds = gdal.Open(dcid_ort_sns_tif_file)

    # Calculate pixel size for DCID
    proj_ds = gdal.Warp('', dcid_ort_tif_file, dstSRS='EPSG:3857', format='VRT')
    transform_3857 = proj_ds.GetGeoTransform()
    xsize_m = transform_3857[1]
    ysize_m = transform_3857[5]
    del proj_ds


    ######## Step 3 ##############
    # Use the manual plumes to come up with a new set of plume masks and labels
    updated_plumes_poly, updated_plumes_point, updated_windspeed = [], [], []
    for newp in plumes_idx_in_dcid:
        feat = manual_annotations['features'][newp]
        logging.info(f'Building output plume {feat["properties"]["Plume ID"]}')

        deliver_emissions = feat['properties']['Simple IME Valid'] == 'Yes'

        rawspace_coords = plume_io.rawspace_coordinate_conversion([], feat['geometry']['coordinates'][0], trans, ortho=True)

        datshape = (ort_ds.RasterYSize, ort_ds.RasterXSize)
        window, newp_trans, local_coords = plume_io.get_window(rawspace_coords, trans, datshape, args.plume_buffer_px)
        if window is None:
            logging.warning(f'Plume {feat["properties"]["Plume ID"]} has invalid window, skipping')
            continue

        cut_plume_data = ort_ds.ReadAsArray(window[0], window[1], window[2], window[3]).squeeze()
        cut_uncdat = unc_ds.ReadAsArray(window[0], window[1], window[2], window[3]).squeeze()
        cut_snsdat = sns_ds.ReadAsArray(window[0], window[1], window[2], window[3]).squeeze()

        manual_mask = rasterize(shapes=[Polygon(local_coords)], out_shape=(cut_plume_data.shape[0],cut_plume_data.shape[1]), dtype=np.uint8) # numpy binary mask for manual IDs

        plumestyle = 'classic'
        if 'Delineation Mode' in feat['properties'].keys():
            plumestyle = feat['properties']['Delineation Mode']

        loc_fid_mask = None # change name to dcid_plume_mask
        if plumestyle == 'classic':
            loc_fid_mask = utils.plume_mask_threshold(cut_plume_data.copy(), manual_mask, style=args.type)
        elif plumestyle == 'manual':    
            loc_fid_mask = manual_mask.astype(bool)
        

        ############  Step 4 ###########
        outmask_finepoly_file, outmask_poly_file, outmask_ort_file = fn.feature_filenames(feat) 

        # Write mask file and save tif reference
        #write_output_file(newp_trans, ort_ds.GetProjection(), cut_plume_mask, outmask_ort_file)
        #write_output_file(newp_trans, ort_ds.GetProjection(), cut_plume_data, outmask_ort_file)
        plume_io.write_cog(outmask_ort_file, loc_fid_mask.reshape((loc_fid_mask.shape[0], loc_fid_mask.shape[1],1)).astype(np.uint8), 
                           newp_trans, ort_ds.GetProjection(), nodata_value=0)
        
        if os.path.isfile(outmask_poly_file):
            os.remove(outmask_poly_file)
        if os.path.isfile(outmask_finepoly_file):
            os.remove(outmask_finepoly_file)
        utils.print_and_call(f'gdal_polygonize {outmask_ort_file} {outmask_finepoly_file} -f GeoJSON -mask {outmask_ort_file} -8 -quiet')
        utils.print_and_call(f'ogr2ogr {outmask_poly_file} {outmask_finepoly_file} -f GeoJSON -lco RFC7946=YES -simplify {trans[1]/2} --quiet')

        # Read back in the polygon we just wrote
        plume_to_add = json.load(open(outmask_poly_file))['features']
        if len(plume_to_add) > 1:
            logging.warning(f'ACK - multiple polygons from one Plume ID in file {outmask_poly_file}')
        plume_to_add[0]['geometry']['coordinates'] = [[[np.round(x[0],5), np.round(x[1],5)] for x in plume_to_add[0]['geometry']['coordinates'][0]]]

        masked_cut_plume_data = cut_plume_data.copy()
        masked_cut_plume_data[~loc_fid_mask] = -9999
        poly_plume, point_plume = utils.build_plume_properties(feat['properties'], plume_to_add[0]['geometry'], masked_cut_plume_data, 
                                                               newp_trans, args.data_version, xsize_m, ysize_m)



        delivery_raster_file, delivery_ql_file, delivery_json_file, delivery_uncert_file, delivery_sens_file = fn.delivery_filenames(poly_plume)

        # Write delivery files
        meta = plume_io.get_metadata(poly_plume, plume_io.global_metadata(data_version=args.data_version, software_version=args.software_build_version))
        plume_io.write_cog(delivery_raster_file, cut_plume_data.astype(np.float32), newp_trans, ort_ds.GetProjection(), nodata_value=-9999, metadata=meta, mask=loc_fid_mask)
        plume_io.write_cog(delivery_uncert_file, cut_uncdat.astype(np.float32), newp_trans, ort_ds.GetProjection(), nodata_value=-9999, metadata=meta, mask=loc_fid_mask)
        plume_io.write_cog(delivery_sens_file, cut_snsdat.astype(np.float32), newp_trans, ort_ds.GetProjection(), nodata_value=-9999, metadata=meta, mask=loc_fid_mask)
        plume_io.write_color_quicklook(cut_plume_data, delivery_ql_file, inmask=loc_fid_mask, trim=True, style=args.type)

        # Write unmasked version of delivery files for quantification (mainly for plotting)
        quant_raster_file, quant_uncert_file, quant_sens_file = fn.quantification_filenames(poly_plume)
        plume_io.write_cog(quant_raster_file, cut_plume_data.astype(np.float32), newp_trans, ort_ds.GetProjection(), nodata_value=-9999, metadata=meta)
        plume_io.write_cog(quant_uncert_file, cut_uncdat.astype(np.float32), newp_trans, ort_ds.GetProjection(), nodata_value=-9999, metadata=meta)
        plume_io.write_cog(quant_sens_file, cut_snsdat.astype(np.float32), newp_trans, ort_ds.GetProjection(), nodata_value=-9999, metadata=meta)

        # Compute Emissions
        emissions_info, windspeed_info = compute_Q_and_uncertainty_utils.single_plume_emissions(
            feat,
            poly_plume,
            fn.quant_dir,
            fn.proc_dir,
            quant_raster_file,
            quant_sens_file,
            quant_uncert_file,
            fn.annotation_file,
            working_windspeed_csv=fn.working_windspeed_csv,
            overrule_simple_ime_flag=True, # we want to run the calc no matter what - we'll discard later per metadata
        )

        # Plume vetting - compute d_norm score and estimated plume length
        pv_cfg = yaml.safe_load(args.pv_config)
        gpd_plume_data = gpd.GeoDataFrame.from_features(manual_annotations['features'])

        out_inoutplume_file, out_spectralmatch_file, out_ch4target_basefile = fn.plume_vetting_filenames(feat)
        pv_result = pv.pv.plume_vetting(
            plume_data=gpd_plume_data,
            plume_id=feat['properties']['Plume ID'],
            cfg=pv_cfg,
            out_inoutplume_file=out_inoutplume_file,
            out_spectralmatch_file=out_spectralmatch_file,
            out_ch4target_basefile=out_ch4target_basefile,
        )[0]

        # TODO: How to save pv_result in MMGIS JSON file?
        # pv_result[0] = d_norm score
        # pv_result[1] = estimated plume length



        poly_plume['properties'].update(emissions_info)
        point_plume['properties'].update(emissions_info)

        # For the delivery file, if not flagged for emissions delivery, don't include
        plume_io.write_delivery_json(delivery_json_file, poly_plume, meta['DAAC Scene Names'], deliver_emissions)

        # Now save for output / archive jsons
        updated_plumes_point.append(point_plume)
        updated_plumes_poly.append(poly_plume)
        if windspeed_info is not None:
            updated_windspeed.append(windspeed_info)


    return updated_plumes_point, updated_plumes_poly, updated_windspeed


def update_features(existing: dict, new_features: List, is_point: bool, add_imgs: dict=None, fn: Filenames=None) -> dict:

    if int(add_imgs is not None) + int(fn is not None) == 1:
        logging.warning('Both add_imgs and fn must be provided, or neither; skipping image addition')

    ########## Step 5 ##########
    for plm in new_features:
        if is_point:
            existing_match_index = [_x for _x, x in enumerate(existing['features']) if plm['properties']['Plume ID'] == x['properties']['Plume ID'] and x['geometry']['type'] == 'Point']
        else:
            existing_match_index = [_x for _x, x in enumerate(existing['features']) if plm['properties']['Plume ID'] == x['properties']['Plume ID'] and x['geometry']['type'] != 'Point']
            
        if len(existing_match_index) > 2:
            logging.warning("HELP! Too many matching indices")

        if add_imgs is not None and fn is not None:
            plm['properties']['images'] = fn.mmgis_q_img_filename_dict(plm)

        if len(existing_match_index) > 0:
            existing['features'][existing_match_index[0]] = plm
        else:
            existing['features'].append(plm)
        
        
    return existing
 


if __name__ == '__main__':
    main()


