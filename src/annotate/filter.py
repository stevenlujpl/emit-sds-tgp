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
from copy import deepcopy
import pandas as pd
import logging
from shapely.geometry import Polygon
import shapely




def spatial_temporal_filter(cov_df, coverage, roi, start_time, end_time):

    temporal_inds = np.where(np.logical_and(cov_df['properties.start_time'] >= pd.to_datetime(start_time) , 
                                            cov_df['properties.end_time'] <= pd.to_datetime(end_time) ))[0]
    spatial_inds = np.where(cov_df['geometry.coordinates'][:][temporal_inds].apply(lambda s,
                        roi=roi: s.intersects(roi)))[0]

    return [coverage['features'][i] for i in temporal_inds[spatial_inds]]

def roi_filter(coverage, roi):
    subdir = deepcopy(coverage) 
    cov_df = pd.json_normalize(coverage['features'])
    cov_df['geometry.coordinates'] = cov_df['geometry.coordinates'].apply(lambda s: Polygon(s[0]) )
    inds = np.where(cov_df['geometry.coordinates'].apply(lambda s,roi=roi: s.intersects(roi)))[0]
    subdir['features'] = [subdir['features'][i] for i in inds]
    return subdir
 
def time_filter(coverage, start_time, end_time):
    subdir = deepcopy(coverage)
    cov_df = pd.json_normalize(coverage['features'])
    inds = np.where(np.logical_and(pd.to_datetime(cov_df['properties.start_time']) >= pd.to_datetime(start_time) , pd.to_datetime(cov_df['properties.end_time']) <= pd.to_datetime(end_time)))[0]
    subdir['features'] = [subdir['features'][i] for i in inds]
    return subdir



def add_fids(manual_annotations, coverage, manual_annotations_previous):
    manual_annotations_fid = deepcopy(manual_annotations)

    previous_plume_ids = []
    if manual_annotations_previous is not None:
        previous_plume_ids = [x['properties']['Plume ID'] for x in manual_annotations_previous['features']]

    updated_plumes=[]
    todel = []
    del_msg = []

    dm_mr1 = 'Missing "R1 - Reviewed" field'
    dm_int = 'Deleting entry from no intersection'
    dm_emis = 'Entries desiring emissions estimate need Origin'

    # do some dataframe conversion once ahead of time to make things faster
    coverage_df = pd.json_normalize(coverage['features'])
    coverage_df['geometry.coordinates'] = coverage_df['geometry.coordinates'].apply(lambda s: Polygon(s[0]) )
    coverage_df['properties.start_time'] = pd.to_datetime(coverage_df['properties.start_time'])
    coverage_df['properties.end_time'] = pd.to_datetime(coverage_df['properties.end_time'])

    for _feat, feat in enumerate(manual_annotations['features']):
        logging.debug(f'Adding new fid {_feat} / {len(manual_annotations["features"])}')
        # If this key isn't present, then the full feature wasn't really added yet
        if 'R1 - Reviewed' not in feat['properties'].keys():
            todel.append(_feat)
            del_msg.append(dm_mr1)
            continue # This is insufficient

        if  feat['properties']['Simple IME Valid'] == 'Yes' and \
            ('Origin' not in feat['properties'].keys() or feat['properties']['Origin'] is None or len(feat['properties']['Origin']) == 0):
            todel.append(_feat)
            del_msg.append(dm_emis)
            continue

        plume_id = feat['properties']['Plume ID']
        if plume_id in previous_plume_ids:
            new_geom = feat['geometry']['coordinates']
            prev_idx = previous_plume_ids.index(plume_id)
            prev_geom = manual_annotations_previous['features'][prev_idx]['geometry']['coordinates']

            # check reviews
            rev_match = True
            for rl in ['R1 - Reviewed','R2 - Reviewed','R1 - VISIONS','R2 - VISIONS', 
                       'Origin', 'Sector', 'Sector Confidence', 'Time Range End', 'Time Range Start', 'Simple IME Valid']:
                if rl in feat['properties'] and rl in manual_annotations_previous['features'][prev_idx]['properties']:
                    if feat['properties'][rl] != manual_annotations_previous['features'][prev_idx]['properties'][rl]:
                        rev_match = False

            if new_geom == prev_geom and rev_match:
                manual_annotations_fid['features'][_feat] = manual_annotations_previous['features'][prev_idx]
                #logging.debug(f'Geometries and properties the same in {feat["properties"]}...skipping safely')
                continue

        roi = Polygon(feat['geometry']['coordinates'][0])
        roi = shapely.buffer(roi, 0.01, join_style='mitre')
        subset_features = spatial_temporal_filter(coverage_df, coverage, roi, 
                                                  feat['properties']['Time Range Start'] + 'Z', 
                                                  feat['properties']['Time Range End'] + 'Z') 
        if len(subset_features) == 0:
            todel.append(_feat)
            del_msg.append(dm_int)
        else:
            fids = [subset_features[x]['properties']['fid'].split('_')[0] for x in range(len(subset_features))]
            manual_annotations_fid['features'][_feat]['properties']['fids'] = fids
            updated_plumes.append(_feat)


    if len(todel) > 0:
        logging.warning('The following plumes have metadata issues that need to be resolved, skipping all for now:')
        for td, rationale in zip(np.array(todel)[::-1], np.array(del_msg)[::-1]):
            msg = f'{manual_annotations_fid["features"][td]["properties"]["Plume ID"]} - Reason: {rationale}'
            logging.warning(msg)
            manual_annotations_fid['features'].pop(td)

        updated_plumes = np.array([x for x in updated_plumes if x not in todel]) # shouldn't be necessary anymore, deosn't hurt
        for td in np.array(todel)[::-1]:
            updated_plumes[updated_plumes >= td] -= 1
        updated_plumes = updated_plumes.tolist()


    #if len(todel) > 0:
    #    logging.warning('Incomplete intersection (likely bad time bounds) for the following plumes:')
    #    for td in np.array(todel)[::-1]:
    #        msg = f'Deleting entry from no intersection - check input {manual_annotations_fid["features"][td]["properties"]["Plume ID"]}'
    #        logging.warning(msg)
    #        manual_annotations_fid['features'].pop(td)

    #updated_plumes = np.array([x for x in updated_plumes if x not in todel]) # shouldn't be necessary anymore, deosn't hurt
    #for td in np.array(todel)[::-1]:
    #    updated_plumes[updated_plumes >= td] -= 1
    #updated_plumes = updated_plumes.tolist()

    return manual_annotations_fid, updated_plumes

def clear_bad(todel, manual_annotations, updated_plumes, rationale=''):
    if len(todel) > 0:
        for td in np.array(todel)[::-1]:
            msg = f'{manual_annotations["features"][td]["properties"]["Plume ID"]} - Reason: {rationale}'
            logging.warning(msg)
            manual_annotations['features'].pop(td)

        updated_plumes = np.array([x for x in updated_plumes if x not in todel]) # shouldn't be necessary anymore, deosn't hurt
        for td in np.array(todel)[::-1]:
            updated_plumes[updated_plumes >= td] -= 1
        updated_plumes = updated_plumes.tolist()
    return manual_annotations, updated_plumes



def add_orbits(annotations, indices_to_update, database):

    ind_to_pop = []
    update_ind_to_pop = []
    for _ind, ind in enumerate(indices_to_update):
        db_ret = [database.find_acquisition_by_id(fid) for fid in annotations['features'][ind]['properties']['fids']]
        orbits = [db_ret[_fid]['orbit'] for _fid, fid in enumerate(annotations['features'][ind]['properties']['fids'])]
        dcids  = [db_ret[_fid]['associated_dcid']  for _fid, fid in enumerate(annotations['features'][ind]['properties']['fids'])]

        scene_numbers  = [db_ret[_fid]['daac_scene'] if 'daac_scene' in db_ret[_fid].keys() else None for _fid, fid in enumerate(annotations['features'][ind]['properties']['fids']) ]

        un_orbits = np.unique(orbits)
        un_dcids = np.unique(dcids)

        if len(db_ret) == 0 or None in scene_numbers:
            logging.info(f'No FIDs or DAAC Scenes at {annotations["features"][ind]["properties"]["Plume ID"]}...skipping')
            annotations['features'][ind]['properties']['orbit'] = []
            ind_to_pop.append(ind)
            update_ind_to_pop.append(_ind)
            continue

        if len(un_dcids) > 1:
            logging.error(f'Ack - entry {annotations["features"][ind]} spans two dcids')
            annotations['features'][ind]['properties']['orbit'] = []
            ind_to_pop.append(ind)
            update_ind_to_pop.append(_ind)
            continue

        annotations['features'][ind]['properties']['orbit'] = un_orbits[0]
        annotations['features'][ind]['properties']['dcid'] = un_dcids[0]
        annotations['features'][ind]['properties']['daac_scenes'] = scene_numbers
    
    if len(ind_to_pop) > 0:
        logging.info('Bad plume list:')
    for ind in np.array(ind_to_pop)[::-1]:
        logging.info(annotations['features'][ind]['properties']['Plume ID'])
        annotations['features'].pop(ind)

    indices_to_update = np.array(indices_to_update)
    for _ind in np.array(update_ind_to_pop)[::-1]:
        indices_to_update[_ind:] -= 1
    indices_to_update = indices_to_update.tolist()

    for _ind in np.array(update_ind_to_pop)[::-1]:
        indices_to_update.pop(_ind)
    return annotations, indices_to_update


