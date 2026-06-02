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
# Authors: Jay Fahlen, philip.brodrick@jpl.nasa.gov

r"""
python compute_flux.py --windms 3 --minppmm 500 --mergedistm 0 \
--maxfetchm 1000 --minareapx 9 cmf_img lng lat

Todo:
- handle distribution skew due to stats excluding nodata pixels:
  e2.g., ime = sum(cmf[mask]) = sum(mask)*mean(cmf[mask]) 
  UNLESS mask contains nodata pixels, then
  sum(mask) > sum(mask & ~nodata)
  and
  sum(cmf[mask]) != sum(mask)*mean(cmf[mask]) 

- for a plume with enhancement \in uniform CMF background,
  variance(ccmean[i<=t]) ~ variance(ccmean[i>t])
"""

from __future__ import absolute_import, division, print_function


from warnings import warn

import sys, os
from os.path import  split as pathsplit
from os.path import splitext

import csv
import numpy as np
import pylab as pl
from osgeo import gdal
from osgeo.gdal import osr
import logging

osr.UseExceptions()

from scipy.spatial.distance import pdist,squareform
from scipy.spatial import ConvexHull, distance_matrix

from skimage.measure import label as imlabel
from skimage.morphology import disk, remove_small_objects
from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import rasterio as rio
import pandas as pd
from annotate import plume_io

#              ppm(m)   L/m^3     mole/L    kg/mole
PPMM2KGCH4 = (1.0/1e6)*(1000.0)*(1.0/22.4)*(0.016042) # 7.161607142857143e-07
KGCH42PPMM = 1.0/PPMM2KGCH4 # 1396334.621618252

# #  CO2!!        ppm(m)   L/m^3     mole/L    kg/mole
# PPMM2KGCH4 = (1.0/1e6)*(1000.0)*(1.0/22.4)*(0.04401)

minareapx = 16
labradpx = 9

plot_diag = False

def mad(a,axis=-1,medval=None,unbiased=False):    
    '''
    computes the median absolute deviation of a list of values
    mad = median(abs(a - medval))/c
    '''
    # try:
    #     from statsmodels.robust.scale import mad as _mad
    #     c = 1.0 if not unbiased else 0.67448975019608171
    #     center = medval or np.median        
    #     return _mad(a, axis=axis, c=c, center=center)
    # except:
    ma = np.asarray(a)
    medval = medval or np.median(ma,axis=axis)
    return np.median(np.abs(ma-medval),axis=axis)

def extrema(a,p=1.0,buf=0.0,axis=None,**kwargs):
    if len(a)==0:
        return np.nan,np.nan
    
    if p==1.0:
        vmin,vmax = np.nanmin(a,**kwargs),np.nanmax(a,**kwargs)
    else:        
        assert(p>0.0 and p<1.0)
        apercent = lambda q: np.nanpercentile(a,axis=axis,q=q,
                                              method='nearest')
        vmin,vmax = apercent((1-p)*100),apercent(p*100)

    if buf !=0:
        vbuf = (vmax-vmin)*buf
        vmin,vmax = vmin-vbuf,vmax+vbuf
    return vmin,vmax

def ppmm2kg(pixels_ppmm,ps,verbose=False):
    """
    ppmm2kg(pixels_ppmm,ps,verbose=False)
    
    Summary: compute ime given a set of cmf pixels each with gsd ps
    
    Arguments:
    - pixels_ppmm: list of cmf pixels (ppmm)
    - ps: pixel size (m)
    
    Keyword Arguments:
    - verbose: verbose (default=False)
    
    Output:
    - pixelwise IME (kg/hr)
    """
    
    #assert (np.isfinite(pixels_ppmm) & (pixels_ppmm>=0)).all()
    scalef = PPMM2KGCH4 * (ps*ps)
    if verbose:
        logging.debug(f'ppmm->kg scalef @ {ps:-5.2f}m: {scalef}')
    
    return pixels_ppmm * scalef

def ime(pixels_ppmm,ps):
    return ppmm2kg(np.asarray(pixels_ppmm),ps).sum()

def ime2flux(ime_kg,fetch_m,wind_ms=5.0):
    """
    ime2flux(ime_kg,fetch_m,wind_ms=5.0)
    
    Summary: compute (point source) plume emission rate given IME
    
    Arguments:
    - ime_kg: plume IME (kg/m)
    - fetch_m: plume fetch (i.e., major axis length, m)
    
    Keyword Arguments:
    - wind_ms: wind speed (m/s, default=5.0)
    
    Output:
    - plume flux rate (kg/hr)
    """
    return (ime_kg * wind_ms * 3600.0) / fetch_m

def ime2cq(ime_kg,fetch_m):
    """
    Summary: compute (point source) plume emission rate given IME
    
    Arguments:
    - ime_kg: plume IME (kg/m)
    - fetch_m: plume fetch (i.e., major axis length, m)
    
    Output:
    - C_Q: for compatibility with Jay's method (C_Q = Q/U) 
    """
    return (3600 * ime_kg) / fetch_m


def geo2utmzone(lng,lat):
    if lng > 180.0:
        lng = lng-360.0
    zone = int((180.0+lng)/6.0)
    hemi = 'N' if (lat >= 0.0) else 'S'
    return zone,hemi

def utmzone2epsg(zone,hemi,astype=str):
    hemiprefix = dict(N='326',S='327')
    epsgstr = hemiprefix[hemi] + '%02d'%int(zone)
    return astype(epsgstr)

def disk(radius,**kwargs):
    from skimage.morphology import disk as _disk
    return _disk(radius,**kwargs)

def findobj(labimg,max_label=0):
    from scipy.ndimage import find_objects as _findobj
    return _findobj(labimg,max_label=max_label)

def bwdist(bwimg,**kwargs):
    metric = kwargs.get('metric','euclidean')
    if metric=='euclidean':
        kwargs.pop('metric',None) # metric is only option for edt
        from scipy.ndimage import distance_transform_edt as _bwdist
    elif metric in ('chessboard','taxicab'):
        from scipy.ndimage import distance_transform_cdt as _bwdist
    kwargs.setdefault('return_distances',True)
    kwargs.setdefault('return_indices',False)
    return _bwdist(bwimg,**kwargs)

def chull(p,return_indices=False,**kwargs):
    from scipy.spatial import ConvexHull as _chull
    hullidx = _chull(p,**kwargs).vertices
    hullp = p[hullidx]
    if return_indices:
        return hullp,hullidx
    return hullp

def mergelabels(labimg,mergedist,return_merged=False,doplot=False):
    # merge labeled regions <= mergedist pixels from each other
    labmask = labimg!=0
    mergereg = bwdist(~labmask,metric='euclidean')<=mergedist
    mergereg = imlabel(mergereg,connectivity=2)
    mergelab = np.unique(mergereg)[1:]
    mergeimg = np.zeros_like(labimg)
    mergemap = {}
    nbefore = 0
    for mlab,mobj in zip(mergelab,findobj(mergereg)):
        mlmask = (mergereg[mobj]==mlab) & labmask[mobj]
        mergeimg[mobj][mlmask] = mlab
        if return_merged:
            mergemap[mlab] = np.unique(labimg[mobj][mlmask])
            nbefore += len(mergemap[mlab]) 
            logging.debug(f'{nbefore} input labels -> {len(mergelab)} merged labels @ mergedist {mergedist}px')
    if doplot:
        import pylab as pl
        fig,ax = pl.subplots(1,2,sharex=True,sharey=True)
        ax[0].imshow(labimg)
        ax[0].set_title(f'{nbefore} labels before')
        ax[1].imshow(mergeimg)
        ax[1].set_title(f'{len(mergelab)} labels after')
        pl.show()
    if return_merged:
        return mergeimg, mergemap
    return mergeimg

def extract_tile(img,ul,tdim,verbose=False,transpose=None,fill_value=0):
    '''
    extract a tile of dims (tdim,tdim,bands) offset from upper-left 
    coordinate ul in img, zero pads when tile overlaps image extent 
    '''
    try:
        if len(tdim)==1:
            tdim = (tdim[0],tdim[0])
    except:
        tdim = (tdim,tdim)

    assert (len(tdim)==2)
            
    ndim = img.ndim
    if ndim==3:
        nr,nc,nb = img.shape
    elif ndim==2:
        nr,nc = img.shape
        nb = 1
    else:
        raise Exception('invalid number of image dims %d'%ndim)

    lr = (ul[0]+tdim[0],ul[1]+tdim[1])

    # get absolute offsets into img (clip if outside image extent)
    ibeg,iend = max(0,ul[0]),min(nr,lr[0])
    jbeg,jend = max(0,ul[1]),min(nc,lr[1])

    # get relative offsets into imgtile (shift if outside image extent)
    padt,padl = max(0,-ul[0]), max(0,-ul[1])
    padb,padr = padt+(iend-ibeg), padl+(jend-jbeg)
    
    if verbose:
        logging.debug(f'img.nrows,img.ncols, {nr}, {nc}')
        logging.debug(f'ul,lr,tdim, {ul}, {lr}, {tdim}')
        logging.debug(f'padt,padb,padl,padr, {padt}, {padb}, {padl}, {padr}')
        logging.debug(f'ibeg,iend,jbeg,jend, {ibeg}, {iend}, {jbeg}, {jend}')

    imgtile = fill_value*np.ones([tdim[0],tdim[1],nb],dtype=img.dtype)
    imgtile[padt:padb,padl:padr] = np.atleast_3d(img[ibeg:iend,jbeg:jend])
    if transpose is not None:
        imgtile = imgtile.transpose(transpose)
    return imgtile


def compute_flux(args):
    
    verbose = args.verbose 

    rgbimgf = args.rgbimgf
    labimgf = args.labimgf
    windms = 1

    mergedistm = args.mergedistm
    maxfetchm = args.maxfetchm
    minaream2 = args.minaream2
    mask_mode = args.mask_mode

    plot_diag = args.plot_diag
    
    cmfimgf = args.cmfimgf
    lng = args.lng
    lat = args.lat
    fid = args.fid

    cmfbase = splitext(pathsplit(cmfimgf)[1])[0]
    imebase = os.path.join(args.plot_path, fid) + args.name_suffix
        
    imemskf = imebase+'_mask.tif'
    imefigf = imebase+'_ql.png'
    imgfigf = imebase+'_diagimg.pdf'
    diagfigf = imebase+'_diagfig.pdf'
    swpfigf = imebase+'_sweep.pdf'
    maskcsvf = imebase+'_mask.csv'
    ccallcsvf = imebase+'_ccall.csv'
    ccposcsvf = imebase+'_ccpos.csv'
    ccthrcsvf = imebase+'_ccthr.csv'

    lid = cmfbase.split('_')[0]
    
    def plume_rois(cmf,minppmm,minareapx,mergedistpx):
        detrois = imlabel(cmf>minppmm,connectivity=2)

        if len(detrois) == 0:
            return f'error: detrois has 0 length in plume_rois()'
        
        try:
            _ = detrois.max()
        except:
            return f'error: detrois.max() fails in plume_rois()'

        if detrois.max()==0:
            logging.debug(f'no CMF pixels > minppmm={minppmm}')
            return detrois
        
        if minareapx>1:
            detrois = remove_small_objects(detrois,minareapx,connectivity=2)
            if detrois.max()==0:
                logging.debug(f'no ROIs with area > minareapx={minareapx}')
                return detrois
            detrois = imlabel(detrois!=0,connectivity=2)
    
        if mergedistpx>0:
            detrois = mergelabels(detrois,mergedistpx)
            
        return detrois

    def fetch_masks(detrois,plumeij,mergedistpx,maxfetchpx):
        fetchpx = maxfetchpx
        plumemask = np.zeros_like(detrois,dtype=np.bool_)
        plumemask[plumeij[0],plumeij[1]] = 1
        plumedist = bwdist(~plumemask,metric='euclidean')
        detidmask = plumedist<=mergedistpx
        plumerois = detrois[detidmask]
        plumerois = plumerois[plumerois!=0]
        if len(plumerois)!=0:
            plumemask = np.isin(detrois,plumerois)
            plumehull = chull(np.c_[np.where(plumemask)])
            hulldists = np.tril(squareform(pdist(plumehull)))
            fetchpx = min(hulldists.max(),2*maxfetchpx)
        else:
            # no plume ROIs found
            plumemask[plumeij[0],plumeij[1]] = 0
            error_msg = f'Warning: no detection ROIs within {mergedistpx}px of plume location {plumeij}'
            logging.debug(error_msg)
            return error_msg, [], [], []

        fetchmask = plumedist<=fetchpx
        return plumemask, detidmask, fetchmask, fetchpx
    
    cmfsrc = rio.open(cmfimgf,'r')
    cmfcrs = cmfsrc.crs

    # make sure we're UTM projected
    cmflng,cmflat = cmfsrc.lnglat()
    zone,hemi = geo2utmzone(cmflng,cmflat)
    cmf_epsg = cmfcrs.to_epsg()
    utm_epsg = int(utmzone2epsg(zone,hemi))

    logging.debug(f'utm zone,hemi: {zone,hemi}')
    
    # if not: use warpedvrt
    if not any([str(cmf_epsg).startswith(prefix) for prefix in ['326','327']]):
        # replace the OGC:CRS84 (lng,lat) projection with conventional WGS-84 (lat,lng) CRS
        logging.debug(f'Converting {str(cmfcrs)} (lng,lat) to EPSG:{utm_epsg} (lat,lng) CRS')
        cmfsrc = gdal.Warp('', cmfimgf, dstSRS=f'EPSG:{utm_epsg}', format='VRT', resampleAlg='bilinear')
        cmfcrs = cmfsrc.GetSpatialRef()
    else:
        cmfsrc = gdal.Open(cmfimgf)

    # Get EPSG code from the spatial reference
    if hasattr(cmfcrs, 'GetAuthorityCode'):
        utm_epsg = int(cmfcrs.GetAuthorityCode(None))
    else:
        utm_epsg = int(utmzone2epsg(zone, hemi))

    cmfimg = cmfsrc.ReadAsArray().squeeze()
    cmfmsk = np.logical_and(cmfimg != cmfsrc.GetRasterBand(1).GetNoDataValue() , np.isfinite(cmfimg))
    cmfimg_sens = cmfimg
    if args.sns_file != 'None':
        senssrc = None
        if not any([str(cmf_epsg).startswith(prefix) for prefix in ['326','327']]):
            senssrc = gdal.Warp('', args.sns_file, dstSRS=f'EPSG:{utm_epsg}', format='VRT', resampleAlg='bilinear')
        else:
            senssrc = gdal.Open(args.sns_file)
        sens = senssrc.ReadAsArray().squeeze()
        cmfimg_sens = cmfimg / sens
        cmfimg_sens[sens == 0] = 0.
    
    if args.unc_file != 'None':
        uncsrc = None
        if not any([str(cmf_epsg).startswith(prefix) for prefix in ['326','327']]):
            uncsrc = gdal.Warp('', args.unc_file, dstSRS=f'EPSG:{utm_epsg}', format='VRT', resampleAlg='bilinear')
        else:
            uncsrc = gdal.Open(args.unc_file)
        unc = uncsrc.ReadAsArray().squeeze()
    else:
        unc = np.zeros_like(cmfimg)

    cmfpix = cmfimg[cmfmsk]
    cmfmin = cmfpix.min()
    cmfmed = np.median(cmfpix)
    cmfmad = mad(cmfpix,medval=cmfmed)
    cmfmax = cmfpix.max()

    logging.debug(f'cmf (med,mad): {cmfmed,cmfmad}')
    logging.debug(f'cmf (min,max): {cmfmin,cmfmax}')
    
    minppmm = args.minppmm or cmfmin
    maxppmm = args.maxppmm or cmfmax

    rgbsrc = cmfsrc
    if cmfsrc.RasterCount ==1 and not rgbimgf:
        rgbimg = np.zeros([cmfimg.shape[0],cmfimg.shape[1],3],dtype=np.uint8)
    else:
        if rgbimgf:
            rgbsrc = rio.open(rgbimgf,'r')
        rgbimg = rgbsrc.read([1,2,3]).transpose([1,2,0])/15

    rgbimg = np.clip(rgbimg,0,1)

    # get (utmx,utmy) and (row,col) coord for (lng,lat))
    geocrs = osr.SpatialReference()
    geocrs.ImportFromEPSG(4326)
    cmfcrs = osr.SpatialReference()
    cmfcrs.ImportFromEPSG(utm_epsg)
    geo2cmf = osr.CoordinateTransformation(geocrs, cmfcrs)
    utmx,utmy,_ = geo2cmf.TransformPoint(lat,lng)
    def index(ds,x,y):
        gt = ds.GetGeoTransform()
        px = int((x - gt[0]) / gt[1])
        py = int((y - gt[3]) / gt[5])
        return py,px

    i,j = index(cmfsrc,utmx,utmy)
    del geocrs, cmfcrs
    logging.debug(f'(lng,lat): {lng,lat}\n(utmx,utmy): {utmx,utmy}\n(i,j): {i,j}')

    manual_boundary_mask = np.ones_like(cmfimg).astype(bool)
    if len(args.manual_boundary_coordinates_lon_lat) > 0:

        bnd_list = list(zip(args.manual_boundary_coordinates_lon_lat[::2], args.manual_boundary_coordinates_lon_lat[1::2]))
        manual_boundary_coordinates_ij = []
        for lon_lat in bnd_list:
            utmx,utmy,_ = geo2cmf.TransformPoint(lon_lat[1],lon_lat[0])
            manual_boundary_coordinates_ij.append(index(cmfsrc,utmx,utmy))
            #manual_boundary_coordinates_ij.append(cmfsrc.index(utmx,utmy))
        p = Path(np.array(manual_boundary_coordinates_ij))

        #Set up the test points
        xp, yp = np.meshgrid(np.arange(cmfimg.shape[0]), np.arange(cmfimg.shape[1]), indexing='ij')
        xp, yp = xp.flatten(), yp.flatten()
        points = np.vstack((xp,yp)).T

        #Test which points are inside the polygon
        manual_boundary_mask = p.contains_points(points).reshape(cmfimg.shape)
    
    #ps = abs(cmfsrc.res[0])
    trans = cmfsrc.GetGeoTransform()
    ps = (abs(trans[1]) + abs(trans[5])) / 2.0
    logging.debug(f'ps: {ps:.3f} (m^2)')

    mergedistpx = int(np.ceil(mergedistm/ps))
    maxfetchpx = int(np.ceil(maxfetchm/ps))    
    minareapx = int(np.ceil(minaream2/ps/ps))

    maxenhance=False
    if maxenhance:
        # move (i,j) to max enhancement location within mergedist-sized window
        ioff = max(mergedistpx,1)
        cmfoff = cmfimg[i-ioff:i+ioff+1,j-ioff:j+ioff+1].copy()
        cmfoff[cmfoff!=cmfoff] = -9999
        oi,oj = np.where(cmfoff == cmfoff.max())
        ni,nj = i+(oi[0]-ioff),j+(oj[0]-ioff)
        logging.debug(f'maxenhance nearest {i,j} (+/- {ioff}px) -> {ni,nj}')
        i,j = ni,nj    

    logging.debug(f'maxfetchm: {maxfetchm} -> maxfetchpx: {maxfetchpx}')
    logging.debug(f'mergedistm: {mergedistm} -> mergedistpx: {mergedistpx}')
    logging.debug(f'minaream2: {minaream2} -> minareapx: {minareapx}')
    if minareapx <= 1:
        logging.debug(f'minarea<=1px with minarea={minaream2}m^2 at image ps={ps}m^2: no area filtering performed ')

    # note: hack
    # do not use connected components > maxfetchpx px from (i,j)
    imin,imax = max(0,i-maxfetchpx),min(cmfimg.shape[0],i+maxfetchpx)
    jmin,jmax = max(0,j-maxfetchpx),min(cmfimg.shape[1],j+maxfetchpx)

    if labimgf:
        labmsk = rio.open(labimgf).read(1).squeeze()!=0
        poslab = imlabel(labmsk,connectivity=2)
    
    logging.debug(f'imin,imax: {imin,imax}, jmin,jmax: {jmin,jmax}')

    if mask_mode=='infer':
        logging.debug("infer new mask")
        detrois = np.zeros_like(cmfimg,dtype=np.int32)
        circle_x, circle_y = np.ogrid[:cmfimg.shape[0], :cmfimg.shape[1]]
        circle_mask = (circle_x - i)**2 + (circle_y - j)**2 < maxfetchpx**2
        cmfimg_manual_boundary = cmfimg * manual_boundary_mask * circle_mask.astype(float)
        ret = plume_rois(cmfimg_manual_boundary[imin:imax,jmin:jmax],
                         minppmm,minareapx,mergedistpx)
        # If ret is a string, that means it holds an error message, so return
        if type(ret) == str:
            return ret, None
        detrois[imin:imax,jmin:jmax] = ret

        logging.debug(f'{len(np.unique(detrois))-1} rois detected')

        try:
            plumemask, detidmask, fetchmask, fetchpx = fetch_masks(detrois,(i,j),
                                                                mergedistpx,
                                                                maxfetchpx)    
        except:
            return f'error: fetch_masks function failed', None

        # If plumemask is a string, that means it holds an error message, so return
        if type(plumemask) == str:
            return plumemask, None
        
        fetchm = fetchpx*ps
        logging.debug(f'estimated fetch: {fetchm} (m)')
        
    elif mask_mode=='label':
        logging.debug("using pre-existing mask")
        plumemask = labmsk
        fetchm = maxfetchm
        fetchpx = maxfetchpx
        
        plumedist = bwdist(~plumemask,metric='euclidean')
        fetchmask = plumedist<=fetchpx
        detidmask = plumedist<=mergedistpx
    
    logging.debug(f'plumemask area: {plumemask.sum()}')

    plume_io.write_cog(imemskf, np.uint8(255*plumemask).reshape(plumemask.shape[0], plumemask.shape[1], 1), 
                       cmfsrc.GetGeoTransform(), cmfsrc.GetProjection(), 
                       nodata_value = 0)
        
    logging.debug(f'saved: {imemskf}')

    plumepix = cmfimg[(plumemask!=0) & cmfmsk]
    plumesens_pix = cmfimg_sens[(plumemask!=0) & cmfmsk] # If not running with sensitivity then cmfimg_sens = cmfimg
    plumeunc_pix = unc[(plumemask!=0) & cmfmsk]
    pltcmf = cmfimg_sens.copy() # save a copy for rendering

    if plot_diag:
        with pl.rc_context({'font.size': 16}):
            make_plot(pltcmf, plumemask, i, j, maxfetchpx, manual_boundary_coordinates_ij, imin, imax, jmin, jmax, imefigf, ps)
    
    if len(plumepix)==0:
        logging.debug(f'no plume pixels > {minppmm}ppmm with area > {minaream2}, exiting')
        plumeime = plumeflux = fetchm = plumearea = plumemean = plumevar = np.nan
        logging.debug('BBB exiting')
        sys.exit(1)
        
################################ 

    ### Simple Mask 
    #plumeime_MASK = ime(plumepix,ps)
    #C_Q_MASK = ime2cq(plumeime_MASK,fetchm)
    #plumeunc_kg = ppmm2kg(plumeunc_pix, ps) * 3600
    #C2_UNC_MASK = np.sum(plumeunc_kg**2)
    #C_Q_MASK =    np.sum( plumepix      * ps*ps * PPMM2KGCH4 * 3600) / fetchm
    C_Q_MASK =    np.sum( plumesens_pix * ps*ps * PPMM2KGCH4 * 3600) / fetchm
    if args.sns_file == 'None':
        # If we didn't do the sensitivity correction, then don't provide an uncertainty because the uncertainty
        # only applies to the sensitivity corrected data.
        C2_UNC_MASK = 0. 
    else:
        C2_UNC_MASK = np.sum((plumeunc_pix  * ps*ps * PPMM2KGCH4 * 3600)**2)
        
    ### CC 
    dmin = min(cmfimg.shape[:2])
    if cmfimg.shape[0]==dmin:
        ccrad = np.abs([dmin-i,i]).max()//2
    else: # cmfimg.shape[1]==dmin
        ccrad = np.abs([dmin-j,j]).max()//2

    ccrad = min(ccrad,2*maxfetchpx)

    # ccirc = d_euc(x,y,x0,y0) for (x0,y0) @ (lat,lon), d_euc<=ccrad
    ccrad = ccrad+1 if ccrad%2==1 else ccrad
    ccirc = bwdist(disk(ccrad),metric='euclidean')
    ccmax = ccirc.max()

    # invert bwdist to get ccirc in correct form
    ccirc = np.float32(ccmax-ccirc) 
    ccinc = np.sqrt(2)
    dists = np.arange(1,ccmax,ccinc)

    # ccloc = top,left coord of ccirc relative to cmf extent
    ccloc = i-ccrad,j-ccrad
    ccdim = ccirc.shape[0]
    ccwin = (ccloc[1],ccloc[0],ccdim,ccdim)

    # extract buffered tiles from [cmf,msk,rgb] to speed up processing
    cmfimg = extract_tile(cmfimg,ccloc,ccdim,fill_value=-9999).squeeze()
    cmfmsk = extract_tile(cmfmsk,ccloc,ccdim,fill_value=0).squeeze()!=0
    rgbimg = extract_tile(rgbimg,ccloc,ccdim,fill_value=0)

    # exclude pixels outside of ccmax ring & nodata pixels
    ccmsk = (ccirc<ccmax) & cmfmsk
    ccirc[~ccmsk] = np.inf

    # exclude outliers in rendering, clip to abs min and center extrema on zero
    cmfpmin,cmfpmax = extrema(cmfimg[cmfmsk & ccmsk],0.9)
    cmfpmin = -(np.abs([cmfpmin,cmfpmax]).min())    

    ###
        
    if labimgf:
        npos = 0

        labradpx = ccrad
        if verbose:
            logging.debug(f'{poslab.shape} labradpx: {labradpx}px')
        
        rmin,rmax = imin,imax+1
        cmin,cmax = jmin,jmax+1
        posrad = extract_tile(poslab,ccloc,ccdim,fill_value=0).squeeze()
            
            
        if verbose:
            logging.debug(f'{posrad.shape} labuse in {rmin,rmax,cmin,cmax}: ')
        labuse = np.unique(posrad*disk(labradpx))
        if labuse.any():
            labmsk = np.isin(poslab,labuse[labuse!=0])            
        else:
            labmsk = np.zeros_like(cmfimg,dtype=np.bool_)

        npos = sum(labmsk.ravel())

        if plot_diag:
            pltcmf = cmfimg.copy()
            pltcmf[cmfimg<250] = np.nan
            pltmsk = np.float32(labmsk)
            pltmsk[labmsk==0] = np.nan
            figrows,figcols,figscale=1,2,5
            figsize=(figcols*figscale,figrows*figscale*1.05)
            fig,ax = pl.subplots(figrows,figcols,figsize=figsize,
                         sharex=True,sharey=True)

            ax[0].imshow(pltcmf,vmin=0,vmax=maxppmm,cmap='YlOrRd')
            ax[1].imshow(pltcmf,vmin=0,vmax=maxppmm,cmap='YlOrRd')    
            ax[1].imshow(pltmsk,vmin=0,vmax=1,cmap='RdBu_r')

            buf = 20
            if npos > 0 and npos < buf**2:
                ii,jj = np.where(labmsk)
                ax[0].set_xlim(jj.min()-buf,jj.max()+buf)
                ax[0].set_ylim(ii.min()-buf,ii.max()+buf)

            pl.tight_layout()
            pl.subplots_adjust(top=0.95,hspace=0.05,wspace=0.05)
            pl.suptitle(f'ps={ps:3.1f}')            
            # pl.savefig(diagfigf)
            pl.close(fig)

        labi,labj = np.where(labmsk)
        
        # hullij = chull(np.c_[labi,labj])
        #logging.debug(f'i,j: {i,j}')
        #i,j = hullij[np.argmin(np.abs(np.c_[i,j]-hullij).sum(axis=1))]
        #logging.debug(f'hull i,j: {i,j}')
    else:
        #poslab = imlabel(cmfimg>minppmm)
        labmsk = np.zeros_like(cmfimg,dtype=np.bool_)
        pass 


    labarea = labmsk.sum() # zero if no label image provided

    # compute concentric circle and (experimental) distance ratio flux stats
    oarea = osum = opsum = oime = 0
    ccmeans,ccareas = [],[]
    cpmeans,cpareas = [],[]
    ctmeans,ctareas = [],[]
    sums,psums,areas,means,pmeans = [],[],[],[],[]
    navg = np.zeros_like(cmfimg,dtype=np.float32) #  px count used in avg / ring
    pavg = np.zeros_like(navg) # percent positive pixels / ring
    cavg = np.zeros_like(navg) # cmf mean / ring
    cmin = np.zeros_like(navg) # cmf min / ring
    cmax = np.zeros_like(navg) # cmf max / ring
    cflx = np.zeros_like(navg) # flux / ring
    cime = np.zeros_like(navg) # ime / ring
    rrat = np.zeros_like(navg) # (dist < d px sum) - (dist \in [d,ccrad] px sum)
    nrat = np.zeros_like(navg) # (dist < d px count) - (dist \in [d,ccrad] px count)

    omsk = np.zeros_like(cmfimg,dtype=np.bool_)
    masklab = maskpos = False
    posmsk = cmfmsk & (cmfimg>0)
    thrmsk = cmfmsk & (cmfimg>minppmm)
    for didx,d in enumerate(dists):
        # dmsk = pixels within d px of (i,j)
        dmsk = (ccirc<=d)
        # pmsk = pixels > d and < maxdist px from (i,j)
        pmsk = (ccirc>d) & (ccirc<=dists[-1])
        if masklab:
             dmsk &= labmsk
             pmsk &= labmsk
        else:
             dmsk &= cmfmsk
             pmsk &= cmfmsk
        # omsk = old dmsk at dists[didx-1]
        # cmsk = current concentric circle at distance d
        cmsk = (np.uint8(dmsk) - np.uint8(omsk))==1
        ncmsk = cmsk.sum()

        nppix = pmsk.sum()
        if verbose:
            logging.debug(f'd: {d}: {ncmsk} {nppix}')#, raw_input()
        if ncmsk==0 or nppix==0:
            # no pixels in current ccirc or no positive pixels in exterior 
            break
        dpix = cmfimg[dmsk]        
        dmean = dpix.mean()
        dsum = dpix.sum()
        dime = ppmm2kg(dsum,ps)
        dflx = ime2flux(dime,d*ps,windms)
        darea = dmsk.sum()

        # plume mask ime/flux
        ppix = cmfimg[pmsk]
        psum = ppix.sum()
        pmean = ppix.mean()

        # cc ime/flux for pixels > 0
        pospix = cmfimg[dmsk & posmsk]
        possum = pospix.sum()
        posime = ppmm2kg(possum,ps)
        posflx = ime2flux(posime,d*ps,windms)

        # cc ime/flux for pixels > minppmm
        thrpix = cmfimg[dmsk & thrmsk]
        thrsum = thrpix.sum()
        thrime = ppmm2kg(thrsum,ps)
        thrflx = ime2flux(thrime,d*ps,windms)
        
        
        if verbose:
            logging.debug(f'd: {d} area: {darea} adiff: {darea-oarea}')
            logging.debug(f'mean: {dmean} sum: {dsum} sdiff: {dsum-osum}')
            logging.debug(f'ime*: {dime:.4f} flux*: {dflx:.4f}')
            logging.debug(f'ime+: {posime:.4f} flux+: {posflx:.4f}')        
            logging.debug(f'ime>: {thrime:.4f} flux>: {thrflx:.4f}')        

        ccpix = cmfimg[cmsk]
        ccmean = ccpix.mean()
        ccmeans.append(ccmean)
        ccareas.append(ncmsk)

        cavg[cmsk] = ccmean
        cmin[cmsk] = ccpix.min()
        cmax[cmsk] = ccpix.max()
        navg[cmsk] = cmsk.sum()
        pavg[cmsk] = (posmsk & cmsk).sum() / navg[cmsk] 
        cime[cmsk] = dime-oime
        cflx[cmsk] = dflx

        rrat[cmsk] = (dsum-osum) - (psum-opsum)
        nrat[cmsk] = len(dpix)-len(ppix)

        if masklab and darea == labarea:
            break


        # note: cpmean/ctmean wrt all pixels in ring (cmsk.sum()),
        #       not just px in mask (cpmsk.sum(), ctmsk.sum())
        cpmsk = cmsk & posmsk
        cppix = cmfimg[cpmsk]
        cpmean = cppix.sum() / cmsk.sum() 
        nposmsk = len(cppix)
        cpmeans.append(cpmean)
        cpareas.append(nposmsk)

        ctmsk = cmsk & thrmsk
        ctpix = cmfimg[ctmsk]
        ctmean = ctpix.sum() / cmsk.sum() 
        nthrmsk = len(ctpix)
        ctmeans.append(ctmean)
        ctareas.append(nthrmsk)        

        # save previous metrics for didx>1 comparisons
        oarea = darea
        osum = dsum
        opsum = psum
        omsk = dmsk
        oime = dime

        means.append(dmean)
        pmeans.append(pmean)
        sums.append(dsum)
        psums.append(psum)
        areas.append(darea)

    # compute cc stats for dists < estimated fetch
    dmask = dists[:-1]<=fetchpx
    try:
        ccmeans = np.float32(ccmeans)[dmask]
    except:
        return f'error: dmask cannot subscript because of dimension error', None
    ccareas = np.float32(ccareas)[dmask]
    ccimes = ppmm2kg(ccmeans*ccareas,ps)
    ccsum = ccmeans.sum()
    ccmean = ccmeans.mean()
    ccvar = ccmeans.var()
    ccime = ccimes.sum()

    C_Q_CC = ime2cq(ccime,fetchm)
    
    ### Write csv for plumes that have an ROI detection
    #csv_file_path = os.path.join(args.csv_path, args.csv_file)
    #status = 'success'
    #if not os.path.exists(csv_file_path): 
    #    with open(csv_file_path, 'w', newline='') as file: 
    #        writer = csv.writer(file)
    #        writer.writerow(['plume_id', 'status', 'C_Q_mask_kg_hr_mpers', 'C_Q_ccirc_kg_hr_mpers', 'lon', 'lat', 'fetch', 
    #                        'merge_dist', 'min_pppmm_thresh', 'max_ppmm_thresh', 
    #                        'min_area_m',  'pixel_size'])
    #        writer.writerow([fid, status, C_Q_MASK, C_Q_CC, lng, lat, fetchm, mergedistm, args.minppmm, args.maxppmm, args.minaream2, ps])
    #else: 
    #    with open(csv_file_path, 'a', newline='') as file: 
    #        writer = csv.writer(file)
    #        logging.debug('Writing row')
    #        writer.writerow([fid, status, C_Q_MASK, C_Q_CC, lng, lat, fetchm, mergedistm, args.minppmm, args.maxppmm, args.minaream2, ps])
    # ## 

    returns = [fid, C_Q_MASK, C_Q_CC, lng, lat, fetchm, mergedistm, args.minppmm, args.maxppmm, args.minaream2, ps, C2_UNC_MASK]
    
    logging.debug([fid, C_Q_MASK, C_Q_CC, lng, lat, fetchm, mergedistm, args.minppmm, args.maxppmm, args.minaream2, ps, C2_UNC_MASK])
#     ccflux = ime2flux(ccime,fetchm,windms)    
#     ccarea = ps*ps*ccareas.sum()
    
#     logging.debug(f'ime(ccirc*):  {ccime} (kg)')
#     logging.debug(f'flux(ccirc*): {ccflux} (kg/hr)')    
#     logging.debug(f'mean(ccirc*): {ccmean} (ppmm)')
#     logging.debug(f'var(ccirc*):  {ccvar} (ppmm)')
#     logging.debug(f'area(ccirc*): {ccarea} (m^2)')

#     plumeime = ime(plumepix,ps)
#     plumefluxf = ime2flux(plumeime,fetchm,windms)
#     plumemean = plumepix.mean()
#     plumevar = plumepix.var()
#     plumearea = ps*ps*plumemask.sum()
    
#     logging.debug(f'ime(mask):  {plumeime} (kg)')
#     logging.debug(f'flux(mask): {plumeflux} (kg/hr)')
#     logging.debug(f'area(mask): {plumearea} (m^2)')
#     logging.debug(f'mean(mask): {plumemean} (ppmm)')
#     logging.debug(f'var(mask):  {plumevar} (ppmm)')

#     # compute cc stats for dists < estimated fetch, positive px only    
#     cpmeans = np.float32(cpmeans)[dmask]
#     cpareas = np.float32(cpareas)[dmask]
#     cpimes = ppmm2kg(cpmeans*cpareas,ps)
#     cpmean = cpmeans.mean()
#     cpvar = cpmeans.var()
#     cpime = cpimes.sum()
#     cpflux = ime2flux(cpime,fetchm,windms)
#     cparea = ps*ps*cpareas.sum()
    
#     logging.debug(f'ime(ccirc+):  {cpime} (kg)')
#     logging.debug(f'flux(ccirc+): {cpflux} (kg/hr)')    
#     logging.debug(f'mean(ccirc+): {cpmean} (ppmm)')
#     logging.debug(f'var(ccirc+):  {cpvar} (ppmm)')
#     logging.debug(f'area(ccirc+): {cparea} (m^2)')

    ccflux = ime2flux(ccime,fetchm,windms)    
    ccarea = ps*ps*ccareas.sum()
    # logging.debug(f'ime(ccirc*):  {ccime} (kg)')
    # logging.debug(f'flux(ccirc*): {ccflux} (kg/hr)')    
    # logging.debug(f'sum(ccirc*): {ccsum} (ppmm)')
    # logging.debug(f'mean(ccirc*): {ccmean} (ppmm)')
    # logging.debug(f'var(ccirc*):  {ccvar} (ppmm)')
    # logging.debug(f'area(ccirc*): {ccarea} (m^2)')

    # df = pd.DataFrame([[lid,lat,lng,windms,ccime,ccflux,fetchm,
    #                    ccarea,ccmean,ccvar]],columns=dfcols)
    # df.to_csv(ccallcsvf,index=False)

    # compute cc stats for dists < estimated fetch, positive px only    
    cpmeans = np.float32(cpmeans)[dmask]
    cpareas = np.float32(cpareas)[dmask]
    cpimes = ppmm2kg(cpmeans*cpareas,ps)
    cpmean = cpmeans.mean()
    cpvar = cpmeans.var()
    cpime = cpimes.sum()
    cpflux = ime2flux(cpime,fetchm,windms)
    cparea = ps*ps*cpareas.sum()
    
    logging.debug(f'ime(ccirc+):  {cpime} (kg)')
    logging.debug(f'flux(ccirc+): {cpflux} (kg/hr)')    
    logging.debug(f'mean(ccirc+): {cpmean} (ppmm)')
    logging.debug(f'var(ccirc+):  {cpvar} (ppmm)')
    logging.debug(f'area(ccirc+): {cparea} (m^2)')
    # df = pd.DataFrame([[lid,lat,lng,windms,cpime,cpflux,fetchm,
                        # cparea,cpmean,cpvar]],columns=dfcols)
    # df.to_csv(ccposcsvf,index=False)


    # compute cc stats for dists < estimated fetch, positive px only    
    ctmeans = np.float32(ctmeans)[dmask]
    ctareas = np.float32(ctareas)[dmask]
    ctimes = ppmm2kg(ctmeans*ctareas,ps)
    ctmean = ctmeans.mean()
    ctvar = ctmeans.var()
    ctime = ctimes.sum()
    ctflux = ime2flux(ctime,fetchm,windms)
    ctarea = ps*ps*ctareas.sum()
    
    logging.debug(f'ime(ccirc>):  {ctime} (kg)')
    logging.debug(f'flux(ccirc>): {ctflux} (kg/hr)')    
    logging.debug(f'mean(ccirc>): {ctmean} (ppmm)')
    logging.debug(f'var(ccirc>):  {ctvar} (ppmm)')
    logging.debug(f'area(ccirc>): {ctarea} (m^2)')    

    # df = pd.DataFrame([[lid,lat,lng,windms,ctime,ctflux,fetchm,
    #                     ctarea,ctmean,ctvar]],columns=dfcols)
    # df.to_csv(ccthrcsvf,index=False)
    
    if plot_diag:
        dists = dists[:len(areas)]
        # binarize rrat,nrat to simplify vis
        rrat = np.float32(rrat>0)
        nrat = np.float32(nrat>0)
        #rrat[rrat==0] = np.nan
        #rrat[nrat==1] = 0

        rrat[~ccmsk] = np.nan
        nrat[~ccmsk] = np.nan    

        means = np.float32(means)
        pmeans = np.float32(pmeans)
        labmsk = np.float32(labmsk)
        labmsk[labmsk!=1] = np.nan

        cmfimg[~(cmfmsk & (cmfimg>minppmm))] = np.nan
        figrows,figcols,figscale=2,4,7.5
        figsize=(figcols*figscale,figrows*figscale)
        fig,ax = pl.subplots(figrows,figcols,figsize=figsize,
                     sharex=True,sharey=True)
        for axi in ax.ravel():
            axi.imshow(rgbimg)

        cavg[~ccmsk] = np.nan
        cmin[~ccmsk] = np.nan
        cmax[~ccmsk] = np.nan
        navg[~ccmsk] = np.nan    
        cime[~ccmsk] = np.nan
        cflx[~ccmsk] = np.nan

        ax[0,0].add_patch(pl.Circle((ccrad,ccrad), fetchpx, ec='w', color='none', lw=2))
        ax[0,0].add_patch(pl.Circle((ccrad,ccrad), maxfetchpx, ec='red', color='none', ls=':', lw=2))
        ax[0,0].imshow(cmfimg,vmin=minppmm,vmax=maxppmm,cmap='YlOrRd');
        ax[0,0].set_title(rf'CMF $\in$ [{minppmm:.0f},{maxppmm:.0f}] ppmm')
        ax[0,1].imshow(ccirc,cmap='RdYlBu_r'); ax[0,1].set_title(rf'CC dists $\in$ [0,{dists.max()*ps:.0f}] m')
        ax[0,2].imshow(cavg,cmap='RdYlBu_r',vmin=cmfpmin,vmax=cmfpmax);
        ax[0,2].set_title(rf'Mean ppmm/circle $\in$ [{cmfpmin:.0f},{cmfpmax:.0f}] ppmm')
        #ax[0,2].imshow(cmin,cmap='RdYlBu_r'); ax[0,2].set_title('Min ppmm/circle')
        #ax[0,2].imshow(cmax,cmap='RdYlBu_r'); ax[0,2].set_title('Max ppmm/circle')

        #ax[0,3].scatter(hullij[:,1],hullij[:,0],50,c='k')
        #ax[0,3].imshow(navg,cmap='RdYlBu_r'); ax[0,3].set_title('Valid px/circle (#)')    
        ax[0,3].imshow(pavg,vmin=0.0,vmax=1.0,cmap='RdBu_r'); ax[0,3].set_title('Positive px/circle (%)')
        ax[1,0].imshow(cime,cmap='RdYlBu_r'); ax[1,0].set_title(r'$\Delta$IME/circle (ppmm)')
        ax[1,0].add_patch(pl.Circle((ccrad,ccrad), fetchpx, ec='k', color='none', lw=2))
        ax[1,1].imshow(cflx,cmap='RdYlBu_r'); ax[1,1].set_title('Flux (cumulative)')

        #for axi in ax.ravel()[3:4]:
        #    axi.imshow(labmsk,vmin=0,vmax=1,cmap='RdBu_r')
        ax[1,2].imshow(rrat,cmap='RdYlBu_r',alpha=0.5); ax[1,2].set_title('IME(C$^-$) > IME(C$^+$)')
        ax[1,3].imshow(nrat,cmap='RdYlBu_r',alpha=0.5); ax[1,3].set_title('Area(C$^-$) > Area(C$^+$)')

        pl.tight_layout()
        ax[0,0].set_xticks([]);
        ax[0,0].set_yticks([]);
        pl.subplots_adjust(top=0.95,hspace=0.05,wspace=0.05)
        # pl.savefig(imgfigf)
        areas = np.float32(areas)
        sums = np.float32(sums)
        psums = np.float32(psums)
        imes = ppmm2kg(sums,ps)

        fluxs = np.float32([ime2flux(ime,d*ps,wind_ms=windms) for ime,d in zip(imes,dists)])
        fluxdiffs = np.float32([0]+list(np.diff(fluxs)))

        meandiffs = np.float32([0]+list(np.diff(means)))
        areadiffs = np.float32([0]+list(np.diff(areas)))
        sumdiffs = np.float32([0]+list(np.diff(sums)))
        rmeandiffs = meandiffs-(sumdiffs/np.maximum(1,areadiffs))
        imediffs = np.float32([0]+list(np.diff(imes)))

        # scaling factor for color mapping only
        sumdiffs = 200*(sumdiffs/sumdiffs.max())

        fluxdiffmin,fluxdiffmax = extrema(fluxdiffs[np.isfinite(fluxdiffs)])
        logging.debug(f'extrema(fluxs): {extrema(fluxs[np.isfinite(fluxs)])}')
        logging.debug(f'extrema(imediffs): {extrema(imediffs[np.isfinite(imediffs)])}')
        logging.debug(f'extrema(fluxdiffs): {fluxdiffmin,fluxdiffmax}')

        fluxdiffmax = np.abs([fluxdiffmin,fluxdiffmax]).min()
        fluxdiffmin = -fluxdiffmax

        rareadiffs = areadiffs / areas
        rareadiffs[0] = np.nan

        figrows,figcols,figscale=4,1,5
        figsize=(2*figcols*figscale,figrows*figscale*1.05)
        fig,ax = pl.subplots(figrows,figcols,figsize=figsize,
                             sharex=True,sharey=False)
        rmin,rmax = extrema(rmeandiffs,p=0.95)
        rmax = min(np.abs([rmin,rmax]))
        rmin = -rmax
        logging.debug(f'extrema(rmeandiffs): {rmin,rmax}')    
        ax[0].scatter(dists,rareadiffs,c=imediffs,cmap='RdYlBu_r')
        optidx1 = np.argmax(np.diff(np.sign(sums-psums)))
        optidx2 = np.argmin(np.abs(sums-psums))
        logging.debug(f'(ime,flux) @ dopt1={dists[optidx1]}: {imes[optidx1],fluxs[optidx1]}')
        logging.debug(f'(ime,flux) @ dopt2={dists[optidx2]}: {imes[optidx2],fluxs[optidx2]}')

        if 0:
            ax[0].set_title('IME diff')
            ax[1].scatter(dists,rareadiffs,c=imediffs,cmap='RdYlBu_r')
            ax[1].set_title('Flux diff')
            ax[1].scatter(dists,rareadiffs,c=fluxdiffs,cmap='RdYlBu_r')
        else:
            ax[0].set_title('IME cumulative')
            ax[1].scatter(dists,rareadiffs,c=imes,cmap='RdYlBu_r')
            ax[1].set_title('Flux cumulative')
            ax[1].scatter(dists,rareadiffs,c=fluxs,cmap='RdYlBu_r')
        ax[-1].set_xlabel('fetch (px)')
        ax[0].set_ylabel(r'$\Delta$area / area (px)')
        ax[1].set_ylabel(r'$\Delta$area / area (px)')
        ax[2].set_title('C$^-$ vs. C$^+$ diff')    
        ax[2].scatter(dists,sums,c='r',marker='_',label='C$^-$')
        ax[2].scatter(dists,psums,c='b',marker='+',label='C$^+$')
        ax[2].axvline(dists[np.argmin(np.abs(sums-psums))])
        ax[3].set_title(r'$\mu^-$ vs. $\mu^+$ diff')
        ax[2].legend()
        ax[3].scatter(dists,means,c='r',marker='_',label=r'$\mu^-$')
        ax[3].scatter(dists,pmeans,c='b',marker='+',label=r'$\mu^+$')
        ax[3].legend()
        ax[3].set_ylim(pmeans.min(),pmeans.max())
        ax[3].axvline(dists[np.argmin(np.abs(means-pmeans))])
        #for axi in ax:
        #    axi.axhline(1.0,c='k',lw=2)
        #pl.legend(loc='upper right')
        pl.tight_layout()
        pl.subplots_adjust(top=0.975,hspace=0.15,wspace=0.05,left=0.05)
        pl.savefig(swpfigf)
        pl.close()
    return 'success', returns

def make_plot(pltcmf, plumemask, i, j, maxfetchpx, manual_boundary_coordinates_ij, imin, imax, jmin, jmax, imefigf, ps):
    off = 5

    circle_min_x, circle_max_x = max(jmin-off, 0), min(jmax+off, pltcmf.shape[1])
    circle_min_y, circle_max_y = max(imin-off, 0), min(imax+off, pltcmf.shape[0])

    if len(manual_boundary_coordinates_ij) > 0:
        plume_x, plume_y = np.array(manual_boundary_coordinates_ij)[:,1], np.array(manual_boundary_coordinates_ij)[:,0]
        plume_min_x, plume_max_x = min([plume_x.min(), jmin-off]), max([plume_x.max(), jmax+off])
        plume_min_y, plume_max_y = min([plume_y.min(), imin-off]), max([plume_y.max(), imax+off])

    figrows,figcols,figscale=3,2,5

    figsize=(2*figcols*figscale,figrows*figscale*1.05)
    mosaic = """.AABB
                .CCDD
                .EEFF"""
    fig, axm = pl.subplot_mosaic(mosaic, layout='constrained', figsize=figsize)
    ax = [[axm['A'], axm['C'], axm['E']],
          [axm['B'], axm['D'], axm['F']]]

    def do_one_imshow(axis, imdata, vmin, vmax, cmap, extent, do_cb = True):
        im = axis.imshow(imdata[::-1,:],vmin=vmin,vmax=vmax,cmap=cmap, extent=extent) # reverse first dim
        if do_cb:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

    ma = np.max(pltcmf[circle_min_y:circle_max_y, 
                       circle_min_x:circle_max_x])

    nx, ny = pltcmf.shape
    # Use of imshow with both extent and set_xlim and set_ylim is confusing.
    # In order to plot with north up, we need to:
    # 1. reverse the min/max for the vertical dimension in the plot, which we call y here
    # 2. reverse the bottom and the top in the extent
    # 3. reverse the first dimension of the array in the imshow command
    plume_min_x_m, plume_max_x_m = (plume_min_x - j) * ps, (plume_max_x - j) * ps # reverse
    plume_min_y_m, plume_max_y_m = [(plume_min_y - i) * ps, (plume_max_y - i) * ps][::-1] # reverse

    circle_min_x_m, circle_max_x_m = (circle_min_x - j) * ps, (circle_max_x - j) * ps
    circle_min_y_m, circle_max_y_m = [(circle_min_y - i) * ps, (circle_max_y - i) * ps][::-1]
    
    # Change axes to meters centered on the pseudo-origin
    l, r, b, t = (0-j)*ps, (ny-1-j)*ps, (nx-1-i)*ps, (0-i)*ps
     
    extent = [l, r, t, b] # reverse top and bottom
    
    # Will be zoomed to show full plume
    do_one_imshow(ax[0][0], pltcmf, 0, 1500, 'inferno', extent, do_cb = False)
    do_one_imshow(ax[1][0], pltcmf, 0, ma,   'inferno', extent, do_cb = False)

    # Will be zoomed to show maxfetchm area around pseudo origin
    do_one_imshow(ax[0][1], pltcmf, 0, 1500, 'inferno', extent)
    do_one_imshow(ax[1][1], pltcmf, 0, ma,   'inferno', extent)

    # Will be zoomed to show maxfetchm area around pseudo origin and plume mask
    do_one_imshow(ax[0][2], pltcmf, 0, 1500, 'YlOrRd', extent, do_cb = False)
    do_one_imshow(ax[1][2], pltcmf, 0, ma,   'YlOrRd', extent, do_cb = False)
    plumemask[plumemask==1] = np.nan
    ax[0][2].imshow(plumemask[::-1,:],vmin=0,vmax=1,cmap='Blues_r', alpha = 0.8, extent = extent)
    ax[1][2].imshow(plumemask[::-1,:],vmin=0,vmax=1,cmap='Blues_r', alpha = 0.8, extent = extent)

    ax[0][0].plot(0, 0, 'o', color = 'lime', markersize = 10)
    ax[0][1].plot(0, 0, 'o', color = 'lime', markersize = 10)
    ax[1][0].plot(0, 0, 'o', color = 'lime', markersize = 10)
    ax[1][1].plot(0, 0, 'o', color = 'lime', markersize = 10)
    ax[0][2].plot(0, 0, 'o', color = 'red', markersize = 10)
    ax[1][2].plot(0, 0, 'o', color = 'red', markersize = 10)

    # The line between the two furthest points (that makes the fetch calculation)
    pt1, pt2, _ = find_max_distance_hull(plumemask)
    pt1_c = np.array([pt1[1] - j, pt1[0] - i])*ps
    pt2_c = np.array([pt2[1] - j, pt2[0] - i])*ps

    for axes in ax:
        for a in axes:
            circle = patches.Circle((0,0), maxfetchpx*ps, fill=False, edgecolor='white', linewidth=2)
            a.add_patch(circle)
            if len(manual_boundary_coordinates_ij) > 0:
                a.plot((plume_x-j)*ps, (plume_y-i)*ps, color = 'white')
            
            # Draw an arrow between the two furthest mask points
            a.annotate("", xy=pt1_c, xytext=pt2_c, arrowprops=dict(arrowstyle="<->", color='cyan'))

    if len(manual_boundary_coordinates_ij) > 0:
        ax[0][0].set_xlim([plume_min_x_m, plume_max_x_m])
        ax[0][0].set_ylim([plume_min_y_m, plume_max_y_m])
        ax[1][0].set_xlim([plume_min_x_m, plume_max_x_m])
        ax[1][0].set_ylim([plume_min_y_m, plume_max_y_m])
    else:
        ax[0][0].set_xlim([circle_min_x_m, circle_max_x_m])
        ax[0][0].set_ylim([circle_min_y_m, circle_max_y_m])
        ax[1][0].set_xlim([circle_min_x_m, circle_max_x_m])
        ax[1][0].set_ylim([circle_min_y_m, circle_max_y_m])

    ax[0][1].set_xlim([circle_min_x_m, circle_max_x_m])
    ax[0][1].set_ylim([circle_min_y_m, circle_max_y_m])
    ax[1][1].set_xlim([circle_min_x_m, circle_max_x_m])
    ax[1][1].set_ylim([circle_min_y_m, circle_max_y_m])

    ax[0][2].set_xlim([circle_min_x_m, circle_max_x_m])
    ax[0][2].set_ylim([circle_min_y_m, circle_max_y_m])
    ax[1][2].set_xlim([circle_min_x_m, circle_max_x_m])
    ax[1][2].set_ylim([circle_min_y_m, circle_max_y_m])

    
    pl.savefig(imefigf)
    pl.close()

def find_max_distance_hull(bool_array):
    # 1. Extract coordinates of all True values
    coords = np.argwhere(bool_array)
    
    if len(coords) < 2:
        return None, None, 0.0

    # 2. Find the Convex Hull vertices
    # This reduces N points to h boundary points (h << N)
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]

    # 3. Calculate pairwise distances only for hull vertices
    dist_matrix = distance_matrix(hull_points, hull_points)
    
    # 4. Find the maximum distance and its endpoint indices
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    p1, p2 = hull_points[i], hull_points[j]
    max_dist = dist_matrix[i, j]
    
    return p1, p2, max_dist

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(os.path.split(__file__)[1])

    # keyword arguments
    parser.add_argument('-v', '--verbose', action='store_true',help='Verbose output')
    parser.add_argument('--plot_diag', action='store_true',help='Plot diagnostic output')
    parser.add_argument('--minppmm', type=float, help='exclude values below minppmm')
    parser.add_argument('--maxppmm', type=float, help='exclude values above maxppmm') 
    parser.add_argument('--mergedistm', type=float, required=True, help='merge dist (m)')
    parser.add_argument('--maxfetchm', type=float, required=True, help='fetch max (m)')
    parser.add_argument('--minaream2', type=int, required=True, help='min area of connnected components (px)')
    parser.add_argument('--rgbimgf', type=str, help='rgbimgf')
    parser.add_argument('--csv_path', type=str, default='/scratch/colemanr/plume_emission_rate/out_csv/')
    parser.add_argument('--plot_path', type=str, default='/scratch/colemanr/plume_emission_rate/out_figs/')
    parser.add_argument('--csv_file', type=str, default='out.csv')
    parser.add_argument('--q_method', type=str, default='mask', help='Options: mask, ccirc') 
    parser.add_argument('--name_suffix', type=str, help='Suffix for plot names') 
    parser.add_argument('--sns_file', type=str, help='Sensitivity filename') 
    parser.add_argument('--unc_file', type=str, help='Uncertainty filename') 
    parser.add_argument('--manual_boundary_coordinates_lon_lat', type=float, nargs = '*', help='List of lon/lat pairs marking the manual plume boundary') 
    
    # positional sarguments 
    parser.add_argument('--cmfimgf', type=str, help='cmfimgf')
    parser.add_argument('--lng', type=float, help='plume longitude (use center by default)')
    parser.add_argument('--lat', type=float, help='plume latitude (use center by default)')
    parser.add_argument('--fid', type=str, help='EMIT FID')
    parser.add_argument('--labimgf', type=str, help='labimgf')
    parser.add_argument('--mask_mode', type=str, default='infer', choices=['infer','label'],
                        help='label=use label image | infer=construct plume mask')

    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--logfile', type=str, default=None)
    args = parser.parse_args()
    status, results = compute_flux(args)

    columns = ['plume_id', 'C_Q_mask_kg_hr_mpers', 'C_Q_ccirc_kg_hr_mpers', 'lon', 'lat', 'fetch', 
               'merge_dist', 'min_pppmm_thresh', 'max_ppmm_thresh', 'min_area_m',  'pixel_size', 'sum_C_sqrd']
    
    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=logging.DEBUG,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    if status == 'success':
        r_dict = {c:[x] for c, x in zip(columns, results)}
        r_dict['status'] = status
    else:
        r_dict = {'plume_id': [args.fid], 'status': [status]}

    ## Write csv for plumes that have an ROI detection
    csv_file_path = os.path.join(args.csv_path, args.csv_file)
    if not os.path.exists(csv_file_path): 
        with open(csv_file_path, 'w', newline='') as file: 
            writer = csv.writer(file)
            writer.writerow(['plume_id', 'status', 'C_Q_mask_kg_hr_mpers', 'C_Q_ccirc_kg_hr_mpers', 'lon', 'lat', 'fetch', 
                            'merge_dist', 'min_pppmm_thresh', 'max_ppmm_thresh', 
                            'min_area_m',  'pixel_size', 'sum_C_sqrd'])
    
    df = pd.read_csv(csv_file_path)
    df = pd.concat([df, pd.DataFrame(r_dict)], ignore_index = True, sort = False)
    #df = df.sort_values('plume_id', key = lambda x: x.str.split('-').str[-1].astype(int))
    df.to_csv(csv_file_path, index = False)

    #with open(csv_file_path, 'a', newline='') as file: 
    #    writer = csv.writer(file)
    #    logging.debug('Writing row')
    #    writer.writerow([fid, status, C_Q_MASK, C_Q_CC, lng, lat, fetchm, mergedistm, args.minppmm, args.maxppmm, args.minaream2, ps])
    # ## 