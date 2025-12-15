import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad
from cv2 import putText, FONT_HERSHEY_SIMPLEX, LINE_AA, getTextSize
import pandas as pd


def add_time(spaced, h_d):
    """Add h m s for RA and DEC coordinates from simbad"""
    toadd = [h_d, 'm', 's']
    spaced = spaced.split(' ')
    timed = ''
    for seg in range(len(spaced)):
        timed += spaced[seg] + toadd[seg]
    return timed


def crop_xy(crop):
    """Parse crop string and return x1, x2, y1, y2"""
    parts = crop.split(';')
    coords = {}
    for part in parts:
        key, val = part.strip().split('=')
        coords[key.strip()] = int(val.strip())
    return coords['x1'], coords['x2'], coords['y1'], coords['y2']


def level_adjust(img):
    """Simple level adjustment for display"""
    img = img - np.nanpercentile(img, 1)
    img = img / np.nanpercentile(img, 99)
    img = np.clip(img, 0, 1)
    return img


def annotate(img_file, fits_file, crop=None, fontScale=0.65, 
                             filter=None, cross=False):
    """
    Annotate astronomical images with SIMBAD catalog objects.
    
    Parameters
    ----------
    img_file : str
        Path to the image file (PNG). If None, will be generated from fits_file.
    fits_file : str
        Path to the FITS file with WCS header.
    crop : str, optional
        Crop specification, e.g., 'y1=54; y2=3176; x1=2067; x2=7156'
    fontScale : float
        Font scale for text annotations (default: 0.65)
    filter : str, optional
        Filter SIMBAD results by name pattern
    cross : bool
        If True, mark objects with crosshairs instead of text.
        If multiple objects found, opens dialog to select which one to plot.
    
    Returns
    -------
    result_table : pandas.DataFrame
        Table of objects found in the frame with their coordinates
    """
    
    # Read FITS header
    header = fits.open(fits_file)[1].header
    
    # Load or generate image
    if img_file is None:
        img = fits.open(fits_file)[1].data
        img = level_adjust(img)
        img_file = fits_file.replace('.fits', '.png')
        img[np.isnan(img)] = 0
        plt.imsave(img_file, img, origin='lower')
    
    img = plt.imread(img_file)
    if img.shape[2] == 4:
        img = img[..., :3]
    
    # Query SIMBAD
    wcs = WCS(header)
    print('querying SIMBAD')
    my_simbad = Simbad()
    my_simbad.TIMEOUT = 300  # seconds
    my_simbad.add_votable_fields('otype')
    result_table = my_simbad.query_region(
        SkyCoord(ra=header['CRVAL1'],
                 dec=header['CRVAL2'],
                 unit=(u.deg, u.deg), frame='fk5'),
        radius=0.1 * u.deg)
    result_table = result_table.to_pandas()
    
    if filter:
        result_table = result_table[result_table['MAIN_ID'].str.contains(filter)]
        result_table.reset_index(drop=True, inplace=True)
    
    print(f'got {len(result_table)} results, converting to pixels')
    if len(result_table) == 0:
        raise Exception('no results')
    
    # Set colors based on object type
    color = (100, 255, 255)
    result_table['color'] = [color] * len(result_table)
    result_table['OTYPE'] = result_table['OTYPE'].str.replace('Star', '*')
    icat = np.where(result_table['OTYPE'].str.contains('\*'))[0]
    for iicat in icat:
        result_table.at[iicat, 'color'] = (255, 255, 255)
    icat = np.where(result_table['OTYPE'].str.contains('ebula'))[0]
    for iicat in icat:
        result_table.at[iicat, 'color'] = (255, 100, 100)
    
    # Convert coordinates to pixels
    pix = np.zeros((len(result_table), 2))
    for ii in range(len(result_table)):
        ra = add_time(result_table['RA'][ii], 'h')
        dec = add_time(result_table['DEC'][ii], 'd')
        c = SkyCoord(ra=ra, dec=dec).to_pixel(wcs)
        pix[ii, :] = [c[0], c[1]]
    
    result_table['pix_x'] = pix[:, 0]
    result_table['pix_y'] = pix[:, 1]
    
    # Determine frame boundaries
    y1 = 0
    y2 = header['NAXIS2']
    x1 = 0
    x2 = header['NAXIS1']
    
    if crop is None:
        if img.shape[:2] != (y2, x2):
            raise Exception('image and hdu not the same size')
    else:
        x1, x2, y1, y2 = crop_xy(crop)
        if img.shape[:2] != (y2-y1, x2-x1):
            raise Exception('image is expected to fit crop size')
    
    inframe = (pix[:, 0] > x1) & (pix[:, 1] > y1) & (pix[:, 0] <= x2) & (pix[:, 1] <= y2)
    
    if np.sum(inframe) == 0:
        print(result_table)
        raise Exception('no results in frame')
    else:
        print(f'{np.sum(inframe)} results in frame')
    
    # Filter to objects in frame
    result_table_inframe = result_table[inframe].copy()
    result_table_inframe.reset_index(drop=True, inplace=True)
    
    # Handle multiple objects with dialog if cross=True
    if cross and len(result_table_inframe) > 1:
        print("\nMultiple objects found:")
        for idx, row in result_table_inframe.iterrows():
            print(f"{idx}: {row['MAIN_ID']} ({row['OTYPE']})")
        
        selected_indices = input("\nEnter indices to plot (comma-separated, or 'all'): ").strip()
        
        if selected_indices.lower() != 'all':
            try:
                selected_indices = [int(i.strip()) for i in selected_indices.split(',')]
                result_table_inframe = result_table_inframe.iloc[selected_indices]
                result_table_inframe.reset_index(drop=True, inplace=True)
            except:
                print("Invalid input, using all objects")
    
    # Prepare image for annotation
    thickness = 2
    if np.nanmax(img) <= 1:
        img = 255 * img
        img = img.astype('uint8')
    
    # Annotate image
    for idx in range(len(result_table_inframe)):
        row = result_table_inframe.iloc[idx]
        px = int(np.round(row['pix_x'] - x1))
        py = int(np.round(y2)) - int(np.round(row['pix_y']))
        
        if cross:
            # Draw crosshair
            crosshair_size = 20
            color_bgr = row['color']
            # Horizontal line
            img = plt.plot([px - crosshair_size, px + crosshair_size], [py, py], 
                          color=np.array(color_bgr)/255, linewidth=2)
            # Vertical line  
            img = plt.plot([px, px], [py - crosshair_size, py + crosshair_size],
                          color=np.array(color_bgr)/255, linewidth=2)
        else:
            # Draw text
            txt = row['MAIN_ID']
            midhight = int(np.floor(getTextSize(txt,
                                                fontFace=FONT_HERSHEY_SIMPLEX,
                                                fontScale=fontScale,
                                                thickness=thickness)[0][1]/2))
            org = (px, py + midhight)
            img = putText(img, txt, org,
                         FONT_HERSHEY_SIMPLEX, fontScale,
                         row['color'], thickness, LINE_AA)
    
    # Save annotated image (always save by default)
    output_file = img_file.replace('.png', '_ann.png')
    if cross:
        # If cross mode, need to save the figure
        plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
        plt.imshow(img, origin='lower')
        plt.axis('off')
        
        for idx in range(len(result_table_inframe)):
            row = result_table_inframe.iloc[idx]
            px = row['pix_x'] - x1
            py = row['pix_y'] - y1
            crosshair_size = 20
            color_rgb = np.array(row['color'])/255
            
            # Horizontal line
            plt.plot([px - crosshair_size, px + crosshair_size], [py, py], 
                    color=color_rgb, linewidth=2)
            # Vertical line  
            plt.plot([px, px], [py - crosshair_size, py + crosshair_size],
                    color=color_rgb, linewidth=2)
        
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.imsave(output_file, img)
    
    print(f"Saved annotated image to: {output_file}")
    
    # Save table to CSV (always save by default)
    csv_file = img_file.replace('.png', '_objects.csv')
    result_table_inframe.to_csv(csv_file, index=False)
    print(f"Saved object table to: {csv_file}")
    
    return result_table_inframe
