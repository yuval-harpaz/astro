import os
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
    
    img = plt.imread(img_file).copy()
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
    
    # Make a writable copy of the image
    img = np.array(img, copy=True)
    
    # Annotate image
    for idx in range(len(result_table_inframe)):
        row = result_table_inframe.iloc[idx]
        px = int(np.round(row['pix_x'] - x1))
        py = int(np.round(y2)) - int(np.round(row['pix_y']))
        
        if cross:
            # Draw crosshair directly in pixels - white color, 3 pixels wide, 20 pixels away from center
            crosshair_length = 20  # Length of each arm from gap to end
            if np.nanmax(img) <= 1:
                color_rgb = np.array([1.0, 1.0, 1.0])  # White for float images
            else:
                color_rgb = np.array([255, 255, 255])  # White for uint8 images
            thickness = 3  # 3 pixels wide
            gap = 20  # 20 pixels away from center
            
            # Horizontal lines (left and right of center)
            for t in range(-thickness//2, thickness//2 + 1):
                y_line = py + t
                if 0 <= y_line < img.shape[0]:
                    # Left side - from (px - gap - crosshair_length) to (px - gap)
                    x_start_left = max(0, px - gap - crosshair_length)
                    x_end_left = max(0, px - gap)
                    if x_end_left > x_start_left:
                        img[y_line, x_start_left:x_end_left] = color_rgb
                    
                    # Right side - from (px + gap) to (px + gap + crosshair_length)
                    x_start_right = min(img.shape[1], px + gap)
                    x_end_right = min(img.shape[1], px + gap + crosshair_length)
                    if x_end_right > x_start_right:
                        img[y_line, x_start_right:x_end_right] = color_rgb
            
            # Vertical lines (top and bottom of center)
            for t in range(-thickness//2, thickness//2 + 1):
                x_line = px + t
                if 0 <= x_line < img.shape[1]:
                    # Bottom side - from (py - gap - crosshair_length) to (py - gap)
                    y_start_bottom = max(0, py - gap - crosshair_length)
                    y_end_bottom = max(0, py - gap)
                    if y_end_bottom > y_start_bottom:
                        img[y_start_bottom:y_end_bottom, x_line] = color_rgb
                    
                    # Top side - from (py + gap) to (py + gap + crosshair_length)
                    y_start_top = min(img.shape[0], py + gap)
                    y_end_top = min(img.shape[0], py + gap + crosshair_length)
                    if y_end_top > y_start_top:
                        img[y_start_top:y_end_top, x_line] = color_rgb
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
    output_file = img_file[:img_file.index('.')] + '_ann.png'
    plt.imsave(output_file, img)
    
    print(f"Saved annotated image to: {output_file}")
    
    # Save table to CSV (always save by default)
    csv_file = img_file.replace('.png', '.csv')
    result_table_inframe.to_csv(csv_file, index=False)
    print(f"Saved object table to: {csv_file}")
    
    return result_table_inframe


if __name__ == "__main__":
    os.chdir('/media/innereye/KINGSTON/JWST/data/SN-2003GD')
    annotate('filt05.jpg', 'jw06049-o001_t001_miri_f560w_i2d.fits', filter='SN 2003', cross=True)

