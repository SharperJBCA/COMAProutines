import numpy as np
from matplotlib import pyplot
from astropy.io import fits
from astropy import wcs
import sys
from scipy import linalg as la
from scipy.ndimage.filters import gaussian_filter, median_filter
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord
from astropy import units as u

from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm
def removeplane(img, slce=0.4):
    """
    Remove a quadratic 2D plane from an image
    """
    xr, yr = np.arange(slce*img.shape[0],(1-slce)*img.shape[0],dtype=int),\
             np.arange(slce*img.shape[1],(1-slce)*img.shape[1],dtype=int)
    x, y = np.meshgrid(xr,yr)

    
    subimg = img[xr[0]:xr[-1]+1,yr[0]:yr[-1]+1]
    imgf = subimg[np.isfinite(subimg)].flatten()

    vecs = np.ones((5,imgf.size))
    vecs[0,:] = x[np.isfinite(subimg)].flatten()
    vecs[1,:] = y[np.isfinite(subimg)].flatten()
    vecs[2,:] = x[np.isfinite(subimg)].flatten()**2
    vecs[3,:] = y[np.isfinite(subimg)].flatten()**2

    C = vecs.dot(vecs.T)
    xv = la.inv(C).dot(vecs.dot(imgf[:,np.newaxis]))
    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    img -= (xv[0]*x    + xv[1]*y    + \
            xv[2]*x**2 + xv[3]*y**2 + \
            xv[4])
    return img

def read_image(filename):
    try:
        hdu = fits.open(filename)
    except OSError:
        print(filename)
        return 0, 0, None
    w = wcs.WCS(hdu[0].header)

    img = hdu[0].data
    wei = hdu[1].data
    hits = hdu[2].data

    img[img == 0] = np.nan
    wei[wei == 0] = np.nan

    #img[np.isnan(img)] = np.nanmedian(img)

    slce = 0.
    xr, yr = np.arange(slce*img.shape[0],(1-slce)*img.shape[0],dtype=int), np.arange(slce*img.shape[1],(1-slce)*img.shape[1],dtype=int)
    x, y = np.meshgrid(xr,yr)
    #img.shape[0]), np.arange(img.shape[1]))

    
    subimg = img[xr[0]:xr[-1]+1,yr[0]:yr[-1]+1]
    gd = np.isfinite(subimg)
    imgf = subimg[gd].flatten()

    if not np.any(gd):
        print(filename)
    pmdl = np.poly1d(np.polyfit(np.arange(imgf.size),imgf,3))
    imgf -= pmdl(np.arange(imgf.size))

    vecs = np.ones((5,imgf.size))
    vecs[0,:] = x[np.isfinite(subimg)].flatten()#*img.flatten()
    vecs[1,:] = y[np.isfinite(subimg)].flatten()#*img.flatten()
    vecs[2,:] = x[np.isfinite(subimg)].flatten()**2#*img.flatten()
    vecs[3,:] = y[np.isfinite(subimg)].flatten()**2#*img.flatten()
    # vecs[4,:] = x[np.isfinite(subimg)].flatten()**3#*img.flatten()
    # vecs[5,:] = y[np.isfinite(subimg)].flatten()**3#*img.flatten()

    C = vecs.dot(vecs.T)
    xv = la.inv(C).dot(vecs.dot(imgf[:,np.newaxis]))
    imgf -= (xv[0]*x[gd].flatten() + xv[1]*y[gd].flatten() +\
            xv[2]*x[gd].flatten()**2 + xv[3]*y[gd].flatten()**2 +\
            xv[4])

    subimg[gd] = imgf
    img[xr[0]:xr[-1]+1,yr[0]:yr[-1]+1] = subimg

    #pyplot.plot(imgf)
    #pyplot.show()

    x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

    #img -= (xv[0]*x + xv[1]*y +\
    #        xv[2]*x**2 + xv[3]*y**2 +\
    #        #xv[4]*x**3 + xv[5]*y**3 +\
    #        xv[4])
    
    #hduout = fits.PrimaryHDU([img], header=w.to_header())
    #hdu1 = fits.HDUList([hduout])
    #hdu1.writeto('test_'+sys.argv[1],overwrite=True)
    #img -= np.nanmedian(img,axis=0)[np.newaxis,:]
    #img -= np.nanmedian(img,axis=1)[:,np.newaxis]


    x0,y0 = w.all_world2pix(283.2,-1.9322,0)
    
    xpos, ypos = np.meshgrid(np.arange(0.25*img.shape[1],0.75*img.shape[1],dtype=int),
                             np.arange(0.25*img.shape[0],0.75*img.shape[0],dtype=int))
    r0 = 30./60. / w.wcs.cdelt[0]
    #circle = Circle((283.2,-1.9322),30/60., edgecolor='red', facecolor='none', linewidth=2,
    #                transform=pyplot.gca().get_transform('fk5'))
    rpos = ((xpos-x0)**2 + (ypos-y0)**2)**0.5
    select = np.where((rpos.flatten() < r0))[0]
    img_flat = img.flatten()
    map_rms = np.nanstd(img_flat[select])
    map_mean=np.nanmean(img_flat[select])


    img[np.isnan(img)] = 0
    wei[np.isnan(wei)] = 0
    return img,wei,hits,w
    

def main1():
    
    m31 = fits.open('fitsfiles/m31cm6i_full_3min_large.fits')
    hdr = m31[0].header
    hdr['NAXIS']=2
    m31_w=  wcs.WCS(naxis=2)#[0,:,:]#.slice((0,slice(0,None),slice(0,None)))
    m31_w.wcs.crpix = [hdr['CRPIX1'],hdr['CRPIX2']]
    m31_w.wcs.cdelt = [hdr['CDELT1'],hdr['CDELT2']]
    m31_w.wcs.crval = [hdr['CRVAL1'],hdr['CRVAL2']]
    m31_w.wcs.ctype = [hdr['CTYPE1'],hdr['CTYPE2']]
    m31_w.wcs.crota = [hdr['CROTA1'],hdr['CROTA2']]
    m31_w.wcs.equinox = hdr['EPOCH']

    for k,v in hdr.items():
        print(k,v)
    #stop

    m31_img = m31[0].data[0,:,:]

    #img_flat[select] = 1000
    #img = np.reshape(img_flat, img.shape)
    #img1, w = read_image('fitsfiles/fg4_feeds15.0-17.0-18.0_offset50.0_band1.0_freq0.0.fits')#sys.argv[1])
    #img2, w = read_image('fitsfiles/fg4_feeds15.0-17.0-18.0_offset50.0_band0.0_freq0.0.fits')#sys.argv[1])
    #img = img1-img2
    img, w = read_image(sys.argv[1])

    cmap = pyplot.get_cmap('RdBu_r')
    pyplot.figure(figsize=(12,8))
    ax = pyplot.subplot(projection=w)

    sources = [SkyCoord('00h38m24.84s','+41d37m06.00s',frame='icrs')]
               #SkyCoord('00h46m48.1s' ,'+41d41m07.00s',frame='icrs'),
               #SkyCoord('00h42m44.33s','+41d16m07.50s',frame='icrs')]

    mimg = img*1
    mimg[np.isnan(img)] = 0
    mimg = gaussian_filter(mimg,sigma=2)
    img[img == 0] =np.nan
    mimg[mimg==0]=np.nan
    zimg = ax.imshow(img,cmap=cmap,origin='lower',aspect='auto')
    cbar = pyplot.colorbar(zimg)
    cbar.set_label('K',size=20)

    for source in sources:
        ax.scatter(source.ra, source.dec, transform=ax.get_transform('icrs'), s=300, edgecolor='k', facecolor='none')


    #print(m31_w)
    #print(w)
    # ax.contour(m31_img, transform=ax.get_transform(m31_w),
    #           origin='lower',
    #           cmap=pyplot.get_cmap('Greys'),
    #           linewidths=3,
    #           alpha=0.85,
    #           levels=[-0.005,0.005,0.010,0.015])
    ##pyplot.contour(mimg, cmap = pyplot.get_cmap('Greys_r'),
    #               levels=[-0.02,-0.015,-0.01,-0.005,0,0.01,0.02,0.045,0.07,0.08,0.09,0.135,0.3,0.4])

    fname = sys.argv[1].split('/')[-1].split('.fit')[0]
    pyplot.gca().invert_xaxis()
    pyplot.grid()
    pyplot.xlabel(r'$\alpha$',size=20)
    pyplot.ylabel(r'$\delta$',size=20)
    pyplot.gca().set_xlim(0.9*img.shape[1],0.1*img.shape[1])
    pyplot.gca().set_ylim(0.1*img.shape[0],0.9*img.shape[0])
    #xpyplot.gca().add_patch(circle)
    pyplot.title(sys.argv[1],size=5)
    #pyplot.savefig('jackknife.png')#.format(fname))
    pyplot.savefig('nooverlay_{}.png'.format(fname))
    pyplot.show()

if __name__ == '__main__':

    import glob
    fileprefix = sys.argv[1]
    filelist = glob.glob('{}*.fits'.format(fileprefix))
    
    imgall = None
    for filename in tqdm(filelist):
        img,wei,hits, w = read_image(filename)
        if isinstance(imgall,type(None)):
            imgall = np.zeros(img.shape)
            weiall = np.zeros(img.shape)
            hitsall= np.zeros(img.shape)
        imgall += img*wei
        weiall += wei
        hitsall+= hits

    img = imgall/weiall
    # img2 = resize(downscale_local_mean(img,(4,4)),img.shape) #resize(img,(img.shape[0]//4,img.shape[1]//4))
    # pyplot.imshow(img2)
    # pyplot.show()
    # img[np.isnan(img)] = 0
    #med_img = median_filter(img,31)
    # pyplot.imshow(med_img)
    # pyplot.show()
    #img -= med_img
    img[img == 0] =np.nan
    #img  = removeplane(img, slce=0.)
    cmap = pyplot.get_cmap('RdBu_r')
    pyplot.figure(figsize=(12,8))
    ax = pyplot.subplot(projection=w)

    sources = [SkyCoord('00h38m24.84s','+41d37m06.00s',frame='icrs')]
               #SkyCoord('00h46m48.1s' ,'+41d41m07.00s',frame='icrs'),
               #SkyCoord('00h42m44.33s','+41d16m07.50s',frame='icrs')]

    mimg = img*1
    mimg[np.isnan(img)] = 0
    mimg = gaussian_filter(mimg,sigma=1)
    mimg[img==0]=np.nan
    img[img == 0] =np.nan

    from astropy.io import fits
    hdu = fits.PrimaryHDU(img,header=w.to_header())
    errs = fits.ImageHDU(np.sqrt(1/weiall), header=w.to_header())
    hits = fits.ImageHDU(hitsall, header=w.to_header())

    hdu1 = fits.HDUList([hdu, errs ,hits])
    hdu1.writeto('fitsfiles/gfield-allfeeds/CombinedMap.fits',overwrite=True)


    zimg = ax.imshow(img,cmap=cmap,origin='lower',aspect='auto',vmax=0.04,vmin=-0.04)
    cbar = pyplot.colorbar(zimg)
    cbar.set_label('K',size=20)
    pyplot.grid()
    pyplot.gca().invert_xaxis()
    pyplot.savefig('nooverlay_{}.png'.format(filelist[0].split('/')[-1].split('_')[0]))
    pyplot.show()
