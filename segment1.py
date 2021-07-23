# Brain tumor segmentation based on deep learning and an attention mechanism using MRI multi-modalities brain images
# Ramin Ranjbarzadeh et al
# https://www.nature.com/articles/s41598-021-90428-8

from argparse          import ArgumentParser
from matplotlib.pyplot import close, cm, savefig, show, subplots, suptitle
from numpy             import array, mean, multiply, std, ones_like, matmul
from numpy.linalg      import norm
from pydicom           import dcmread
from mri3d             import Study
from os.path           import join

Threshold = {'FLAIR' : 0.7,
             'T1wCE' : 0.9,
             'T2w'   : 0.7}

def get_Z_score(pixel_array):
    mu    = mean(pixel_array)
    sigma = std(pixel_array)
    return (pixel_array - mu)/sigma

def threshold_pixels(pixel_array,threshold=0.7):
    return (pixel_array>threshold*pixel_array.max())*ones_like(pixel_array)

def declutter(ax,spines=['top','right','bottom', 'left']):
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for spine in spines:
        ax.spines[spine].set_visible(False)

# https://nipy.org/nibabel/dicom/dicom_orientation.html

def create_matrix(dcim):
    delta_i, delta_j       = dcim.PixelSpacing
    Sx, Sy, Sz             = dcim.ImagePositionPatient
    Xx, Xy, Xz, Yx, Yy, Yz = dcim.ImageOrientationPatient
    return array([
        [Xx*delta_i, Yx*delta_j, 0, Sx],
        [Xy*delta_i, Yy*delta_j, 0, Sy],
        [Xz*delta_i, Yz*delta_j, 0, Sz],
        [0,          0,          0, 1]
    ])


def create_vector(i,j):
    return array([i,j,0,1]).transpose()

def create_midpoint(dcim):
    return create_vector(dcim.Rows/2, dcim.Columns/2)

def get_closest(series,centre_pixel,centre_pos):
    min_distance = float('inf')
    closest_dcim = None
    for dcim in series.dcmread():
        M1 = create_matrix(dcim)
        c  = matmul(M1,centre_pixel)
        dist = norm(centre_pos - c)
        if dist<min_distance:
            min_distance = dist
            closest_dcim = dcim
    return closest_dcim

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--study')
    parser.add_argument('--path',    default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--show',    default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    args=parser.parse_args()

    study        = Study(args.study,join(args.path,'train',args.study))
    dcims        = {f'{series}':series[series.seqs[len(series.seqs)//2]] for series in study.get_series()}
    M            = create_matrix(dcims['FLAIR'])
    centre_pixel = create_midpoint(dcims['FLAIR'])
    centre_pos   = matmul(M,centre_pixel)
    for series in study.get_series():
        if series.name=='FLAIR': continue
        dcims[series.name] = get_closest(series,centre_pixel,centre_pos)
    Z_normalized = {name:get_Z_score(dcim.pixel_array) for name,dcim in dcims.items() }
    thresholded  = {name:threshold_pixels(z_norm,threshold=Threshold[name]) for name,z_norm in Z_normalized.items() if name in Threshold}
    hadamard      = multiply(thresholded['FLAIR'],thresholded['T2w'])
    fig,axs      = subplots(nrows = 4, ncols = 4, figsize = (20,20))

    axs[0,0].set_title('Raw')
    for i,series in enumerate(study.get_series()):
        axs[i,0].imshow(dcims[f'{series}'].pixel_array,
                        cmap = cm.gray)
        axs[i,0].set_ylabel(f'{series}')


    axs[0,1].set_title('Z-normalized')
    for i,series in enumerate(study.get_series()):
        axs[i,1].imshow(Z_normalized[series.name],
                        vmin = Z_normalized[series.name].min(),
                        vmax = Z_normalized[series.name].max(),
                        cmap = cm.gray)


    axs[0,2].set_title('Thresholded')
    for i,series in enumerate(study.get_series()):
        if series.name in thresholded:
            axs[i,2].imshow(thresholded[series.name],
                            vmin = 0,
                            vmax = thresholded[series.name].max(),
                            cmap = cm.gray)
        else:
            declutter(axs[i][2])

    axs[0,3].set_title('TO DO')
    for i,series in enumerate(study.get_series()):
        if series.name=='FLAIR':
            axs[0,3].imshow(hadamard, vmin=0,vmax=hadamard.max(),cmap=cm.gray)
        else:
            declutter(axs[i][3])

    suptitle(f'{study}')
    if args.show:
        show()
