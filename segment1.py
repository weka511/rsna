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


class MRI_Geometry:
    # https://nipy.org/nibabel/dicom/dicom_orientation.html
    @staticmethod
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

    @staticmethod
    def create_vector(i,j):
        return array([i,j,0,1]).transpose()

    @staticmethod
    def create_midpoint(dcim):
        return MRI_Geometry.create_vector(dcim.Rows/2, dcim.Columns/2)

    @staticmethod
    def get_closest(series,centre_pixel,centre_pos):
        min_distance = float('inf')
        closest_dcim = None
        for dcim in series.dcmread():
            M1 = MRI_Geometry.create_matrix(dcim)
            c  = matmul(M1,centre_pixel)
            dist = norm(centre_pos - c)
            if dist<min_distance:
                min_distance = dist
                closest_dcim = dcim
            elif dist>min_distance:
                return closest_dcim

class SimpleSegmenter:

    class Z_score_failed(Exception):
        pass

    def __init__(self,
                 Threshold = {
                     'FLAIR' : 0.7,
                     'T1wCE' : 0.9,
                     'T2w'   : 0.7}):
        self.Threshold = {name:threshold for name,threshold in Threshold.items()}

    def segment(self,dcims):
        M            = MRI_Geometry.create_matrix(dcims['FLAIR'])
        centre_pixel = MRI_Geometry.create_midpoint(dcims['FLAIR'])
        centre_pos   = matmul(M,centre_pixel)
        for series in study.get_series():
            if series.name!='FLAIR':
                dcims[series.name] = MRI_Geometry.get_closest(series,centre_pixel,centre_pos)
        Z_normalized = {name:self.get_Z_score(dcim.pixel_array) for name,dcim in dcims.items() }
        thresholded  = {name:self.threshold_pixels(z_normalized,
                                                   threshold=self.Threshold[name])          \
                        for name,z_normalized in Z_normalized.items() if name in self.Threshold}
        hadamard     = multiply(thresholded['FLAIR'],thresholded['T2w'])
        return Z_normalized, thresholded,hadamard


    def get_Z_score(self,pixel_array):
        mu    = mean(pixel_array)
        sigma = std(pixel_array)
        if sigma==0: raise SimpleSegmenter.Z_score_failed
        return (pixel_array - mu)/sigma

    def threshold_pixels(self,pixel_array,threshold=0.7):
        return (pixel_array>threshold*pixel_array.max())*ones_like(pixel_array)

def declutter(ax,spines=['top','right','bottom', 'left']):
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for spine in spines:
        ax.spines[spine].set_visible(False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--study',   default = '00098')
    parser.add_argument('--path',    default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--show',    default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    args      = parser.parse_args()
    study     = Study(args.study,join(args.path,'train',args.study))
    segmenter = SimpleSegmenter()
    for k in range(1,200):
        try:
            dcims = {f'{series}':series[series.seqs[k]] for series in study.get_series() if series.name=='FLAIR'}
            if dcims['FLAIR'].pixel_array.sum()==0: continue
            Z_normalized, thresholded,hadamard = segmenter.segment(dcims)

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
            savefig(f'{study}-{k}')
            close (fig)
        except SimpleSegmenter.Z_score_failed:
            print (f'Could not process {k}')
        except IndexError:
            break
        finally:
            if k%5==0: print (f'Processed {k}')

    if args.show:
        show()
