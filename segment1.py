# MIT License

# Copyright (c) 2021 Simon Crase -- simon@greenweaves.nz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Brain tumor segmentation based on deep learning and an attention mechanism using MRI multi-modalities brain images
# Ramin Ranjbarzadeh et al
# https://www.nature.com/articles/s41598-021-90428-8

from argparse          import ArgumentParser
from matplotlib.pyplot import close, cm, savefig, show, subplots, suptitle
from numpy             import mean, multiply, std, ones_like, matmul
from numpy.linalg      import norm
from pydicom           import dcmread
from mri3d             import declutter, Labelled_MRI_Dataset, MRI_Geometry, Study
from os.path           import join

# SimpleSegmenter
#
# THis is the segmented described in Section 6 of Brain tumor segmentation based on deep learning
# and an attention mechanism using MRI multi-modalities brain images--Ramin Ranjbarzadeh et al
# https://www.nature.com/articles/s41598-021-90428-8
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
        IJ_to_pos    = MRI_Geometry.create_matrix(dcims['FLAIR'])
        centre_pixel = MRI_Geometry.create_midpoint(dcims['FLAIR'])
        centre_pos   = matmul(IJ_to_pos,centre_pixel)
        for series in study.get_series():
            if series.name!='FLAIR':
                dcims[series.name],min_distance,min_distance2,_ = MRI_Geometry.get_closest(series,centre_pixel,centre_pos)
                if min_distance>0:
                    delta_i, delta_j = dcims['FLAIR'].PixelSpacing
                    print (f'Min distance={min_distance}, projected={min_distance2}, delta_i={delta_i}, delta_j={delta_j} (mm)')
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

# BoundingBox
#
# This class represents the smallest box that includes all the non-trivial data in an image
#
# Note: imshow transposed image, as explained in
#       https://stackoverflow.com/questions/18237169/flip-x-and-y-axes-for-matplotlib-imshow
#       This class follows the numpy array, and transposes the box to match the image

class BoundingBox:

    def __init__(self,Rows,Columns,pixel_array):
        self.i0,self.i1,self.j0,self.j1 = 0,0,0,0

        for i in range(Rows):
            if sum(pixel_array[i,:]>0):
                self.i0 = i
                break
        for i in range(Rows-1,0,-1):
            if sum(pixel_array[i,:]>0):
                self.i1 = i
                break
        for j in range(Columns):
            if sum(pixel_array[:,j]>0):
                self.j0 = j
                break
        for j in range(Columns-1,0,-1):
            if sum(pixel_array[:,j]>0):
                self.j1 = j
                break

    def draw(self,ax=None, color = 'red', linewidth = 3, linestyle='--'):
        ax.plot([self.j0,self.j0], [self.i0,self.i1], color=color, linewidth=linewidth, linestyle=linestyle)
        ax.plot([self.j1,self.j1], [self.i0,self.i1], color=color, linewidth=linewidth, linestyle=linestyle)
        ax.plot([self.j0,self.j1], [self.i0,self.i0], color=color, linewidth=linewidth, linestyle=linestyle)
        ax.plot([self.j0,self.j1], [self.i1,self.i1], color=color, linewidth=linewidth, linestyle=linestyle)

if __name__ == '__main__':
    parser = ArgumentParser('Brain tumor segmentation based on deep learning and an attention mechanism using MRI multi-modalities brain images')
    parser.add_argument('--study',   default = '00098')
    parser.add_argument('--path',    default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--show',    default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--figs',    default = './figs',                     help = 'Path to store plots')
    args      = parser.parse_args()
    dataset   = Labelled_MRI_Dataset(args.path)
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
                axs[i,0].imshow(dcims[f'{series}'].pixel_array, cmap = cm.gray)
                axs[i,0].set_ylabel(f'{series}')
                bounds = BoundingBox(dcims[f'{series}'].Rows,dcims[f'{series}'].Columns,dcims[f'{series}'].pixel_array)
                bounds.draw(ax=axs[i,0])

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

            suptitle(f'{study} MGMT={dataset.get_label(str(study))}')
            savefig(join(args.figs,f'{study}-{k}'))
            close (fig)
        except SimpleSegmenter.Z_score_failed:
            print (f'Could not calculate Z-score for {k}')
        except IndexError:
            break
        finally:
            if k%5==0: print (f'Processed {k}')

    if args.show:
        show()
