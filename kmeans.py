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

from argparse          import ArgumentParser
from colorsys          import hsv_to_rgb
from matplotlib.cm     import get_cmap
from matplotlib.pyplot import figure, legend, plot, show, suptitle
from mri3d             import ImagePlane, Labelled_MRI_Dataset
from numpy             import argmax, convolve, count_nonzero, ones, zeros
from scipy.stats       import tmean
from skimage.measure   import regionprops

def get_mean_intensities(dcims,is_left,L,include_zeros=True):
    def get_mean(pixels):
        try:
            return tmean(pixels[:,0:L] if is_left else pixels[:,L:-1],
                         limits    = (0,None),
                         inclusive = (include_zeros,False))
        except ValueError:
            return 0

    return [get_mean(dcim.pixel_array) for dcim in dcims]

 # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

def get_running_averages(means,h=8, mode='full'):
    return  convolve(means, ones(2*h+1)/(2*h+1), mode=mode)

def verify_axial(series):
    for dcim in series.dcmread():
        assert str(ImagePlane.get(dcim.ImageOrientationPatient))=='Axial'
        return


def get_centroid(series):
    def get_biggest_slice():
        best_seq = None
        best_count = -1
        for i,dcim in enumerate(series.dcmread()):
            non_zero_count = count_nonzero(dcim.pixel_array)
            if non_zero_count>best_count:
                best_count = non_zero_count
                best_seq = i
        return series[series.seqs[best_seq]].pixel_array

    image = get_biggest_slice()
    # https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image
    labeled_foreground = (image > 0).astype(int)
    properties         = regionprops(labeled_foreground, image)
    return  [int(z) for z in properties[0].centroid]


def determine_hemisphere(series):
    L     = get_centroid(series)[1]
    Area1 = 0
    Area2 = 0
    for dcim in series.dcmread():
        Area1 += dcim.pixel_array[:,0:L].sum()
        Area2 += dcim.pixel_array[:,L:-1].sum()
    return Area1>Area2,L

def pseudocolor(pixel_array): #https://github.com/NikosMouzakitis/Brain-tumor-detection-using-Kmeans-and-histogram
    M,N    = pixel_array.shape
    RGB    = zeros((M,N,3))
    maxval = pixel_array.max()
    for i in range(M):
        for j in range(N):
            if pixel_array[i,j]>0:
                RGB[i,j,:] = hsv_to_rgb(pixel_array[i,j]/maxval, 1, 1)

    return RGB

if __name__=='__main__':
    parser = ArgumentParser('Visualize & segment in 3D')
    parser.add_argument('actions',      choices=['slice', 'kmeans'], nargs='+')
    parser.add_argument('--path',       default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--figs',       default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',       default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--study',      default = '00098',                      help = 'Name of Studies to be processed' )
    parser.add_argument('--window',     default=8,  type=int)
    parser.add_argument('--nrows',      default=4,  type=int)
    parser.add_argument('--ncols',      default=4,  type=int)
    parser.add_argument('--slices',     default=[], type=int, nargs = '+')
    args       = parser.parse_args()

    dataset    = Labelled_MRI_Dataset(args.path,'train')
    study      = dataset[args.study]
    label      = dataset.get_label(args.study)
    slices     = args.slices
    if 'slice' in args.actions:
        for series in study.get_series(types=['FLAIR']):
            verify_axial(series)
            is_left,L = determine_hemisphere(series)
            dcim      = series.dcmread()
            means     = get_mean_intensities(dcim,is_left,L)
            averages  = get_running_averages(means,h=args.window)
            index_max = argmax(averages)
            print (study,series.description, len(means),len(averages),index_max, is_left,L)
            centre    = args.window+index_max
            nrows    = args.nrows
            ncols = args.ncols
            fig       = figure(figsize=(20,20))
            suptitle(args.study)
            ax1 = fig.add_subplot(nrows,ncols,1)
            ax1.plot(means[args.window:], label='means',color='xkcd:blue')
            ax1.plot(averages, label='averages',color='xkcd:red')
            ax1.legend()

            slices = [series.seqs[centre+i-nrows*ncols//2] for i in range(2,nrows*ncols+1)]
            for i in range(2,nrows*ncols+1):
                try:
                    ax2         = fig.add_subplot(nrows,ncols,i)
                    pixel_array = series[slices[i]].pixel_array
                    # rgb         = pseudocolor(pixel_array)
                    ax2.imshow(pixel_array,cmap='gray')
                    ax2.set_title(slices[i])
                except IndexError:
                    break

    if 'kmeans' in args.actions:
        print (f'Slices: {slices}')
        for seq in slices:
            for series in study.get_series(types=['FLAIR']):
                fig = figure(figsize=(20,20))
                ax1 = fig.add_subplot(1,1,1)
                pixel_array = series[seq].pixel_array
                rgb         = pseudocolor(pixel_array)
                ax1.imshow(rgb)
                ax1.set_title(seq)
    if args.show:
        show()
