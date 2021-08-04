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
from matplotlib.pyplot import figure, legend, plot, savefig, show, suptitle
from mri3d             import ImagePlane, Labelled_MRI_Dataset
from numpy             import argmax, convolve, count_nonzero, ones, zeros
from scipy.stats       import tmean
from skimage.color     import rgb2lab
from skimage.measure   import regionprops
from sklearn.cluster   import KMeans

# get_mean_intensities

def get_mean_intensities(dcims,is_left,L,include_zeros=True):
    def get_mean(pixels):
        try:
            return tmean(pixels[:,0:L] if is_left else pixels[:,L:-1],
                         limits    = (0,None),
                         inclusive = (include_zeros,False))
        except ValueError:
            return 0

    return [get_mean(dcim.pixel_array) for dcim in dcims]

# get_running_averages
#
# Calculate a running average
#
# https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

def get_running_averages(xs, h=8, mode='full'):
    return  convolve(xs, ones(2*h+1)/(2*h+1), mode=mode)

# verify_axial

def verify_axial(series):
    for dcim in series.dcmread():
        assert str(ImagePlane.get(dcim.ImageOrientationPatient))=='Axial'
        return

# get_centroid_biggest_slice
#
# Find centroid of the largest slice of the brain

def get_centroid_biggest_slice(series):
    def get_biggest_slice():
        best_seq   = None
        best_count = -1
        for i,dcim in enumerate(series.dcmread()):
            non_zero_count = count_nonzero(dcim.pixel_array)
            if non_zero_count>best_count:
                best_count = non_zero_count
                best_seq = i
        return series[series.seqs[best_seq]].pixel_array

    def get_centroid(image): # https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image
        labeled_foreground = (image > 0).astype(int)
        properties         = regionprops(labeled_foreground, image)
        return  [int(z) for z in properties[0].centroid]

    return get_centroid(get_biggest_slice())

# determine_hemisphere

def determine_hemisphere(series):
    L     = get_centroid_biggest_slice(series)[1]
    Area1 = 0
    Area2 = 0
    for dcim in series.dcmread():
        Area1 += dcim.pixel_array[:,0:L].sum()
        Area2 += dcim.pixel_array[:,L:-1].sum()
    return Area1>Area2,L

# get_pseudocolour
#
# It isn't clear exactly how Wu et al do this, but I have adapted code that I found at
# https://github.com/NikosMouzakitis/Brain-tumor-detection-using-Kmeans-and-histogram
def get_pseudocolour(pixel_array):
    M,N    = pixel_array.shape
    rgb    = zeros((M,N,3))
    maxval = pixel_array.max()
    for i in range(M):
        for j in range(N):
            if pixel_array[i,j]>0:
                rgb[i,j,:] = hsv_to_rgb(pixel_array[i,j]/maxval, 1, 1)

    return rgb

# https://github.com/NikosMouzakitis/Brain-tumor-detection-using-Kmeans-and-histogram/
def cluster(Lab,K = 10):
    M,N,C  = Lab.shape
    Z      = Lab[:,:,1:3].reshape(M*N,2)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(Z)
    return kmeans.labels_.reshape(M,N)



if __name__=='__main__':
    parser = ArgumentParser('Segment using kmeans and false colours')
    parser.add_argument('actions',      choices=['slice', 'kmeans'], nargs='+')
    parser.add_argument('--path',       default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--figs',       default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',       default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--study',      default = '00098',                      help = 'Name of Studies to be processed' )
    parser.add_argument('--window',     default=8,  type=int,                   help = 'Window will lead and trail by this amount')
    parser.add_argument('--nrows',      default=4,  type=int)
    parser.add_argument('--ncols',      default=4,  type=int)
    parser.add_argument('--slices',     default=[], type=int, nargs = '+')
    parser.add_argument('--K',          default=10,  type=int,                 help = 'Number of clusters for kMeans')
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
            savefig('{study}-slice')
            slices = [series.seqs[centre+i-nrows*ncols//2] for i in range(2,nrows*ncols+1)]
            for i in range(2,nrows*ncols+1):
                try:
                    ax2         = fig.add_subplot(nrows,ncols,i)
                    ax2.imshow(series[slices[i]].pixel_array,cmap='gray')
                    ax2.set_title(slices[i])
                except IndexError:
                    break

    if 'kmeans' in args.actions:
        print (f'Slices: {slices}')
        for seq in slices:
            for series in study.get_series(types=['FLAIR']):
                pixel_array = series[seq].pixel_array
                rgb         = get_pseudocolour(pixel_array)
                Lab         = rgb2lab(rgb)
                Labels      = cluster(Lab,K=args.K)

                fig = figure(figsize=(20,20))
                ax1 = fig.add_subplot(2,2,1)
                ax1.imshow(pixel_array,cmap='gray')
                ax2 = fig.add_subplot(2,2,2)
                ax2.imshow(rgb)
                ax3 = fig.add_subplot(2,2,3)
                ax3.imshow(Lab)
                suptitle(f'{args.study} {seq}')

                M,N = Labels.shape
                fig = figure(figsize=(20,20))
                desimg = zeros((M,N))

                m,n = 2,args.K+1

                for k in range(args.K):
                    Ls = []
                    blanks = zeros((M,N))
                    for i in range(M):
                        for j in range(N):
                            if Labels[i,j]==k:
                                blanks[i,j] = 1
                                Ls.append(Lab[i,j,0])
                                if k==args.K-1:
                                    desimg[i,j] = pixel_array[i,j] # FIXME
                    ax = fig.add_subplot(m,n,k+1)
                    ax.imshow(blanks,cmap='gray')
                    ax = fig.add_subplot(m,n,k+n+1)
                    ax.hist(Ls)

                ax = fig.add_subplot(m,n,args.K+1)
                ax.imshow(blanks,cmap='afmhot')
            savefig('{study}-{seq}')

            fig = figure(figsize=(20,20))
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(Labels, cmap='coolwarm', interpolation='nearest')
            ax2 = fig.add_subplot(2,2,2)
            n,bins,_ = ax2.hist(Lab[:,:,0].flatten(),bins=256)
            savefig('{study}-final')

    if args.show:
        show()
