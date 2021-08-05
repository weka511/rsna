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
from math              import isqrt
from matplotlib.pyplot import close, figure, legend, plot, savefig, show, suptitle
from mri3d             import ImagePlane, MRI_Dataset
from numpy             import argmax, convolve, count_nonzero, mean, ones, std, zeros
from os.path           import join
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
#
# At this stage we are only processing axial series

def verify_axial(series):
    for dcim in series.dcmread():
        assert str(ImagePlane.get(dcim.ImageOrientationPatient))=='Axial'
        return

# get_centroid
#
# Get centroid of an image

def get_centroid(image): # https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image
    labeled_foreground = (image > 0).astype(int)
    properties         = regionprops(labeled_foreground, image)
    return  [int(z) for z in properties[0].centroid]

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

    return get_centroid(get_biggest_slice())

# determine_hemisphere

def determine_hemisphere(series):
    # L     = get_centroid_biggest_slice(series)[1]
    Area1 = 0
    Area2 = 0
    for dcim in series.dcmread():
        print (dcim.ImagePositionPatient)
        try:
            L      = get_centroid(dcim.pixel_array)[1]
            Area1 += dcim.pixel_array[:,0:L].sum()
            Area2 += dcim.pixel_array[:,L:-1].sum()
        except IndexError:
            continue
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

# cluster

# https://github.com/NikosMouzakitis/Brain-tumor-detection-using-Kmeans-and-histogram/
def cluster(Lab,K = 10):
    M,N,_  = Lab.shape
    Z      = Lab[:,:,1:3].reshape(M*N,2)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(Z)
    return kmeans.labels_.reshape(M,N)

# find_limit

def find_limit(index_max,cutoff,averages,direction=+1):
    index = index_max
    while index > 0 and index < len(averages) and averages[index]/averages[index_max]>cutoff:
        index += direction
    return index

# partition_figure
def partition_figure(total_cells):
    m = isqrt(total_cells)
    n = total_cells//m
    while m*n<total_cells:
        n+=1
    return m,n

# get_axes
#
# This allows us to iterate through clusters and produce either a set of detailed plots or a single summary plot

def get_axes(width=20,height=20,detailed=False,rows=2,columns=1,show=False):
    fig = figure(figsize=(width,height))
    m   = rows
    n   = 1 if detailed else columns
    for j in range(columns):
        if detailed and j>0:
            if not show:
                close(fig)
            fig = figure(figsize=(width,height))
        yield [fig.add_subplot(m,n,1+j%n+i*n) for i in range(m)]

if __name__=='__main__':
    parser = ArgumentParser('Segment using kmeans and false colours')
    parser.add_argument('actions',      choices=['slice', 'kmeans'], nargs='+')
    parser.add_argument('--path',       default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--figs',       default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',       default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--study',      default = '00098',                      help = 'Name of Studies to be processed' )
    parser.add_argument('--window',     default = 8,  type=int,                 help = 'Window will lead and trail by this amount')
    parser.add_argument('--slices',     default = [], type=int, nargs = '+')
    parser.add_argument('--K',          default = 10,  type=int,                help = 'Number of clusters for kMeans')
    parser.add_argument('--cutoff',     default = 0.95, type=float)
    parser.add_argument('--test',       default = False, action = 'store_true')
    parser.add_argument('--modality',   default = 'FLAIR')
    parser.add_argument('--summary',    default = False, action = 'store_true')
    args       = parser.parse_args()

    dataset    = MRI_Dataset(args.path,
                             'test' if args.test else 'train')
    study      = dataset[args.study]
    slices     = args.slices

    if 'slice' in args.actions:
        for series in study.get_series(types=[args.modality]):
            print (study,series.description)
            verify_axial(series)
            is_left,L = determine_hemisphere(series)
            dcim      = series.dcmread()
            means     = get_mean_intensities(dcim,is_left,L)
            m         = 2*args.window+1
            averages  = [0]*args.window + [mean(means[i:i+m+1]) for i in range(len(means)-m)]
            stds      = [0]*args.window + [std(means[i:i+m+1]) for i in range(len(means)-m)]
            index_max = argmax(averages)
            centre    = args.window+index_max
            fig       = figure(figsize=(20,20))
            suptitle(args.study)
            ax1 = fig.add_subplot(2,1,1)
            ax1.plot(means[args.window:], label='means',color='xkcd:blue')
            ax1.plot(averages, label='averages',color='xkcd:red')
            ax1.legend()
            ax2 = fig.add_subplot(2,1,2)
            ax2.plot(stds, label='running std',color='xkcd:red')
            savefig(join(args.figs,f'{study}-{args.modality}-average-intensity'))
            ax2.legend()
            if not args.show:
                close(fig)
            i0     = find_limit(index_max,args.cutoff,averages,direction=-1)
            i1     = find_limit(index_max,args.cutoff,averages,direction=+1)
            slices = [series.seqs[i] for i in range(i0,i1+1)]

            m,n    = partition_figure(len(slices))
            fig    = figure(figsize=(20,20))
            suptitle(f'{args.study} {"Left" if is_left else "Right"}')
            for i in range(len(slices)):
                try:
                    ax2 = fig.add_subplot(m,n,i+1)
                    ax2.imshow(series[slices[i]].pixel_array,cmap='gray')
                    ax2.set_title(slices[i])
                except IndexError:
                    break
            savefig(join(args.figs,f'{study}-{args.modality}-slices'))
            if not args.show:
                close(fig)

    if 'kmeans' in args.actions:
        print (f'Slices: {slices}')
        for seq in slices:
            for series in study.get_series(types=[args.modality]):
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
                savefig(join(args.figs,f'{study}-{args.modality}-image'))
                if not args.show:
                    close(fig)
                M,N = Labels.shape

                detailed = not args.summary
                for k,axes in enumerate(get_axes(detailed=detailed,columns=args.K,show=args.show)):
                    Luminosities = []
                    blanks = zeros((M,N))
                    for i in range(M):
                        for j in range(N):
                            if Labels[i,j]==k:
                                blanks[i,j] = 1
                                Luminosities.append(Lab[i,j,0])

                    axes[0].imshow(blanks,cmap='gray')
                    axes[0].set_title(f'{k}')
                    if detailed or k==0:
                        axes[0].set_ylabel('Clusters')
                    axes[1].hist(Luminosities,bins=50)
                    if detailed or k==0:
                        axes[1].set_ylabel('Luminosities')
                    if detailed or k==args.K-1:
                        savefig(join(args.figs,f'{study}-{args.modality}-{seq}-{k}'),dpi=250) #FIXME

            if not args.show:
                close(fig)

            fig = figure(figsize=(20,20))
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(Labels, cmap='coolwarm', interpolation='nearest')
            ax2 = fig.add_subplot(2,2,2)
            n,bins,_ = ax2.hist(Lab[:,:,0].flatten(),bins=256)
            ax2.set_xlabel('Luminosity')
            savefig(join(args.figs,f'{study}-{args.modality}-final'))
            if not args.show:
                close(fig)

    if args.show:
        show()
