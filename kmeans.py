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
#
# Segment brain tumours using kmeans and false colours
#
# References:
#            Wu, M.-N.; Lin, C.-C. & Chang, C.-C.
#            Brain Tumor Detection Using Color-Based K-Means Clustering Segmentation
#            IEEE Comput Soc, 2007, 2, 245-250
#
#            Eltayeb, E.; Salem, N. & Al-Atabany, W.
#            Automated brain tumor segmentation from multi-slices FLAIR MRI images
#            Bio-Medical Materials and Engineering, 2019, 30, 1-13
#
#            Mouzakitis, N.
#            Brain tumor detection using Kmeans and histogram
#            2021
#            https://github.com/NikosMouzakitis/Brain-tumor-detection-using-Kmeans-and-histogram/

from argparse          import ArgumentParser
from colorsys          import hsv_to_rgb
from math              import isqrt
from matplotlib.pyplot import close, figure, legend, plot, savefig, show, suptitle
from mri3d             import ImagePlane, MRI_Dataset, MRI_Geometry
from numpy             import argmax, asarray, convolve, count_nonzero, full, maximum, mean,  ones, std, where, zeros
from os.path           import join
from scipy.stats       import tmean
from skimage.color     import rgb2lab
from sklearn.cluster   import KMeans
from sklearn.metrics   import calinski_harabasz_score
from warnings          import warn

# get_mean_intensities
#
# Used to compute mean of intensities in specified hemisphere
# Eltayeb et al, figure 3

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
# Eltayeb et al, figure 3
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


# determine_hemisphere
#
# Following Eltayeb et al, determine which hemisphere has largest srea under curve
def determine_hemisphere(series):
    Area1 = 0
    Area2 = 0
    for dcim in series.dcmread():
        try:
            L      = MRI_Geometry.get_centroid(dcim)[1]
            Area1 += dcim.pixel_array[:,0:L].sum()
            Area2 += dcim.pixel_array[:,L:-1].sum()
        except IndexError:
            continue
    return Area1>Area2,L

# get_pseudocolour
#
# Wu et al speak of the "standard RGB colour map", but it isn't clear to me whar this is.
# I have adapted code that I found at
# https://github.com/NikosMouzakitis/Brain-tumor-detection-using-Kmeans-and-histogram
#
# I have inspected the code for hsv_to_rsv and observed that parameter 'v' is
# is always included in one position in every return statement.
#        def hsv_to_rgb(h, s, v):
#           ...
#                return v, v, v
#           ...
#                return v, t, p
#            ...
#                return q, v, p
#            ...
#                return p, v, t
#           ...
#                return p, q, v
#            ...
#                return t, p, v
#            ...
#               return v, p, q
# Conclusion: since v=1 in get_pseudocolour(...), hsv_to_rgb never returns (0,0,0),
#             whence get_pseudocolour(...) never returns (0,0,0) unless pixel_array is zero

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
#
# Perform clustering on a,b from L*a*b
#
# See https://github.com/NikosMouzakitis/Brain-tumor-detection-using-Kmeans-and-histogram/
def cluster(Lab,K = 10):
    M,N,_  = Lab.shape
    Z      = Lab[:,:,1:3].reshape(M*N,2)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(Z)
    print (f'Calinski Harabasz={calinski_harabasz_score(Z, kmeans.labels_):.0f}')
    return kmeans.labels_.reshape(M,N)

# find_limit
#
# Given the index of the peak, search back or forward to determine the limits
# of the slices where tumpor is likely to be

def find_limit(index_max,averages,cutoff=0.95,direction=+1):
    index = index_max
    try:
        while averages[index]/averages[index_max]>cutoff:
            index += direction
    except IndexError:
        warn(f'Ran off end: index={index}. Stepping back.')
        index -= direction
    return index

# detect_slice_range
#
# Search through the average maeans of Intensity (Eltayeb et al) to establish the index of the peak value

def detect_slice_range(series,half_width=8):
    is_left,L = determine_hemisphere(series)
    dcim      = series.dcmread()
    means     = get_mean_intensities(dcim,is_left,L)
    m         = 2*half_width+1
    averages  = [0]*half_width + [mean(means[i:i+m+1]) for i in range(len(means)-m)]
    stds      = [0]*half_width + [std(means[i:i+m+1]) for i in range(len(means)-m)]
    return is_left, means, averages,stds,argmax(averages)

# segment
#
# Cluster on a,b from L*a*b values, the split each cluster further by reclustering on L

def segment(dcim,K=10,K2=2):
    def luminosity_cluster(k):
        Mask         = where(Labels==k,Labels,zeros((M,N)))
        Luminosities = where(Mask==k,Lab[:,:,0],full((M,N),-1))
        kmeans       = KMeans(n_clusters=K2, random_state=0).fit(Luminosities.reshape(M*N,1))
        return maximum(kmeans.labels_.reshape(M,N),zeros((M,N)))

    rgb    = get_pseudocolour(dcim.pixel_array)
    Lab    = rgb2lab(rgb)
    Labels = cluster(Lab,K=K)
    M,N    = Labels.shape

    return rgb, Lab, Labels, [luminosity_cluster(k) for k in range(K)]

# partition_figure
#
# Used to organize a figure into subplots, arranged in a rectangle that is as close to a square as possible
# Rectangle will grow in horizontal direction if necessary

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
            fig = figure(figsize=(width,height))
        yield [fig.add_subplot(m,n,1+j%n+i*n) for i in range(m)]
        if not show:
            close(fig)

if __name__=='__main__':
    parser = ArgumentParser('Segment brain tumours using kmeans and false colours')
    parser.add_argument('actions',      choices=['slice', 'kmeans'], nargs='+', help = 'List of actions: identify slices; segment using kmeans')
    parser.add_argument('--path',       default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--figs',       default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',       default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--study',      default = '00098',                      help = 'Name of Studies to be processed' )
    parser.add_argument('--window',     default = 8,     type=int,              help = 'Window will lead and trail by this amount')
    parser.add_argument('--slices',     default = [],    type=int, nargs = '+', help = 'Slices to be processed if "kmeans" specified without "slice"')
    parser.add_argument('--K',          default = 10,    type=int,              help = 'Number of clusters for *a*b')
    parser.add_argument('--K2',         default = 3,     type=int,              help = 'Number of clusters for L')
    parser.add_argument('--cutoff',     default = 0.95,  type=float,            help = 'Determines which slices should be processed if "slice"')
    parser.add_argument('--test',       default = False, action = 'store_true', help = 'Use test or tarn data')
    parser.add_argument('--modality',   default = 'FLAIR',                      help = 'Modality to be used')
    parser.add_argument('--summary',    default = False, action = 'store_true', help = 'One summary plot for all clusters?')
    parser.add_argument('--cmap',       default = 'viridis',                    help = 'To use for greyscale images')
    args       = parser.parse_args()

    dataset    = MRI_Dataset(args.path,
                             'test' if args.test else 'train')
    study      = dataset[args.study]
    slices     = args.slices

    if 'slice' in args.actions:
        for series in study.get_series(types=[args.modality]):
            print (study,series.description)
            verify_axial(series)
            is_left, means, averages,stds,index_max = detect_slice_range(series,half_width=args.window)
            index_first_slice                       = find_limit(index_max,averages,cutoff=args.cutoff,direction=-1)
            index_last_slice                        = find_limit(index_max,averages,cutoff=args.cutoff,direction=+1)
            slices                                  = [series.seqs[i] for i in range(index_first_slice,index_last_slice+1)]

            fig = figure(figsize=(20,20))
            ax = fig.add_subplot(1,1,1)
            ax.plot(means[args.window:], label='means',color='xkcd:blue')
            ax.plot(averages, label='averages',color='xkcd:red')
            ax.plot(stds, label='std',color='xkcd:blue',linestyle='--')
            ax.plot(list(range(index_first_slice,index_last_slice)),
                    [averages[index_max]*args.cutoff for i in range(index_first_slice,index_last_slice)],
                    label     = 'slices of interest',
                    color     = 'xkcd:red',
                    linestyle = ':')
            ax.legend()
            suptitle(f'Study: {args.study}, peak at {index_max}--slice {series.seqs[index_max]}')
            savefig(join(args.figs,f'{study}-{args.modality}-average-intensity'))

            if not args.show:
                close(fig)

            m,n    = partition_figure(len(slices))
            fig    = figure(figsize=(20,20))

            for i in range(len(slices)):
                try:
                    ax = fig.add_subplot(m,n,i+1)
                    ax.imshow(series[slices[i]].pixel_array,cmap=args.cmap)
                    ax.set_title(slices[i])
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                except IndexError:
                    break

            suptitle(f'Study: {args.study} slices of interest for tumour in {"Left" if is_left else "Right"} hemisphere')
            savefig(join(args.figs,f'{study}-{args.modality}-slices'))
            if not args.show:
                close(fig)

    if 'kmeans' in args.actions:
        print (f'Slices: {slices}')
        for seq in slices:
            for series in study.get_series(types=[args.modality]):
                rgb, Lab, Labels,LuminosityClusters = segment(series[seq],K=args.K,K2=args.K2)

                fig = figure(figsize=(20,20))
                ax1 = fig.add_subplot(2,2,1)
                ax1.imshow(series[seq].pixel_array,cmap='gray')
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
                for k,axes in enumerate(get_axes(detailed=detailed,columns=args.K,show=args.show,rows=2+args.K2)):
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

                    Luminosity = LuminosityClusters[k]
                    for kk in range(args.K2):
                        Mask = where(Labels==kk,Labels,zeros((M,N)))
                        img  = where(Mask==kk,Luminosity[:,:],zeros((M,N)))
                        axes[2+kk].imshow(img,vmin=0,vmax=img.max())
                        axes[2+kk].set_ylabel(f'{img.min()} {img.max()}')
                    if detailed or k==args.K-1:
                        savefig(join(args.figs,f'{study}-{args.modality}-{seq}-{k}'))

            fig      = figure(figsize=(20,20))
            ax1      = fig.add_subplot(2,2,1)
            ax1.imshow(Labels, cmap='coolwarm', interpolation='nearest')
            ax2      = fig.add_subplot(2,2,2)
            n,bins,_ = ax2.hist(Lab[:,:,0].flatten(),bins=256)
            ax2.set_xlabel('Luminosity')
            savefig(join(args.figs,f'{study}-{args.modality}-final'))
            if not args.show:
                close(fig)

    if args.show:
        show()
