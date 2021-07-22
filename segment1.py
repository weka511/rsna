from argparse          import ArgumentParser
from matplotlib.pyplot import axes, close, cm, figure, get_cmap, savefig, show, subplots, suptitle, title
from numpy             import mean, std, ones_like
from pydicom           import dcmread
from mri3d             import Study
from os.path           import join

Threshold = {'FLAIR':0.7,
             'T1wCE':0.9,
             'T2w': 0.7}

def get_Z_score(pixel_array):
    mu    = mean(pixel_array)
    sigma = std(pixel_array)
    return (pixel_array - mu)/sigma

def threshold_pixels(pixel_array,threshold=0.7):
    return (pixel_array>threshold)*ones_like(pixel_array)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--study')
    parser.add_argument('--path',    default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--show',    default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    args=parser.parse_args()

    study   = Study(args.study,join(args.path,'train',args.study))
    fig,axs = subplots(nrows = 4, ncols = 4, figsize = (20,20))
    dcims   = {f'{series}':series[series.seqs[len(series)//2]] for series in study.get_series()}

    for i,series in enumerate(study.get_series()):
        axs[i,0].imshow(dcims[f'{series}'].pixel_array)
        axs[i,0].set_ylabel(f'{series}')
    axs[0,0].set_title('Raw')

    Z_normalized = {name:get_Z_score(dcim.pixel_array) for name,dcim in dcims.items() }
    for i,series in enumerate(study.get_series()):
        axs[i,1].imshow(Z_normalized[series.name],vmin=0,vmax=1,cmap=cm.gray)
    axs[0,1].set_title('Z-normalized')

    for i,series in enumerate(study.get_series()):
        if series.name in Threshold:
            axs[i,2].imshow(threshold_pixels(Z_normalized[series.name],threshold=Threshold[series.name]),vmin=0,vmax=1,cmap=cm.gray)
        else:
            axs[i][2].axes.xaxis.set_visible(False)
            axs[i][2].axes.yaxis.set_visible(False)
            axs[i][2].spines['top'].set_visible(False)
            axs[i][2].spines['right'].set_visible(False)
            axs[i][2].spines['bottom'].set_visible(False)
            axs[i][2].spines['left'].set_visible(False)

    if args.show:
        show()
