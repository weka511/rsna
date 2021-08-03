from argparse          import ArgumentParser
from matplotlib.pyplot import figure, legend, plot, show, title
from mri3d             import Labelled_MRI_Dataset
from numpy             import convolve, ones
from scipy.stats       import tmean

def get_mean_intensities(dcims):
    def get_mean_non_zero(pixels):
        try:
            return tmean(pixels,limits=(0,None),inclusive=(True,False))
        except ValueError:
            return 0

    return [get_mean_non_zero(dcim.pixel_array) for dcim in dcims]

 # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

def get_running_averages(means,h=8, mode='full'):
    return  convolve(means, ones(2*h+1)/(2*h+1), mode=mode)

if __name__=='__main__':
    parser = ArgumentParser('Visualize & segment in 3D')
    parser.add_argument('--path',       default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--figs',       default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',       default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--study',      default = '00098',                      help = 'Name of Studies to be processed' )
    parser.add_argument('--window',    default=8, type=int)
    args       = parser.parse_args()

    dataset    = Labelled_MRI_Dataset(args.path,'train')
    study      = dataset[args.study]
    label      = dataset.get_label(args.study)

    for series in study.get_series(types=['FLAIR']):
        print (study,series.description)
        dcim     = series.dcmread()
        means    = get_mean_intensities(dcim)
        averages = get_running_averages(means,h=args.window)
        print (len(means),len(averages))
        fig = figure(figsize=(20,20))
        plot(means[args.window:], label='means')
        plot(averages, label='averages')
        title(args.study)
        legend()

    if args.show:
        show()
