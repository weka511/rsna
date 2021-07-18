from argparse          import ArgumentParser
from matplotlib.pyplot import axes, close, cm, figure, savefig, show, title
from pydicom           import dcmread
from os                import sep, listdir, walk
from os.path           import join, normpath
from pandas            import read_csv
from re                import match

# Study
#
# This class represents a study for one patient
#
class Study:
    # Series
    #
    # Each Study comprises several (in practic 4) Series
    class Series:
        def __init__(self,name):
            self.name             = name
            self.missing_images   = set()
            self.dirpath          = None
            self.image_plane      = None
            self.description      = None
            self.patient_position = None

        # add_images
        #
        # Add a collection of images to Series

        def add_images(self,dirpath,filenames):
            def extract_digits(s):
                m = match(r'\D*(\d+)\D+',s)
                if m:
                    return int(m.group(1))

            self.dirpath          = dirpath
            seqs                  = sorted([extract_digits(name) for name in filenames])
            self.N                = seqs[-1]
            self.missing_images   = set([i for i in range(1,self.N) if i not in seqs])
            dcim                  = dcmread(join(dirpath,filenames[0]))
            self.image_plane      = self.get_image_plane(dcim.ImageOrientationPatient)
            self.description      = dcim.SeriesDescription
            self.patient_position = dcim.PatientPosition

        # dcmread
        #
        # A generator to iterate through all the images.
        #
        # parameters:
        #     stop_before_pixels    Used if we just want to analyze metadata

        def dcmread(self, stop_before_pixels = False):
            for i in range(1,len(self)+1):
                if i not in self.missing_images:
                    yield  dcmread(join(self.dirpath,f'Image-{i}.dcm'), stop_before_pixels = stop_before_pixels)

        # get_image_plane
        #
        # Snarfed from https://www.kaggle.com/davidbroberts/determining-mr-image-planes
        def get_image_plane(self,loc):
            row_x = round(loc[0])
            row_y = round(loc[1])
            row_z = round(loc[2])
            col_x = round(loc[3])
            col_y = round(loc[4])
            col_z = round(loc[5])

            if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 0:  return "Coronal"

            if row_x == 0 and row_y == 1 and col_x == 0 and col_y == 0:  return "Sagittal"

            if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 1:  return "Axial"

            return "Unknown"

        def __len__(self):
            return self.N

        # image_files
        #
        # Generator for iterating through image files

        def image_files(self):
            for i in range(1,self.N+1):
                if i not in self.missing_images:
                    yield join(self.dirpath,f'Image-{i}.dcm')

    def __init__(self,name,path):
        self.series        = None
        self.name          = name
        for dirpath, dirnames, filenames in walk(path):
            if self.series == None:
                self.series = {series_name: Study.Series(series_name) for series_name in dirnames}
            else:
                path_components = normpath(dirpath).split(sep)
                series = self.series[path_components[-1]]
                series.add_images(dirpath,filenames)

    def get_series(self):
        for name in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
            yield self.series[name]

    def get_image_planes(self):
        def get_image_plane(series):
            path_name   = next(series.image_files())
            dcim        = dcmread(path_name,stop_before_pixels=True)
            return  series.get_image_plane(dcim.ImageOrientationPatient)
        return [get_image_plane(series)  for series in self.series.values()]

    def __str__(self):
        return self.name

# MRI_Dataset
#
# An MRI Dataset comprises sevral stidies, either test or training

class MRI_Dataset:
    def __init__(self,path,folder):
        self.studies = {name:Study(name,join(path,folder,name)) for name in listdir(join(path,folder))}

    def get_studies(self):
        for study in self.studies.values():
            yield study

# Labelled_MRI_Dataset
#
# A Labelled_MRI_Dataset is a MRI_Dataset accompanied by labels for training

class Labelled_MRI_Dataset(MRI_Dataset):

    def __init__(self,path,folder,labels='train_labels.csv'):
        super().__init__(path,folder)
        self.labels = read_csv(join(path,labels),dtype={'BraTS21ID':str})

# plot_orbit
#
# Show how the patient moves through the MRI instrument during one Study

def plot_orbit(study,path='./'):
    fig       = figure(figsize=(20,20))
    ax        = axes(projection='3d')

    for series in study.get_series():
        orbit = []
        sizes = []
        for dcim in series.dcmread():
            orbit.append(dcim.ImagePositionPatient)
            sizes.append(10 if dcim.pixel_array.sum()> 0 else 1)
        ax.scatter(*[[float(a) for a in p] for p in list(zip(*orbit))],
                   label = f'{dcim.SeriesDescription}: {dcim.PatientPosition} {series.get_image_plane(dcim.ImageOrientationPatient)}',
                   s     = sizes)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title(dcim.PatientID)
    ax.legend()
    savefig(join(path,dcim.PatientID))
    return fig

if __name__=='__main__':
    parser = ArgumentParser('Determine trajectories for all studies')
    parser.add_argument('--path',   default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--unique', default = 'unique.csv',                 help = 'File name for list of studies whose planes are identical')
    parser.add_argument('--figs',   default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',   default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    args = parser.parse_args()

    training = Labelled_MRI_Dataset(args.path,'train')
    with open(args.unique,'w') as out:
        for study in training.get_studies():
            image_planes = study.get_image_planes()
            if len(set(image_planes))==1:
                print (study, image_planes[0])
                out.write(f'{study}, {image_planes[0]}\n')
                fig = plot_orbit(study, path=args.figs)
                if not args.show:
                    close(fig)

    if args.show:
        show()
