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
from matplotlib.pyplot import axes, close, cm, figure, get_cmap, savefig, show, subplots, suptitle, title
from numpy             import array, matmul, sign
from numpy.linalg      import inv, norm
from operator          import itemgetter
from pydicom           import dcmread
from os                import sep, listdir, walk
from os.path           import join, normpath
from pandas            import read_csv
from re                import match

# http://www.aboutcancer.com/mri_gbm.htm

# ImagePlane

class ImagePlane:
    names = ['Sagittal','Axial','Coronal' ]

    # get_image_plane
    #
    # Adapted from David Roberts -- https://www.kaggle.com/davidbroberts/determining-mr-image-planes
    @staticmethod
    def get(loc):
        orientation = [round(ll) for ll in loc]

        if orientation[0] == 1 and orientation[1] == 0 and orientation[3] == 0 and orientation[4] == 0:  return "Coronal"

        if orientation[0] == 0 and orientation[1] == 1 and orientation[3] == 0 and orientation[4] == 0:  return "Sagittal"

        if orientation[0] == 1 and orientation[1] == 0 and orientation[3] == 0 and orientation[4] == 1:  return "Axial"

        return "Unknown"


# Study
#
# This class represents a study for one patient
#
class Study:

    # Series
    #
    # Each Study comprises several (in practic 4) Series
    class Series:

        Types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        def __init__(self,name):
            self.name             = name
            self.missing_images   = set()
            self.dirpath          = None
            self.image_plane      = None
            self.description      = None
            self.patient_position = None
            self.slices           = []

        def __str__(self):
            return self.description

        # add_images
        #
        # Add a collection of images to Series

        def add_images(self,dirpath,filenames):
            def extract_digits(s):
                m = match(r'\D*(\d+)\D+',s)
                if m:
                    return int(m.group(1))

            self.dirpath          = dirpath
            self.seqs             = sorted([extract_digits(name) for name in filenames])
            self.N                = self.seqs[-1]
            self.missing_images   = set([i for i in range(1,self.N) if i not in self.seqs])
            dcim                  = dcmread(join(dirpath,filenames[0]))
            self.image_plane      = ImagePlane.get(dcim.ImageOrientationPatient)
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

        def __getitem__(self,i):
            return  dcmread(join(self.dirpath,f'Image-{i}.dcm'))


        def __len__(self):
            return self.N

        # slice_files
        #
        # Generator for iterating through image files

        def slice_files(self):
            for i in range(1,self.N+1):
                if i not in self.missing_images:
                    yield join(self.dirpath,f'Image-{i}.dcm')

        # get_values_from_meta
        #
        # Convert meta data (list of tuples of ASCII data) to a list of data
        # sequences, each being the time series for one datum
        @staticmethod
        def get_values_from_meta(orbit):
            return [[float(a) for a in p] for p in list(zip(*orbit))]


        def get_orbit(self):
            return [[float(a) for a in dcim.ImagePositionPatient] for dcim in self.dcmread(stop_before_pixels=True)]

        # get_orbit_for_plot
        #
        # Get trajectory of patient

        def get_orbit_for_plot(self):
            orbit     = []
            trivial   = []
            brightest = 1
            for dcim in self.dcmread():
                orbit.append(dcim.ImagePositionPatient)
                trivial.append(dcim.pixel_array.sum()==0)
                brightest = max(brightest,dcim.pixel_array.max())
            return self.get_values_from_meta(orbit),trivial,dcim.SeriesDescription, dcim.PatientPosition, dcim.ImageOrientationPatient,brightest

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
        for name in Study.Series.Types:
            yield self.series[name]

    @staticmethod
    def get_image_plane(series):
        path_name   = next(series.slice_files())
        dcim        = dcmread(path_name,stop_before_pixels=True)
        return  ImagePlane.get(dcim.ImageOrientationPatient)

    def get_image_planes(self):
        return [self.get_image_plane(series)  for series in self.series.values()]

    def __str__(self):
        return self.name

    def __getitem__(self, key):
        return self.series[key]

# MRI_Dataset
#
# An MRI Dataset comprises sevral stidies, either test or training

class MRI_Dataset:
    def __init__(self,path,folder):
        self.studies = {name:Study(name,join(path,folder,name)) for name in listdir(join(path,folder))}

    # __getitem__
    #
    # Get value of label

    def __getitem__(self, key):
        return self.studies[key]

    # get_studies
    #
    # A generator to iterate through studies
    #
    # Parameters:
    #     study_names     Specified list of studies (or empty for all studies in dataset)

    def get_studies(self, study_names = []):
        if len(study_names)==0:
            for study in self.studies.values():
                yield study
        else:
            for study_name in study_names:
                yield self.studies[study_name]

# Labelled_MRI_Dataset
#
# A Labelled_MRI_Dataset is a MRI_Dataset accompanied by labels for training

class Labelled_MRI_Dataset(MRI_Dataset):

    def __init__(self,path,folder,labels='train_labels.csv'):
        super().__init__(path,folder)
        self.labels = read_csv(join(path,labels),dtype={'BraTS21ID':str})

    # get_counts
    #
    # Get number of negative and positive labels

    def get_counts(self):
        ones = sum(max(0,label) for label in self.labels['MGMT_value'])
        return len(self.labels) - ones, ones

    # __getitem__
    #
    # Get value of label

    def get_label(self, key):
        return int(self.labels[self.labels.BraTS21ID==key].MGMT_value)

# plot_orbit
#
# Show how the patient moves through the MRI instrument during one Study

def plot_orbit(study,
               path   = './',
               weight = 10):
    fig       = figure(figsize=(20,20))
    ax        = axes(projection='3d')

    for series in study.get_series():
        orbit, trivial, SeriesDescription, PatientPosition, ImageOrientationPatient,_ = series.get_orbit_for_plot()
        ax.scatter(*orbit,
                   label = f'{SeriesDescription}: {PatientPosition} {ImagePlane.get(ImageOrientationPatient)}',
                   s     = [1 if t else weight for t in trivial])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title(f'{study}')
    ax.legend()
    savefig(join(path,f'{study}'))
    return fig

# hide_decorations
#
# Remove boxes and other decorations from plot

def declutter(ax,spines=['top','right','bottom', 'left']):
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for spine in spines:
        ax.spines[spine].set_visible(False)


# create_cells
#
# A generator for iterating through cells, left to right, then start next row

def create_cells(ncols,axs):
    i = 0
    j = 0
    while True:
        yield axs[i,j]
        j +=1
        if j==ncols:
            j = 0
            i+= 1

# plot_series
#
# Plot an entire series of images

def plot_series(series,
               path   = './',
               study  = '',
               ncols  = 6,
               cmap   = get_cmap('coolwarm'),
               width  = 20,
               height = 20):

    _, trivial, Description, _, ImageOrientation,brightest = series.get_orbit_for_plot()
    non_trivial_slices                                     = sum([0 if t else 1 for t in trivial])
    nrows                                                  = max(2,non_trivial_slices // ncols)
    while nrows*ncols < non_trivial_slices: nrows +=1

    fig,axs   = subplots(nrows = nrows, ncols = ncols, figsize = (width,height))
    cell      = create_cells(ncols,axs)

    for k,dcim in enumerate(series.dcmread()):
        if not trivial[k]:
            next(cell).imshow(dcim.pixel_array/brightest,cmap = cmap)

    for i in range(nrows):
        for j in range(ncols):
            declutter(axs[i][j])

    suptitle(f'{study} {Description}: {ImagePlane.get(ImageOrientation)}')
    savefig(join(path,f'{study}-{Description}-{ImagePlane.get(ImageOrientation)}'))
    return fig

# MRI_Geometry
#
# Utilites for mapping coordinates
#
# https://nipy.org/nibabel/dicom/dicom_orientation.html
class MRI_Geometry:

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
    def create_mapping(dcim):
        def row(i):
            return [array([i,j]) for j in range(dcim.Columns)]
        return [row(i) for i in range(dcim.Rows)]

    @staticmethod
    def get_closest(series,centre_pixel,centre_pos):
        min_distance = float('inf')
        closest_dcim = None
        for dcim in series.dcmread():
            M    = MRI_Geometry.create_matrix(dcim)
            c    = matmul(M,centre_pixel)
            dist = norm(centre_pos - c)
            if dist<min_distance:
                min_distance = dist
                closest_dcim = dcim
            elif dist>min_distance:
                return closest_dcim,min_distance,norm(centre_pos[0:2] - c[0:2]),M
        return closest_dcim,min_distance,norm(centre_pos[0:2] - c[0:2]),M

def format_group(plane,group):
    return f'{plane}: [{",".join(series for series in group)}]'

# get_end_distances
#
# Check coplanar groups to see whether observations are collinear

def get_end_distances(coplanar_groups,study):

    # get_distance
    #
    # Get distance between point p and the point that corresponds to p on the line[P0,P1],
    #
    # Parameters:
    #     p         The point
    #     index     Identifies the axis used to establish correspondence
    #     P0        One of the two endpoints used to determine the line
    #     P1        One of the two endpoints used to determine the line
    #     slope     +/1, depending in whether P1[index]?P0[index]

    def get_distance(p,index,P0,P1,slope):
        def q(i):
            alpha = (p[index]-P0[index])/(P1[index]-P0[index])
            return alpha*slope*(P1[i]-P0[i])+P0[i]
        return norm([p[i]-q(i) for i in range(len(P0))])

    Distances  = {}
    for name,group in coplanar_groups.items():
        Orbits    = {series_name:study[series_name].get_orbit() for series_name in group}
        first_key = list(Orbits.keys())[0]
        P0        = Orbits[first_key][0]
        P1        = Orbits[first_key][-1]
        Delta     = [P1[i]-P0[i] for i in range(len(P0))]
        index, _  = max(enumerate([abs(delta) for delta in Delta]), key=itemgetter(1))
        slope     = sign(Delta[index])

        for series_name,Orbit in Orbits.items():
            if series_name != first_key:
                Distances[f'{first_key}-{series_name}'] = (get_distance(Orbits[series_name][0],index,P0,P1,slope),
                                                           get_distance(Orbits[series_name][-1],index,P0,P1,slope))
        return Distances

if __name__=='__main__':
    parser = ArgumentParser('Determine trajectories for all studies')
    parser.add_argument('--path',     default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--unique',   default = 'unique.csv',                 help = 'File name for list of studies whose planes are identical')
    parser.add_argument('--figs',     default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',     default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--studies',  default = [],    nargs = '*',           help = 'Names of Studies to be processed (omit to process all)' )
    parser.add_argument('--cmap',     default = 'gray',                       help = 'Colour map for displaying greyscale images')
    parser.add_argument('--coplanar', default = 'coplanar.txt')
    args = parser.parse_args()

    training = Labelled_MRI_Dataset(args.path,'train')

    with open(args.unique,'w') as out,              \
         open(args.coplanar,'w') as coplanar:
        for study in training.get_studies(study_names = args.studies):
            image_planes = study.get_image_planes()
            groups       = {name:[] for name in ImagePlane.names}
            for series_type, image_plane in zip(Study.Series.Types,image_planes):
                groups[image_plane].append(series_type)
            coplanar_groups = {name: groups[name] for name in ImagePlane.names if len(groups[name])>1}

            output_line = ';'.join(format_group(plane,coplanar_groups[plane]) for plane in sorted(coplanar_groups.keys()))
            print(f'{study.name}  {output_line}' )
            coplanar.write(f'{study.name}  {output_line}\n' )
            for key,value in get_end_distances(coplanar_groups,study).items():
                print (key,value)
                coplanar.write(f'{key}  {value}\n' )
            if len(set(image_planes))==1:
                out.write(f'{study}, {image_planes[0]}\n')
                fig = plot_orbit(study, path = args.figs)
                if not args.show:
                    close(fig)
                for series in study.get_series():
                    fig = plot_series(series,
                                      study = study,
                                      path  = args.figs,
                                      cmap  = args.cmap)
                    if not args.show:
                        close(fig)

    if args.show:
        show()
