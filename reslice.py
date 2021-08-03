
from pydicom           import dcmread
from numpy             import zeros
from matplotlib.pyplot import figure, imshow, show, subplot
from sys               import argv
from glob              import glob

# load the DICOM files

def read_slices(paths):
    files = [dcmread(fname) for fname in paths]

    print("file count: {}".format(len(files)))

    return sorted([f for f in files if hasattr(f, 'SliceLocation')],
                    key=lambda s: s.SliceLocation)

def get_pixel_aspects(slices):
    # pixel aspects, assuming all slices are the same
    ps         = slices[0].PixelSpacing
    ss         = slices[0].SliceThickness
    return ps[1]/ps[0], ps[1]/ss, ss/ps[0]

def create_image(slices):
    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = zeros(img_shape)

    # fill 3D array with the images from the files
    for i, slice in enumerate(slices):
        img2d          = slice.pixel_array
        img3d[:, :, i] = img2d
    return img3d, img_shape

if __name__=='__main__':

    slices = read_slices(glob(argv[1], recursive=False))

    ax_aspect, sag_aspect, cor_aspect = get_pixel_aspects(slices)

    img3d, img_shape = create_image(slices)
    # plot 3 orthogonal slices
    fig = figure()
    a1 = subplot(2, 2, 1)
    imshow(img3d[:, :, img_shape[2]//2])
    a1.set_aspect(ax_aspect)

    a2 = subplot(2, 2, 2)
    imshow(img3d[:, img_shape[1]//2, :])
    a2.set_aspect(sag_aspect)

    a3 = subplot(2, 2, 3)
    imshow(img3d[img_shape[0]//2, :, :].T)
    a3.set_aspect(cor_aspect)

    show()
