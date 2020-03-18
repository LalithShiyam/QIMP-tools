from utils.NiftiDataset import *
import matplotlib.pyplot as plt
from utils.data_generator import *
import utils.NiftiDataset as NiftiDataset
import argparse

''' The script show as the output of the data loader (volume and label patches) fed to the network. Before running it, the .txt files with the list
of samples used for the training must be created (--Create_training_test_dataset from the command line of the main.py script)'''

parser = argparse.ArgumentParser()

parser.add_argument("--save_dir", type=str, default='./Data_folder/', help='path to early frames and late frames dose folders')
parser.add_argument("--resample", action='store_true', default=False, help='Decide or not to resample the images to a new resolution')
parser.add_argument("--new_resolution", type=float, default=(1.5, 1.5, 1.5), help='New resolution')
parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64], help="Input dimension for the generator")
parser.add_argument("--batch_size", type=int, nargs=1, default=1, help="Batch size to feed the network (currently supports 1)")
parser.add_argument("--drop_ratio", type=float, nargs=1, default=0, help="Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1")
parser.add_argument("--min_pixel", type=int, nargs=1, default=20, help="Percentage of minimum non-zero pixels in the cropped label")

args = parser.parse_args()

min_pixel = int(args.min_pixel*((args.patch_size[0]*args.patch_size[1]*args.patch_size[2])/100))

trainTransforms = [
    NiftiDataset.Resample(args.new_resolution, args.resample),
    NiftiDataset.Augmentation(),
    NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
    NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]),
                            args.drop_ratio, min_pixel)
]

train_gen = data_generator(images_list=args.save_dir + '/' + 'train.txt', labels_list=args.save_dir + '/' + 'train_labels.txt',
                         batch_size=args.batch_size, Transforms=trainTransforms)


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap= 'gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def plot3d(image):
    original=image
    original = np.rot90(original, k=-1)
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, original)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


batch1 = next(train_gen)
vol = batch1[0]
mask = batch1[1]

print(vol.shape)

vol = np.squeeze(vol, axis=4)
vol = np.squeeze(vol, axis=0)
mask = np.squeeze(mask, axis=4)
mask = np.squeeze(mask, axis=0)

plot3d(vol)
plot3d(mask)

