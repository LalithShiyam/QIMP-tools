import SimpleITK as sitk
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset


def data_generator(images_list, labels_list, batch_size, Transforms):

    f = open(images_list, 'r')
    images = f.readlines()
    f.close()

    f = open(labels_list, 'r')
    labels = f.readlines()
    f.close()

    c = 0

    while True:

        mapIndexPosition = list(zip(images, labels))  # shuffle order list
        random.shuffle(mapIndexPosition)
        images, labels = zip(*mapIndexPosition)

        for i in range(c, c + batch_size):

            TrainDataset = NiftiDataset.NiftiDataset(
                image_filename=images[i],
                label_filename=labels[i],
                transforms=Transforms,
                train=True
            )

            trainDataset = TrainDataset.get_dataset()
            # trainDataset = trainDataset.batch(batch_size)

        c += batch_size

        if c + batch_size >= len(images):
            c = 0

        yield (trainDataset)

        # # read batch_size of data
        # for i in range(0, batch_size):
        #     source_img = read_image(files_image_dir[count])
        #     target_img = read_image(files_label_dir[count])
        #     count = (count+1)%len(files_image_dir)
        #
        # yield source_img, target_img

