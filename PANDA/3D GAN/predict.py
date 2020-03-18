#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from tqdm import tqdm
import datetime
from predict_single_image import from_numpy_to_itk, prepare_batch, inference
import math


def inference_all(model, image_path, resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size):

    case = image_path
    case = case.split('/')
    case = case[3]
    case = case.split('.')
    case = case[0]

    # case = image_path
    # case = case.split('/')
    # case = case[2]
    # case = case.split('.')
    # case = case[0]
    # case = case.split('\\')
    # case = case[1]

    if not os.path.isdir('./Data_folder/results'):
        os.mkdir('./Data_folder/results')

    label_directory = os.path.join('./Data_folder/results', case)

    if not os.path.isdir(label_directory):  # create folder
        os.mkdir(label_directory)

    result = inference(False, model, image_path, './prova.nii', resample, resolution, patch_size_x,
                       patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size)

    # save segmented label
    writer = sitk.ImageFileWriter()
    label_directory = os.path.join(label_directory, 'label_prediction.nii.gz')
    writer.SetFileName(label_directory)
    writer.Execute(result)
    # print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), label_path))
    print('************* Next image coming... *************')


def check_accuracy_model(model, images_list, resample, new_resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size):

    f = open(images_list, 'r')
    images = f.readlines()
    f.close()

    print("0/%i (0%%)" % len(images))
    for i in range(len(images)):

       inference_all(model=model, image_path=images[i].rstrip(), resample=resample, resolution=new_resolution, patch_size_x=patch_size_x,
                                        patch_size_y=patch_size_y, patch_size_z=patch_size_z,  stride_inplane=stride_inplane, stride_layer=stride_layer, batch_size=batch_size)






