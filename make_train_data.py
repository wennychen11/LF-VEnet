import os
import random
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


def getPatch(train_data, patch_size, data_type, scale):  # [49, H, W]
    D, h, w = train_data.shape
    gt_patch_size = patch_size * scale  # patch_size is for lr, and gt_patch_size is for gt
    # whether the data is from hci or lytro?
    if train_data[:,:,-1].any() == 0:   #  the data is from hci, which lytro is filled with non-zero values here
        # randomly select a coordinate
        randh = random.randrange(0, 512 - gt_patch_size + 1)
        randw = random.randrange(0, 512 - gt_patch_size + 1)
        lab = train_data[:, randh:randh + gt_patch_size, randw:randw + gt_patch_size]  # crop the gt
    else:
        randh = random.randrange(0, 372 - gt_patch_size + 1)
        randw = random.randrange(0, 540 - gt_patch_size + 1)
        lab = train_data[:, randh:randh + gt_patch_size, randw:randw + gt_patch_size]
    rresized = np.zeros((D, gt_patch_size, gt_patch_size), dtype=np.float32)
    lr = np.zeros((D, patch_size, patch_size), dtype=np.float32)
    for i in range(D):
        lr[i, :, :] = cv2.resize(lab[i, :, :], (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)  # down-sampling
        # the data type "1" is for the data which is re-resized
        if data_type == 1:
            rresized[i, :, :] = cv2.resize(lr[i, :, :], (gt_patch_size, gt_patch_size), interpolation=cv2.INTER_CUBIC)
    # the data type "2" and "3" is for the data which is 2x and 4x down-sampled
    if data_type == 1:
        return augmentation(rresized, lab)  # [49, ps*scale, ps*scale],[[49, ps*scale, ps*scale]
    else:
        return augmentation(lr, lab)    # [49, ps, ps],[[49, ps*scale, ps*scale]


def augmentation(lf_y, lab):
    # Mirror flip
    if random.random() < 0.5:
        lf_y = lf_y[:, :, ::-1]
        lab = lab[:, :, ::-1]
    if random.random() < 0.5:
        lf_y = lf_y[:, ::-1, :]
        lab = lab[:, ::-1, :]
    # rotate
    if random.random() < 0.5:
        rot_num = random.randrange(1, 4, 1)
        lf_y = np.rot90(lf_y, rot_num, (1, 2))
        lab = np.rot90(lab, rot_num, (1, 2))
    return lf_y, lab


class trainDataSet(Dataset):
    def __init__(self, file_path, patch_size=64, data_type=1, scale=2, n_view=7):
        super(trainDataSet, self).__init__()
        f = h5py.File(file_path)
        self.gt_data = f.get('train_data')
        self.patch_size = patch_size
        self.data_type = data_type
        self.scale = scale
        self.n_view = n_view

    def __len__(self):
        return self.gt_data.shape[0]

    def __getitem__(self, index):
        gt_data = self.gt_data[index]   # [49, H, W]
        lf_y, lab = getPatch(gt_data, self.patch_size, self.data_type, self.scale)
        lf_y, lab = np.float32(lf_y / 255.0), np.float32(lab / 255.0)
        # to 3D Tensor
        return torch.from_numpy(lf_y), torch.from_numpy(lab)


def make_train_data_h5(imageFolder, n_view=7):
    # the training LF image are raw LF image
    images = os.listdir(imageFolder)
    # storage location of the result h5
    save_path = 'D:/datasets/train_data/LF-VEnet_training_data.h5'
    if os.path.exists(save_path):
        os.remove(save_path)
    f = h5py.File(save_path, 'a')
    for index, name in enumerate(images):
        print('[{}/{}]'.format(index + 1, len(images)), name)
        lf_name = os.path.join(imageFolder, name)
        lf_img = cv2.imread(lf_name)
        ycrcb = color.rgb2ycbcr(lf_img[:, :, ::-1]) # bgr -> rgb
        lf_img_y = ycrcb[:, :, 0]
        h, w = lf_img_y.shape[0], lf_img_y.shape[1]  # [an*H, an*W]
        gt_data = np.zeros((1, n_view ** 2, 512, 540), dtype=np.uint8)
        # The size is suitable for both real-world LF images and synthetic LF images.
        if h > 5000:  # real-world LF images
            n_view_ori = 14
            img_h, img_w = h // n_view_ori, w // n_view_ori  # [H, W]

            # Crop images to a fixed size
            lf_img_y = lf_img_y[:-(img_h - 372) * n_view_ori, :]  # Crop the spatial resolution
            if img_w > 540:
                lf_img_y = lf_img_y[:, :-(img_w - 540) * n_view_ori]
            img_h, img_w = 372, 540
            # Crop the angular resolution (4 for the up and left, 3 for the bottom and right) : 14×14 --> 7×7
            cut_n = ((n_view_ori + 1) - n_view) // 2
            for i in range(0, n_view, 1):
                for j in range(0, n_view, 1):
                    if i == 0 and j == 0:
                        train_data = np.expand_dims(lf_img_y[i + cut_n::n_view_ori, j + cut_n::n_view_ori], 0)
                    else:
                        lf_y_temp = np.expand_dims(lf_img_y[i + cut_n::n_view_ori, j + cut_n::n_view_ori], 0)
                        train_data = np.concatenate((train_data, lf_y_temp), 0)
            train_data = np.expand_dims(train_data, 0)
            gt_data[:, :, 0:img_h, 0:img_w] = train_data
        else:   # synthetic LF images
            n_view_ori = 9
            img_h, img_w = h // n_view_ori, w // n_view_ori  # [H, W]
            # Crop the angular resolution: 9×9 --> 7×7
            cut_n = ((n_view_ori + 1) - n_view) // 2
            for i in range(0, n_view, 1):
                for j in range(0, n_view, 1):
                    if i == 0 and j == 0:
                        train_data = np.expand_dims(lf_img_y[i + cut_n::n_view_ori, j + cut_n::n_view_ori], 0)
                    else:
                        lf_y_temp = np.expand_dims(lf_img_y[i + cut_n::n_view_ori, j + cut_n::n_view_ori], 0)
                        train_data = np.concatenate((train_data, lf_y_temp), 0)
            train_data = np.expand_dims(train_data, 0)  # [1, 49, H, W]
            gt_data[:, :, 0:img_h, 0:img_w] = train_data

        if index == 0:
            len_data = 1
            dataset_train_data = f.create_dataset("train_data", (1, n_view ** 2, 540, 540),
                                                  maxshape=(None, n_view ** 2, 540, 540), dtype='float32')
            dataset_train_data[0:len_data] = gt_data
        else:
            len_data_sta = len_data
            len_data += 1
            dataset_train_data.resize([len_data, n_view ** 2, 540, 540])
            dataset_train_data[len_data_sta:len_data] = gt_data
    print(f['train_data'].shape)
    f.close()


if __name__ == '__main__':
    # "imageFolder" is for the storage location of training LF images
    make_train_data_h5(imageFolder='D:/datasets/train_data/training_data/', n_view=7)