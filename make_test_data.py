import os
import cv2
import h5py
import torch
import numpy as np
from skimage import color
from torch.utils.data import Dataset


class testDataSet(Dataset):
    def __init__(self, file_path):
        super(testDataSet, self).__init__()
        f = h5py.File(file_path)
        self.gt = f.get('gt')
        self.rresized = f.get('rresized')
        self.lr_2 = f.get('lr_2')
        self.lr_4 = f.get('lr_4')

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, index):
        gt = self.gt[index]  # [49, H, W]
        rresized = self.rresized[index] # [49, H, W]
        lr_2 = self.lr_2[index] # [49, H/2, W/2]
        lr_4 = self.lr_4[index] # [49, H/4, W/4]
        gt, rresized, lr_2, lr_4 = np.float32(gt / 255.0), np.float32(rresized / 255.0), \
                                   np.float32(lr_2 / 255.0), np.float32(lr_4 / 255.0)
        # to 3D Tensor
        return torch.from_numpy(gt), torch.from_numpy(rresized), torch.from_numpy(lr_2), torch.from_numpy(lr_4)


def make_test_data_h5(imageFolder, n_view=7):
    # the test LF image are raw LF image
    images = os.listdir(imageFolder)
    test_name = imageFolder.split('/')
    test_name = test_name[-2]
    # storage location of the result h5, which are classified by the category of test LF
    save_path = 'D:/datasets/LF-VEnet_test_data_{}.h5'.format(test_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    f = h5py.File(save_path, 'a')
    for index, name in enumerate(images):
        print('[{}/{}]'.format(index + 1, len(images)), name)
        lf_name = os.path.join(imageFolder, name)
        lf_img = cv2.imread(lf_name)
        lf_ycrcb = color.rgb2ycbcr(lf_img[:, :, ::-1])  # bgr -> rgb
        h, w, c = lf_ycrcb.shape  # [an*H, an*W, 3]
        if w > 7000:  # real-world LF images
            n_view_ori = 14
            img_h, img_w = h // n_view_ori, w // n_view_ori  # [H, W]
            # Crop images to a fixed size
            lf_ycrcb = lf_ycrcb[:-(img_h - 372) * n_view_ori, :, :]  # Crop the spatial resolution
            if img_w > 540:
                lf_ycrcb = lf_ycrcb[:, :-(img_w - 540) * n_view_ori, :]
            # Crop the angular resolution (4 for the up and left, 3 for the bottom and right) : 14×14 --> 7×7
            cut_n = ((n_view_ori + 1) - n_view) // 2
            for i in range(0, n_view, 1):
                for j in range(0, n_view, 1):
                    if i == 0 and j == 0:
                        gt_data = np.expand_dims(lf_ycrcb[i + cut_n::n_view_ori, j + cut_n::n_view_ori, :], 0)
                    else:
                        lf_y_temp = np.expand_dims(lf_ycrcb[i + cut_n::n_view_ori, j + cut_n::n_view_ori, :], 0)
                        gt_data = np.concatenate((gt_data, lf_y_temp), 0)   # input LF: [an*H, an*W, 3] -> [49, H, W, 3]
            rresized_data, lr_2_data, lr_4_data, img_h, img_w, img_h_2, img_w_2, img_h_4, img_w_4 = downSampling(gt_data)
            gt_data = np.expand_dims(gt_data, 0)  # [1, 49, H, W, 3]
        else:
            n_view_ori = 9
            # Crop the angular resolution: 9×9 --> 7×7
            cut_n = ((n_view_ori + 1) - n_view) // 2
            for i in range(0, n_view, 1):
                for j in range(0, n_view, 1):
                    if i == 0 and j == 0:
                        gt_data = np.expand_dims(lf_ycrcb[i + cut_n::n_view_ori, j + cut_n::n_view_ori, :], 0)
                    else:
                        lf_y_temp = np.expand_dims(lf_ycrcb[i + cut_n::n_view_ori, j + cut_n::n_view_ori, :], 0)
                        gt_data = np.concatenate((gt_data, lf_y_temp), 0)   # input LF: [an*H, an*W, 3] -> [49, H, W, 3]
            rresized_data, lr_2_data, lr_4_data, img_h, img_w, img_h_2, img_w_2, img_h_4, img_w_4 = downSampling(gt_data)
            gt_data = np.expand_dims(gt_data, 0)  # [1, 49, H, W, 3]

        if index == 0:
            len_data = 1
            dataset_gt = f.create_dataset("gt", (1, n_view ** 2, img_h, img_w, 3),
                                          maxshape=(None, n_view ** 2, img_h, img_w, 3), dtype='float32')
            dataset_rresized = f.create_dataset("rresized", (1, n_view ** 2, img_h, img_w, 3),
                                                maxshape=(None, n_view ** 2, img_h, img_w, 3), dtype='float32')
            dataset_lr_2 = f.create_dataset("lr_2", (1, n_view ** 2, img_h_2, img_w_2, 3),
                                            maxshape=(None, n_view ** 2, img_h_2, img_w_2, 3), dtype='float32')
            dataset_lr_4 = f.create_dataset("lr_4", (1, n_view ** 2, img_h_4, img_w_4, 3),
                                            maxshape=(None, n_view ** 2, img_h_4, img_w_4, 3), dtype='float32')
            dataset_gt[0:len_data] = gt_data
            dataset_rresized[0:len_data] = rresized_data
            dataset_lr_2[0:len_data] = lr_2_data
            dataset_lr_4[0:len_data] = lr_4_data
        else:
            len_data_sta = len_data
            len_data += 1
            dataset_gt.resize([len_data, n_view ** 2, img_h, img_w, 3])
            dataset_gt[len_data_sta:len_data] = gt_data
            dataset_rresized.resize([len_data, n_view ** 2, img_h, img_w, 3])
            dataset_rresized[len_data_sta:len_data] = rresized_data
            dataset_lr_2.resize([len_data, n_view ** 2, img_h_2, img_w_2, 3])
            dataset_lr_2[len_data_sta:len_data] = lr_2_data
            dataset_lr_4.resize([len_data, n_view ** 2, img_h_4, img_w_4, 3])
            dataset_lr_4[len_data_sta:len_data] = lr_4_data
    print(f.keys())
    print(f['gt'].shape)
    print(f['rresized'].shape)
    print(f['lr_2'].shape)
    print(f['lr_4'].shape)
    f.close()


def downSampling(test_data):  # [49, H, W, 3]
    D, img_h, img_w, c = test_data.shape
    img_h_2, img_w_2 = img_h // 2, img_w // 2  # [H/2, W/2]
    img_h_4, img_w_4 = img_h // 4, img_w // 4  # [H/4, W/4]
    rresized = np.zeros((D, img_h, img_w, c), dtype=np.float32)
    lr_2 = np.zeros((D, img_h_2, img_w_2, c), dtype=np.float32)
    lr_4 = np.zeros((D, img_h_4, img_w_4, c), dtype=np.float32)
    for i in range(D):
        for j in range(c):
            lr_2[i, :, :, j] = cv2.resize(test_data[i, :, :, j], (img_w_2, img_h_2),
                                          interpolation=cv2.INTER_CUBIC)  # the 2x downsampled data [49, H/2, W/2, 3]
            lr_4[i, :, :, j] = cv2.resize(test_data[i, :, :, j], (img_w_4, img_h_4),
                                          interpolation=cv2.INTER_CUBIC)  # the 4x downsampled data [49, H/4, W/4, 3]
            rresized[i, :, :, j] = cv2.resize(lr_2[i, :, :, j], (img_w, img_h),
                                              interpolation=cv2.INTER_CUBIC)  # the re-resized data [49, H, W, 3]
    return np.expand_dims(rresized, 0), np.expand_dims(lr_2, 0), np.expand_dims(lr_4, 0),\
           img_h, img_w, img_h_2, img_w_2, img_h_4, img_w_4


if __name__ == '__main__':
    # "imageFolder" is for the storage location of test LF images of one specific category
    make_test_data_h5(imageFolder='D:/datasets/test_data/sta_general/', n_view=7)
