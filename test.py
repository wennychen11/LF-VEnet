import argparse
import math
import os
import sys
import time
import cv2
import torch
import numpy as np
import pandas as pd
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from make_test_data import testDataSet
from model_VEnet import VEnet

# Training settings
# test datasets include kalantari, hci_test, sta_general, sta_occlusions, hci_old
parser = argparse.ArgumentParser(description="LF-VEnet")
parser.add_argument("--test_dataset", type=str, default="kalantari", help="Dataset file for testing")
parser.add_argument("--dataset_path", type=str, default="D:/datasets/", help="Dataset path for testing")
parser.add_argument("--result_path", type=str, default="result/", help="image save path")
parser.add_argument("--save_path", type=str, default="D:/VE_models/model_1.0/", help="model save path")
parser.add_argument("--n_view", type=int, default=7, help="Size of angular dim for testing")
parser.add_argument("--an", type=int, default=7, help="Size of original angular dim")
parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--is_save", type=int, default=0, help="save result lf image or not")
parser.add_argument('--gpu_no', type=int, default=2, help='GPU used: (default: %(default)s)')
parser.add_argument('--layer_num', type=int, default=10, help="number of layers in resBlocks")
parser.add_argument('--network_name', type=str, default='model_VEnet', help='name of testing network')
parser.add_argument('--data_type', type=int, default=2, help="1: rresized, 2: lr_2, 3:lr_4")
opt = parser.parse_args()


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_no)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('creating save and result directory...')
    save_path = opt.save_path + 'test_detail/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_path = opt.result_path + '{}/'.format(opt.test_dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    sys.stdout = Logger(
        save_path + '{}_test_{}_{}.log'.format(opt.test_dataset, opt.network_name, int(time.time())), sys.stdout)

    print(opt)
    print('loading datasets...')
    test_path = opt.dataset_path + 'LF-VEnet_test_data_{}.h5'.format(opt.test_dataset)
    test_set = testDataSet(test_path)
    test_loader = DataLoader(dataset=test_set, batch_size=1)
    print('loaded {} LFIs from {}'.format(len(test_loader), test_path))

    print('using network {}'.format(opt.network_name))
    model = SeNet(n_view=opt.n_view, scale=opt.scale, layer_num=opt.layer_num).to(device)

    print('loading pretrained model')
    resume_path = os.path.join(opt.save_path, 'model_x{}_bicubic.pth'.format(opt.scale))
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'], strict=False)

    print('testing...')
    model.eval()
    xls_list = []
    psnr_list = []
    ssim_list = []
    time_list = []
    for image_index, batch in enumerate(test_loader):
        if opt.is_save:
            lf_save_path = opt.save_path + 'results/{}/test_image_{}/'.format(opt.test_dataset, image_index)
            if not os.path.exists(lf_save_path):
                os.makedirs(lf_save_path)
        lf_ycrcb = batch[opt.data_type]  # [1, 49, H/scale, W/scale,3]
        lab = batch[0]  # [1, 49, H, W, 3]
        lf_y = lf_ycrcb[:, :, :, :, 0].to(device)
        lf_ycrcb = lf_ycrcb[0, :, :, :, :].numpy()  # for the HR result
        lab = lab[0, :, :, :, :].numpy()  # for the calculation PNSR and SSIM
        start = time.time()
        with torch.no_grad():
            sr_y = model(lf_y)
        end = time.time()
        run_time = end - start
        sr_y = sr_y.cpu().numpy().squeeze(0)
        time_list.append(run_time)
        psnr_image = np.zeros((opt.n_view, opt.n_view))
        ssim_image = np.zeros((opt.n_view, opt.n_view))
        lab = lab[:, :, :, 0]
        sr_y = np.clip(sr_y, 16. / 255., 235. / 255.)
        for view_index in range(opt.n_view ** 2):
            psnr = peak_signal_noise_ratio(sr_y[view_index, :, :], lab[view_index, :, :])
            ssim = structural_similarity(sr_y[view_index, :, :], lab[view_index, :, :])
            u = view_index // opt.n_view
            v = view_index % opt.n_view
            psnr_image[u, v] = psnr
            ssim_image[u, v] = ssim
            if opt.is_save:
                _, h, w, c = lf_ycrcb.shape
                if opt.data_type > 1:
                    # the input is downsampled LF image
                    sr_ycrcb = np.zeros((h * opt.scale, w * opt.scale, c), dtype=np.float32)
                else:
                    # the input is re-resized LF image
                    sr_ycrcb = np.zeros((h, w, c), dtype=np.float32)
                for c_index in range(c):
                    if opt.data_type > 1:
                        sr_ycrcb[:, :, c_index] = cv2.resize(lf_ycrcb[view_index, :, :, c_index],
                                                             (w * opt.scale, h * opt.scale),
                                                             interpolation=cv2.INTER_CUBIC)
                    else:
                        sr_ycrcb[:, :, c_index] = lf_ycrcb[view_index, :, :, c_index]
                sr_ycrcb[:, :, 0] = sr_y[view_index, :, :]
                sr_image = color.ycbcr2rgb(sr_ycrcb * 255.) * 255.0
                lf_name = '{}_{}.png'.format(u, v)
                cv2.imwrite(lf_save_path + lf_name, sr_image[:, :, ::-1])
        print('[{}/{}] test_image_{}'.format(image_index + 1, len(test_loader), image_index + 1))
        for i in range(opt.n_view):
            for j in range(opt.n_view):
                print('{:6.4f}/{:6.4f}'.format(psnr_image[i, j], ssim_image[i, j]), end='\t\t')
            print('')
        print(
            'PSNR Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}, SSIM: Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}, TIME: {:.4f}'
                .format(np.mean(psnr_image), np.max(psnr_image), np.min(psnr_image),
                        np.mean(ssim_image), np.max(ssim_image), np.min(ssim_image), run_time))

        psnr_ = np.mean(psnr_image)
        psnr_list.append(psnr_)
        ssim_ = np.mean(ssim_image)
        ssim_list.append(ssim_)
        xls_list.append(['test_image_{}'.format(image_index + 1), psnr_, ssim_, run_time])

    psnr_average = np.mean(psnr_list)
    ssim_average = np.mean(ssim_list)
    time_average = np.mean(time_list)
    xls_list.append(['average', psnr_average, ssim_average, time_average])
    xls_list = np.array(xls_list)
    result = pd.DataFrame(xls_list, columns=['image', 'psnr', 'ssim', 'time'])
    result.to_csv(save_path + '{}_result_epoch_{}_{}.csv'.format(opt.test_dataset, epoch, int(time.time())))

    print('-' * 100)
    print('AVR: PSNR: {:.4f}, SSIM: {:.4f}, TIME: {:.4f}'.format(np.mean(psnr_list), np.mean(ssim_list),
                                                                 np.mean(time_list)))
    print('all done')


if __name__ == '__main__':
    main()