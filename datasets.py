import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils import *
import torch
from astropy.io import fits


class SpeDataset(Dataset):

    def __init__(self, root, unaligned=False, mode="train"):
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        f = open(self.files_A[index % len(self.files_A)])
        data = f.read()
        f.close()
        lineList = data.splitlines()
        wave = []
        flux = []

        for line in lineList:
            lTemp = line.split()
            wave.append(float(lTemp[0]))
            flux.append(float(lTemp[1]))

        spectra_A = np.asarray(flux)
        if self.unaligned:
            fitsFile_B = fits.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
            spectra_B = fitsFile_B[1].data[0][5]
            z = fitsFile_B[0].header['z']
            spectra_B_wave = fitsFile_B[1].data[0][2] / (1 + z)
            # shift = rv(self.files_B[random.randint(0, len(self.files_B) - 1)])
            # spectra_B_wave = shiftToRest(shift, spectra_B_wave)


        else:
            fitsFile_B = fits.open(self.files_B[index % len(self.files_B)])
            spectra_B = fitsFile_B[1].data[0][5]
            z = fitsFile_B[0].header['z']
            spectra_B_wave = fitsFile_B[1].data[0][2] / (1 + z)
            # shift = rv(self.files_B[index % len(self.files_B)])
            # print(shift)
            # spectra_B_wave = shiftToRest(shift, spectra_B_wave)

        data_raw_A = spectra_A.astype(np.float32)
        data_raw_A = torch.tensor(data_raw_A)
        data_raw_A = torch.reshape(data_raw_A, (1, -1))
        data_raw_A = data_raw_A[0, 900:5700]
        # min_a = torch.min(data_raw_A)
        # max_a = torch.max(data_raw_A)
        # data_raw_A = (data_raw_A - min_a) / (max_a - min_a)
        data_raw_A = torch.reshape(data_raw_A, (1, -1))
        item_A = data_raw_A

        fitsFile_B.close()
        wave, data_raw_B = interpOntoGrid(waveGrid, spectra_B_wave, spectra_B)
        data_raw_B = data_raw_B.astype(np.float32)
        data_raw_B = torch.from_numpy(data_raw_B)
        data_raw_B = torch.reshape(data_raw_B, (1, -1))
        # print(data_raw_B.shape,'b1')
        data_raw_B = data_raw_B[0, 900:5700]
        # print(data_raw_B.shape,'b2')
        # min_b = torch.min(data_raw_B)
        # max_b = torch.max(data_raw_B)
        # data_raw_B = (data_raw_B - min_b) / (max_b - min_b)
        data_raw_B = torch.reshape(data_raw_B, (1, -1))
        # print(data_raw_B.shape,'b3')

        item_B = data_raw_B


        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))