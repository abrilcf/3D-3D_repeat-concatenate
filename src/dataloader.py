import os
import glob
import torch
import torch.nn as nn
import numpy as np
import torchvision
import SimpleITK as sitk
import scipy.ndimage
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from src.utils import Resize_image, Normalization_min_max, ToTensor, Normalization



class BagXCT_dataset(Dataset):
    """
    Class for loading CT scans
    and paired Digitaly Reconstructed Radiographs
    (DRR) images
    """
    def __init__(self, data_dir, train, dataset='bags',
                 use_f16=False, scale=128, projections=2, p_rand=0.1,
                 align=False):
        self.data_dir = data_dir
        self.p_rand = p_rand
        self.scale = scale
        self.projections = projections
        self.dataset = dataset
        self.f16 = use_f16
        self.train = train
        self.align = align
        self.view_list = [0, 2] #  dummy to keep dataclasses consistent

        self.x_dirs, self.ct_dirs = self._get_dirs()

        self.xray_tx = [torchvision.transforms.Resize(scale),
                        Normalization(0, 255),
                        ToTensor(f16=self.f16)]

        self.ct_tx = [Normalization_min_max(0., 1., remove_noise=True),
                      ToTensor(f16=self.f16)]

    def __len__(self):
        return len(self.x_dirs)

    def __getitem__(self, idx):
        x_object_dir = os.path.dirname(self.x_dirs[idx])
        ct_object_dir = os.path.dirname(self.ct_dirs[idx])
        xray_files = []
        xrays_list = []
        
        xray_files.append(f'{x_object_dir}/projections_grayscale2/{os.path.basename(x_object_dir)}_elevation_90.png')
        xray_files.append(f'{x_object_dir}/projections_grayscale2/{os.path.basename(x_object_dir)}_azimuth_90.png')

        for x in xray_files:
            xray = Image.open(x).convert('L')
            for transf in self.xray_tx:
                xray = transf(xray)
            xrays_list.append(xray)
        xrays = torch.stack(xrays_list, 0)
        

        pattern = '*f16.npz'
        ct_path = f'{ct_object_dir}/**/{pattern}'
            
        ct_file = glob.glob(ct_path, recursive=True)
        ct_scan = np.load(ct_file[0])['ct']
        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        data = {
            "xrays": xrays,
            "ct": ct_scan.unsqueeze(0),
            "x_dir_name": x_object_dir,
            "ct_dir_name": ct_object_dir,
            "view_list": self.view_list}

        return data

    def _get_dirs(self, shuffle=True):    
        data_dirs = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        data_dirs.sort()
        files = []
        
        pattern = '*f16.npz' if self.f16 else '*.npy'
        for folder in data_dirs:
            file_list = list(Path(folder).rglob(pattern))
            if len(file_list) > 0 and "projections_grayscale2" in set(os.listdir(folder)):
                files.append(os.path.dirname(file_list[0]))

        if shuffle:
            np.random.seed(500)
            np.random.shuffle(files)
        split = self.get_splits(len(files))
        if self.train:
            return files[split:] if not self.align else self.get_scramble(files[split:]), files[split:]
        return files[:split] if not self.align else self.get_scramble(files[:split]), files[:split]

    def get_splits(self, total_examples, val_ratio=0.1):
        assert ((val_ratio >= 0) and (val_ratio <= 1)), "[!] valid_size should be in the range [0, 1]."
        return int(np.floor(val_ratio * total_examples))
        

    def get_scramble(self, file_list):
        list_copy = file_list.copy()
        np.random.seed(200)
        np.random.shuffle(list_copy)
        return list_copy


class XCT_dataset(Dataset):
    """
    Class for loading chest CT scans
    and paired (for evaluation) Digitaly Reconstructed Radiographs
    (DRR) images
    """
    def __init__(self,
                 data_dir,
                 train,
                 dataset='chest',
                 xray_scale=128,
                 projections=2,
                 f16=False,
                 separate_xrays=True,
                 scale=128,
                 scale_ct=False,
                 load_res=None,
                 use_synthetic=True,
                 norm=0.0):
        """
        :param data_dir: (str) path to data folder
        :param train: (bool) are we training or testing.
        :param scale: (int) resize both x-ray images and ct volumes to this value

        :param projections: (int) how many xray projections to use
          3 projections (sagittal, one at 45 deg and coronal)
          2 projections (sagittal & coronal)
          4 projections (one every 90 degrees)
          8 projections (one every 45 degrees)

        :param ct_min, ct_max: (int)  min and max values to adjust the intensities. 
        These values are taken from the preprocessing of the scans.
        should only be changed if the data preprocessing changes

        :return data: (dict)
        'xrays': stacked x-ray projections torch.Tensor(projections, 1, scale, scale)
        'ct': ct volume torch.Tensor(1, scale, scale, scale)
        'dir_name': full path to that ct scan (str)
        """
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.dataset = dataset
        self.ext = ('*.jpeg', '*.png')
        split_file = f'data/{self.dataset}_train_split.txt' if train else f'data/{self.dataset}_test_split.txt'
        self.projections = projections
        self.data_dir = data_dir
        self.f16 = f16
        self.load_res = load_res
        self.separate_xrays = separate_xrays

        view_lists = {
            0: [1, 3, 5, 7], # testing non-orthogonal views
            1: [0],
            2: [0, 2],
            4: [0, 2, 4, 6],
        }
        self.view_list = view_lists.get(projections, [0, 1, 2, 3, 4, 5, 6, 7])
        if dataset == 'knee' and not use_synthetic:
            self.view_list = [0, 1]
            
        self.split_list = self._get_split(split_file) if self.dataset == 'chest' else None
        self.object_dirs = self._get_dirs()
        print("dirs: ", len(self.object_dirs))
        
        self.xray_tx = [torchvision.transforms.Resize(xray_scale),
                        Normalization(0, 255),
                        ToTensor(f16=self.f16)]

        self.ct_tx = [Resize_image((scale, scale, scale)) if scale_ct else nn.Identity(),
                      Normalization_min_max(norm, 1.), 
                      ToTensor(f16=self.f16)]


    def __len__(self):
        return len(self.object_dirs)

    def __getitem__(self, idx):
        object_dir = self.object_dirs[idx]
        xray_files = []
        for ext in self.ext:
            xray_files.extend(sorted(glob.glob(os.path.join(object_dir, ext))))
        xrays_list = []
        for i in self.view_list:
            xray = Image.open(xray_files[i]).convert('L')
            for transf in self.xray_tx:
                xray = transf(xray)
            xrays_list.append(xray)
        xrays = torch.stack(xrays_list, 0)

        if self.load_res is not None:
            ct_path = f'{object_dir}/*_{self.load_res}.npz'
            ct_file = glob.glob(ct_path)
            ct_scan = torch.from_numpy(np.load(ct_file[0])["ct"]).squeeze(0)
        else:
            pattern = f'*chest_64_f16.npz' if self.f16 else f'*chest_.npz'
            ct_path = f'{object_dir}/{pattern}'
            ct_file = glob.glob(ct_path)
            ct_scan = np.flip(np.load(ct_file[0])['ct'])

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        if self.separate_xrays:
            separate_xrays = ()
            for x in xrays:
                separate_xrays = separate_xrays + (x.unsqueeze(0),)
            xrays = separate_xrays
            
        ct_scan = ct_scan.unsqueeze(0)

        data = {
            "xrays": xrays,
            "ct": ct_scan,
            "dir_name": os.path.basename(object_dir),
            "view_list": self.view_list,
            "idx": idx,
        }

        return data

    def _get_dirs(self):
        data_dirs = glob.glob(f"{self.data_dir}/**/*_chest", recursive=True)
        if self.split_list == None:
            return data_dirs
        dirs = [x for x in data_dirs if os.path.basename(x).split('_')[0] in set(self.split_list)]
        return dirs

    @staticmethod
    def _get_split(split_file):
        with open(split_file) as file:
            data = [n.rstrip() for n in file]
        return data


class XCT_Unpaired_dataset(Dataset):
    """
    Class for loading chest CT scans
    and paired (for evaluation) Digitaly Reconstructed Radiographs
    (DRR) images
    """
    def __init__(self,
                 data_dir,
                 train,
                 dataset='chest',
                 xray_scale=128,
                 projections=2,
                 f16=False,
                 separate_xrays=False,
                 scale=128,
                 scale_ct=False,
                 load_res=None,
                 use_synthetic=True,
                 align=False,
                 seed=500):
        """
        :param data_dir: (str) path to data folder
        :param train: (bool) are we training or testing.
        :param scale: (int) resize both x-ray images and ct volumes to this value

        :param projections: (int) how many xray projections to use
          3 projections (sagittal, one at 45 deg and coronal)
          2 projections (sagittal & coronal)
          4 projections (one every 90 degrees)
          8 projections (one every 45 degrees)

        :param ct_min, ct_max: (int)  min and max values to adjust the intensities. 
        These values are taken from the preprocessing of the scans.
        should only be changed if the data preprocessing changes

        :return data: (dict)
        'xrays': stacked x-ray projections torch.Tensor(projections, 1, scale, scale)
        'ct': ct volume torch.Tensor(1, scale, scale, scale)
        'dir_name': full path to that ct scan (str)
        """
        assert os.path.exists(data_dir), f"Error: {data_dir} not found!"
        self.dataset = dataset
        self.ext = '*.jpeg' if use_synthetic and dataset=='knee' else '*.png'
        split_file = f'/home2/xvqk44/x2ct/ot_ae/data/{self.dataset}_train_split.txt' if train else f'/home2/xvqk44/x2ct/ot_ae/data/{self.dataset}_test_split.txt'
        self.projections = projections
        self.data_dir = data_dir
        self.f16 = f16
        self.load_res = load_res
        self.separate_xrays = separate_xrays
        self.align = align
        self.seed = seed

        view_lists = {
            1: [0],
            2: [0, 2],
            4: [0, 2, 4, 6],
        }
        self.view_list = view_lists.get(projections, [0, 1, 2, 3, 4, 5, 6, 7])
        if dataset == 'knee' and not use_synthetic:
            self.view_list = [0, 1]

        self.split_list = self._get_split(split_file)
        self.x_dirs, self.ct_dirs = self._get_dirs()
        
        self.xray_tx = [torchvision.transforms.Resize(xray_scale),
                        Normalization(0, 255),
                        ToTensor(f16=self.f16)]

        self.ct_tx = [Resize_image((scale, scale, scale)) if scale_ct else nn.Identity(),
                      Normalization_min_max(0., 1.), 
                      ToTensor(f16=self.f16)]


    def __len__(self):
        return len(self.x_dirs)

    def __getitem__(self, idx):
        x_object_dir = self.x_dirs[idx]
        ct_object_dir = self.ct_dirs[idx]
        xray_files = sorted(glob.glob(os.path.join(x_object_dir, self.ext)))
        xrays_list = []
        for i in self.view_list:
            xray = Image.open(xray_files[i]).convert('L')
            for transf in self.xray_tx:
                xray = transf(xray)
            xrays_list.append(xray)
        xrays = torch.stack(xrays_list, 0)

        if self.load_res is not None:
            ct_path = f'{ct_object_dir}/*_{self.load_res}.npz'
            ct_file = glob.glob(ct_path)
            ct_scan = torch.from_numpy(np.load(ct_file[0])["ct"]).squeeze(0)
        else:
            pattern = f'*{self.dataset}_64_f16.npz' if self.f16 else f'*{self.dataset}_.npz'
            ct_path = f'{ct_object_dir}/{pattern}'
            ct_file = glob.glob(ct_path)
            ct_scan = np.flip(np.load(ct_file[0])['ct'])

        for transf in self.ct_tx:
            ct_scan = transf(ct_scan)

        if self.separate_xrays:
            separate_xrays = ()
            for x in xrays:
                separate_xrays = separate_xrays + (x.unsqueeze(0),)
            xrays = separate_xrays
            
        ct_scan = ct_scan.unsqueeze(0)

        data = {
            "xrays": xrays,
            "ct": ct_scan,
            "x_dir_name": x_object_dir,
            "ct_dir_name": ct_object_dir,
            "view_list": self.view_list}

        return data

    def _get_dirs(self):
        data_dirs = glob.glob(f"{self.data_dir}/**/*_{self.dataset}", recursive=True)
        ct_dirs = [x for x in data_dirs if os.path.basename(x).split('_')[0] in set(self.split_list)]
        x_dirs = ct_dirs.copy()
        if not self.align:
            np.random.seed(self.seed)
            np.random.shuffle(x_dirs)
        return x_dirs, ct_dirs

    @staticmethod
    def _get_split(split_file):
        with open(split_file) as file:
            data = [n.rstrip() for n in file]
        return data


class CT_dataset(Dataset):
    """
    Class for loading CT scans
    some functions are from:
    https://github.com/kylekma/X2CT
    """
    def __init__(self, data_dir, scale=128):
        self.ct_files = self._get_files(data_dir)

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct_scan, ct_spacing = load_scan(self.ct_files[idx])
        resampled_scan, _ = ct_resample(ct_scan, ct_spacing)
        cropped_ct = ct_crop_to_standard(resampled_scan, scale=128)
        return cropped_ct

    @staticmethod
    def _get_files(data_dir):
        print("CT scans data dir: ", data_dir)
        ct_files = sorted(glob.glob(f'{data_dir}/*.mha'))
        return ct_files


class Xrays_dataset(Dataset):
    """Class for getting data
    Args:
        data_dir = path of input images
    Output:
        xray image"""

    def __init__(self, data_dir, img_size):
        self.xrays_files = self._get_files(data_dir)
        self.tx = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
#            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
 #                                              saturation=0.4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5],
                                             std=[0.5])
                  ])

    def __len__(self):
        return len(self.xrays_files)

    @staticmethod
    def _get_files(data_dir):
        image_files = sorted(glob.glob(f'{data_dir}/*.png'))
        return image_files

    def __getitem__(self, idx):
        xray = self.xrays_files[idx]
        xray = Image.open(xray)
        xray = self.tx(xray)

        return xray


class List_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_list):
        for t_list in self.transforms:
            # deal with separately
            if len(t_list) > 1:
                new_img_list = []
                for img, t in zip(img_list, t_list):
                    if t is None:
                        new_img_list.append(img)
                    else:
                        new_img_list.append(t(img))
                img_list = new_img_list
            # deal with combined
            else:
                img_list = t_list[0](img_list)

        return img_list


def load_scan(path):
    # input could be .mhd/.mha format
    img_itk = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img_itk)
    return img, img_itk.GetSpacing()


def ct_resample(image, spacing, new_spacing=[1,1,1]):
    # resample to stantard spacing (keep physical scale stable, change pixel numbers)
    # .mhd image order : z, y, x
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)
    spacing = spacing[::-1]
    new_spacing = new_spacing[::-1]

    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing        


def ct_crop_to_standard(scan, scale):
    # crop to stantard shape (scale x scale x scale) (keep pixel scale consistency)
    z, y, x = scan.shape    
    if z >= scale:
        ret_scan = scan[z-scale:z, :, :]
    else:        
        temp1 = np.zeros(((scale-z)//2, y, x))
        temp2 = np.zeros(((scale-z)-(scale-z)//2, y, x))
        ret_scan = np.concatenate((temp1, scan, temp2), axis=0)
    z, y, x = ret_scan.shape
    if y >= scale:
        ret_scan = ret_scan[:, (y-scale)//2:(y+scale)//2, :]
    else:
        temp1 = np.zeros((z, (scale-y)//2, x))
        temp2 = np.zeros((z, (scale-y)-(scale-y)//2, x))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)
    z, y, x = ret_scan.shape
    if x >= scale:
        ret_scan = ret_scan[:, :, (x-scale)//2:(x+scale)//2]
    else:
        temp1 = np.zeros((z, y, (scale-x)//2))
        temp2 = np.zeros((z, y, (scale-x)-(scale-x)//2))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=2)
    return ret_scan


def get_xrays(data_dir, batch_size, img_size):
    if not os.path.isdir(data_dir):
        print("Xrays data dir missing!")
        exit()
        
    training_data = Xrays_dataset(data_dir, img_size)

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(training_data,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=True, sampler=None,
                                             pin_memory=False)

    return dataloader


def get_ct_scans(data_dir, batch_size, img_size):
    if not os.path.isdir(data_dir):
        print("CT scans data dir missing!")
        exit()
        
    training_data = CT_dataset(data_dir, scale=img_size)

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(training_data,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=True, sampler=None,
                                             pin_memory=False)

    return dataloader
