from cmath import phase
from email.mime import image
import os
from pickletools import uint8
import numpy as np
import torch, torchvision
from tqdm import tqdm
import cv2
import pandas as pd

import random
from torchvision import transforms
from PIL import Image

random.seed(42)

def get_spatial_fragments(
    video,
    fragments_h=7,
    fragments_w=7,
    fsize_h=32,
    fsize_w=32,
    aligned=32,
    nmini_patches=1,
    random=False,
    fallback_type="upsample",
):
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w

    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:

        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)


    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                    :, t_s:t_e, h_so:h_eo, w_so:w_eo
                ]
    return target_video


class QMM_Dataset(torch.utils.data.Dataset):                                   
    def __init__(
        self,
        csv_file,
        data_prefix,
        fragments=7,
        fsize=32,
        nmini_patches=1,
        phase="test",
        img_length_read = 6,
    ):
        self.csv_file = csv_file
        self.data_prefix = data_prefix
        self.fragments = fragments
        self.fsize = fsize
        self.nmini_patches = nmini_patches
        self.image_infos = []
        self.phase = phase
        self.img_length_read = img_length_read
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        df = pd.read_csv(csv_file)
        images = df['Image'].values 
        if_test = df['if_test'].values 
        mos = df['MOS'].values 
        

        self.image_infos = []
        if phase == 'test':
            used_images = images[if_test==1]
            used_mos = mos[if_test==1]
        else:
            used_images = images[if_test==0]
            used_mos = mos[if_test==0]


        for i in range(len(used_images)):
            self.image_infos.append(dict(filename=os.path.join(data_prefix,used_images[i]),label = used_mos[i]))


    def __getitem__(self, index):
        # define mini-patch parameters
        fragments = self.fragments
        fsize = self.fsize
        image_info = self.image_infos[index]
        filename = image_info["filename"]
        label = image_info["label"]

        img_channel = 3
        img_height = 224
        img_width = 224
        # define the number of generated quality mini-patch map (QMM), 1 as default
        QMM_num = 1
        # define the number of mini-patch maps uesd for mini-patch selection
        img_length_read = self.img_length_read  
        # define the shape of mini_patch_maps and QMM   
        mini_patch_maps = torch.zeros([img_length_read, img_channel, img_height, img_width])
        QMM = torch.zeros([QMM_num, img_channel, img_height, img_width])
        img_read_index = 0

        # return mini-patch maps for the projections
        for i in range(img_length_read):
            img_name = os.path.join(filename, str(i) + '.png')
            if os.path.exists(img_name):
                img = cv2.imread(img_name)
                img = torch.from_numpy(img[:, :, [2, 1, 0]]).permute(2, 0, 1)            
                image = img.unsqueeze(1)
                ifrag = get_spatial_fragments(image, fragments, fragments, fsize, fsize)
                ifrag = (((ifrag.permute(1, 2, 3, 0) - self.mean) / self.std).squeeze(0).permute(2, 0, 1))
                mini_patch_maps[i] = ifrag
                img_read_index += 1
            else:
                print(img_name)
                print('Image do not exist!')
        
        
        # get the number of mini_patches picked from each of the first (img_length_read-1) viewpoint
        pick_mini_patches_num = fragments**2 // img_length_read
        for i in range(img_length_read):
            # get the random ids of mini_patches from each of the first (img_length_read-1) viewpoint
            mini_patches_list = list(range(fragments**2))
            random.shuffle(mini_patches_list)
            mini_patch_id = 0
            # pick the rest mini_patches from the last viewpoint
            if i == img_length_read - 1: 
                pick_mini_patches_num = fragments**2 - (img_length_read - 1) * pick_mini_patches_num
            # assign the picked mini_patches to QMM
            for pick_id in range(pick_mini_patches_num):
                # get the merge frag id
                QMM_id = i * fragments**2 // img_length_read + pick_id
                QMM_h = (QMM_id % fragments)*fsize
                QMM_w = (QMM_id // fragments)*fsize
                # get the pick mini_patch_id
                for merge_id in range(QMM_num):
                    # if the pick_mini_patch_id exceeds fragments**2 (limits of the mini_patches_list), the last frag will be repeatedly assigned
                    while mini_patch_id < fragments**2:
                        pick_mini_patch_id = mini_patches_list[mini_patch_id]
                        pick_frag_h = (pick_mini_patch_id % fragments)*fsize
                        pick_frag_w = (pick_mini_patch_id // fragments)*fsize
                        mini_patch_id = mini_patch_id + 1
                        # ignore the blank patch 
                        if torch.mean(mini_patch_maps[i][:,  pick_frag_h : pick_frag_h + fsize, pick_frag_w : pick_frag_w + fsize ]) <  2.2: break
                    QMM[merge_id][:,  QMM_h : QMM_h + fsize, QMM_w : QMM_w + fsize ] = \
                        mini_patch_maps[i][:,  pick_frag_h : pick_frag_h + fsize, pick_frag_w : pick_frag_w + fsize ] 




        data = {
            "image": QMM,
            "gt_label": label,
            "name": filename,
        }

        return data

    def __len__(self):
        return len(self.image_infos)

