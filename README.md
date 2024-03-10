# GMS-3DQA
Official repo for "GMS-3DQA: Projection-based Grid Mini-patch Sampling for 3D Model Quality Assessment", accepted by ACM TOMM.


## Quick Start

Get the 6 face projection of the point clouds and replace the folder path in the `train.sh'. 

Please refer to [MM-PCQA repo](https://github.com/zzc-1998/MM-PCQA/blob/main/utils/get_projections.py) for information about generating projections for point clouds. 
The code is used to generate 4 projections, to modify it for 6 face projections please change the code on line 60-61:
```
rotate_para = [[0,0],[90*interval,0],[90*interval,0],[90*interval,0]]
    for i in range(4):
```
into:
```
rotate_para = [[0,0],[90*interval,0],[90*interval,0],[90*interval,0],[0,90*interval],[0,180*interval]]
    for i in range(6):
```

We also provide all the projections of SJTU-PCQA, WPC, and WPC2.0 here on [Baiduyunpan](https://pan.baidu.com/s/1R_OrdH_90eGlXkNi5SpQ8A?pwd=rmec).
The file structure is like:

```
├── sjtupcqa_wpc_wpc2.0_6face
│   ├── hhi_0.ply
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── 3.png
│   │   ├── 4.png
│   │   ├── 5.png
...
```

Then use `train.sh' to conduct your own training !
The pretrained `swin_tiny_patch4_window7_224_22k.pth' can be downloaded at [here](https://1drv.ms/u/s!AjaDoj_-yWgggttCGTboscTuOExeVQ?e=ECpGtR).
## Citation
If you find our work useful, please give us star and cite our paper as:
```
@article{zhang2023gms,
  title={Gms-3dqa: Projection-based grid mini-patch sampling for 3d model quality assessment},
  author={Zhang, Zicheng and Sun, Wei and Wu, Haoning and Zhou, Yingjie and Li, Chunyi and Chen, Zijian and Min, Xiongkuo and Zhai, Guangtao and Lin, Weisi},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2023},
  publisher={ACM New York, NY}
}
```
