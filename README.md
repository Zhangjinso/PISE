# PISE





# Requirement

```
conda create -n pise python=3.6
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install scikit-image pillow pandas tqdm dominate natsort 
```



# Data

Data preparation for images and keypoints can follow [Pose Transfer](https://github.com/tengteng95/Pose-Transfer)


Parsing data can be found from [baidu](https://pan.baidu.com/s/19boQPJnrq2wASSMqzl27NQ) (fectch code: abcd) or [Google drive](https://drive.google.com/file/d/1AcK4fuYOZw0i2Gi_X7kGdO3ffosIIUnj/view?usp=sharing).



# Train

```
python train.py --name=fashion --model=painet --gpu_ids=0
```



# Test

You can directly download our test results from [baidu](https://pan.baidu.com/s/16HiFP6hExXVSzbs9A_Bhbw) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1u62gyQ46_qZGB6BlESpk0WLjcZ-NH8-F/view?usp=sharing). <br>
Pre-trained checkpoint reported in our paper can be found from [baidu](https://pan.baidu.com/s/14v3LaCCGCHJUoqQ_wlyNpA) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1gcdzahJ-pE-bSQfcnrW__iXIViH_y-FB/view?usp=sharing) and put it in the folder (-->results-->fashion). 

**Test by yourself** <br>


```
python test.py --name=fashion --model=painet --gpu_ids=0 
```




# Citation

If you use this code, please cite our paper.

```
@inproceedings{PISE,
  title={{PISE}: Person Image Synthesis and Editing with Decoupled GAN},
  author={Jinsong, Zhang and Kun, Li and Yu-Kun, Lai and Jingyu, Yang},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

# Acknowledgments

Our code is based on [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention).









