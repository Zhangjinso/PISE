# PISE

The code for our CVPR paper [PISE: Person Image Synthesis and Editing with Decoupled GAN](https://arxiv.org/abs/2103.04023), [Project Page](http://cic.tju.edu.cn/faculty/likun/projects/PISE/index.html), [supp.](http://cic.tju.edu.cn/faculty/likun/projects/PISE/assets/supp.pdf)

# Requirement

```
conda create -n pise python=3.6
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install scikit-image pillow pandas tqdm dominate natsort 
```

# Data

Data preparation for images and keypoints can follow [Pose Transfer](https://github.com/tengteng95/Pose-Transfer) and [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention/blob/master/PERSON_IMAGE_GENERATION.md).



1. Download [deep fashion dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). You will need to ask a password from dataset maintainers. Unzip 'Img/img.zip' and put the folder named 'img' in the './fashion_data' directory.

2. Download train/test key points annotations and the dataset list from [Google Drive](https://drive.google.com/open?id=1BX3Bxh8KG01yKWViRY0WTyDWbJHju-SL), including **fashion-pairs-train.csv**, **fashion-pairs-test.csv**, **fashion-annotation-train.csv**, **fashion-annotation-train.csv,** **train.lst**, **test.lst**. Put these files under the  `./fashion_data` directory.

3. Run the following code to split the train/test dataset.

   ```
   python data/generate_fashion_datasets.py
   ```

4. Download parsing data, and put these files under the  `./fashion_data` directory. Parsing data for testing can be found from [baidu](https://pan.baidu.com/s/19boQPJnrq2wASSMqzl27NQ) (fectch code: abcd) or [Google drive](https://drive.google.com/file/d/1AcK4fuYOZw0i2Gi_X7kGdO3ffosIIUnj/view?usp=sharing).
   Parsing data for training can be found from [baidu](https://pan.baidu.com/s/1WHWk2Kz2JUEyFXC-g_LnvA) (fectch code: abcd) or [Google drive](https://drive.google.com/file/d/1dmW1NX9UZS8jTEjhP3364ktbSVIespIU/view?usp=sharing). You can get the data follow with [PGN](https://github.com/Engineering-Course/CIHP_PGN), and re-organize the labels as you need.



# Train

```
python train.py --name=fashion --model=painet --gpu_ids=0
```
**Note that if you want to train a pose transfer model as well as texture transfer and region editing, just comments the line 177 and 178, and uncomments line 162-176.**

**For training using multi-gpus, you can refer to [issue in GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention/issues/22)** 


# Test

You can directly download our test results from [baidu](https://pan.baidu.com/s/16HiFP6hExXVSzbs9A_Bhbw) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1u62gyQ46_qZGB6BlESpk0WLjcZ-NH8-F/view?usp=sharing). <br>
Pre-trained checkpoint of human pose transfer reported in our paper can be found from [baidu](https://pan.baidu.com/s/14v3LaCCGCHJUoqQ_wlyNpA) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1gcdzahJ-pE-bSQfcnrW__iXIViH_y-FB/view?usp=sharing) and put it in the folder (-->results-->fashion). 

Pre-Trained checkpoint of texture transfe, region editing, style interpolation used in our paper can be found from [baidu](https://pan.baidu.com/s/1E025k57INvL0O8cdLi87og)(fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1fMFBIkU1AEQaa3vbhba3oV0rU5YSr7GR/view?usp=sharing). Note that the model need to be changed.

**Test by yourself** <br>


```
python test.py --name=fashion --model=painet --gpu_ids=0 
```


# Citation

If you use this code, please cite our paper.

```
@InProceedings{Zhang_2021_CVPR,
    author    = {Zhang, Jinsong and Li, Kun and Lai, Yu-Kun and Yang, Jingyu},
    title     = {{PISE}: Person Image Synthesis and Editing With Decoupled GAN},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {7982-7990}
}
```

# Acknowledgments

Our code is based on [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention).
