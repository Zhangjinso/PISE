# PISE





# Requirement

```
conda create -n pise python=3.6
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install scikit-image pillow pandas tqdm dominate natsort 
```



# Data

Data preparation for images and keypoints can follow [Pose Transfer](https://github.com/tengteng95/Pose-Transfer)


Parsing data can be found from [baidu](https://pan.baidu.com/s/19boQPJnrq2wASSMqzl27NQ) (fectch code: abcd).



# Train

```
python train.py --name=fashion --model=painet --gpu_ids=0
```



# Test

You can directly download our test results from [baidu](https://pan.baidu.com/s/16HiFP6hExXVSzbs9A_Bhbw) (fetch code: abcd). <br>
Pre-trained checkpoint reported in our paper can be found from [baidu](https://pan.baidu.com/s/14v3LaCCGCHJUoqQ_wlyNpA) (fetch code: abcd) and put it in the folder (-->results-->fashion). 

**Test by yourself** <br>


```
python test.py --name=fashion --model=painet --gpu_ids=0 
```




# Citation

If you use this code, please cite our paper.



# Acknowledgments

Our code is based on [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention).









