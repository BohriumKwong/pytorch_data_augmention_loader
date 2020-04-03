# 基于pytorch下使用的能同时实现增强和增量的dataloader #

## 主要改进功能 ##

1. 基于pytorch 官方的提供的data.Dataset类实现一个自己的数据加载器

2. 该数据加载器可以实现内部真正意义上的数据增量增强(官方的数据增强只是进行图像转换但实际上的数量并没有增加)

3. 除了支持传入数据根目录之外,还支持传入读取好的图像列表作为代替,这样的话无需在目录和路径上对数据有所限制;甚至只需要自己分别生成**train**和**val**的`list`分别传入就实现了不同数据集的读取,而不需要手工在根目录上设置单独的**train**和**val**文件夹

****

##  Installation ##

```pip install -r requirements.txt```
****

## 主脚本 ##

> **data_augmention_loader.py**


## Run ##

创建data set类和dataloader的语句如下:

```python
import torch.utils.data as data
from data_augmention_loader import augmention_dataset

sub_dir = 'dataloader_test'
train_dset = augmention_dataset(sub_dir = sub_dir,class_to_idx = None, image_list = None,
                                transform=train_trans)
train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=32, shuffle=False,
    num_workers=0, pin_memory=False)
# 因为我写的代码中已经自带数据打乱功能,所以上述的shuffle=False可以设置为False
# 如果是在docker中运行时需注意,因为容器设定的shm内存不够会出现相关报错,此时将num_workers设为0则可
```
然后设置`train_dset`的方法如下:

```python
train_dset.shuffle_data(True)#通过调用这个方法可以将整个数据集打乱
train_dset.setmode(2)
train_dset.maketraindata(3)
```
其中`setmode(2)`是将数据集设置为训练模式,只有在这个模式下才能进行数据增强的扩展。具体可参考**data_augmention_loader.py**代码。

之后调用`maketraindata(3)`可以实现额外3倍的增强,传参的数字代表额外增强的倍数(一般要求是奇数,传参不是奇数也会处理为奇数)。



## testing ##
已经在`main()`函数设置好相关的运行demo,直接运行`data_augmention_loader.py`则可。

#### 输出如下 ####

Finded class:cat,total 200 files.

Finded class:dog,total 200 files.

Number of images: 400

Augmention data_loader testing: 100%|█████████████████████| 50/50 [00:08<00:00,  5.78it/s, test info :((32, 3, 224, 224), 0.5625)]

Bacth count is 50, len of dataset is 1600

> **原来的只有400张图片,但是用我这个脚本进行扩展之后,在batch=32的情况下能读取50个batch,约扩展到1600张。**
