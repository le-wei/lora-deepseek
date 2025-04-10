
#### 1、训练
torchrun --nproc_per_node=4 train.py --distributed True   #分布式训练
torchrun --nproc_per_node=1 train.py --distributed False   #非分布式训练
### 2、模型合并
python merge_model.py   #注意模型路径
### 3、测试
python test.py   #注意模型路径

参考网页：
https://www.cnblogs.com/shanren/p/18719435
https://zhuanlan.zhihu.com/p/17628689019
https://zhuanlan.zhihu.com/p/25054526736