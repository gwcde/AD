#### 所作修改

1. json文件需要自己创建

   > 格式参考AD/Data/English.json

   🎈`IEMOCAP_process.py`未作修改，不适用目前数据集

2. yaml中路径需要修改

   > - 将原来的`root`改为`train_root`和`test_root`
   >
   > - 新增`train_path`、`test_path`用于记录json路径

3. 修改`dataset.py`中部分代码，设置指定大小

   > 截取960000， train_batch_size = 2 显存7.+G
   >
   > ​						  train_batch_size = 4 显存12G

   🎈 函数`_load_wav`返回`Tensor`

4. 修改了`expert.py`中部分关于路径的代码



#### 执行

```python
python run_downstream.py -n ExpName -m train -u hubert_large_ll60k -d emotion -c emotion/emotion_config.yaml -o "config.downstream_expert.datarc.test_fold='fold1'"
```

