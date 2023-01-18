1. `runner.py`替换`s3prl/downstream/runner.py`(用于保存embeddings)

- 函数`load_features`生成/加载embedings
- 🎈如果分别在train、dev和test的时候生成embeddings，会出现显存不足的问题--->因此，直接在train的过程中生成embeddings
  > train/dev -- EN: 设置evaluate_ratio(config中) = 0  
  > test -- SP: 将config中的train、test路径互换  
  > 🎈注意，npz文件保存的路径 需指定

2. `data_aug.ipynb`(用于data aug)
- 可以设置aug后保存的路径
- 保存的aug后的wav文件命名：原来的名字 + i（第几个segment）
- 可以选择aug的模式（default=“random”）和个数（default=1）

3. `emotion_train.ipynb`(用于生成emotion所需的json文件)
- 如果将aug的文件放到Data/train文件夹下（英语和西班牙语在一起），不用更改get_train_json()
- 如果将aug的文件与原先的英语文件分开，可先将train/下的英语文件新建一个文件夹保存，然后参考get_test_json()
