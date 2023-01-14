替换`s3prl/downstream/runner.py`

- 函数`load_features`生成/加载embedings
- 🎈如果分别在train、dev和test的时候生成embeddings，会出现显存不足的问题--->因此，直接在train的过程中生成embeddings
  > train/dev -- EN: 设置evaluate_ratio(config中) = 0
  > test -- SP: 将config中的train、test路径互换
  > 🎈注意，npz文件保存的路径
