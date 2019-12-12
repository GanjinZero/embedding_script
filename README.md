# embedding-script
一个简单的在大的文档进行word2vec和glove embedding计算的python脚本。

# 使用说明

Word2Vec:
```shell
python train_word2vec.py -f ../text/ -o ./word2vec/ -d 300 -c 12 -i 2 -s 2 -u ./dict.txt
```

Glove:
```shell
python train_glove.py -f ../text/ -o ./word2vec/ -d 300 -c 12 -i 2 -s 2 -u ./dict.txt
```

# 参数解释
参数|解释
-|-
-f|要计算embedding的文本文件或文件夹
-o|保存的embedding模型文件名或目录（如果是目录会自动指定文件名）
-d|embedding维数
-c|使用的CPU核数
-i|文本迭代次数
-s|是否使用分词，0代表不分词（已经用空格分好了词或者做字级别embedding），1使用jieba分词，2使用jieba分词加上自定义词典
-u|自定义词典地址

# 依赖
- gensim: 用于Word2Vec模型训练，通过`pip install gensim`安装。
- jieba: 用于分词，通过`pip install jieba`安装。
- pyserverchan: 用于训练完进行微信通知，通过`pip install pyserverchan`安装，需要去Server酱官网配置一下用户名，不需要微信推送功能可以将该部分代码注释。
- glove\_python: 用于Glove模型训练，通过`pip install glove_python`安装。

