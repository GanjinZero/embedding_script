import argparse
import os
import time
from pyserverchan import pyserver
import gensim
from gensim.models import Word2Vec
import jieba


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_path")
parser.add_argument("-o", "--output_path")
parser.add_argument("-w", "--window_size", type=int, default=5)
parser.add_argument("-d", "--embedding_size", type=int, default=300)
parser.add_argument("-c", "--cpu_count", type=int, default=4)
parser.add_argument("-i", "--iters", type=int, default=2)
parser.add_argument("-s", "--use_segment", type=int, default=0)

args = parser.parse_args()
if os.path.isfile(args.file_path):
    file_path_list = [args.file_path] 
else:
    file_path_list = []
    for root, dirs, files in os.walk(args.file_path):
        for f in files:
            now_file_path = os.path.join(root, f)
            if os.path.isfile(now_file_path):
                file_path_list.append(now_file_path)

print(f"Found {len(file_path_list)} file(s) under {args.file_path}")
start_time = time.time()

gensim.models.Word2Vec.FAST_VERSION = 1

# args.use_segment: 0-not use 1-use_jieba 2-use_jieba_with_ditionary`
if args.use_segment == 2:
    jieba.load_userdict("../medical_term_judger/dictionary/all_dictionary.txt")    

class Text(object):
    def __init__(self, file_path_list):
        self.file_path_list = file_path_list

    def __iter__(self):
        file_count = 0
        for file_path in file_path_list:
            file_count += 1
            print(f"Now file name:{file_path}, now file count:{file_count}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if args.use_segment == 0:
                        yield list(line.strip())
                    else:
                        yield list(jieba.cut(line.strip()))

sentences = Text(file_path_list)
model = Word2Vec(sentences, size=args.embedding_size, window=args.window_size, iter=args.iters, workers=args.cpu_count)
end_time = time.time()
use_time = round(end_time - start_time, 2)

model_name = args.output_path
if not os.path.isdir(model_name):
    model.save(model_name)
else:
    model_name = os.path.join(model_name, f"word2vec_{args.window_size}_{args.embedding_size}_{args.use_segment}.model")
    model.save(model_name)

log_text = f"Job_name:Word2Vec, Use_time:{str(use_time)}, Save_path:{model_name})"
svc = pyserver.ServerChan("https://sc.ftqq.com/SCU40533T8d2ed497f1ac4a681a08a566d986e8ac5c3c258135d4e.send")
svc.output_to_weixin(log_text)

