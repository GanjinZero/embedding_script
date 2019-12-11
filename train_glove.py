from glove import Glove, Corpus
import argparse
import os
import time
from pyserverchan import pyserver
# import ipdb


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_path")
parser.add_argument("-o", "--output_path")
parser.add_argument("-w", "--window_size", type=int, default=10)
parser.add_argument("-d", "--embedding_size", type=int, default=300)
parser.add_argument("-c", "--cpu_count", type=int, default=4)
parser.add_argument("-i", "--iters", type=int, default=2)

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
                    yield list(line.strip())

sentences = Text(file_path_list)
corpus_model = Corpus()
corpus_model.fit(sentences, window=args.window_size)

model = Glove(no_components=args.embedding_size)
model.fit(corpus_model.matrix, epochs=args.iters, no_threads=args.cpu_count)
model.add_dictionary(corpus_model.dictionary)

end_time = time.time()
use_time = round(end_time - start_time, 2)

model_name = args.output_path
if not os.path.isdir(model_name):
    model.save(model_name)
else:
    model_name = os.path.join(model_name, f"glove_{args.window_size}_{args.embedding_size}.model")
    model.save(model_name)

log_text = f"Job_name:Glove, Use_time:{str(use_time)}, Save_path:{model_name})"
svc = pyserver.ServerChan("https://sc.ftqq.com/SCU40533T8d2ed497f1ac4a681a08a566d986e8ac5c3c258135d4e.send")
svc.output_to_weixin(log_text)

