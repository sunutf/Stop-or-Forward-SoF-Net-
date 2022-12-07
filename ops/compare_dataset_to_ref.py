import os
import time 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir_path", type = str)
parser.add_argument("txt_path", type=str)

args = parser.parse_args()

if __name__ == "__main__":
    dir_path = args.dir_path
    txt_path = args.txt_path
    

    txts = [x.strip().split(',')[0] for x in open(txt_path).readlines()]
    files = [os.path.splitext(x)[0] for x in os.listdir(dir_path)]
    print(files[0]) 

    total_videos = len(files)
    total = len(txts)
    remain = total
    for target in txts :
        if target not in files:
            print("target : %s" % target )
        else :
            remain -=1

    print("ref :%d --- %d / %d" % (total_videos, total, remain) )
