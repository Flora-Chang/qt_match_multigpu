import sys

with open("../data/dev_with_score.txt", "r") as f, open("../data/dev_pool.txt", "w") as out_f:
    for line in f:
        ori_line = line.strip()
        line = line.strip().split("\t")
        pos_score = float(line[-2])
        neg_score = float(line[-1])
        if pos_score < neg_score:
            out_f.write(ori_line + '\n')