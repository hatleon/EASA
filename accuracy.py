
import argparse
from collections import Counter
import numpy as np
from numpy import linalg as LA
# import torch.nn.functional as F
import os


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--res_dir', type=str, default='./res')  # 向量输出文件夹
    parser.add_argument('--embedding_dim', type=int, default=100)
    # parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=100)
    # parser.add_argument('--batch_size', type=int, default=4800)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_generator', type=int, default=4)
    # parser.add_argument('--n_generator', type=int, default=8)
    # parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=3000)
    # parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--context_size', type=int, default=100)  # add
    args = parser.parse_args()
    # print(args)

    print("train over")

    # embedding in
    ent_vecs = []
    with open(os.path.join(args.res_dir, "entity2vec.bern_my2"), "r", encoding="utf-8") as fp:
        for line in fp:
            vec = [float(i) for i in line.strip().split()]
            ent_vecs.append(np.array(vec))

    # rel_vecs = []
    # with open(os.path.join(args.res_dir, "relation2vec.bern"), "r", encoding="utf-8") as fp:
    #     for line in fp:
    #         vec = [float(i) for i in line.strip().split()]
    #         rel_vecs.append(np.array(vec))

    # from tensorboardX import SummaryWriter
    # # https://blog.csdn.net/JNingWei/article/details/79740825
    # writer = SummaryWriter()
    # ---------------cal accuracy and plot------------------
    # step4
    cnt = 0
    rank = 0
    hits10 = hits1 = 0

    baidu_idxs = []
    hudong_idxs = []
    with open(os.path.join(args.data_dir, "baidu2id.txt"), "r", encoding="utf-8") as fp:
        for line in fp:
            b_idx = line.strip().split()[1]
            baidu_idxs.append(int(b_idx))
    pair_id_dict = {}
    with open(os.path.join(args.data_dir, "hudong2id.txt"), "r", encoding="utf-8") as fp:
        for line in fp:
            h_idx = line.strip().split()[1]
            hudong_idxs.append(int(h_idx))
    with open(os.path.join(args.data_dir, "pair2id.txt"), "r", encoding="utf-8") as fp:
        for line in fp:
            b_idx = int(line.strip().split()[0])
            h_idx = int(line.strip().split()[1])
            pair_id_dict[b_idx] = h_idx

    num_b = len(baidu_idxs)
    print(len(ent_vecs))
    for i in baidu_idxs:
        distance = []  # pos
        for j in hudong_idxs:
            dis = LA.norm(ent_vecs[i] - ent_vecs[j], 1)  # numpy L1
            distance.append((dis, i, j))

        distance.sort()  # sort
        for k in range(10):
            (dis, i, j) = distance[k]
            if i == pair_id_dict[j]:
                hits10 += 1
                break

        for k in range(1):
            (dis, i, j) = distance[k]
            if i == pair_id_dict[j]:
                hits1 += 1

        for k in range(len(distance)):
            (dis, i, j) = distance[k]
            if i == pair_id_dict[j]:
                rank += k + 1  # rank sum
                cur_rank = k + 1
                break
        cnt += 1
        if cnt % 1000 == 0:
            print("align info" + str(cnt))
            print(cur_rank, float(hits10) / cnt, float(hits1) / cnt, float(rank) / cnt, cnt)
            print("\n")
        # if cnt == 2563:
        if cnt == num_b:
            print("\naccuracy\n")
            print(cur_rank, float(hits10) / cnt, float(hits1) / cnt, float(rank) / cnt, cnt)

    #     writer.add_scalar("result/cur_rank", cur_rank, cnt)
    #     writer.add_scalar("result/hits10", float(hits10) / cnt, cnt)
    #     writer.add_scalar("result/hits1", float(hits1) / cnt, cnt)
    #     writer.add_scalar("result/rank", rank, cnt)
    # writer.close()

    # plot


if __name__ == '__main__':
    main()
