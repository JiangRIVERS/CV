"""
Filename: CDSSM.py

A CDSSM(CLSM) implement based on Pytorch


reference https://github.com/nishnik/Deep-Semantic-Similarity-Model-PyTorch/blob/master/cdssm.py

@author: Jiang Rivers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
英文的处理方式（word-trigram letter-trigram）在中文中并不可取，
因为英文中虽然用了 word-ngram 把样本空间拉成了百万级，
但是经过 letter-trigram 又把向量空间降到可控级别，只有 3*30K（9 万）。
而中文如果用 word-trigram，那向量空间就是百万级的了，
显然还是字向量（1.5 万维）比较可控。
"""
# 网络参数设置
LETTER_GRAM_SIZE=3 # 每个letter被分为n-gram
WINDOW_SIZE=3 # 滑动窗口,将每个句子分为n-gram

TOTAL_LETTER_GRAMS=3e4 # 总共的letter gram的维度
WORD_DEPTH=WINDOW_SIZE*TOTAL_LETTER_GRAMS #输入的维度

HT=300 # Convolution layer ht维度
SE=128 # Semantic layer se 维度
J=4 # 反例个数

# 网络结构
def max_pool(x):
    # 输入为tensor
    r, idx = x.topk(1, dim=2)  # 在dim=2上的top1
    return r


class CDSSM(nn.Module):
    """
    输入query为tensor,尺寸应为[batch_size,每个entity的word_tri_gram表示,query中entity的个数] 例如:[1,90k,5]
    """

    def __init__(self):
        super(CDSSM, self).__init__()

        self.query_conv = nn.Conv1d(WORD_DEPTH, HT, kernel_size=1)
        self.query_fc = nn.Linear(HT, SE)

        self.doc_conv = nn.Conv1d(WORD_DEPTH, HT, kernel_size=1)
        self.doc_fc = nn.Linear(HT, SE)
        self.cosine = nn.CosineSimilarity(dim=2)


    def forward(self, q, pos, negs):
        """
        :params q: query 例如:[1,90k,5]
        :params pos: doc中的正例，即和query相似的doc
        :params negs: doc中的反例，反例会有多个
        """
        # Query
        q_ht = torch.tanh(self.query_conv(q))
        q_v = max_pool(q_ht)
        q_y = torch.tanh(self.query_fc(q_v.permute((0, 2, 1))))  # 对Tensor进行转置并放入fc层中

        # 正例
        pos_ht = torch.tanh(self.doc_conv(pos))
        pos_v = max_pool(pos_ht)
        pos_y = torch.tanh(self.doc_fc(pos_v.permute((0, 2, 1))))

        # 反例
        negs_ht = [torch.tanh(self.doc_conv(neg)) for neg in negs]
        negs_v = [max_pool(neg_ht) for neg_ht in negs_ht]
        negs_y = [torch.tanh(self.doc_fc(neg_v.permute(0, 2, 1))) for neg_v in negs_v]

        # 匹配层
        R_list = [self.cosine(q_y, pos_y)]
        R_list += [self.cosine(q_y, neg_y) for neg_y in negs_y]

        R_vec = torch.stack(R_list)  # 维度为[J+1,Batch_size,Batch_size]，J为反例数
        # prob=F.softmax(with_gamma,dim=0) # 之后使用nn.CrossEntropy，因而这一步省略softmax

        return R_vec.view(-1,J+1)

# demo实验
if __name__=='__main__':
    # 数据制作
    import numpy as np

    Batch_size = 1
    query = torch.randn((Batch_size, 100, 5))
    pos = torch.randn((Batch_size, 100, 20))
    negs = []
    for i in range(J):  # J个反例
        neg_len = np.random.randint(10, 30)
        neg = torch.torch.randn((Batch_size, 100, neg_len))
        negs.append(neg)

    label = torch.zeros(Batch_size)  # 第0维是优化目标
    label = label.long()  # CrossEntropy 的target需要是long型

    # 跑模型
    model = CDSSM()
    r = model(query, pos, negs)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(r, label)
    loss.backward()
