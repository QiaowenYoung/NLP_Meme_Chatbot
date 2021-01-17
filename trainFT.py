import fasttext
import os, csv
import numpy as np

path_train = "/Users/qiaowenyang/Desktop/newpj/fastText/smalltrain.csv"
path_test = "/Users/qiaowenyang/Desktop/newpj/fastText/smalltest.csv"

f = open(path_train, 'r')
reader = csv.reader(f)
ft_trainset = list(reader)

f = open(path_test, 'r')
reader = csv.reader(f)
ft_testset = list(reader)

def train_model(ipt=None, opt=None, model='', dim=100, epoch=5, lr=0.1, loss='softmax', ngram=2):
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
                                         lr=lr, wordNgrams=ngram, loss=loss)
        """
          训练一个监督模型, 返回一个模型对象
          
          @param input:           训练数据文件路径
          @param lr:              学习率
          @param dim:             向量维度
          @param ws:              cbow模型时使用
          @param epoch:           次数
          @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
          @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
          @param minn:            构造subword时最小char个数
          @param maxn:            构造subword时最大char个数
          @param neg:             负采样
          @param wordNgrams:      n-gram个数
          @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
          @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
          @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
          @param lrUpdateRate:    学习率更新
          @param t:               负采样阈值
          @param label:           类别前缀
          @param verbose:         ??
          @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
          @return model object
        """
        classifier.save_model(opt)
    return classifier

dim = 200
lr = 1
epoch = 1000
ngram = 4
model = f'data_dim{str(dim)}_lr{str(lr)}_iter{str(epoch)}_ngram{ngram}.model'
ipt = 'smalltrain.csv'

classifier = train_model(ipt=ipt, opt=model, model=model, dim=dim, epoch=epoch, lr=lr, ngram=ngram)
result = classifier.test('smalltest.csv')
print(f"dim: {dim}, lr: {lr}, epoch: {epoch}")
print("(测试数据量，precision，recall):")
print(result)
