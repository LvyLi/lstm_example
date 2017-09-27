#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}


# Set the random number generators' seeds for consistency.
SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def _p(pp, name):
    """
    初始化
    :param pp:
    :param name:
    :return:
    """
    return '%s_%s' % (pp, name)

def zipp(params, tparams):
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    对数据Shuffle随机排列，首先获取数据集所有句子数n，按照batch_size对所有句子划分处n/batch_size+1个列表
    拼在一起构成一个二重列表，返回一个zip(batch索引，Batch内容)
    :param n:
    :param minibatch_size:
    :param shuffle:
    :return:
    """
    idx_list = numpy.arange(n, dtype='int32')

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0

    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start : minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def get_layer(name):
    """
    调用layer={'lstm':}

    """
    fns = layers[name]
    return fns

def ortho_weight(ndim):
    """
    生成正交矩阵，先生成一个n*n的矩阵，然后进行svd分解

    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def init_params(options):
    '''
    Global parameter. For the embedding and the classifier.
    embedding, softmax和lstm层的参数不是一起初始化的，这里先初始化了embedding和softmax分类器的参数
    :param options:
    :return:
    '''

    params = OrderedDict()

    # 随机生成embedding矩阵，这里为10000*128维的，因为词典的大小是10000，也就是说词的ID范围是1-10000，
    # 我们将每个词转换成一个128维的向量，所以这里生成了一个10000*128的矩阵，每个词转换成它的ID的那一行的128
    # 维向量。比如“我”这个词的ID是5，那么“我”就用param['Wenb']矩阵的第5行表示，是一个随机生成的128维向量。
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    # 调用get_layer和layer，返回layer['lstm']的第一项param_init_lstm函数
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])

    # classifier 初始化softmax分类器的参数
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params

    初始化LSTM的参数，输入门、输出门、遗忘门和记忆单元的参数维度表示相同，所以一同初始化，运算的时候再分开
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U

    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    """
    lstm的核心功能，隐藏层计算
    :param tparams:
    :param state_below:
    :param options:
    :param prefix:
    :param mask:
    :return:
    """
    # state_below是输入x和w,b计算后的输入节点。第一维代表step
    nsteps = state_below.shape[0]
    # 如果输入三维的x,那么样本数就是第二维的长度，否则就是只有一个样本
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask 非 None
    assert mask is not None

    def _slice(_x, n, dim):
        """
        切片，计算的时候几个门一起计算，切片将各个门分开
        :param _x:
        :param n:
        :param dim:
        :return:
        """
        if _x.ndim == 3:
            return _x[:, :, n + dim:(n + 1 ) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        """
        隐藏层计算，i:输入门，f:忘记门，o:输出门，c:cell
        :param m_: mask
        :param x_: x
        :param h_: 上一时刻隐藏层输出
        :param c_: 上一时刻cell输出
        :return:
        """
        # 将前一时序h与4倍的U矩阵并行计算，并加上4倍化Wx的预计算
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        # 当m_为0是，第一项为0，c会一直保持前一时刻c_的状态，当m_中元素为1，c保持计算的结果不变，对于长度小于
        # maxlen的句子，空位补0，但是在这些0位置处，memory cell的状态采用了句子最后一个单词计算的状态进行填充
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    # 因为每一个step都需要做Wx+b运算，预计算好所有step，并行量大
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']

    # 循环执行_step函数,它的输入是m_, x_, h_, c_，迭代的是sequences中的mask，state_below，
    # 每次拿出他们的一行，作为输入的m_和x_，h_和c_的初始值设置为0（outputs_info设置初始值），
    # 每次计算，_step返回的h和c分别赋给下次迭代的h_和c_，迭代次数为nsteps，这样就实现了隐藏层节点的传递，
    # 最后函数返回h和c给rval
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    # 每个step里，h结果是一个2D矩阵，[BatchSize, Emb_Dim]
    # 而rval[0]是一个3D矩阵，[n_step, BatchSize, Emb_Dim]
    return rval[0]

# ff: Feed Forward, only useful to put after lstm before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def get_dataset(name):
    return datasets[name][0], datasets[name][1]

def init_tparams(params):
    """
    将参数转化为theano.shared类型，参数从params替换为tparams
    :param params:
    :return:
    """
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer
    :param lr:
    :param tparams:
    :param grads:
    :param x:
    :param mask:
    :param y:
    :param cost:
    :return:
    """
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    """
    建立模型
    """
    # 随机数生成器
    trng = RandomStreams(SEED)

    # dropout时的选项
    use_noise = theano.shared(numpy_floatX(0.))

    # 为x,y,mask生成占位符
    x = tensor.matrix('x',dtype='int64')
    mask = tensor.matrix('mask',dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    # x的行代表steps
    n_timesteps = x.shape[0]
    # x的列代表不同的样本
    n_samples = x.shape[1]

    # 将词用向量表示，.flatten(ndim=1)返回原始变量的一个view，将变量降为ndim维
    # emb转换为一个三维矩阵[n_step, BitchSize, Emb_Dim]
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_proj']])
    # 隐藏层的计算，这里调用了lstm_layer函数
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    # 计算样本每个时刻的h的均值，对proj的n_step维度求和，如果这个状态有值，那么相应的mask值为1，否则就是0，
    # 然后除以mask每列的和，也就是样本一共的step个数，求出平均值
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    # 如果使用dropout，调用dropout_layer随机丢弃一些隐藏层
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    # pred为预测值，隐藏层h的均值输入softmax得到的
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    # 将预测输出编译成x,mask的函数
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    # 损失函数
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def train_lstm(
        dim_proj = 128, # word embedding的维数和隐藏层的维数
        patience = 10, # 用于earlystop,如果10轮迭代的误差没有降低，就进行earlystop.
        max_epochs = 5000, # 迭代次数
        dispFreq = 10, # 每更新10次显示训练过程（训练、验证、测试误差）
        decay_c=0, # 参数U的正则权重，U为隐藏层到输出层的参数
        lrate=0.0001, # sgd用的学习率
        n_words = 10000, # 词典大小，用于数据预处理部分，将词用该词在词典中的ID表示，超过10000的用1表示
        optimizer=adadelta, # 优化方法，代码提供了sgd,adatdlta和rmsprop
        encoder='lstm', # 一个标识符
        saveto='lstm_model.npz', # 保存最好模型的文件，保存训练误差，验证误差和测试误差等
        validFreq=370, # 验证频率
        saveFreq=1110, # 保存频率
        batch_size=16, # 训练的batch大小
        valid_batch_size=64, # 验证集用的batch大小
        maxlen = 100, # 序列的最大长度，超出长度的数据被抛弃，数据预处理
        dataset = 'imdb', # 用于数据预处理的参数，全局变量datasets的key'imdb'的value为两个处理数据的函数

        # Parameter for extra option.
        test_size=-1, # If >0, we keep only this number of test example.
        reload_model=None, # Path to a saved model we want to start from.
        noise_std=0,
        use_dropout=True,

):
    # 将当先的函数局部的参数copy到字典model_options中，作为参数传递
    model_options = locals().copy()
    print("model_options", model_options)

    load_data, prepare_data = get_dataset(dataset) # dataset = 'imdb'

    print ('Loading data...')
    # 获取预处理数据的函数
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    if test_size >0:
        # 如果设置了test_size >0,从测试集中随机找出test_size个作为测试数据。
        # 如果没有设置test_size，会用所有的测试集数据做测试。
        # 原来的测试集是根据长度排列的，这里做了一次打散。
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    # ydim为标签y的维数，因为是从0开始的，所以后面+1，并将它加入模型参数中
    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim
    print ('Building model')

    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options) # 参数初始化

    # 如果设置了reload，将从保存的文件中加载参数
    if reload_model:
        load_params('lstm_model.npz', params)
    # init_tparams将上一步初始化的参数转化为theano.shared类型
    tparams = init_tparams(params)

    # 建立模型
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    # 如果加入正则，在损失函数里加上L2损失
    if decay_c > 0:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    # 编译损失函数
    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    # 求导，编译
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)
    print ('Optimization')

    # 将验证集和测试集分成batches，返回batchID和对应的样本序号
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print ("%d train examples" % len(train[0]))
    print ("%d valid examples" % len(valid[0]))
    print ("%d test examples" % len(test[0]))

    history_errs = []  # 记录误差
    best_p = None  # 记录最好的结果
    bad_count = 0  # 计数

    # 如果未设置验证频率和保存频率，就设置为一个epoch
    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # 记录更新的次数
    estop = False  # early stop
    start_time = time.time()

    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # 将训练集随机分成batches
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1  # 更新次数+1
                use_noise.set_value(1.)  # 设置dropout

                # 从minibatch里读取y,x
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # 获取数据，调用prepare_data函数
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                # 判断cost是否超出范围
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print ('Epoch', eidx, 'Update', uidx, 'Cost', cost)
                    return 1., 1., 1.
                # 判断是否到了显示频率
                if numpy.mod(uidx, dispFreq) == 0:
                    print ('Epoch', eidx, 'Updata', uidx, 'Cost', cost)
                # 判断是否到了保存频率
                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print ('Saving.....')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)

                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print ("Done")
                # 判断是否到了验证频率
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    # 记录验证误差和测试误差
                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,0].min()):
                        best_p = unzip(tparams)
                        bad_count = 0

                    print ('Train', train_err, 'Valid', valid_err, 'Test', test_err)
                    # 如果当前的验证误差大于前patience次验证误差的最小值（误差没有降低）
                    # bad_counter计数+1，当bad_counter>patience， 就early stop
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,0].min()):
                        bad_count +=1
                        if bad_count > patience:
                            print ("Early Stop!")
                            estop = True
                            break

            print ('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print ('Training interupted')

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print ('Train', train_err, 'Valid', valid_err, 'Test', test_err)
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print ('The code run for %d epochs , with %f sec/epochs'
           % (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))

    print (('Training took %.1fs' % (end_time - start_time)), file=sys.stderr)

    return train_err, valid_err, test_err


if __name__ == '__main__':
    train_lstm(
        max_epochs=100,
        test_size=500
    )