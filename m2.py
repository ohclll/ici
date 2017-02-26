# __author__ = 'lllcho'
# __date__ = '2017/2/18'
import sys
import time
import os.path as osp
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from utils import *

tf.set_random_seed(0)
np.random.seed(0)


class LastNTimeLayer(tl.layers.Layer):
    def __init__(self, layer, ntime=2, name='lastntime'):
        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = tf.reshape(self.inputs[:, -ntime:], [-1, ntime * int(self.inputs.get_shape()[2])])

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


def get_sample(input_len=14):
    npzfile = np.load(osp.join('dataset', 'pay_ftr.npz'))
    pay_cnt, = [npzfile[name] for name in ['arr_{}'.format(i) for i in range(1)]]
    train_idx=list(range(67*7,68*7))
    train_idx.append(69*7)

    x_data=[]
    y_data=[]
    for idx in train_idx:
        x_data.append(pay_cnt[:,idx-input_len:idx])
        y_data.append(pay_cnt[:,idx:idx+7].sum(axis=2))
    x_data=np.concatenate(x_data)
    y_data=np.concatenate(y_data)
    x_test=pay_cnt[:,-input_len:]
    return [x_data,y_data],x_test


def get_model(n_step):
    pay_cnt=tf.placeholder(dtype=tf.float32, shape=[None, n_step, 12])

    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 7])

    inputs_ = [pay_cnt]

    pay_cnt_in = tl.layers.InputLayer(inputs=pay_cnt, name='input_pay_cnt')
    pay_cnt_in2 = tl.layers.InputLayer(inputs=tf.reshape(tf.reduce_mean(pay_cnt,axis=-1),(-1,n_step//7,7)),
                                       name='input_pay_cnt2')
    rnn1 = RNNLayer(pay_cnt_in,
                   cell_fn=GRUCell,
                   n_hidden=64,
                   n_steps=n_step,
                   attend=True,
                   attn_length=15,
                   # dropout=0.9,
                   name='rnn1')
    res1=tl.layers.RNNLayer(rnn1,
                           cell_fn=GRUCell,
                           cell_init_args={'activation': tf.nn.relu},
                           n_hidden=1,
                           n_steps=n_step,
                            name='res1'
                           )
    res1 = tl.layers.LambdaLayer(res1, lambda x: x[:, -7:, 0] * 12)
    res2=RNNLayer(pay_cnt_in2,
                   cell_fn=GRUCell,
                   cell_init_args={'activation': tf.nn.relu},
                   n_hidden=7,
                   n_steps=n_step//7,
                   attend=True,
                   attn_length=3,
                   # dropout=0.9,
                  return_last=True,
                   name='rnn2')

    # res2=tl.layers.RNNLayer(res2,
    #                         cell_fn=GRUCell,
    #                         cell_init_args={'activation': tf.nn.relu},
    #                         n_hidden=7,
    #                         n_steps=n_step//7,
    #                         return_last=True,
    #                         name='res2')

    net=tl.layers.ElementwiseLayer(layer=[res1,res2],combine_fn=tf.add)

    # net = tl.layers.DenseLayer(net, n_units=64, act=tf.nn.relu, name='dense1',
    #                            W_init_args={'regularizer': tf.contrib.layers.l2_regularizer(.0)},)
    # # variable_summaries(net.outputs, 'ftr_dense1')
    # net = tl.layers.DropoutLayer(net, keep=0.5, name='drop2')
    # net = tl.layers.DenseLayer(net, n_units=64, act=tf.nn.relu, name='dense2',
    #                            W_init_args={'regularizer': tf.contrib.layers.l2_regularizer(.0)})
    # # variable_summaries(net.outputs, name='ftr_dense2')
    # net = tl.layers.DenseLayer(net, n_units=7, act=tf.nn.relu, name='out',)
    # # variable_summaries(net.outputs, name='y_pred')
    return inputs_, y_, net



n_step=21
inputs, y_, net = get_model(n_step)
inputs.append(y_)
y = net.outputs
cost=loss_func(y_,y)

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


data,x_test = get_sample(n_step)
train_data=[d[:-2000] for d in data]
val_data=[d[-2000:] for d in data]


test = False
if test:
    sess.run(tf.global_variables_initializer())
    params = tl.files.load_npz(name='model/model_0226.npz')
    tl.files.assign_params(sess, params, net)
    val_data = [x_test]
    val_data_index = gen_val_idx(val_data[0].shape[0], 32)
    preds = []  # 0.111
    for idx in val_data_index:
        data_batch = [d[idx] for d in val_data]
        dp_dict = tl.utils.dict_to_one(net.all_drop)
        feed_dict = dict(zip(inputs[:-1], data_batch))
        feed_dict.update(dp_dict)
        pred = sess.run(y, feed_dict=feed_dict)
        preds.append(pred)
    preds = np.concatenate(preds)
    if preds.shape[1] == 7:
        preds = np.concatenate([preds, preds], axis=1)
    t = pd.DataFrame(data=np.round(preds), index=range(1, 2001))
    t.to_csv('result/0226_t21_85.csv', header=False, float_format='%d')
    sess.close()
    sys.exit(0)

nb_train = len(train_data[-1])
nb_val = len(val_data[-1])

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.001, global_step, 500, 0.95, staircase=False)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step, var_list=net.all_params)
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('model/train', sess.graph)

batch_size = 64
steps = 100000
print_freq = 500
global_loss = 1e10
save_path = 'model/model_0226.npz'
train_data_index = gen_train_idx(nb_train, batch_size)
errs=0
for step in range(steps):
    idx = next(train_data_index)
    data_batch = [d[idx] for d in train_data]
    feed_dict = dict(zip(inputs, data_batch))
    feed_dict.update(net.all_drop)
    _, err = sess.run([train_op, cost], feed_dict=feed_dict)
    errs+=err
    print('\r{} step:{:0>5}, cost:{:.5f}, '.format(time_now(), step + 1, err), end='')
    print('{} step:{:0>5}, cost:{:.5f}, '.format(time_now(), step + 1, errs/print_freq), end='')
    # train_writer.add_summary(summary, step)

    if step + 1 == 0 or (step + 1) % print_freq == 0:
        errs=0
        val_data_index = gen_val_idx(nb_val, batch_size)
        y_preds = []
        y_val = val_data[-1]
        for idx in val_data_index:
            data_batch = [d[idx] for d in val_data]
            dp_dict = tl.utils.dict_to_one(net.all_drop)
            feed_dict = dict(zip(inputs, data_batch))
            feed_dict.update(dp_dict)
            pred = sess.run(y, feed_dict=feed_dict)
            y_preds.append(pred)
        y_pred = np.concatenate(y_preds)
        val_cost = get_loss(y_val, y_pred)
        print('v_loss1:{:.6f}'.format(val_cost))
        if val_cost < global_loss:
            global_loss = val_cost
            tl.files.save_npz(net.all_params, save_path, sess=sess)
# train_writer.close()
sess.close()
