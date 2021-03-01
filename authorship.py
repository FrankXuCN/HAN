import time
import numpy as np
import tensorflow as tf

from models import GAT, HeteGAT, HeteGAT_multi
from utils import process
import utils as utl

import json
# 禁用gpu
import os, sys


# jhy data
import scipy.io as sio
import scipy.sparse as sp

import linecache

def loadDataByNo(filePath, endLine, startLine=0):
    endNo = startLine
    dataset = []
    with open(filePath, 'r') as df:
        for i, line in enumerate(df):
            if i+1 >= startLine and i+1 <= endLine:
                temp = json.loads(line)
                endNo += 1
                dataset.append(temp)
            if endNo >= endLine:
                break
    return dataset
    # while endNo != endLine:
    #     temp = linecache.getline(filePath, endNo)
    #     # temp = loadOneDatasetEntry(filePath, endNo)
    #     dataset.append(json.loads(temp))
    # dataset : list of list
    return dataset

def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y



def buildEmbeddingLayer(ftr_in_list, bias_in_list, attn_drop, ffd_drop, OneDatasetEntry):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    fea_list, _, _ = OneDatasetEntry
    # do not allow to use gpu by yhxu
    # nb :- NeiBie
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # dataset = 'acm'
    featype = 'fea'
    # checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
    # print('model: {}'.format(checkpt_file))
    # training params
    batch_size = 1
    nb_epochs = 200
    patience = 100
    lr = 0.005  # learning rate
    l2_coef = 0.001  # weight decay
    # numbers of hidden units per each attention head in each layer
    hid_units = [8]
    n_heads = [8, 1]  # additional entry for the output layer
    residual = False
    nonlinearity = tf.nn.elu
    model = HeteGAT_multi

    # print('Dataset: ' + dataset)
    print('----- Opt. hyperparams -----')
    print('lr: ' + str(lr))
    print('l2_coef: ' + str(l2_coef))
    print('----- Archi. hyperparams -----')
    print('nb. layers: ' + str(len(hid_units)))
    print('nb. units per layer: ' + str(hid_units))
    print('nb. attention heads: ' + str(n_heads))
    print('residual: ' + str(residual))
    print('nonlinearity: ' + str(nonlinearity))
    print('model: ' + str(model))

    # adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp()

    if featype == 'adj':
        fea_list = adj_list

    # nb_nodes = fea_list[0].shape[0] # N，样本量
    ft_size = len(fea_list[0]) # D，结点的原始维度

    print('build HAN...')
    # with tf.Graph().as_default():
    final_embedding, att_val = model.inference(ftr_in_list, attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    embed = tf.expand_dims(final_embedding, axis = 0)
    return embed

def modelInputs(batch_size, OneDatasetEntry):#(batch_size, classNum):
    """
    Create the model inputs
    """
    f, _, l = OneDatasetEntry
    ft_size = len(f[0])
    labelShape = np.array(l).shape

    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, None, ft_size),
                                      name='ftr_in_{}'.format(i))
                       # for i in range(len(fea_list))]
                       for i in range(1)]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, None, None),# nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        # for i in range(len(biases_list))]
                        for i in range(1)]
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
    # labels_ = tf.placeholder(tf.int32, [batch_size, classNum], name='labels')
    labels_ = tf.placeholder(tf.float32, labelShape, name='labels')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
    
    # return inputs_, labels_, keep_prob_
    return labels_, keep_prob_, ftr_in_list, bias_in_list, attn_drop, ffd_drop


def buildLSTMLayer(lstm_sizes, embed, keep_prob_, batch_size):
    """
    Create the LSTM layers
    """
    print("building LSTM...")
    lstms = tf.contrib.rnn.BasicLSTMCell(lstm_sizes)
    # Add dropout to the cell
    drops = tf.contrib.rnn.DropoutWrapper(lstms, output_keep_prob=keep_prob_)
    # Getting an initial state of all zeros
    initial_state = drops.zero_state(batch_size, tf.float32)
    
    lstm_outputs, final_state = tf.nn.dynamic_rnn(drops, embed, initial_state=initial_state)
    
    return initial_state, lstm_outputs, drops, final_state
    # lstmCell = tf.keras.layers.LSTMCell(lstm_sizes)
    # initial_state = lstmCell.get_initial_state(batch_size, tf.float32)
    # rnn = tf.keras.layers.RNN(cell = lstmCell)
    # print("lstm_sizes is {}, rnn is {}, embed is {}".format(lstm_sizes, rnn, embed))
    # output = rnn(embed)
    # return  output, initial_state

def buildCostFnAndOpt(lstm_outputs, labels_, learning_rate, OneDatasetEntry):
    """
    Create the Loss function and Optimizer
    """
    _, _, label = OneDatasetEntry
    predictions = tf.contrib.layers.fully_connected(lstm_outputs[:, -1],
                                                    num_outputs=len(label),
                                                    activation_fn=tf.sigmoid)
    # loss = tf.losses.mean_squared_error(labels_, predictions)
    # predictions = tf.nn.softmax(predictions, name='softmax')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=predictions))
    # optimzer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    
    return predictions, loss, optimzer

def buildAccuracy(predictions, labels_):
    """
    Create accuracy
    """
    correct_pred = tf.cast(tf.equal(tf.argmax(predictions, 0), tf.argmax(labels_, 0)),tf.float32)
    # correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return correct_pred


def build_and_train_network(lstm_sizes, epochs, batch_size,
                             learning_rate, keep_prob, totalLines, file):#, train_x, val_x, train_y, val_y):
    fixlines = 100
    trainSteps = int((totalLines*0.6) / fixlines)
    validateSteps = int((totalLines*.02) / fixlines)

    # miniDataset: [[fea, adj, lab], ... ]
    miniDataset = loadDataByNo(file, fixlines)

    labels_, keep_prob_, ftr_in_list, bias_in_list, attn_drop, ffd_drop = modelInputs(batch_size, miniDataset[0])

    embed = buildEmbeddingLayer(ftr_in_list, bias_in_list, attn_drop, ffd_drop, miniDataset[0])

    initial_state, lstm_outputs, lstm_cell, final_state = buildLSTMLayer(lstm_sizes, embed, keep_prob_, batch_size)

    predictions, loss, optimizer = buildCostFnAndOpt(lstm_outputs, labels_, learning_rate, miniDataset[0])

    accuracy = buildAccuracy(predictions, labels_)
    print("build Network over")
    saver = tf.train.Saver()
    
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        
        sess.run(init_op)
        state = sess.run(initial_state)
        # print(initial_state)

        # dataset will be over 20k entries and its size will be over 1GB
        # In order to save memory, we will read the dataset in a batch way

        for epoch in range(epochs):
            # ================== train =====================
            train_acc = []
            for step in range(1,trainSteps+1):
                batch_acc = []
                for entry in miniDataset:
                    # f: N*D; a: X*N*N; l: (Y,)
                    f = entry[0]
                    a = entry[1]
                    l = entry[2]
                    features = [np.array(f)]
                    adjMetricx = [np.array(a[0])]
                    y_train = np.array(l)
                    # print("features.shape is {}; a.shape is {}, y_train.shape is {}".format(
                    #     np.array(features).shape, np.array(a).shape, np.array(y_train).shape))

                    # fea_list: 1*N*N; adj_list: 1*N*N; lab_list: 1*N
                    fea_list = [fea[np.newaxis] for fea in features]
                    adj_list = [adj[np.newaxis] for adj in adjMetricx]
                    graphNodes = features[0].shape[0]
                    # print("fea_list.shape is {}".format(np.array(fea_list[0]).shape), 
                    #     "adj_list.shape is {}".format(np.array(adj_list[0]).shape))
                    biases_list = [process.adj_to_bias(adj, [graphNodes], nhood=1) for adj in adj_list]

                    fd1 = {i: d[:] for i, d in zip(ftr_in_list, fea_list)}
                    fd2 = {i: d[:] for i, d in zip(bias_in_list, biases_list)}
                    fd3 = {
                           attn_drop: 0.0,
                           ffd_drop: 0.0}
                
                    fd = fd1
                    fd.update(fd2)
                    fd.update(fd3)
                    fd4 = {labels_: y_train, keep_prob_:keep_prob, initial_state: state}
                    fd.update(fd4)
                    loss_, state, _,  acc = sess.run([loss, final_state, optimizer, accuracy
                                                                                    ], feed_dict=fd)
                    batch_acc.append(acc)
                train_acc.append(np.mean(batch_acc))
                # print("{} records feeded for trainning".format(step*fixlines))
                # miniDataset: [[fea, adj, lab], ... ]
                miniDataset = loadDataByNo(file, (step+1)*fixlines, step*fixlines)

            # ================== validate =====================
            val_acc = []
            val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
            for step in range(trainSteps+1,trainSteps+validateSteps+1):
                val_batch_acc = []
                for entry in miniDataset:
                    # f: N*D; a: X*N*N; l: (Y,)
                    f = entry[0]
                    a = entry[1]
                    l = entry[2]
                    features = [np.array(f)]
                    adjMetricx = [np.array(a[0])]
                    y_train = np.array(l)

                    # fea_list: 1*N*N; adj_list: 1*N*N; lab_list: 1*N
                    fea_list = [fea[np.newaxis] for fea in features]
                    adj_list = [adj[np.newaxis] for adj in adjMetricx]
                    nb_nodes = features[0].shape[0]
                    biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

                    fd1 = {i: d[:] for i, d in zip(ftr_in_list, fea_list)}
                    fd2 = {i: d[:] for i, d in zip(bias_in_list, biases_list)}
                    fd3 = {
                           attn_drop: 0.0,
                           ffd_drop: 0.0}
                
                    fd = fd1
                    fd.update(fd2)
                    fd.update(fd3)
                    fd4 = {labels_: y_train, keep_prob_:1, initial_state: val_state}
                    fd.update(fd4)
                    acc, val_state = sess.run([accuracy, final_state], feed_dict=fd)
                    val_batch_acc.append(acc)

                val_acc.append(np.mean(val_batch_acc))
                # miniDataset: [[fea, adj, lab], ... ]
                miniDataset = loadDataByNo(file, (step+1)*fixlines, step*fixlines)

            print(
                  "epoch : {}".format(epoch),
                  "Train Loss: {:.5f}...".format(loss_),
                  "Train Accruacy: {:.5f}...".format(np.mean(train_acc)),
                  "Val Accuracy: {:.5f}".format(np.mean(val_acc))
                  )
        saver.save(sess, "checkpoints/sentiment.ckpt")

def networkTest(model_dir, batch_size, totalLines, file):
    fixlines = 100
    trainSteps = int((totalLines*0.6) / fixlines)
    testSteps = int((totalLines*.02) / fixlines)

    # miniDataset: [[fea, adj, lab], ... ]
    miniDataset = loadDataByNo(file, fixlines)

    labels_, keep_prob_, ftr_in_list, bias_in_list, attn_drop, ffd_drop = modelInputs(batch_size, miniDataset[0])

    embed = buildEmbeddingLayer(ftr_in_list, bias_in_list, attn_drop, ffd_drop, miniDataset[0])

    initial_state, lstm_outputs, lstm_cell, final_state = buildLSTMLayer(lstm_sizes, embed, keep_prob_, batch_size)

    predictions, loss, optimizer = buildCostFnAndOpt(lstm_outputs, labels_, learning_rate, miniDataset[0])

    accuracy = buildAccuracy(predictions, labels_)
    print("build Network over")
    saver = tf.train.Saver()
    
    test_acc = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        test_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
        # for i in range(trainSteps+1,testSteps+1):
        for step in range(1,trainSteps+1):
            for entry in miniDataset:
                # f: N*D; a: X*N*N; l: (Y,)
                f = entry[0]
                a = entry[1]
                l = entry[2]
                features = [np.array(f)]
                adjMetricx = [np.array(a[0])]
                y_train = np.array(l)

                # fea_list: 1*N*N; adj_list: 1*N*N; lab_list: 1*N
                fea_list = [fea[np.newaxis] for fea in features]
                adj_list = [adj[np.newaxis] for adj in adjMetricx]
                nb_nodes = features[0].shape[0]
                biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

                fd1 = {i: d[:] for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[:] for i, d in zip(bias_in_list, biases_list)}
                fd3 = {
                       attn_drop: 0.0,
                       ffd_drop: 0.0}
            
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                fd4 = {labels_: y_train, keep_prob_:1, initial_state: test_state}
                fd.update(fd4)
                acc, test_state = sess.run([accuracy, final_state], feed_dict=fd)
                test_acc.append(acc)
            
            # miniDataset: [[fea, adj, lab], ... ]
            miniDataset = loadDataByNo(file, (step+1)*fixlines, step*fixlines)

        print(
              "Val Accuracy: {:.3f}".format(np.mean(test_acc))
              )
 
if __name__ == '__main__':
    tflag = len(sys.argv)
    # Define Inputs and Hyperparameters
    lstm_sizes = 64
    # vocab_size = 200 # len(vocab_to_int) + 1 #add one for padding
    # embed_size = 300
    epochs = 100
    batch_size = 1
    learning_rate = 0.15
    keep_prob = 0.5
    totalLines = 5000
    file = '/mnt/hgfs/IIT/authroship/gcj-dataset-master/gcj2020/json_binaries/5kdatasetFromAST.json'

    if tflag < 2:
        # load our data set 60% for trainning, 20% for validate, 20 for test
        with tf.Graph().as_default():
            build_and_train_network(lstm_sizes, epochs, batch_size,
                                    learning_rate, keep_prob, totalLines, file)#, train_x, val_x, train_y, val_y)

    networkTest('checkpoints', batch_size, totalLines, file)

    print("finish!")