# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2019/5/21 9:49
@ Author  : Yaoming Cai
@ FileName: GRegConvAE_Base.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import os
import sys

from munkres import Munkres
from scipy.sparse import csgraph
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.classification import cohen_kappa_score, accuracy_score

sys.path.append('/home/caiyaom/python_codes/')
import numpy as np
import tensorflow as tf


class GRegConvAE:

    def __init__(self, epoch, n_cluster, image_name, lr=0.01, reg_lap=1., reg_latent=1., weight_decay=0.,
                 verb_per_iter=None, random_state=None):
        """
        :param epoch: the maximum iteration for updating parameters
        :param n_cluster: the number of clusters
        :param image_name: HSI data name, see demo.py
        :param lr: learning rate (float)
        :param reg_lap: laplasian regularization coefficient
        :param reg_latent: self-expression term regularization coefficient
        :param weight_decay: self-expression coef. regularization coefficient (L2 reg.)
        :param verb_per_iter: print clustering accuracy after n iterations (default None means print nothing)
        :param random_state: graph-level random seed (default is None)
        """
        tf.reset_default_graph()
        self.epoch = epoch
        self.reg_lap = reg_lap
        self.image_name = image_name
        self.reg_latent = reg_latent
        self.weight_decay = weight_decay
        self.n_cluster = n_cluster
        self.lr = lr
        self.verb_per_iter = verb_per_iter
        if random_state is not None:
            tf.set_random_seed(random_state)
        if not os.path.exists(self.image_name):
            os.mkdir(self.image_name)
        self.model_root_dir = self.image_name
        self.model_path = self.model_root_dir + '/' + self.image_name + '-model'

    def net(self, x, n_batch, is_training):
        # X = tf.layers.batch_normalization(X, training=is_training)

        # ============ encoder =================
        embed_1, embed_2, code = self.encoder(x, is_training, 'encoder', reuse=tf.AUTO_REUSE)

        # ============ self expression =============
        Z, Z_hat, C, latent_z = self.self_expression(code, n_batch, 'self-expression', reuse=tf.AUTO_REUSE)

        # ============ encoder =================
        decode = self.decoder(latent_z, [embed_1, embed_2, code], x.get_shape().as_list()[-1], is_training, 'decoder', reuse=tf.AUTO_REUSE)

        return code, decode, Z, Z_hat, C

    def self_expression(self, x, n_batch, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            Z = tf.layers.flatten(x)  # tf.reshape(X, (tf.shape(X)[0], -1))
            C = tf.Variable(1.0e-8 * tf.ones([n_batch, n_batch], tf.float32), name='Coef')
            Z_hat = tf.matmul(C, Z)
            latent_z = tf.reshape(Z_hat, tf.shape(x))
        return Z, Z_hat, C, latent_z

    def encoder(self, x, is_training, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            """========= Conv 1 ============"""
            hidden = tf.layers.conv2d(x, 24, (3, 3), strides=(1, 1), padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      bias_initializer=tf.initializers.zeros())
            embed_1 = tf.nn.relu(tf.layers.batch_normalization(hidden, training=is_training))

            """========= Conv 2 ============"""
            hidden = tf.layers.conv2d(embed_1, 24, (3, 3), strides=(1, 1), padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      bias_initializer=tf.initializers.zeros())
            embed_2 = tf.nn.relu(tf.layers.batch_normalization(hidden, training=is_training))

            """========= Conv 3 ============"""
            hidden = tf.layers.conv2d(embed_2, 32, (3, 3), strides=(1, 1), padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      bias_initializer=tf.initializers.zeros())
            code = tf.nn.relu(tf.layers.batch_normalization(hidden, training=is_training))
        return embed_1, embed_2, code

    def decoder(self, latent_z, x, out_channel, is_training, name, reuse):
        with tf.variable_scope(name, reuse=reuse):
            """========= Conv 1 ============"""
            hidden = tf.layers.conv2d_transpose(latent_z, 32, (3, 3), strides=(1, 1), padding='same',
                                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                bias_initializer=tf.initializers.zeros())
            embed_1 = tf.layers.batch_normalization(hidden, training=is_training)
            res_1 = tf.nn.relu(tf.add(x[2], embed_1))
            # embed_1 = tf.nn.relu(tf.layers.batch_normalization(hidden, training=is_training))
            # concat_1 = tf.concat([X[0], embed_1], axis=4)

            """========= Conv 2 ============"""
            hidden = tf.layers.conv2d_transpose(res_1, 24, (3, 3), strides=(1, 1), padding='same',
                                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                bias_initializer=tf.initializers.zeros())
            embed_2 = tf.layers.batch_normalization(hidden, training=is_training)
            res_2 = tf.nn.relu(tf.add(x[1], embed_2))
            # embed_2 = tf.nn.relu(tf.layers.batch_normalization(hidden, training=is_training))
            # concat_2 = tf.concat([X[1], embed_2], axis=4)

            """========= Conv 3 ============"""
            hidden = tf.layers.conv2d_transpose(res_2, 24, (3, 3), strides=(1, 1), padding='same',
                                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                bias_initializer=tf.initializers.zeros())
            embed_3 = tf.layers.batch_normalization(hidden, training=is_training)
            res_3 = tf.nn.relu(tf.add(x[0], embed_3))
            # embed_3 = tf.nn.relu(tf.layers.batch_normalization(hidden, training=is_training))
            # concat_3 = tf.concat([X[2], embed_3], axis=4)

            """========= Conv output ============"""
            decode = tf.layers.conv2d(res_3, out_channel, (1, 1), strides=(1, 1), padding='same',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      bias_initializer=tf.initializers.zeros())
            # decode = tf.nn.relu(tf.layers.batch_normalization(hidden, training=is_training))
        return decode

    def __loss(self, x_true, x_predict, z, z_hat, C, L):
        """
        compute loss
        :param x_true: training labels
        :param x_predict: all samples' prediction
        :param C: adjacent matrix
        :return:
        """
        # =========== model reconstruction loss ==============
        loss_recon = tf.reduce_mean(tf.losses.mean_squared_error(x_true, x_predict))
        tf.summary.scalar('loss-recon', loss_recon)

        # =========== coefficient L2 loss ==============
        loss_l2 = tf.nn.l2_loss(C)
        tf.summary.scalar('loss-l2', loss_l2)

        # =========== latent reconstruction loss ==============
        loss_recon_latent = tf.reduce_mean(tf.losses.mean_squared_error(z, z_hat))
        tf.summary.scalar('loss-latent', loss_recon_latent)

        # =========== laplacian loss ==============
        loss_lap = tf.trace(
            tf.matmul(tf.matmul(tf.transpose(C), tf.constant(L, dtype=tf.float32)), C))
        # loss_lap = tf.trace(
        #     tf.matmul(tf.matmul(tf.transpose(z), tf.constant(L, dtype=tf.float32)), z))
        tf.summary.scalar('loss-lap', loss_lap)

        loss = loss_recon + self.reg_lap * loss_lap + self.reg_latent * loss_recon_latent + self.weight_decay * loss_l2
        tf.summary.scalar('loss-total', loss)
        return loss

    def lap_matrix(self, x):
        # A = kneighbors_graph(X.reshape(X.shape[0], -1), n_neighbors=10, include_self=True, n_jobs=3).toarray()
        # A_ = kneighbors_graph(X.reshape(X.shape[0], -1), n_neighbors=5, include_self=True, n_jobs=8).toarray()
        # A_ = 0.5 * (A_ + A_.T)
        A_ = pairwise_kernels(x.reshape(x.shape[0], -1), metric='rbf', gamma=1., n_jobs=8)
        A = 0.5 * (A_ + A_.T)
        # A_[np.nonzero(A_)] = A[np.nonzero(A_)]
        L = csgraph.laplacian(A, normed=True)
        return L

    def __init_net__(self, X):
        x_placeholder = tf.placeholder(tf.float32, shape=(None, X.shape[1], X.shape[2], X.shape[3]))
        is_training = tf.placeholder(tf.bool)
        n_batch = X.shape[0]
        code, decode, Z, Z_hat, C = self.net(x_placeholder, n_batch, is_training)
        L = self.lap_matrix(X)
        loss_p = self.__loss(x_placeholder, decode, Z, Z_hat, C, L)
        tf.summary.histogram('C', C)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss_p)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        self.train_op = train_op
        self.C = C
        self.x_placeholder = x_placeholder
        self.is_training = is_training
        self.prediction_p = decode
        self.loss_p = loss_p
        self.sess = sess

    def fit(self, X, y):
        self.__init_net__(X)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.model_root_dir + '/logs', self.sess.graph)
        saver = tf.train.Saver()
        loss_his = []
        acc_his = {'oa':[], 'nmi':[], 'kappa':[], 'ca':[]} #[]
        for step_i in range(self.epoch):
            train_feed_dict = {self.x_placeholder: X, self.is_training: True}
            _, loss, summary = self.sess.run([self.train_op, self.loss_p, merged], feed_dict=train_feed_dict)
            print('epoch %s ==> loss=%s' % (step_i, loss))
            loss_his.append(loss)
            writer.add_summary(summary, step_i)
            # =============== test ==================
            # loss_eval, C_eval = self.sess.run([loss_p, C], feed_dict={self.x_placeholder: X, self.is_training: False})
            # # print logs after self.verb_per_iter iterations
            if self.verb_per_iter is not None and (step_i + 1) % self.verb_per_iter == 0:
                loss_test, y_pre = self.predict(X)
                acc, nmi, kappa, ca = self.cluster_accuracy(y, y_pre)
                print('epoch %s ==> loss=%s, acc=%s' % (step_i, loss_test, (acc, nmi, kappa, ca)))
                acc_his['oa'].append(acc)
                acc_his['nmi'].append(nmi)
                acc_his['kappa'].append(kappa)
                acc_his['ca'].append(ca)
                saver.save(self.sess, self.model_path, write_meta_graph=False)
        np.savez(self.model_root_dir + '/history.npz', loss=loss_his, acc=acc_his)
        saver.save(self.sess, self.model_path)
        if self.verb_per_iter is not None:
            return acc_his

    def predict(self, X, alpha=0.25):
        loss, Coef = self.sess.run([self.loss_p, self.C], feed_dict={self.x_placeholder: X, self.is_training: False})
        Coef = self.thrC(Coef, alpha)
        y_pre, C = self.post_proC(Coef, self.n_cluster, 8, 18)
        np.savez(self.model_root_dir + '/Affinity.npz', coef=C)
        np.savez(self.model_root_dir + '/y_pre.npz', y_pre=y_pre)
        # missrate_x = self.err_rate(y, y_x)
        # acc = 1 - missrate_x
        return loss, y_pre

    def predict_from_model(self, X, y):
        if not os.path.exists(self.model_root_dir + '/checkpoint'):
            raise Exception('model cannot be found !')
        else:
            self.__init_net__(X)
            # saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)
            Coef = self.sess.run(self.C, feed_dict={self.x_placeholder: X, self.is_training: False})
            Coef = self.thrC(Coef, 0.25)
            y_pre, C = self.post_proC(Coef, self.n_cluster, 8, 18)
            np.savez(self.model_root_dir + '/Affinity.npz', coef=C)
            acc = self.cluster_accuracy(y, y_pre)
            return y_pre, acc

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def build_aff(self, C):
        N = C.shape[0]
        Cabs = np.abs(C)
        ind = np.argsort(-Cabs, 0)
        for i in range(N):
            Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
        Cksym = Cabs + Cabs.T
        return Cksym

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L

    def cluster_accuracy(self, y_true, y_pre):
        Label1 = np.unique(y_true)
        nClass1 = len(Label1)
        Label2 = np.unique(y_pre)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = y_true == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = y_pre == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        y_best = np.zeros(y_pre.shape)
        for i in range(nClass2):
            y_best[y_pre == Label2[i]] = Label1[c[i]]

        # # calculate accuracy
        err_x = np.sum(y_true[:] != y_best[:])
        missrate = err_x.astype(float) / (y_true.shape[0])
        acc = 1. - missrate
        nmi = normalized_mutual_info_score(y_true, y_pre)
        kappa = cohen_kappa_score(y_true, y_best)
        ca = self.class_acc(y_true, y_best)
        return acc, nmi, kappa, ca

    def class_acc(self, y_true, y_pre):
        """
        calculate each classes's acc
        :param y_true:
        :param y_pre:
        :return:
        """
        ca = []
        for c in np.unique(y_true):
            y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
            y_c_p = y_pre[np.nonzero(y_true == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        return ca

    @staticmethod
    def build_laplacian(C):
        C = 0.5 * (np.abs(C) + np.abs(C.T))
        W = np.sum(C, axis=0)
        W = np.diag(1.0 / W)
        L = W.dot(C)
        return L