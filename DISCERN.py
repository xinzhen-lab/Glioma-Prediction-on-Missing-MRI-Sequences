from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import *
import tensorflow as tf
import random
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softplus
import sys

import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore')
python_version = sys.version_info

"说明：进行3个MRI序列缺失的情景模拟：(1)缺失T1WI, (2)缺失CE-T1WI, (3)缺失T2-FLAIR, (4)缺失T1WI+CE-T1WI,(5)缺失T1WI+T2-FLAIR, (6)缺失CE-T1WI+T2-FLAIR" \
", (7)全部4个序列的缺失场景模拟：缺失CE-T1WI，只有3个平扫序列，对CE-TWI进行补全，与使用全部4个序列的场景进行比较"

def load_data(Data, Label, tr_idx, va_idx):
    # 删除特征为0的特征列
    # Data = np.delete(Data, zero_columns, axis=1)

    X_train = Data[tr_idx]
    y_train = Label[tr_idx]
    X_valid = Data[va_idx]
    y_valid = Label[va_idx]

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    sm = SMOTE()  # kind = ['regular', 'borderline1', 'borderline2', 'svm']
    if python_version >= (3, 9):
        X_train, y_train = sm.fit_resample(X_train, y_train)
    else:
        X_train, y_train = sm.fit_sample(X_train, y_train)
    # X_train, y_train = sm.fit_resample(X_train, y_train)

    y_valid = np.reshape(y_valid, (len(y_valid), 1))
    y_train = np.reshape(y_train, (len(y_train), 1))
    return X_train, y_train, X_valid, y_valid
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon  # 均值+标准差*服从正态分布N(0, 1)的噪声

class MDVAEmodel_multiple(Model):
    def __init__(self, inp_shape, latent_dim, private_dim, shared_dim, **kwargs):
        super(MDVAEmodel_multiple, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.private_dim = private_dim
        self.shared_dim = shared_dim
        self.inp_shape = inp_shape

        self.Flair_encoder = self.build_encoder_network(name='Flair')
        self.T1ce_encoder = self.build_encoder_network(name='T1ce')
        self.T1_encoder = self.build_encoder_network(name='T1')
        self.T2_encoder = self.build_encoder_network(name='T2')

        self.Flair_decoder = self.build_decoder_network(output_shape=self.inp_shape[0], name='Flair')
        self.T1ce_decoder = self.build_decoder_network(output_shape=self.inp_shape[0], name='T1ce')
        self.T1_decoder = self.build_decoder_network(output_shape=self.inp_shape[0], name='T1')
        self.T2_decoder = self.build_decoder_network(output_shape=self.inp_shape[0], name='T2')

        self.grade_classifier = self.build_classification_network()

        # self.Flair_encoder.summary()
        # self.grade_classifier.summary()

        self.MMDR_model = self.build_MMDR_model()
        self.custom_metrics = METRICS
        # self.MMDR_model.summary()

    def build_MMDR_model(self):

        MMDR_input_Flair = Input(shape=self.inp_shape, name='MMDR_input_FLair')
        MMDR_input_T1ce = Input(shape=self.inp_shape, name='MMDR_input_T1ce')
        MMDR_input_T1 = Input(shape=self.inp_shape, name='MMDR_input_T1')
        MMDR_input_T2 = Input(shape=self.inp_shape, name='MMDR_input_T2')

        Flair_encoder_out = self.Flair_encoder(MMDR_input_Flair)
        T1ce_encoder_out = self.T1ce_encoder(MMDR_input_T1ce)
        T1_encoder_out = self.T1_encoder(MMDR_input_T1)
        T2_encoder_out = self.T2_encoder(MMDR_input_T2)

        Flair_decoder_out = self.Flair_decoder([Flair_encoder_out[2], Flair_encoder_out[5]])
        T1ce_decoder_out = self.T1ce_decoder([T1ce_encoder_out[2], T1ce_encoder_out[5]])
        T1_decoder_out = self.T1_decoder([T1_encoder_out[2], T1_encoder_out[5]])
        T2_decoder_out = self.T2_decoder([T2_encoder_out[2], T2_encoder_out[5]])

        "2. Stacking method"
        z_mean_private_out = [T1_encoder_out[0], T1ce_encoder_out[0], T2_encoder_out[0], Flair_encoder_out[0]]
        z_log_var_private_out = [T1_encoder_out[1], T1ce_encoder_out[1], T2_encoder_out[1], Flair_encoder_out[1]]
        z_private_out = [T1_encoder_out[2], T1ce_encoder_out[2], T2_encoder_out[2], Flair_encoder_out[2]]  # latent private representation
        z_mean_shared_out = [T1_encoder_out[3], T1ce_encoder_out[3], T2_encoder_out[3], Flair_encoder_out[3]]
        z_log_var_shared_out = [T1_encoder_out[4], T1ce_encoder_out[4], T2_encoder_out[4], Flair_encoder_out[4]]
        z_shared_out = [T1_encoder_out[5], T1ce_encoder_out[5], T2_encoder_out[5], Flair_encoder_out[5]]  # latent shared representation
        z_mean_private_add = []
        z_log_var_private_add = []
        z_private_add = []
        z_mean_shared_add = []
        z_log_var_shared_add = []
        z_shared_add = []
        for i in range(len(fusion_idx)):
            z_mean_private_add.append(z_mean_private_out[fusion_idx[i]])
            z_log_var_private_add.append(z_log_var_private_out[fusion_idx[i]])
            z_private_add.append(z_private_out[fusion_idx[i]])
            z_mean_shared_add.append(z_mean_shared_out[fusion_idx[i]])
            z_log_var_shared_add.append(z_log_var_shared_out[fusion_idx[i]])
            z_shared_add.append(z_shared_out[fusion_idx[i]])

        "==============Mean========================"
        z_common_fusion = Lambda(lambda x: tf.reduce_mean(tf.stack([x[i] for i in list(range(len(fusion_idx)))], axis=0), axis=0))(z_shared_add)

        # 多/单模态的特异表示+共享表示
        z_private_add.append(z_common_fusion)

        "增加共享表示和每个模态的特异表示的解码器"
        Flair_fusion_decoder_out = self.Flair_decoder([Flair_encoder_out[2], z_common_fusion])
        T1ce_fusion_decoder_out = self.T1ce_decoder([T1ce_encoder_out[2], z_common_fusion])
        T1_fusion_decoder_out = self.T1_decoder([T1_encoder_out[2], z_common_fusion])
        T2_fusion_decoder_out = self.T2_decoder([T2_encoder_out[2], z_common_fusion])
        "===================================================================================================="

        # 多个特异部分+1个共享部分 或者每个模态的特异部分+共享部分
        common_code = tf.concat(z_private_add, axis=1)

        grade_out = self.grade_classifier(common_code)

        model = Model(inputs=[MMDR_input_T1, MMDR_input_T1ce, MMDR_input_T2, MMDR_input_Flair],
                      outputs=[T1_encoder_out, T1ce_encoder_out, T2_encoder_out, Flair_encoder_out,
                               T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out,
                               T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out, Flair_fusion_decoder_out,
                               common_code, grade_out],
                      name="MMDR")
        return model

    def train_step(self, x, y, missing_idxs):
        with tf.GradientTape() as tape:
            T1_encoder_out, T1ce_encoder_out, T2_encoder_out, Flair_encoder_out, \
            T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out, \
            T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out, Flair_fusion_decoder_out, common_code, grade_out = self.MMDR_model(x)

            "获取每个单一模态的特异表示的均值、方差、特异表示，以及单一模态的共享表示的均值、方差和共享表示"
            z_mean_private_Flair, z_log_var_private_Flair, z_private_Flair, z_mean_shared_Flair, z_log_var_shared_Flair, z_shared_Flair = Flair_encoder_out
            z_mean_private_T1, z_log_var_private_T1, z_private_T1, z_mean_shared_T1, z_log_var_shared_T1, z_shared_T1  = T1_encoder_out
            z_mean_private_T1ce, z_log_var_private_T1ce, z_private_T1ce, z_mean_shared_T1ce, z_log_var_shared_T1ce, z_shared_T1ce  = T1ce_encoder_out
            z_mean_private_T2, z_log_var_private_T2, z_private_T2, z_mean_shared_T2, z_log_var_shared_T2, z_shared_T2 = T2_encoder_out

            "单一模态特异部分+单一模态共享部分的列表"
            Decoders_out = [T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out]  # 自身特异+自身共享
            z_log_vars_private = [z_log_var_private_T1, z_log_var_private_T1ce, z_log_var_private_T2, z_log_var_private_Flair]
            z_means_private = [z_mean_private_T1, z_mean_private_T1ce, z_mean_private_T2, z_mean_private_Flair]
            z_log_vars_shared = [z_log_var_shared_T1, z_log_var_shared_T1ce, z_log_var_shared_T2, z_log_var_shared_Flair]
            z_means_shared = [z_mean_shared_T1, z_mean_shared_T1ce, z_mean_shared_T2, z_mean_shared_Flair]

            "单一模态特异部分+PoE融合共享部分的列表"
            Decoders_fusion_out = [T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out, Flair_fusion_decoder_out]  # 自身特异+融合共享享

            "所有模态的共享表示和特异表示的列表"
            z_private_out = [z_private_T1, z_private_T1ce, z_private_T2, z_private_Flair]
            z_shared_out = [z_shared_T1, z_shared_T1ce, z_shared_T2, z_shared_Flair]

            "计算重构损失、KL损失和分级损失"
            reconstruction_loss = 0
            kl_loss = 0
            com_loss = 0
            spe_loss = 0
            epsilon = 1e-10  # 一个很小的常数，用于防止除零错误
            if len(fusion_idx) > 1:
                for i in range(len(fusion_idx)):
                    kl_loss = kl_loss + (-0.5 * tf.reduce_mean(1 + z_log_vars_private[fusion_idx[i]] - tf.square(z_means_private[fusion_idx[i]]) - tf.exp(z_log_vars_private[fusion_idx[i]]))) \
                              + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared[fusion_idx[i]] - tf.square(z_means_shared[fusion_idx[i]]) - tf.exp(z_log_vars_shared[fusion_idx[i]])))

                    "20231108: 针对完整样本计算重构损失"
                    missing_idxs_tr = missing_idxs[:, i] # 这是原来959个病人的索引
                    smote_idxs_tr = np.ones((x[fusion_idx[i]].shape[0] - len(missing_idxs_tr)))
                    new_idxs_tr = np.hstack((missing_idxs_tr, smote_idxs_tr))
                    reconstruction_loss = reconstruction_loss + \
                    tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[i]][new_idxs_tr != 0], Decoders_out[fusion_idx[i]][new_idxs_tr != 0]))) \
                    + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[i]][new_idxs_tr != 0], Decoders_fusion_out[fusion_idx[i]][new_idxs_tr != 0])))

                    # reconstruction_loss = reconstruction_loss + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[i]], Decoders_out[fusion_idx[i]]))) \
                    #                + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[i]], Decoders_fusion_out[fusion_idx[i]])))  # 单一模态特异+融合共享重构

                    "3. 两两的共享表示、特异表示的损失"
                    if i < len(fusion_idx) - 1:
                        for j in range(i + 1, len(fusion_idx)):
                            com_loss += tf.reduce_mean(tf.sqrt(mean_squared_error(z_shared_out[fusion_idx[i]], z_shared_out[fusion_idx[j]])))
                            spe_loss += tf.reduce_mean(tf.sqrt(mean_squared_error(z_private_out[fusion_idx[i]], z_private_out[fusion_idx[j]])))
                com_spec_loss = com_loss / spe_loss
            else:
                kl_loss = (-0.5 * tf.reduce_mean(1 + z_log_vars_private[fusion_idx[0]] - tf.square(z_means_private[fusion_idx[0]]) - tf.exp(z_log_vars_private[fusion_idx[0]]))) \
                          + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared[fusion_idx[0]] - tf.square(z_means_shared[fusion_idx[0]]) - tf.exp(z_log_vars_shared[fusion_idx[0]])))
                reconstruction_loss = tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[0]], Decoders_out[fusion_idx[0]])))
                com_spec_loss = com_loss / (spe_loss + epsilon)  # 防止除0

            grade_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, grade_out))

            # total_loss = params[0] * reconstruction_loss + params[1] * kl_loss + params[2] * grade_loss
            total_loss = params[0] * reconstruction_loss + params[1] * kl_loss + params[2] * com_spec_loss + params[3] * grade_loss

        "计算梯度"
        grads = tape.gradient(total_loss, self.trainable_variables)

        "使用优化器更新模型参数"
        # self.optimizer.learning_rate = lr_schedule(epoch)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        print(">>>>>>>>>>>>>>metrics_names:", self.metrics_names)
        # 输出当前 epoch 的学习率
        print(f"Epoch {epoch + 1}, Learning Rate: {self.optimizer.lr.numpy()}")

        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, grade_out)

        for metric in self.custom_metrics:
            metric.update_state(y, grade_out)
            metric_value = metric.result().numpy()
            # print(f'{metric.name}: {metric_value}')

        return {
            "Decoders_out": Decoders_out,
            "Decoders_fusion_out": Decoders_fusion_out,
            "grade_out": grade_out,
            "common_code": common_code,
            "loss": total_loss,
            "rec_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "grade_loss": grade_loss,
            self.custom_metrics[0].name: self.custom_metrics[0].result(),
            self.custom_metrics[1].name: self.custom_metrics[1].result(),
            # 当IDH时去掉以下部分
        }

    def test_step(self, x_, y_, missing_idxs):
        T1_encoder_out_, T1ce_encoder_out_, T2_encoder_out_, Flair_encoder_out_, \
        T1_decoder_out_, T1ce_decoder_out_, T2_decoder_out_, Flair_decoder_out_, \
        T1_fusion_decoder_out_, T1ce_fusion_decoder_out_, T2_fusion_decoder_out_, Flair_fusion_decoder_out_,common_code_, grade_out_ = self.MMDR_model(x_)

        "获取每个单一模态的特异表示的均值、方差、特异表示，以及单一模态的共享表示的均值、方差和共享表示"
        z_mean_private_Flair_, z_log_var_private_Flair_, z_private_Flair_, z_mean_shared_Flair_, z_log_var_shared_Flair_, z_shared_Flair_= Flair_encoder_out_
        z_mean_private_T1_, z_log_var_private_T1_, z_private_T1_, z_mean_shared_T1_, z_log_var_shared_T1_, z_shared_T1_ = T1_encoder_out_
        z_mean_private_T1ce_, z_log_var_private_T1ce_, z_private_T1ce_, z_mean_shared_T1ce_, z_log_var_shared_T1ce_, z_shared_T1ce_ = T1ce_encoder_out_
        z_mean_private_T2_, z_log_var_private_T2_, z_private_T2_, z_mean_shared_T2_, z_log_var_shared_T2_, z_shared_T2_ = T2_encoder_out_

        "单一模态特异部分+单一模态共享部分的列表"
        Decoders_out_ = [T1_decoder_out_, T1ce_decoder_out_, T2_decoder_out_, Flair_decoder_out_]  # 自身特异+自身共享
        z_log_vars_private_ = [z_log_var_private_T1_, z_log_var_private_T1ce_, z_log_var_private_T2_,  z_log_var_private_Flair_]
        z_means_private_ = [z_mean_private_T1_, z_mean_private_T1ce_, z_mean_private_T2_, z_mean_private_Flair_]
        z_log_vars_shared_ = [z_log_var_shared_T1_, z_log_var_shared_T1ce_, z_log_var_shared_T2_, z_log_var_shared_Flair_]
        z_means_shared_ = [z_mean_shared_T1_, z_mean_shared_T1ce_, z_mean_shared_T2_, z_mean_shared_Flair_]

        "单一模态特异部分+PoE融合共享部分的列表"
        Decoders_fusion_out_ = [T1_fusion_decoder_out_, T1ce_fusion_decoder_out_, T2_fusion_decoder_out_, Flair_fusion_decoder_out_]  # 自身特异+融合共享

        "所有模态的共享表示和特异表示的列表"
        z_private_out_ = [z_private_T1_, z_private_T1ce_, z_private_T2_, z_private_Flair_]
        z_shared_out_ = [z_shared_T1_, z_shared_T1ce_, z_shared_T2_, z_shared_Flair_]

        "计算重构损失、KL损失和分级损失"
        reconstruction_loss_ = 0
        kl_loss_ = 0
        com_loss_ = 0
        spe_loss_ = 0
        epsilon = 1e-10  # 一个很小的常数，用于防止除零错误
        if len(fusion_idx) > 1:
            for i in range(len(fusion_idx)):
                kl_loss_ = kl_loss_ + (-0.5 * tf.reduce_mean(1 + z_log_vars_private_[fusion_idx[i]] - tf.square(z_means_private_[fusion_idx[i]]) - tf.exp(z_log_vars_private_[fusion_idx[i]]))) \
                          + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared_[fusion_idx[i]] - tf.square(z_means_shared_[fusion_idx[i]]) - tf.exp(z_log_vars_shared_[fusion_idx[i]])))
                # reconstruction_loss_ = reconstruction_loss_ + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]], Decoders_out_[fusion_idx[i]]))) \
                #                       + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]], Decoders_fusion_out_[fusion_idx[i]])))

                "20231108: 针对完整样本计算重构损失"
                missing_idxs_te = missing_idxs[:, i]
                reconstruction_loss_ = reconstruction_loss_ + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]][missing_idxs_te != 0],
                Decoders_out_[fusion_idx[i]][missing_idxs_te != 0]))) + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]][missing_idxs_te != 0],
                Decoders_fusion_out_[fusion_idx[i]][missing_idxs_te != 0])))

                "3. 两两的共享表示、特异表示的损失"
                if i < len(fusion_idx) - 1:
                    for j in range(i + 1, len(fusion_idx)):
                        com_loss_ += tf.reduce_mean(tf.sqrt(mean_squared_error(z_shared_out_[fusion_idx[i]], z_shared_out_[fusion_idx[j]])))
                        spe_loss_ += tf.reduce_mean(tf.sqrt(mean_squared_error(z_private_out_[fusion_idx[i]], z_private_out_[fusion_idx[j]])))
            com_spec_loss_ = com_loss_ / spe_loss_
        else:
            kl_loss_ = (-0.5 * tf.reduce_mean(1 + z_log_vars_private_[fusion_idx[0]] - tf.square(z_means_private_[fusion_idx[0]]) - tf.exp(z_log_vars_private_[fusion_idx[0]]))) \
                      + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared_[fusion_idx[0]] - tf.square(z_means_shared_[fusion_idx[0]]) - tf.exp(z_log_vars_shared_[fusion_idx[0]])))
            reconstruction_loss_ = tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[0]], Decoders_out_[fusion_idx[0]])))
            com_spec_loss_ = com_loss_ / (spe_loss_ + epsilon)  # 防止除0

        grade_loss_ = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_, grade_out_))

        # total_loss = params[0] * reconstruction_loss + params[1] * kl_loss + params[2] * grade_loss
        total_loss_ = params[0] * reconstruction_loss_ + params[1] * kl_loss_ + params[2] * com_spec_loss_ + params[3] * grade_loss_

        # self.compiled_metrics.update_state(y, grade_out)
        # Update metrics manually
        "更新指标"
        grade_out_ = tf.where(tf.math.is_nan(grade_out_), tf.zeros_like(grade_out_), grade_out_)

        for metric in self.custom_metrics:
            metric.update_state(y_, grade_out_)
            metric_value = metric.result().numpy()
            # print(f'{metric.name}: {metric_value}')

        return {
            "Decoders_out": Decoders_out_,
            "Decoders_fusion_out": Decoders_fusion_out_,
            "grade_out": grade_out_,
            "common_code": common_code_,
            "loss": total_loss_,
            "rec_loss": reconstruction_loss_,
            "kl_loss": kl_loss_,
            "grade_loss": grade_loss_,
            self.custom_metrics[0].name: self.custom_metrics[0].result(),
            self.custom_metrics[1].name: self.custom_metrics[1].result(),
            # IDH时去掉以下部分
        }

    def build_encoder_network(self, disentangled=True, name=''):
        """
        create encoder network
        :param input_shape:
        :param disentangled:
        :param name:
        :return:
        """

        input_layer = Input(shape=self.inp_shape)
        x = Dense(80, activation='relu')(input_layer)  # 1024
        x = BatchNormalization()(x)
        x = Dense(40, activation='relu')(x)  # 256
        x = BatchNormalization()(x)
        x = Dense(self.latent_dim, activation='relu')(x)  # 32
        if disentangled:
            mean_private = Dense(self.private_dim)(x)   # 10.14
            log_var_private = softplus(Dense(self.private_dim)(x))  # 10.14
            mean_shared = Dense(self.shared_dim)(x)  # 10.14
            log_var_shared = softplus(Dense(self.shared_dim)(x))  # 10.14

            z_private = Sampling()([mean_private, log_var_private])  # 10.14
            z_shared = Sampling()([mean_shared, log_var_shared])  # 10.14
            return Model(inputs=input_layer, outputs=[mean_private, log_var_private, z_private, mean_shared, log_var_shared, z_shared], name='encoder_{}'.format(name))
        else:
            return Model(inputs=input_layer, outputs=x, name='encoder_{}'.format(name))

    def build_decoder_network(self, output_shape, name=''):
        private_input = Input(shape=(self.private_dim,), name='input_1_{}'.format(name))
        shared_input = Input(shape=(self.shared_dim,), name='input_2_{}'.format(name))
        inp = Concatenate(axis=-1)([private_input, shared_input])
        l = BatchNormalization()(inp)
        # z_input = Input(shape=(self.private_dim+self.shared_dim,), name='input_1_{}'.format(name))
        # l = BatchNormalization()(z_input)
        l = Dense(40, activation='relu')(l)
        l = BatchNormalization()(l)
        l = Dense(80, activation='relu')(l)
        l = Dense(output_shape, activation='tanh')(l)
        return Model(inputs=[private_input, shared_input], outputs=l, name='decoder_{}'.format(name))

    def build_classification_network(self):
        """

        :return:
        """
        # fusion_feature = Input((self.latent_dim*len(fusion_idx),), name='fused_input')  # Stacking fusion method
        fusion_feature = Input((len(fusion_idx)*self.private_dim+self.shared_dim,), name='fused_input')  # Sum average and PoE method
        # x = Dense(5, activation='relu')(fusion_feature)
        # x = BatchNormalization()(x)
        # x = Dense(10, activation='relu')(fusion_feature)
        # x = BatchNormalization()(x)

        x = Dense(1, activation='sigmoid', name='grading')(fusion_feature)
        model = Model(inputs=fusion_feature, outputs=x)
        return model
def evalution_metric(y_label, y_probas):
    fpr, tpr, thresholds = roc_curve(y_label, y_probas[:, 0])
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]

    y_pred_class = np.where(y_probas[:, 0] > best_threshold, 1, 0)

    accuracy = accuracy_score(y_label, y_pred_class)
    # print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_label, y_pred_class)
    # print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_label, y_pred_class)
    # print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_label, y_pred_class)
    # print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_label, y_pred_class)
    # print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc_value = roc_auc_score(y_label, y_probas[:, 0])
    # print('ROC AUC: %f' % auc)
    # confusion matrix
    confusion = confusion_matrix(y_label, y_pred_class)
    # print(confusion)

    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    # print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    # print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    # print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    # print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_score(y_label, y_pred_class)
    # print("\nJaccard similarity score: " + str(jaccard_index))

    npv = confusion[0, 0]/(confusion[0, 0]+confusion[1, 0])

    labels = [0, 1]
    target_names = ["LrGG", "HGG"]
    # print(classification_report(y_label, y_pred_class, labels=labels, target_names=target_names))

    return auc_value, accuracy, sensitivity, specificity, precision, recall, npv, f1, confusion, jaccard_index, kappa

def Generate_missing_data(N, M, missing_percentages, y_label):
    "设定缺失一个模态的样本比例为r1, 缺失2个模态的样本比例为r2...缺失（M-1)个模态的样本比例为r(M-1)"
    missing_matrix = np.ones((N, M))  # 初始化为全1矩阵

    remaining_indices = list(range(N))
    for i in range(len(missing_percentages)):
        num_missing = int(N * missing_percentages[i])  # 缺失样本的数量

        indices = np.random.choice(remaining_indices, num_missing, replace=False)  # 选择的缺失的样本索引

        if len(remaining_indices) < num_missing:
            raise ValueError(f"缺失数量过多，无法满足条件。")

        if i == 0:  # 只缺失一个模态的情况
            for j in range(num_missing):
                # modalities_to_missing = np.random.randint(0, M)  # 随机选择要缺失的模态

                # 对于3个模态，T1WI:0, CE-T1WI:1, T2-FLAIR:2
                modalities_to_missing = 0  # 0, 1, 2
                missing_matrix[indices[j], modalities_to_missing] = 0

        else:  # 缺失多个模态的情况
            num_modalities_to_missing = i + 1  # 缺失的模态数量
            for j in range(num_missing):
                "这里到时候也可以指定固定哪两个模态缺失，比如T1WI+CE-T1WI， 但是我这里是假定同一个病人同时缺失2种模态的情况，" \
                "没有考虑那种部分病人缺失T1WI，部分病人缺失另一个模态如CE-T1WI，然后这两个缺失的比例加起来才是10%，这种情况太复杂先不考虑"

                # modalities_to_missing = np.random.choice(range(M), num_modalities_to_missing, replace=False)
                modalities_to_missing = [0, 1]  # T1WI:0, CE-T1WI:1, T2-FLAIR:2

                missing_matrix[indices[j], modalities_to_missing] = 0

        # 剔除上一步已经缺失的样本索引
        remaining_indices = np.setdiff1d(remaining_indices, indices)

    return missing_matrix
def preprocess_data(Datas, fusion_idx, missing_idxs):
    new_Data = []
    for i in range(len(Datas)):
        Datai = Datas[i]

        # 判断当前模态i是否要融合的模态，是的话就进行缺失，否则不处理
        if i in fusion_idx:
            # 找到缺失那个模态的位置
            idx = fusion_idx.index(i)

            # 将对应模态的列扩充为跟Data一样的形状
            idxi = np.expand_dims(missing_idxs[:, idx], axis=1)
            expanded_idxi = np.tile(idxi, (1, Datai.shape[1]))

            # Datai和idxi对应位置元素相乘
            masked_data = Datai * expanded_idxi

            # 添加到new_Data列表中
            new_Data.append(masked_data)
        else:
            new_Data.append(Datai)

    return new_Data
def Standardization(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
def apply_smote(X_train, y_train, X_test, y_test):
    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    y_resampled = np.reshape(y_resampled, (len(y_resampled), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    return X_resampled, y_resampled, X_test, y_test


if __name__ == '__main__':
    "Load data: 只使用4个常规的MRI序列"
    data_Flair = (pd.read_csv('D:/Glioma/2-Features/zhengyi/Grade/FLAIR.csv')).values
    data_T1ce = (pd.read_csv('D:/Glioma/2-Features/zhengyi/Grade/T1C.csv')).values
    data_T1 = (pd.read_csv('D:/Glioma/2-Features/zhengyi/Grade/T1WI.csv')).values
    data_T2 = (pd.read_csv('D:/Glioma/2-Features/zhengyi/Grade/T2WI.csv')).values

    X_Flair, y_Flair = data_Flair[:, 2:111], (data_Flair[:, 1]).astype(int)
    X_T1ce, y_T1ce = data_T1ce[:, 2:111], (data_T1ce[:, 1]).astype(int)
    X_T1, y_T1 = data_T1[:, 2:111], (data_T1[:, 1]).astype(int)
    X_T2, y_T2 = data_T2[:, 2:111], (data_T2[:, 1]).astype(int)
    origin_label = [y_T1, y_T1ce, y_T2, y_Flair]

    data_Flair_test = (pd.read_csv('D:/Glioma/2-Features/tiantan/Grade/FLAIR.csv')).values  # 分级用combat，IDH用原始的
    data_T1ce_test = (pd.read_csv('D:/Glioma/2-Features/tiantan/Grade/T1C.csv')).values
    data_T1_test = (pd.read_csv('D:/Glioma/2-Features/tiantan/Grade/T1WI.csv')).values
    data_T2_test = (pd.read_csv('D:/Glioma/2-Features/tiantan/Grade/T2WI.csv')).values
    X_Flair_test, y_Flair_test = data_Flair_test[:, 2:111], (data_Flair_test[:, 1]).astype(int)
    X_T1ce_test, y_T1ce_test = data_T1ce_test[:, 2:111], (data_T1ce_test[:, 1]).astype(int)
    X_T1_test, y_T1_test = data_T1_test[:, 2:111], (data_T1_test[:, 1]).astype(int)
    X_T2_test, y_T2_test = data_T2_test[:, 2:111], (data_T2_test[:, 1]).astype(int)
    origin_label_te = [y_T1_test, y_T1ce_test, y_T2_test, y_Flair_test]

    '标准化'
    X_T1, X_T1_test = Standardization(X_T1, X_T1_test)
    X_T1ce, X_T1ce_test = Standardization(X_T1ce, X_T1ce_test)
    X_T2, X_T2_test = Standardization(X_T2, X_T2_test)
    X_Flair, X_Flair_test = Standardization(X_Flair, X_Flair_test)
    "=================================================="

    'Define model parameters'
    num_epochs = 100
    nfolds = 5
    fusion_idx = [0, 1, 3]  # 各种融合组合的训练和测试,0: T1WI, 1: T1ce, 2: T2WI, 3: T2-FLAIR， 缺3个序列的索引为0， 1， 3
    folder_name = "+".join(str(idx) for idx in fusion_idx)
    params = [0.01, 0.00001, 0.001, 0.001]  # weight of each loss
    feature_dim = 10
    private_dim = 4
    shared_dim = 6
    N, d = X_T1ce.shape[0], X_T1ce.shape[1]  # 样本数
    N_test, d_test = X_T1ce_test.shape[0], X_T1ce_test.shape[1]
    M = len(fusion_idx)  # 模态数

    METRICS = [
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.BinaryAccuracy(name='acc')
    ]

    "Randomly initialize some modalities of some patients to 0 (missing modalities) and retu''rn the missing index"
    ratio = '10%'
    # missing_percentages = [0, 0.1]  # 各种情况的缺失比例[缺失1个序列的比例，缺失2个序列比例，缺失3个序列比例], 维度是M-1
    missing_percentages = [0.1, 0]

    total_best_result = []

    for ite in [1]:

        'The original features and generated features'
        origin_features_total_tr = np.zeros((4, N, d))
        generate_features_total_tr = np.zeros((4, N, d))
        origin_features_total_te = np.zeros((4, N_test, d))
        generate_features_total_te = np.zeros((4, N_test, d))
        for l in range(4):
            origin_features_total_tr[l] = [X_T1, X_T1ce, X_T2, X_Flair][l]
            origin_features_total_te[l] = [X_T1_test, X_T1ce_test, X_T2_test, X_Flair_test][l]

        "Train的缺失索引"
        missing_idxs_tr = Generate_missing_data(N, M, missing_percentages, y_T1)
        missing_datas_tr = preprocess_data([X_T1, X_T1ce, X_T2, X_Flair], fusion_idx, missing_idxs_tr)
        new_X_T1, new_X_T1ce, new_X_T2, new_X_Flair = missing_datas_tr[0], missing_datas_tr[1], missing_datas_tr[2], missing_datas_tr[3]  # 4个序列中，部分序列是缺失的

        "Test的缺失索引计算"
        missing_idxs_te = Generate_missing_data(N_test, M, missing_percentages, y_T1_test)
        missing_datas_te = preprocess_data([X_T1_test, X_T1ce_test, X_T2_test, X_Flair_test], fusion_idx, missing_idxs_te)
        new_X_T1_test, new_X_T1ce_test, new_X_T2_test, new_X_Flair_test = missing_datas_te[0], missing_datas_te[1], missing_datas_te[2],  missing_datas_te[3]

        "对Train进行SMOTE增强"
        sm = SMOTE()
        new_X_T1_sm, y_T1_sm, new_X_T1_test, y_T1_test = apply_smote(new_X_T1, y_T1, new_X_T1_test, y_T1_test)
        new_X_T1ce_sm, y_T1ce_sm, new_X_T1ce_test, y_T1ce_test = apply_smote(new_X_T1ce, y_T1ce, new_X_T1ce_test, y_T1ce_test)
        new_X_T2_sm, y_T2_sm, new_X_T2_test, y_T2_test = apply_smote(new_X_T2, y_T2, new_X_T2_test, y_T2_test)
        new_X_Flair_sm, y_Flair_sm, new_X_Flair_test, y_Flair_test = apply_smote(new_X_Flair, y_Flair, new_X_Flair_test, y_Flair_test)

        inputs = [new_X_T1_sm, new_X_T1ce_sm, new_X_T2_sm, new_X_Flair_sm]
        inputs_test = [new_X_T1_test, new_X_T1ce_test, new_X_T2_test, new_X_Flair_test]

        'Define model'
        initial_learning_rate = 0.01  # 初始学习率, TCIA:0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=40, decay_rate=0.5, staircase=True)
        model = MDVAEmodel_multiple(inp_shape=inputs[0].shape[1:], latent_dim=feature_dim, private_dim=private_dim, shared_dim=shared_dim)
        model.compile(optimizer=Adam(lr=initial_learning_rate), metrics=METRICS)

        "==================2023.11.25: 保存训练集和测试集的结果====================="
        fusion_features_tr = np.zeros((num_epochs, len(y_T1_sm), M * private_dim + shared_dim))
        confusion_matrix_tr = np.zeros((num_epochs, 2, 2))
        y_probas_tr = np.zeros((num_epochs, len(y_T1_sm))) # 这里设置了smote数据增强，长度不再是train了
        y_preds_tr = np.zeros((num_epochs, len(y_T1_sm)))
        y_trues_tr = np.zeros((num_epochs, len(y_T1_sm)))
        y_true_tr = y_T1_sm

        fusion_features_va = np.zeros((num_epochs, len(y_T1_test), M * private_dim + shared_dim))
        confusion_matrix_va = np.zeros((num_epochs, 2, 2))
        y_probas_va = np.zeros((num_epochs, len(y_T1_test)))
        y_preds_va = np.zeros((num_epochs, len(y_T1_test)))
        y_trues_va = np.zeros((num_epochs, len(y_T1_test)))
        y_true_va = y_T1_test
        "==================2023.11.25: 保存训练集和验证集的结果====================="

        "==================Begin Training=================="
        auc_list = []  # 存储每一折100次迭代的值
        acc_list = []
        sen_list = []
        spe_list = []
        precision_list = []
        f1score_list = []
        npv_list = []
        jaccard_list = []
        cohen_list = []
        total_list = []
        rec_list = []
        kl_list = []
        grade_list = []

        auc_list_ = []
        acc_list_ = []
        sen_list_ = []
        spe_list_ = []
        precision_list_ = []
        f1score_list_ = []
        npv_list_ = []
        jaccard_list_ = []
        cohen_list_ = []
        total_list_ = []
        rec_list_ = []
        kl_list_ = []
        grade_list_ = []

        loss_not_improved_count = 0
        best_loss = float('inf')
        global epoch
        for epoch in range(num_epochs):

            "Training"
            result_train = model.train_step(inputs, y_T1_sm, missing_idxs_tr)  # 这个缺失的索引只有未smote前的，在train_step时需要增加smote后的长度才能和inputs相对应

            total_loss = result_train['loss']
            rec_loss = result_train['rec_loss']
            kl_loss = result_train['kl_loss']
            grade_loss = result_train['grade_loss']
            decoder_out = result_train['Decoders_out']  # 11.08 增加解码器的输出
            decoder_out_fusion = result_train["Decoders_fusion_out"]  # 01.30 增加自身特异+融合表示的编码输出
            common_code = result_train['common_code']
            grade_out = result_train['grade_out']  # 11.20 增加编码器的输出概率
            auc_tr, acc_tr, sen_tr, spe_tr, precision_tr, recall_tr, npv_tr, f1score_tr, confusion_tr, jaccard_tr, cohen_tr = evalution_metric(y_T1_sm, grade_out)  # IDH

            # print(f'Epoch {epoch + 1}, AUC: {auc_tr}, ACC:{acc_tr}, SEN:{sen_tr}, SPE:{spe_tr}, Precision: {precision_tr}, '
            #       f'Recall: {recall_tr}, F1-score:{f1score_tr}, NPV:{npv_tr}, Loss: {total_loss}')
            # print(f'Epoch {epoch + 1}, AUC: {auc_tr}, ACC:{acc_tr}, SEN:{sen_tr}, SPE:{spe_tr}, PPV:{precision_tr}, Loss: {total_loss}')

            "Testing"
            results_valid = model.test_step(inputs_test, y_T1_test, missing_idxs_te)

            total_loss_ = results_valid['loss']
            rec_loss_ = results_valid['rec_loss']
            kl_loss_ = results_valid['kl_loss']
            grade_loss_ = results_valid['grade_loss']
            decoder_out_ = results_valid['Decoders_out']  # 11.08 增加解码器的输出
            decoder_out_fusion_ = results_valid["Decoders_fusion_out"]  # 01.30 增加自身特异+融合表示的编码输出
            common_code_ = results_valid['common_code']  # 11.17 增加编码后输入分类器的共享潜在表示
            grade_out_ = results_valid['grade_out']  # 11.20 增加编码器的输出概率
            auc_, acc_, sen_, spe_, precision_, recall_, npv_, f1score_, confusion_, jaccard_, cohen_ = evalution_metric(y_T1_test, grade_out_)  # IDH

            print(f'Valid Result - AUC: {auc_}, ACC: {acc_}, SEN:{sen_}, SPE:{spe_}, PPV:{precision_}, total_Loss: {total_loss_}')

            "Save values to plot"
            auc_list.append(auc_tr)
            auc_list_.append(auc_)
            acc_list.append(acc_tr)
            acc_list_.append(acc_)
            sen_list.append(sen_tr)
            sen_list_.append(sen_)
            spe_list.append(spe_tr)
            spe_list_.append(spe_)
            precision_list.append(precision_tr)
            precision_list_.append(precision_)
            f1score_list.append(f1score_tr)
            f1score_list_.append(f1score_)
            npv_list.append(npv_tr)
            npv_list_.append(npv_)
            jaccard_list.append(jaccard_tr)
            jaccard_list_.append(jaccard_)
            cohen_list.append(cohen_tr)
            cohen_list_.append(cohen_)
            total_list.append(total_loss)
            total_list_.append(total_loss_)
            rec_list.append(rec_loss)
            rec_list_.append(rec_loss_)
            kl_list.append(kl_loss)
            kl_list_.append(kl_loss_)
            grade_list.append(grade_loss)
            grade_list_.append(grade_loss_)

            "2023.11.25:计算训练集和验证集的混淆矩阵、预测概率"
            fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_true_tr, grade_out)
            youden_index = tpr_tr - fpr_tr
            best_threshold_tr = thresholds_tr[np.argmax(youden_index)]
            y_pred_tr = np.where(grade_out > best_threshold_tr, 1, 0)
            # y_pred_tr = np.where(grade_out > 0.5, 1, 0)
            y_preds_tr[epoch] = np.reshape(y_pred_tr, y_pred_tr.shape[0])
            y_trues_tr[epoch] = np.reshape(y_true_tr, y_true_tr.shape[0])
            y_probas_tr[epoch] = np.reshape(grade_out, grade_out.shape[0])
            confusion_matrix_tr[epoch] = confusion_matrix(y_true_tr, y_pred_tr)
            fusion_features_tr[epoch] = common_code

            fpr_va, tpr_va, thresholds_va = roc_curve(y_true_va, grade_out_)
            youden_index_ = tpr_va - fpr_va
            best_threshold_va = thresholds_va[np.argmax(youden_index_)]
            y_pred_va = np.where(grade_out_ > best_threshold_va, 1, 0)
            # y_pred_va = np.where(grade_out_ > 0.5, 1, 0)
            y_preds_va[epoch] = np.reshape(y_pred_va, y_pred_va.shape[0])
            y_trues_va[epoch] = np.reshape(y_true_va, y_pred_va.shape[0])
            y_probas_va[epoch] = np.reshape(grade_out_, grade_out_.shape[0])
            confusion_matrix_va[epoch] = confusion_matrix(y_true_va, y_pred_va)
            fusion_features_va[epoch] = common_code_

            "2024.01.30： 下一次迭代开始前，将缺失模态自身重建+联合重建的均值替换原始的缺失部分"
            for j in range(len(fusion_idx)):
                "2023.11.09: 使用重构的数据补全缺失数据"
                missing_idxs_train = missing_idxs_tr[:, j]  # 959, 原始未SMOTE的训练集样本数
                smote_idxs_train = np.ones((inputs[fusion_idx[j]].shape[0] - len(missing_idxs_train)))  # 进行SMOET后增加的样本
                new_idxs_train = np.hstack((missing_idxs_train, smote_idxs_train))

                inputs[fusion_idx[j]][new_idxs_train == 0] = (decoder_out[fusion_idx[j]][new_idxs_train == 0] + decoder_out_fusion[fusion_idx[j]][new_idxs_train == 0])/2

                missing_idxs_test = missing_idxs_te[:, j]  # 240
                inputs_test[fusion_idx[j]][missing_idxs_test == 0] = (decoder_out_[fusion_idx[j]][missing_idxs_test == 0] + decoder_out_fusion_[fusion_idx[j]][missing_idxs_test == 0])/2

                "2023.11.20：用重构的数据替代原始的数据"
                generate_features_total_tr[fusion_idx[j]] = inputs[fusion_idx[j]][0:len(missing_idxs_train)]
                generate_features_total_te[fusion_idx[j]] = inputs_test[fusion_idx[j]]

            "2024.03.08"
            # Check if validation loss has improved
            if total_loss_ < best_loss:
                best_loss = total_loss_
                loss_not_improved_count = 0
            else:
                loss_not_improved_count += 1

            # Print validation result
            print(f'Valid Result - total_Loss: {total_loss_}')

            # Check if validation loss has not improved for 10 epochs
            if loss_not_improved_count >= 10:
                print('Validation loss has not improved for 10 epochs. Stopping training.')
                break

        epoch = np.argmax(auc_list_)
        # print(f'Best Train Result of 5 folds - AUC: {auc_list[epoch]}, ACC: {acc_list[epoch]}, SEN/Recall:{sen_list[epoch]}, SPE:{spe_list[epoch]},'
        #       f'Precision/PPV:{precision_list[epoch]}, F1-score:{f1score_list[epoch]}, NPV:{npv_list[epoch]}, Jaccard_score:{jaccard_list[epoch]}, '
        #       f'Cohen__kappa:{cohen_list[epoch]}, total_Loss: {total_list[epoch]}')
        print(f'Best Valid Result of 5 folds - AUC: {auc_list_[epoch]}, ACC: {acc_list_[epoch]}, SEN/Recall:{sen_list_[epoch]}, SPE:{spe_list_[epoch]},'
              f'Precision/PPV:{precision_list_[epoch]}, F1-score:{f1score_list_[epoch]}, NPV:{npv_list_[epoch]},Jaccard_score:{jaccard_list_[epoch]},'
              f'Cohen__kappa:{cohen_list_[epoch]}, total_Loss: {total_list_[epoch]}')

        "保存训练集和验证集结果（AUC, ACC, SEN, SPE)"
        auc_values = auc_list_[epoch]

        best_result = []
        best_result.append(auc_list[epoch])
        best_result.append(acc_list[epoch])
        best_result.append(sen_list[epoch])
        best_result.append(spe_list[epoch])
        best_result.append(precision_list[epoch])
        best_result.append(f1score_list[epoch])
        best_result.append(npv_list[epoch])
        best_result.append(jaccard_list[epoch])
        best_result.append(cohen_list[epoch])
        best_result.append(auc_list_[epoch])
        best_result.append(acc_list_[epoch])
        best_result.append(sen_list_[epoch])
        best_result.append(spe_list_[epoch])
        best_result.append(precision_list_[epoch])
        best_result.append(f1score_list_[epoch])
        best_result.append(npv_list_[epoch])
        best_result.append(jaccard_list_[epoch])
        best_result.append(cohen_list_[epoch])

        # Path = 'D:/Glioma/3-Results/Train and Test (IDH)/Incomplete/' + folder_name +'/missing(0-T1WI)/'+ ratio +'/TCIA'
        # sub_Path = Path + '/Metrics/' + str(ite)
        # if not os.path.exists(sub_Path):
        #     os.makedirs(sub_Path)
        # np.save(sub_Path + '/' + 'best_result.npy', best_result)

        "2023.11.25: 分别保存最后那一次epoch的训练样本以及验证样本的预测概率、预测标签、混淆矩阵、融合特征"
        Y_probas_list_train_best = y_probas_tr[epoch]
        Y_preds_list_train_best = np.where(y_probas_tr[epoch]> 0.5, 1, 0)
        Y_trues_list_train_best = y_trues_tr[epoch]
        Fusion_feature_list_train_best = fusion_features_tr[epoch]
        Confusion_matrix_list_train_best = confusion_matrix_tr[epoch]

        Y_probas_list_valid_best = y_probas_va[epoch]
        Y_preds_list_valid_best = np.where(y_probas_va[epoch] > 0.5, 1, 0)
        Y_trues_list_valid_best = y_trues_va[epoch]
        Fusion_feature_list_valid_best = fusion_features_va[epoch]
        Confusion_matrix_list_valid_best = confusion_matrix_va[epoch]

        # sub_Path1 = Path + '/Train/' + str(ite)
        # if not os.path.exists(sub_Path1):
        #     os.makedirs(sub_Path1)
        # np.save(sub_Path1 + '/Y_probas.npy', Y_probas_list_train_best)
        # np.save(sub_Path1 + '/Y_preds.npy', Y_preds_list_train_best)
        # np.save(sub_Path1 + '/Y_trues.npy', Y_trues_list_train_best)
        # np.save(sub_Path1 + '/fusion_features.npy', Fusion_feature_list_train_best)
        # np.save(sub_Path1 + '/confusion_matrix.npy', Confusion_matrix_list_train_best)

        # sub_Path2 = Path + '/Test/' + str(ite)
        # if not os.path.exists(sub_Path2):
        #     os.makedirs(sub_Path2)
        # np.save(sub_Path2 + '/Y_probas.npy', Y_probas_list_valid_best)
        # np.save(sub_Path2 + '/Y_preds.npy', Y_preds_list_valid_best)
        # np.save(sub_Path2 + '/Y_trues.npy', Y_trues_list_valid_best)
        # np.save(sub_Path2 + '/fusion_features.npy', Fusion_feature_list_valid_best)
        # np.save(sub_Path2 + '/confusion_matrix.npy', Confusion_matrix_list_valid_best)


        # "2023.11.27: 保持重构前后的完整数据、缺失数据"
        # sub_Path3 = Path + '/Missing samples/' + str(ite)
        # if not os.path.exists(sub_Path3):
        #     os.makedirs(sub_Path3)

        for k in range(len(fusion_idx)):
            "Train: 原始的缺失数据+补全后的缺失数据"
            origin_missing_data = origin_features_total_tr[fusion_idx[k]][missing_idxs_tr[:, k] == 0]
            generate_missing_data = generate_features_total_tr[fusion_idx[k]][missing_idxs_tr[:, k] == 0]
            missing_label = origin_label[fusion_idx[k]][missing_idxs_tr[:, k] == 0]

            "Train:原始的完整数据+补全后的完整数据"
            origin_complete_data = origin_features_total_tr[fusion_idx[k]][missing_idxs_tr[:, k] != 0]
            generate_complete_data = generate_features_total_tr[fusion_idx[k]][missing_idxs_tr[:, k] != 0]
            complete_label = origin_label[fusion_idx[k]][missing_idxs_tr[:, k] != 0]

            "Test: 原始的缺失数据+补全后的缺失数据"
            origin_missing_data_te = origin_features_total_te[fusion_idx[k]][missing_idxs_te[:, k] == 0]
            generate_missing_data_te = generate_features_total_te[fusion_idx[k]][missing_idxs_te[:, k] == 0]
            missing_label_te = origin_label_te[fusion_idx[k]][missing_idxs_te[:, k] == 0]

            "Test:原始的完整数据+补全后的完整数据"
            origin_complete_data_te = origin_features_total_te[fusion_idx[k]][missing_idxs_te[:, k] != 0]
            generate_complete_data_te = generate_features_total_te[fusion_idx[k]][missing_idxs_te[:, k] != 0]
            complete_label_te = origin_label_te[fusion_idx[k]][missing_idxs_te[:, k] != 0]

        #     # 使用 np.savez 保存多个数组到一个文件
        #     np.savez(sub_Path3 + '/' + 'train_missing_samples_' + str(fusion_idx[k]) + '_' + str(ite) + '.npz',
        #              origin_missing_data=origin_missing_data,
        #              generate_missing_data=generate_missing_data,
        #              label=missing_label)
        #     np.savez(sub_Path3 + '/' + 'train_complete_samples_' + str(fusion_idx[k]) + '_' + str(ite) + '.npz',
        #              origin_complete_data=origin_complete_data,
        #              generate_complete_data=generate_complete_data,
        #              label=complete_label)
        #     np.savez(sub_Path3 + '/' + 'test_missing_samples_' + str(fusion_idx[k]) + '_' + str(ite) + '.npz',
        #              origin_missing_data=origin_missing_data_te,
        #              generate_missing_data=generate_missing_data_te,
        #              label=missing_label_te)
        #     np.savez(sub_Path3 + '/' + 'test_complete_samples_' + str(fusion_idx[k]) + '_' + str(ite) + '.npz',
        #              origin_complete_data=origin_complete_data_te,
        #              generate_complete_data=generate_complete_data_te,
        #              label=complete_label_te)
        #
        # best_result = np.array(best_result)
        # best_result = np.reshape(best_result, (1, len(best_result)))
        # df = pd.DataFrame(best_result)
        # df.to_excel(Path + '/Metrics/best_result.xlsx', header=False, index=False)










