from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
from imblearn.over_sampling import SMOTE
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
import shap
import matplotlib.pyplot as plt
import sys
import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore')
python_version = sys.version_info


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(
            0.5 * z_log_var) * epsilon  # Noise with mean + standard deviation obeying a normal distribution N(0, 1)

class Completion_model(Model):
    def __init__(self, inp_shape, latent_dim, private_dim, shared_dim, **kwargs):
        super(Completion_model, self).__init__(**kwargs)
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

        self.predict_classifier = self.build_classification_network()

        # self.Flair_encoder.summary()
        # self.predict_classifier.summary()

        self.MMDR_model = self.build_MMDR_model()
        self.custom_metrics = METRICS
        # self.MMDR_model.summary()

    def build_MMDR_model(self):

        MMDR_input_Flair = Input(shape=self.inp_shape, name='MMDR_input_FLair')
        MMDR_input_T1ce = Input(shape=self.inp_shape, name='MMDR_input_T1ce')
        MMDR_input_T1 = Input(shape=self.inp_shape, name='MMDR_input_T1')
        MMDR_input_T2 = Input(shape=self.inp_shape, name='MMDR_input_T2')

        "Load the output of each sequence's encoder and decoder"
        Flair_encoder_out = self.Flair_encoder(MMDR_input_Flair)
        T1ce_encoder_out = self.T1ce_encoder(MMDR_input_T1ce)
        T1_encoder_out = self.T1_encoder(MMDR_input_T1)
        T2_encoder_out = self.T2_encoder(MMDR_input_T2)

        Flair_decoder_out = self.Flair_decoder([Flair_encoder_out[2], Flair_encoder_out[5]])
        T1ce_decoder_out = self.T1ce_decoder([T1ce_encoder_out[2], T1ce_encoder_out[5]])
        T1_decoder_out = self.T1_decoder([T1_encoder_out[2], T1_encoder_out[5]])
        T2_decoder_out = self.T2_decoder([T2_encoder_out[2], T2_encoder_out[5]])

        z_mean_private_out = [T1_encoder_out[0], T1ce_encoder_out[0], T2_encoder_out[0], Flair_encoder_out[0]]
        z_log_var_private_out = [T1_encoder_out[1], T1ce_encoder_out[1], T2_encoder_out[1], Flair_encoder_out[1]]
        z_private_out = [T1_encoder_out[2], T1ce_encoder_out[2], T2_encoder_out[2],
                         Flair_encoder_out[2]]  # sequence-specific representation
        z_mean_shared_out = [T1_encoder_out[3], T1ce_encoder_out[3], T2_encoder_out[3], Flair_encoder_out[3]]
        z_log_var_shared_out = [T1_encoder_out[4], T1ce_encoder_out[4], T2_encoder_out[4], Flair_encoder_out[4]]
        z_shared_out = [T1_encoder_out[5], T1ce_encoder_out[5], T2_encoder_out[5],
                        Flair_encoder_out[5]]  # sequence-neutral representation
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

        z_common_fusion = Lambda(
            lambda x: tf.reduce_mean(tf.stack([x[i] for i in list(range(len(fusion_idx)))], axis=0), axis=0))(
            z_shared_add)
        z_private_add.append(z_common_fusion)

        "The reconstructed output of each sequence's specific and the fusion component"
        Flair_fusion_decoder_out = self.Flair_decoder([Flair_encoder_out[2], z_common_fusion])
        T1ce_fusion_decoder_out = self.T1ce_decoder([T1ce_encoder_out[2], z_common_fusion])
        T1_fusion_decoder_out = self.T1_decoder([T1_encoder_out[2], z_common_fusion])
        T2_fusion_decoder_out = self.T2_decoder([T2_encoder_out[2], z_common_fusion])

        "Multi-sequences fusion"
        fusion_code = tf.concat(z_private_add, axis=1)

        "The predictive probability"
        predict_out = self.predict_classifier(fusion_code)

        model = Model(inputs=[MMDR_input_T1, MMDR_input_T1ce, MMDR_input_T2, MMDR_input_Flair],
                      outputs=[T1_encoder_out, T1ce_encoder_out, T2_encoder_out, Flair_encoder_out,
                               T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out,
                               T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out,
                               Flair_fusion_decoder_out,
                               fusion_code, predict_out],
                      name="MMDR")
        return model

    def train_step(self, x, y, missing_idxs):
        with tf.GradientTape() as tape:
            T1_encoder_out, T1ce_encoder_out, T2_encoder_out, Flair_encoder_out, \
                T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out, \
                T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out, Flair_fusion_decoder_out, fusion_code, predict_out = self.MMDR_model(
                x)

            "Get the mean, variance, and latant representation of sequence-specific and sequence-neutral representation, respectively"
            z_mean_private_Flair, z_log_var_private_Flair, z_private_Flair, z_mean_shared_Flair, z_log_var_shared_Flair, z_shared_Flair = Flair_encoder_out
            z_mean_private_T1, z_log_var_private_T1, z_private_T1, z_mean_shared_T1, z_log_var_shared_T1, z_shared_T1 = T1_encoder_out
            z_mean_private_T1ce, z_log_var_private_T1ce, z_private_T1ce, z_mean_shared_T1ce, z_log_var_shared_T1ce, z_shared_T1ce = T1ce_encoder_out
            z_mean_private_T2, z_log_var_private_T2, z_private_T2, z_mean_shared_T2, z_log_var_shared_T2, z_shared_T2 = T2_encoder_out

            z_log_vars_private = [z_log_var_private_T1, z_log_var_private_T1ce, z_log_var_private_T2,
                                  z_log_var_private_Flair]
            z_means_private = [z_mean_private_T1, z_mean_private_T1ce, z_mean_private_T2, z_mean_private_Flair]
            z_log_vars_shared = [z_log_var_shared_T1, z_log_var_shared_T1ce, z_log_var_shared_T2,
                                 z_log_var_shared_Flair]
            z_means_shared = [z_mean_shared_T1, z_mean_shared_T1ce, z_mean_shared_T2, z_mean_shared_Flair]
            Decoders_out = [T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out]
            Decoders_fusion_out = [T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out,
                                   Flair_fusion_decoder_out]
            z_private_out = [z_private_T1, z_private_T1ce, z_private_T2, z_private_Flair]
            z_shared_out = [z_shared_T1, z_shared_T1ce, z_shared_T2, z_shared_Flair]

            "Calculation of the total loss consisting of reconstruction loss, KL loss and classification loss"
            reconstruction_loss = 0
            kl_loss = 0
            com_loss = 0
            spe_loss = 0
            epsilon = 1e-10

            "For multi-sequences"
            if len(fusion_idx) > 1:
                for i in range(len(fusion_idx)):
                    "The kl loss"
                    kl_loss = kl_loss + (-0.5 * tf.reduce_mean(
                        1 + z_log_vars_private[fusion_idx[i]] - tf.square(z_means_private[fusion_idx[i]]) - tf.exp(
                            z_log_vars_private[fusion_idx[i]]))) \
                              + (-0.5 * tf.reduce_mean(
                        1 + z_log_vars_shared[fusion_idx[i]] - tf.square(z_means_shared[fusion_idx[i]]) - tf.exp(
                            z_log_vars_shared[fusion_idx[i]])))

                    "The reconstruction loss for complete samples"
                    missing_idxs_tr = missing_idxs[:, i]  # Unenhanced patient index
                    smote_idxs_tr = np.ones((x[fusion_idx[i]].shape[0] - len(missing_idxs_tr)))
                    new_idxs_tr = np.hstack((missing_idxs_tr, smote_idxs_tr))
                    reconstruction_loss = reconstruction_loss + \
                                          tf.reduce_mean(tf.sqrt(
                                              tf.keras.losses.mean_squared_error(x[fusion_idx[i]][new_idxs_tr != 0],
                                                                                 Decoders_out[fusion_idx[i]][
                                                                                     new_idxs_tr != 0]))) \
                                          + tf.reduce_mean(tf.sqrt(
                        tf.keras.losses.mean_squared_error(x[fusion_idx[i]][new_idxs_tr != 0],
                                                           Decoders_fusion_out[fusion_idx[i]][new_idxs_tr != 0])))

                    "The disentangled loss"
                    if i < len(fusion_idx) - 1:
                        for j in range(i + 1, len(fusion_idx)):
                            com_loss += tf.reduce_mean(
                                tf.sqrt(mean_squared_error(z_shared_out[fusion_idx[i]], z_shared_out[fusion_idx[j]])))
                            spe_loss += tf.reduce_mean(
                                tf.sqrt(mean_squared_error(z_private_out[fusion_idx[i]], z_private_out[fusion_idx[j]])))
                com_spec_loss = com_loss / spe_loss
            else:
                "For single sequence"
                kl_loss = (-0.5 * tf.reduce_mean(
                    1 + z_log_vars_private[fusion_idx[0]] - tf.square(z_means_private[fusion_idx[0]]) - tf.exp(
                        z_log_vars_private[fusion_idx[0]]))) \
                          + (-0.5 * tf.reduce_mean(
                    1 + z_log_vars_shared[fusion_idx[0]] - tf.square(z_means_shared[fusion_idx[0]]) - tf.exp(
                        z_log_vars_shared[fusion_idx[0]])))
                reconstruction_loss = tf.reduce_mean(
                    tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[0]], Decoders_out[fusion_idx[0]])))
                com_spec_loss = com_loss / (spe_loss + epsilon)  # Preventing division by 0

            "The classification loss"
            classification_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, predict_out))

            "The total loss"
            total_loss = params[0] * reconstruction_loss + params[1] * kl_loss + params[2] * com_spec_loss + params[
                3] * classification_loss

        "Calculate the gradient"
        grads = tape.gradient(total_loss, self.trainable_variables)

        "Updating model parameters and learning rates using the optimizer"
        self.optimizer.learning_rate = lr_schedule(epoch)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        print(">>>>>>>>>>>>>>metrics_names:", self.metrics_names)
        print(f"Epoch {epoch + 1}, Learning Rate: {self.optimizer.lr.numpy()}")

        "Update metrics manually"
        for metric in self.custom_metrics:
            metric.update_state(y, predict_out)

        return {
            "Decoders_out": Decoders_out,
            "Decoders_fusion_out": Decoders_fusion_out,
            "predict_out": predict_out,
            "fusion_code": fusion_code,
            "loss": total_loss,
            "rec_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "classification_loss": classification_loss,
            self.custom_metrics[0].name: self.custom_metrics[0].result(),
            self.custom_metrics[1].name: self.custom_metrics[1].result()
        }

    def test_step(self, x_, y_, missing_idxs):
        T1_encoder_out_, T1ce_encoder_out_, T2_encoder_out_, Flair_encoder_out_, \
            T1_decoder_out_, T1ce_decoder_out_, T2_decoder_out_, Flair_decoder_out_, \
            T1_fusion_decoder_out_, T1ce_fusion_decoder_out_, T2_fusion_decoder_out_, Flair_fusion_decoder_out_, fusion_code_, predict_out_ = self.MMDR_model(
            x_)

        "Get the mean, variance, and latant representation of sequence-specific and sequence-neutral representation, respectively"
        z_mean_private_Flair_, z_log_var_private_Flair_, z_private_Flair_, z_mean_shared_Flair_, z_log_var_shared_Flair_, z_shared_Flair_ = Flair_encoder_out_
        z_mean_private_T1_, z_log_var_private_T1_, z_private_T1_, z_mean_shared_T1_, z_log_var_shared_T1_, z_shared_T1_ = T1_encoder_out_
        z_mean_private_T1ce_, z_log_var_private_T1ce_, z_private_T1ce_, z_mean_shared_T1ce_, z_log_var_shared_T1ce_, z_shared_T1ce_ = T1ce_encoder_out_
        z_mean_private_T2_, z_log_var_private_T2_, z_private_T2_, z_mean_shared_T2_, z_log_var_shared_T2_, z_shared_T2_ = T2_encoder_out_

        z_log_vars_private_ = [z_log_var_private_T1_, z_log_var_private_T1ce_, z_log_var_private_T2_,
                               z_log_var_private_Flair_]
        z_means_private_ = [z_mean_private_T1_, z_mean_private_T1ce_, z_mean_private_T2_, z_mean_private_Flair_]
        z_log_vars_shared_ = [z_log_var_shared_T1_, z_log_var_shared_T1ce_, z_log_var_shared_T2_,
                              z_log_var_shared_Flair_]
        z_means_shared_ = [z_mean_shared_T1_, z_mean_shared_T1ce_, z_mean_shared_T2_, z_mean_shared_Flair_]
        Decoders_out_ = [T1_decoder_out_, T1ce_decoder_out_, T2_decoder_out_, Flair_decoder_out_]
        Decoders_fusion_out_ = [T1_fusion_decoder_out_, T1ce_fusion_decoder_out_, T2_fusion_decoder_out_,
                                Flair_fusion_decoder_out_]
        z_private_out_ = [z_private_T1_, z_private_T1ce_, z_private_T2_, z_private_Flair_]
        z_shared_out_ = [z_shared_T1_, z_shared_T1ce_, z_shared_T2_, z_shared_Flair_]

        "Calculation of the total loss consisting of reconstruction loss, KL loss and classification loss"
        reconstruction_loss_ = 0
        kl_loss_ = 0
        com_loss_ = 0
        spe_loss_ = 0
        epsilon = 1e-10

        "For multi-sequences"
        if len(fusion_idx) > 1:
            for i in range(len(fusion_idx)):
                "The kl loss"
                kl_loss_ = kl_loss_ + (-0.5 * tf.reduce_mean(
                    1 + z_log_vars_private_[fusion_idx[i]] - tf.square(z_means_private_[fusion_idx[i]]) - tf.exp(
                        z_log_vars_private_[fusion_idx[i]]))) \
                           + (-0.5 * tf.reduce_mean(
                    1 + z_log_vars_shared_[fusion_idx[i]] - tf.square(z_means_shared_[fusion_idx[i]]) - tf.exp(
                        z_log_vars_shared_[fusion_idx[i]])))

                "The reconstruction loss for complete samples"
                missing_idxs_te = missing_idxs[:, i]
                reconstruction_loss_ = reconstruction_loss_ + tf.reduce_mean(
                    tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]][missing_idxs_te != 0],
                                                               Decoders_out_[fusion_idx[i]][
                                                                   missing_idxs_te != 0]))) + tf.reduce_mean(
                    tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]][missing_idxs_te != 0],
                                                               Decoders_fusion_out_[fusion_idx[i]][
                                                                   missing_idxs_te != 0])))

                "The disentangled loss"
                if i < len(fusion_idx) - 1:
                    for j in range(i + 1, len(fusion_idx)):
                        com_loss_ += tf.reduce_mean(
                            tf.sqrt(mean_squared_error(z_shared_out_[fusion_idx[i]], z_shared_out_[fusion_idx[j]])))
                        spe_loss_ += tf.reduce_mean(
                            tf.sqrt(mean_squared_error(z_private_out_[fusion_idx[i]], z_private_out_[fusion_idx[j]])))
            com_spec_loss_ = com_loss_ / spe_loss_
        else:
            "For single sequence"
            kl_loss_ = (-0.5 * tf.reduce_mean(
                1 + z_log_vars_private_[fusion_idx[0]] - tf.square(z_means_private_[fusion_idx[0]]) - tf.exp(
                    z_log_vars_private_[fusion_idx[0]]))) \
                       + (-0.5 * tf.reduce_mean(
                1 + z_log_vars_shared_[fusion_idx[0]] - tf.square(z_means_shared_[fusion_idx[0]]) - tf.exp(
                    z_log_vars_shared_[fusion_idx[0]])))
            reconstruction_loss_ = tf.reduce_mean(
                tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[0]], Decoders_out_[fusion_idx[0]])))
            com_spec_loss_ = com_loss_ / (spe_loss_ + epsilon)

        "The classification loss"
        classification_loss_ = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_, predict_out_))

        "The total loss"
        total_loss_ = params[0] * reconstruction_loss_ + params[1] * kl_loss_ + params[2] * com_spec_loss_ + params[
            3] * classification_loss_

        "Update metrics manually"
        grade_out_ = tf.where(tf.math.is_nan(predict_out_), tf.zeros_like(predict_out_), predict_out_)
        for metric in self.custom_metrics:
            metric.update_state(y_, predict_out_)

        return {
            "Decoders_out": Decoders_out_,
            "Decoders_fusion_out": Decoders_fusion_out_,
            "predict_out": predict_out_,
            "fusion_code": fusion_code_,
            "loss": total_loss_,
            "rec_loss": reconstruction_loss_,
            "kl_loss": kl_loss_,
            "classification_loss": classification_loss_,
            self.custom_metrics[0].name: self.custom_metrics[0].result(),
            self.custom_metrics[1].name: self.custom_metrics[1].result()
        }

    def build_encoder_network(self, disentangled=True, name=''):
        input_layer = Input(shape=self.inp_shape)
        x = Dense(80, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dense(40, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(self.latent_dim, activation='relu')(x)
        if disentangled:
            mean_private = Dense(self.private_dim)(x)
            log_var_private = softplus(Dense(self.private_dim)(x))
            mean_shared = Dense(self.shared_dim)(x)
            log_var_shared = softplus(Dense(self.shared_dim)(x))

            z_private = Sampling()([mean_private, log_var_private])
            z_shared = Sampling()([mean_shared, log_var_shared])
            return Model(inputs=input_layer,
                         outputs=[mean_private, log_var_private, z_private, mean_shared, log_var_shared, z_shared],
                         name='encoder_{}'.format(name))
        else:
            return Model(inputs=input_layer, outputs=x, name='encoder_{}'.format(name))

    def build_decoder_network(self, output_shape, name=''):
        private_input = Input(shape=(self.private_dim,), name='input_1_{}'.format(name))
        shared_input = Input(shape=(self.shared_dim,), name='input_2_{}'.format(name))
        inp = Concatenate(axis=-1)([private_input, shared_input])
        l = BatchNormalization()(inp)
        l = Dense(40, activation='relu')(l)
        l = BatchNormalization()(l)
        l = Dense(80, activation='relu')(l)
        l = Dense(output_shape, activation='tanh')(l)
        return Model(inputs=[private_input, shared_input], outputs=l, name='decoder_{}'.format(name))

    def build_classification_network(self):
        fusion_feature = Input((len(fusion_idx) * self.private_dim + self.shared_dim,), name='fused_input')
        x = Dense(1, activation='sigmoid', name='grading')(fusion_feature)
        model = Model(inputs=fusion_feature, outputs=x)
        return model

def Standardization(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def Generate_missing_data(N, M, missing_percentages):
    missing_matrix = np.ones((N, M))
    remaining_indices = list(range(N))

    for i in range(len(missing_percentages)):
        num_missing = int(N * missing_percentages[i])

        # Check BEFORE random choice
        if len(remaining_indices) < num_missing:
            raise ValueError(f"Missing too many to meet conditions.")

        indices = np.random.choice(remaining_indices, num_missing, replace=False)

        "The case where only one sequence is missing"
        if i == 0:
            for j in range(num_missing):
                modalities_to_missing = 0  # T1WI:0, CE-T1WI:1, T2-FLAIR:2
                missing_matrix[indices[j], modalities_to_missing] = 0
        else:
            "The case where multiple sequences are missing"
            for j in range(num_missing):
                modalities_to_missing = [0, 1]  # T1WI:0, CE-T1WI:1, T2-FLAIR:2
                missing_matrix[indices[j], modalities_to_missing] = 0

        remaining_indices = np.setdiff1d(remaining_indices, indices)

    return missing_matrix

def preprocess_data(Datas, fusion_idx, missing_idxs):
    new_Data = []
    for i in range(len(Datas)):
        Datai = Datas[i]

        "Determine if the current sequence i is the sequence to be fused, if yes, missing is performed, otherwise it is not processed"
        if i in fusion_idx:

            "The index of missing sequence"
            idx = fusion_idx.index(i)

            idxi = np.expand_dims(missing_idxs[:, idx], axis=1)
            expanded_idxi = np.tile(idxi, (1, Datai.shape[1]))

            masked_data = Datai * expanded_idxi

            new_Data.append(masked_data)
        else:
            new_Data.append(Datai)

    return new_Data

def apply_smote(X_train, y_train, X_test, y_test):
    sm = SMOTE()
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    y_resampled = np.reshape(y_resampled, (len(y_resampled), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    return X_resampled, y_resampled, X_test, y_test

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

    npv = confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])

    return auc_value, accuracy, sensitivity, specificity, precision, recall, npv, f1, confusion, jaccard_index, kappa


if __name__ == '__main__':
    "Step 1: Load data of the training and test set"
    data_Flair = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Train/T2-FLAIR.csv')).values
    data_T1ce = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Train/CE-T1WI.csv')).values
    data_T1 = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Train/T1WI.csv')).values
    data_T2 = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Train/T2WI.csv')).values
    X_Flair, y_Flair = data_Flair[:, 2:111], (data_Flair[:, 1]).astype(int)
    X_T1ce, y_T1ce = data_T1ce[:, 2:111], (data_T1ce[:, 1]).astype(int)
    X_T1, y_T1 = data_T1[:, 2:111], (data_T1[:, 1]).astype(int)
    X_T2, y_T2 = data_T2[:, 2:111], (data_T2[:, 1]).astype(int)
    origin_label = [y_T1, y_T1ce, y_T2, y_Flair]

    data_Flair_test = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Test/T2-FLAIR.csv')).values
    data_T1ce_test = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Test/CE-T1WI.csv')).values
    data_T1_test = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Test/T1WI.csv')).values
    data_T2_test = (pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Test/T2WI.csv')).values
    X_Flair_test, y_Flair_test = data_Flair_test[:, 2:111], (data_Flair_test[:, 1]).astype(int)
    X_T1ce_test, y_T1ce_test = data_T1ce_test[:, 2:111], (data_T1ce_test[:, 1]).astype(int)
    X_T1_test, y_T1_test = data_T1_test[:, 2:111], (data_T1_test[:, 1]).astype(int)
    X_T2_test, y_T2_test = data_T2_test[:, 2:111], (data_T2_test[:, 1]).astype(int)
    origin_label_te = [y_T1_test, y_T1ce_test, y_T2_test, y_Flair_test]

    "The feature name for plotting SHAP values"
    feature_name = pd.read_csv('H:/Glioma/github-Code/test_files/Grading/Test/T2-FLAIR.csv').columns[2: 111]
    t1_feature_name = [i + '_T1WI' for i in feature_name]
    t1ce_feature_name = [i + '_CE-T1WI' for i in feature_name]
    t2_feature_name = [i + '_T2WI' for i in feature_name]
    flair_feature_name = [i + '_T2-FLAIR' for i in feature_name]
    feature_names = [t1_feature_name, t1ce_feature_name, t2_feature_name, flair_feature_name]
    feature_names_1 = np.concatenate(feature_names, axis=-1)

    "Step 2: Standardized pre-processing"
    X_T1, X_T1_test = Standardization(X_T1, X_T1_test)
    X_T1ce, X_T1ce_test = Standardization(X_T1ce, X_T1ce_test)
    X_T2, X_T2_test = Standardization(X_T2, X_T2_test)
    X_Flair, X_Flair_test = Standardization(X_Flair, X_Flair_test)
    total_samples = [X_T1, X_T1ce, X_T2, X_Flair]

    "Step 3: Define model parameters"
    num_epochs = 100
    nfolds = 5
    fusion_idx = [0, 1, 2, 3]  # The index of each sequence, 0: T1WI, 1: T1ce, 2: T2WI, 3: T2-FLAIR
    folder_name = "+".join(str(idx) for idx in fusion_idx)
    params = [0.01, 0.00001, 0.001, 0.001]  # The weight coefficient of each loss
    feature_dim = 10  # The dimension of latent feature space
    private_dim = 4  # The dimension of sequence-specific component
    shared_dim = 6  # The dimension of sequence-neutral component
    N, d = X_T1ce.shape[0], X_T1ce.shape[1]
    N_test, d_test = X_T1ce_test.shape[0], X_T1ce_test.shape[1]
    M = len(fusion_idx)  # The number of MRI sequences
    METRICS = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.BinaryAccuracy(name='acc')]
    missing_ratio = '10%'
    missing_percentages = [0.1,0]  # The first number represents the proportion of sequences with only one missing, and the second is the proportion of sequences with 2 missing at the same time
    total_best_result = []

    "Step 4: Start iterative training"
    "Split training data into train and validation sets (70% train, 30% validation)"
    val_ratio = 0.3
    X_T1_train, X_T1_val, y_T1_train, y_T1_val = train_test_split(X_T1, y_T1, test_size=val_ratio, random_state=42,stratify=y_T1)
    X_T1ce_train, X_T1ce_val, y_T1ce_train, y_T1ce_val = train_test_split(X_T1ce, y_T1ce, test_size=val_ratio,random_state=42, stratify=y_T1ce)
    X_T2_train, X_T2_val, y_T2_train, y_T2_val = train_test_split(X_T2, y_T2, test_size=val_ratio, random_state=42,stratify=y_T2)
    X_Flair_train, X_Flair_val, y_Flair_train, y_Flair_val = train_test_split(X_Flair, y_Flair, test_size=val_ratio,random_state=42, stratify=y_Flair)

    N_train = X_T1_train.shape[0]
    N_val = X_T1_val.shape[0]

    "The original features and generated features"
    origin_features_total_tr = np.zeros((4, N_train, d))
    generate_features_total_tr = np.zeros((4, N_train, d))
    origin_features_total_val = np.zeros((4, N_val, d))
    generate_features_total_val = np.zeros((4, N_val, d))
    origin_features_total_te = np.zeros((4, N_test, d))
    generate_features_total_te = np.zeros((4, N_test, d))
    for l in range(4):
        origin_features_total_tr[l] = [X_T1_train, X_T1ce_train, X_T2_train, X_Flair_train][l]
        origin_features_total_val[l] = [X_T1_val, X_T1ce_val, X_T2_val, X_Flair_val][l]
        origin_features_total_te[l] = [X_T1_test, X_T1ce_test, X_T2_test, X_Flair_test][l]

    "The missing index of train"
    missing_idxs_tr = Generate_missing_data(N_train, M, missing_percentages)
    missing_datas_tr = preprocess_data([X_T1_train, X_T1ce_train, X_T2_train, X_Flair_train], fusion_idx,missing_idxs_tr)
    new_X_T1, new_X_T1ce, new_X_T2, new_X_Flair = missing_datas_tr[0], missing_datas_tr[1], missing_datas_tr[2], \
    missing_datas_tr[3]

    "The missing index of validation"
    missing_idxs_val = Generate_missing_data(N_val, M, missing_percentages)
    missing_datas_val = preprocess_data([X_T1_val, X_T1ce_val, X_T2_val, X_Flair_val], fusion_idx, missing_idxs_val)
    new_X_T1_val, new_X_T1ce_val, new_X_T2_val, new_X_Flair_val = missing_datas_val[0], missing_datas_val[1], \
    missing_datas_val[2], missing_datas_val[3]

    "The missing index of test"
    missing_idxs_te = Generate_missing_data(N_test, M, missing_percentages)
    missing_datas_te = preprocess_data([X_T1_test, X_T1ce_test, X_T2_test, X_Flair_test], fusion_idx, missing_idxs_te)
    new_X_T1_test, new_X_T1ce_test, new_X_T2_test, new_X_Flair_test = missing_datas_te[0], missing_datas_te[1], \
    missing_datas_te[2], missing_datas_te[3]

    "SMOTE data enhancement for training set only"
    sm = SMOTE()
    new_X_T1_sm, y_T1_sm = sm.fit_resample(new_X_T1, y_T1_train)
    y_T1_sm = np.reshape(y_T1_sm, (len(y_T1_sm), 1))
    new_X_T1ce_sm, y_T1ce_sm = sm.fit_resample(new_X_T1ce, y_T1ce_train)
    y_T1ce_sm = np.reshape(y_T1ce_sm, (len(y_T1ce_sm), 1))
    new_X_T2_sm, y_T2_sm = sm.fit_resample(new_X_T2, y_T2_train)
    y_T2_sm = np.reshape(y_T2_sm, (len(y_T2_sm), 1))
    new_X_Flair_sm, y_Flair_sm = sm.fit_resample(new_X_Flair, y_Flair_train)
    y_Flair_sm = np.reshape(y_Flair_sm, (len(y_Flair_sm), 1))

    "Reshape validation and test labels"
    y_T1_val = np.reshape(y_T1_val, (len(y_T1_val), 1))
    y_T1_test_reshaped = np.reshape(y_T1_test, (len(y_T1_test), 1))

    inputs = [new_X_T1_sm, new_X_T1ce_sm, new_X_T2_sm, new_X_Flair_sm]
    inputs_val = [new_X_T1_val, new_X_T1ce_val, new_X_T2_val, new_X_Flair_val]
    inputs_test = [new_X_T1_test, new_X_T1ce_test, new_X_T2_test, new_X_Flair_test]

    "Define model"
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=40, decay_rate=0.5,staircase=True)
    model = Completion_model(inp_shape=inputs[0].shape[1:], latent_dim=feature_dim, private_dim=private_dim, shared_dim=shared_dim)
    model.compile(optimizer=Adam(lr=initial_learning_rate), metrics=METRICS)

    "The matrix for saving training and validation results"
    fusion_features_tr = np.zeros((num_epochs, len(y_T1_sm), M * private_dim + shared_dim))
    confusion_matrix_tr = np.zeros((num_epochs, 2, 2))
    y_probas_tr = np.zeros((num_epochs, len(y_T1_sm)))
    y_preds_tr = np.zeros((num_epochs, len(y_T1_sm)))
    y_trues_tr = np.zeros((num_epochs, len(y_T1_sm)))
    y_true_tr = y_T1_sm

    fusion_features_val = np.zeros((num_epochs, len(y_T1_val), M * private_dim + shared_dim))
    confusion_matrix_val = np.zeros((num_epochs, 2, 2))
    y_probas_val = np.zeros((num_epochs, len(y_T1_val)))
    y_preds_val = np.zeros((num_epochs, len(y_T1_val)))
    y_trues_val = np.zeros((num_epochs, len(y_T1_val)))
    y_true_val = y_T1_val

    "The lists for saving the results of all iterations of training and validation"
    auc_list = []
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
    classification_list = []

    "Validation metrics lists"
    auc_list_val = []
    acc_list_val = []
    sen_list_val = []
    spe_list_val = []
    precision_list_val = []
    f1score_list_val = []
    npv_list_val = []
    jaccard_list_val = []
    cohen_list_val = []
    total_list_val = []
    rec_list_val = []
    kl_list_val = []
    classification_list_val = []

    loss_not_improved_count = 0
    best_loss = float('inf')
    best_epoch = 0
    best_model_weights = None

    "Begin training and validation"
    global epoch
    for epoch in range(num_epochs):

        "Training"
        result_train = model.train_step(inputs, y_T1_sm, missing_idxs_tr)
        total_loss = result_train['loss']
        rec_loss = result_train['rec_loss']
        kl_loss = result_train['kl_loss']
        classification_loss = result_train['classification_loss']
        decoder_out = result_train['Decoders_out']
        decoder_out_fusion = result_train["Decoders_fusion_out"]
        fusion_code = result_train['fusion_code']
        predict_out = result_train['predict_out']
        auc_tr, acc_tr, sen_tr, spe_tr, precision_tr, recall_tr, npv_tr, f1score_tr, confusion_tr, jaccard_tr, cohen_tr = evalution_metric(
            y_T1_sm, predict_out)

        print(f'Train Result - AUC: {auc_tr}, ACC: {acc_tr}, SEN:{sen_tr}, SPE:{spe_tr}, NPV:{npv_tr}, PPV:{precision_tr}, F1:{f1score_tr}, total_Loss: {total_loss}')

        "Validation (instead of testing every epoch)"
        results_val = model.test_step(inputs_val, y_T1_val, missing_idxs_val)

        total_loss_val = results_val['loss']
        rec_loss_val = results_val['rec_loss']
        kl_loss_val = results_val['kl_loss']
        classification_loss_val = results_val['classification_loss']
        decoder_out_val = results_val['Decoders_out']
        decoder_out_fusion_val = results_val["Decoders_fusion_out"]
        fusion_code_val = results_val['fusion_code']
        predict_out_val = results_val['predict_out']
        auc_val, acc_val, sen_val, spe_val, precision_val, recall_val, npv_val, f1score_val, confusion_val, jaccard_val, cohen_val = evalution_metric(
            y_T1_val, predict_out_val)

        print(f'Validation Result - AUC: {auc_val}, ACC: {acc_val}, SEN:{sen_val}, SPE:{spe_val}, NPV:{npv_val}, PPV:{precision_val}, F1:{f1score_val}, total_Loss: {total_loss_val}')

        "Save values"
        auc_list.append(auc_tr)
        auc_list_val.append(auc_val)
        acc_list.append(acc_tr)
        acc_list_val.append(acc_val)
        sen_list.append(sen_tr)
        sen_list_val.append(sen_val)
        spe_list.append(spe_tr)
        spe_list_val.append(spe_val)
        precision_list.append(precision_tr)
        precision_list_val.append(precision_val)
        f1score_list.append(f1score_tr)
        f1score_list_val.append(f1score_val)
        npv_list.append(npv_tr)
        npv_list_val.append(npv_val)
        jaccard_list.append(jaccard_tr)
        jaccard_list_val.append(jaccard_val)
        cohen_list.append(cohen_tr)
        cohen_list_val.append(cohen_val)
        total_list.append(total_loss)
        total_list_val.append(total_loss_val)
        rec_list.append(rec_loss)
        rec_list_val.append(rec_loss_val)
        kl_list.append(kl_loss)
        kl_list_val.append(kl_loss_val)
        classification_list.append(classification_loss)
        classification_list_val.append(classification_loss_val)

        "Calculating predictive probabilities, predictive labels, and confusion matrices for training"
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_true_tr, predict_out)
        youden_index = tpr_tr - fpr_tr
        best_threshold_tr = thresholds_tr[np.argmax(youden_index)]
        y_pred_tr = np.where(predict_out > best_threshold_tr, 1, 0)
        y_preds_tr[epoch] = np.reshape(y_pred_tr, y_pred_tr.shape[0])
        y_trues_tr[epoch] = np.reshape(y_true_tr, y_true_tr.shape[0])
        y_probas_tr[epoch] = np.reshape(predict_out, predict_out.shape[0])
        confusion_matrix_tr[epoch] = confusion_matrix(y_true_tr, y_pred_tr)
        fusion_features_tr[epoch] = fusion_code

        "Calculating predictive probabilities, predictive labels, and confusion matrices for validation"
        y_pred_val = np.where(predict_out_val > best_threshold_tr, 1, 0)
        y_preds_val[epoch] = np.reshape(y_pred_val, y_pred_val.shape[0])
        y_trues_val[epoch] = np.reshape(y_true_val, y_pred_val.shape[0])
        y_probas_val[epoch] = np.reshape(predict_out_val, predict_out_val.shape[0])
        confusion_matrix_val[epoch] = confusion_matrix(y_true_val, y_pred_val)
        fusion_features_val[epoch] = fusion_code_val

        "Before the start of the next iteration, the original missing part is replaced by the mean of the missing " \
        "sequence's own reconstruction + joint reconstruction"
        for j in range(len(fusion_idx)):
            "Training"
            missing_idxs_train = missing_idxs_tr[:, j]
            smote_idxs_train = np.ones((inputs[fusion_idx[j]].shape[0] - len(missing_idxs_train)))
            new_idxs_train = np.hstack((missing_idxs_train, smote_idxs_train))
            inputs[fusion_idx[j]][new_idxs_train == 0] = (decoder_out[fusion_idx[j]][new_idxs_train == 0] +
                                                          decoder_out_fusion[fusion_idx[j]][new_idxs_train == 0]) / 2

            "Validation"
            missing_idxs_validation = missing_idxs_val[:, j]
            inputs_val[fusion_idx[j]][missing_idxs_validation == 0] = (decoder_out_val[fusion_idx[j]][
                                                                           missing_idxs_validation == 0] +
                                                                       decoder_out_fusion_val[fusion_idx[j]][
                                                                           missing_idxs_validation == 0]) / 2

            "Replacing original data with reconstructed data"
            generate_features_total_tr[fusion_idx[j]] = inputs[fusion_idx[j]][0:len(missing_idxs_train)]
            generate_features_total_val[fusion_idx[j]] = inputs_val[fusion_idx[j]]

        "Check if validation loss has improved and save best model"
        if total_loss_val < best_loss:
            best_loss = total_loss_val
            best_epoch = epoch
            loss_not_improved_count = 0
            "Save best model weights"
            best_model_weights = model.get_weights()
        else:
            loss_not_improved_count += 1

        if loss_not_improved_count >= 10:
            print('Validation loss has not improved for 10 epochs. Stopping training.')
            break

    "Use the best model based on validation loss for final test set evaluation"
    print(f'\n=== Best epoch based on validation loss: {best_epoch + 1} ===')
    print(
        f'Best Train Result - AUC: {auc_list[best_epoch]}, ACC: {acc_list[best_epoch]}, SEN/Recall:{sen_list[best_epoch]}, SPE:{spe_list[best_epoch]},'
        f'NPV:{npv_list[best_epoch]}, Precision/PPV:{precision_list[best_epoch]}, F1-score:{f1score_list[best_epoch]}, total_Loss: {total_list[best_epoch]}')
    print(
        f'Best Validation Result - AUC: {auc_list_val[best_epoch]}, ACC: {acc_list_val[best_epoch]}, SEN/Recall:{sen_list_val[best_epoch]}, SPE:{spe_list_val[best_epoch]},'
        f'NPV:{npv_list_val[best_epoch]}, Precision/PPV:{precision_list_val[best_epoch]}, F1-score:{f1score_list_val[best_epoch]}, total_Loss: {total_list_val[best_epoch]}')

    "Restore best model weights for test set evaluation"
    if best_model_weights is not None:
        model.set_weights(best_model_weights)

    "Perform ONE test set evaluation with the best model"
    print(f'\n=== Final Test Set Evaluation (using best model from epoch {best_epoch + 1}) ===')
    results_test = model.test_step(inputs_test, y_T1_test_reshaped, missing_idxs_te)

    total_loss_te = results_test['loss']
    rec_loss_te = results_test['rec_loss']
    kl_loss_te = results_test['kl_loss']
    classification_loss_te = results_test['classification_loss']
    decoder_out_te = results_test['Decoders_out']
    decoder_out_fusion_te = results_test["Decoders_fusion_out"]
    fusion_code_te = results_test['fusion_code']
    predict_out_te = results_test['predict_out']
    auc_te, acc_te, sen_te, spe_te, precision_te, recall_te, npv_te, f1score_te, confusion_te, jaccard_te, cohen_te = evalution_metric(
        y_T1_test_reshaped, predict_out_te)

    print(f'Test Result - AUC: {auc_te}, ACC: {acc_te}, SEN:{sen_te}, SPE:{spe_te}, NPV:{npv_te}, PPV:{precision_te}, F1:{f1score_te}, total_Loss: {total_loss_te}')

    "Calculate SHAP values on the test set"
    print(f'\n=== Computing SHAP values on validation set with the best model ===')
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough

    shap.initjs()
    explainer = shap.DeepExplainer((model.MMDR_model.inputs, model.MMDR_model.outputs[-1]), inputs)
    shap_values = explainer.shap_values(inputs_test, check_additivity=False)
    print(f'SHAP expected value: {explainer.expected_value}')

    plt.figure(dpi=1200)
    fig, ax = plt.gcf(), plt.gca()
    shap.summary_plot(shap_values=np.concatenate(shap_values[0], axis=-1),
                      features=np.concatenate(inputs_test, axis=-1),
                      feature_names=feature_names_1,
                      plot_type='dot',
                      max_display=10, show=False, color_bar=False)
    best_shap_values = np.concatenate(shap_values[0], axis=-1)

    'Save figure'
    fig.set_facecolor('white')
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',  # normal
             'size': 15,
             }
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_xlabel(ax.get_xlabel(), fontdict=font1)


    ax.yaxis.set_label_coords(3, 2)

    # rotate y-axis labels
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_fontname('Times New Roman')
        label.set_weight('bold')
        label.set_color('gray')

    "colorbar setting"
    rect_10 = [1, 0.12, 0.013, 0.85]
    cbar_ax = fig.add_axes(rect_10)
    colorbar = plt.colorbar(shrink=0.2, pad=0.05, fraction=0.05, orientation='vertical', spacing='uniform', cax=cbar_ax)
    # Customize colorbar font size and style
    colorbar.set_label('Feature values', fontdict=font1)  # Change label properties
    colorbar.ax.tick_params(labelsize=12, labelcolor='black', pad=8)  # Change label properties

    # Set font family for tick labels
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    for patch in ax.patches:
        patch.set_edgecolor('none')

    fig.savefig(r'H:/Glioma/github-Code/Feature_importance.png', bbox_inches='tight', dpi=600)  # , pil_kwargs={'compression': 'tiff_lzw'}

