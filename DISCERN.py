from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
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
        return z_mean + K.exp(0.5 * z_log_var) * epsilon  # Noise with mean + standard deviation obeying a normal distribution N(0, 1)
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
        z_private_out = [T1_encoder_out[2], T1ce_encoder_out[2], T2_encoder_out[2], Flair_encoder_out[2]]  # sequence-specific representation
        z_mean_shared_out = [T1_encoder_out[3], T1ce_encoder_out[3], T2_encoder_out[3], Flair_encoder_out[3]]
        z_log_var_shared_out = [T1_encoder_out[4], T1ce_encoder_out[4], T2_encoder_out[4], Flair_encoder_out[4]]
        z_shared_out = [T1_encoder_out[5], T1ce_encoder_out[5], T2_encoder_out[5], Flair_encoder_out[5]]  # sequence-neutral representation
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

        z_common_fusion = Lambda(lambda x: tf.reduce_mean(tf.stack([x[i] for i in list(range(len(fusion_idx)))], axis=0), axis=0))(z_shared_add)
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
                               T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out, Flair_fusion_decoder_out,
                               fusion_code, predict_out],
                      name="MMDR")
        return model

    def train_step(self, x, y, missing_idxs):
        with tf.GradientTape() as tape:
            T1_encoder_out, T1ce_encoder_out, T2_encoder_out, Flair_encoder_out, \
            T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out, \
            T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out, Flair_fusion_decoder_out, fusion_code,predict_out = self.MMDR_model(x)

            "Get the mean, variance, and latant representation of sequence-specific and sequence-neutral representation, respectively"
            z_mean_private_Flair, z_log_var_private_Flair, z_private_Flair, z_mean_shared_Flair, z_log_var_shared_Flair, z_shared_Flair = Flair_encoder_out
            z_mean_private_T1, z_log_var_private_T1, z_private_T1, z_mean_shared_T1, z_log_var_shared_T1, z_shared_T1  = T1_encoder_out
            z_mean_private_T1ce, z_log_var_private_T1ce, z_private_T1ce, z_mean_shared_T1ce, z_log_var_shared_T1ce, z_shared_T1ce  = T1ce_encoder_out
            z_mean_private_T2, z_log_var_private_T2, z_private_T2, z_mean_shared_T2, z_log_var_shared_T2, z_shared_T2 = T2_encoder_out

            z_log_vars_private = [z_log_var_private_T1, z_log_var_private_T1ce, z_log_var_private_T2, z_log_var_private_Flair]
            z_means_private = [z_mean_private_T1, z_mean_private_T1ce, z_mean_private_T2, z_mean_private_Flair]
            z_log_vars_shared = [z_log_var_shared_T1, z_log_var_shared_T1ce, z_log_var_shared_T2, z_log_var_shared_Flair]
            z_means_shared = [z_mean_shared_T1, z_mean_shared_T1ce, z_mean_shared_T2, z_mean_shared_Flair]
            Decoders_out = [T1_decoder_out, T1ce_decoder_out, T2_decoder_out, Flair_decoder_out]
            Decoders_fusion_out = [T1_fusion_decoder_out, T1ce_fusion_decoder_out, T2_fusion_decoder_out, Flair_fusion_decoder_out]
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
                    kl_loss = kl_loss + (-0.5 * tf.reduce_mean(1 + z_log_vars_private[fusion_idx[i]] - tf.square(z_means_private[fusion_idx[i]]) - tf.exp(z_log_vars_private[fusion_idx[i]]))) \
                              + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared[fusion_idx[i]] - tf.square(z_means_shared[fusion_idx[i]]) - tf.exp(z_log_vars_shared[fusion_idx[i]])))

                    "The reconstruction loss for complete samples"
                    missing_idxs_tr = missing_idxs[:, i]  # Unenhanced patient index
                    smote_idxs_tr = np.ones((x[fusion_idx[i]].shape[0] - len(missing_idxs_tr)))
                    new_idxs_tr = np.hstack((missing_idxs_tr, smote_idxs_tr))
                    reconstruction_loss = reconstruction_loss + \
                    tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[i]][new_idxs_tr != 0], Decoders_out[fusion_idx[i]][new_idxs_tr != 0]))) \
                    + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[i]][new_idxs_tr != 0], Decoders_fusion_out[fusion_idx[i]][new_idxs_tr != 0])))

                    "The disentangled loss"
                    if i < len(fusion_idx) - 1:
                        for j in range(i + 1, len(fusion_idx)):
                            com_loss += tf.reduce_mean(tf.sqrt(mean_squared_error(z_shared_out[fusion_idx[i]], z_shared_out[fusion_idx[j]])))
                            spe_loss += tf.reduce_mean(tf.sqrt(mean_squared_error(z_private_out[fusion_idx[i]], z_private_out[fusion_idx[j]])))
                com_spec_loss = com_loss / spe_loss
            else:
                "For single sequence"
                kl_loss = (-0.5 * tf.reduce_mean(1 + z_log_vars_private[fusion_idx[0]] - tf.square(z_means_private[fusion_idx[0]]) - tf.exp(z_log_vars_private[fusion_idx[0]]))) \
                          + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared[fusion_idx[0]] - tf.square(z_means_shared[fusion_idx[0]]) - tf.exp(z_log_vars_shared[fusion_idx[0]])))
                reconstruction_loss = tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x[fusion_idx[0]], Decoders_out[fusion_idx[0]])))
                com_spec_loss = com_loss / (spe_loss + epsilon)  # Preventing division by 0

            "The classification loss"
            classification_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, predict_out))

            "The total loss"
            total_loss = params[0] * reconstruction_loss + params[1] * kl_loss + params[2] * com_spec_loss + params[3] * classification_loss

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
        T1_fusion_decoder_out_, T1ce_fusion_decoder_out_, T2_fusion_decoder_out_, Flair_fusion_decoder_out_, fusion_code_, predict_out_ = self.MMDR_model(x_)

        "Get the mean, variance, and latant representation of sequence-specific and sequence-neutral representation, respectively"
        z_mean_private_Flair_, z_log_var_private_Flair_, z_private_Flair_, z_mean_shared_Flair_, z_log_var_shared_Flair_, z_shared_Flair_= Flair_encoder_out_
        z_mean_private_T1_, z_log_var_private_T1_, z_private_T1_, z_mean_shared_T1_, z_log_var_shared_T1_, z_shared_T1_ = T1_encoder_out_
        z_mean_private_T1ce_, z_log_var_private_T1ce_, z_private_T1ce_, z_mean_shared_T1ce_, z_log_var_shared_T1ce_, z_shared_T1ce_ = T1ce_encoder_out_
        z_mean_private_T2_, z_log_var_private_T2_, z_private_T2_, z_mean_shared_T2_, z_log_var_shared_T2_, z_shared_T2_ = T2_encoder_out_

        z_log_vars_private_ = [z_log_var_private_T1_, z_log_var_private_T1ce_, z_log_var_private_T2_,  z_log_var_private_Flair_]
        z_means_private_ = [z_mean_private_T1_, z_mean_private_T1ce_, z_mean_private_T2_, z_mean_private_Flair_]
        z_log_vars_shared_ = [z_log_var_shared_T1_, z_log_var_shared_T1ce_, z_log_var_shared_T2_, z_log_var_shared_Flair_]
        z_means_shared_ = [z_mean_shared_T1_, z_mean_shared_T1ce_, z_mean_shared_T2_, z_mean_shared_Flair_]
        Decoders_out_ = [T1_decoder_out_, T1ce_decoder_out_, T2_decoder_out_, Flair_decoder_out_]
        Decoders_fusion_out_ = [T1_fusion_decoder_out_, T1ce_fusion_decoder_out_, T2_fusion_decoder_out_, Flair_fusion_decoder_out_]
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
                kl_loss_ = kl_loss_ + (-0.5 * tf.reduce_mean(1 + z_log_vars_private_[fusion_idx[i]] - tf.square(z_means_private_[fusion_idx[i]]) - tf.exp(z_log_vars_private_[fusion_idx[i]]))) \
                          + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared_[fusion_idx[i]] - tf.square(z_means_shared_[fusion_idx[i]]) - tf.exp(z_log_vars_shared_[fusion_idx[i]])))

                "The reconstruction loss for complete samples"
                missing_idxs_te = missing_idxs[:, i]
                reconstruction_loss_ = reconstruction_loss_ + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]][missing_idxs_te != 0],
                Decoders_out_[fusion_idx[i]][missing_idxs_te != 0]))) + tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[i]][missing_idxs_te != 0],
                Decoders_fusion_out_[fusion_idx[i]][missing_idxs_te != 0])))

                "The disentangled loss"
                if i < len(fusion_idx) - 1:
                    for j in range(i + 1, len(fusion_idx)):
                        com_loss_ += tf.reduce_mean(tf.sqrt(mean_squared_error(z_shared_out_[fusion_idx[i]], z_shared_out_[fusion_idx[j]])))
                        spe_loss_ += tf.reduce_mean(tf.sqrt(mean_squared_error(z_private_out_[fusion_idx[i]], z_private_out_[fusion_idx[j]])))
            com_spec_loss_ = com_loss_ / spe_loss_
        else:
            "For single sequence"
            kl_loss_ = (-0.5 * tf.reduce_mean(1 + z_log_vars_private_[fusion_idx[0]] - tf.square(z_means_private_[fusion_idx[0]]) - tf.exp(z_log_vars_private_[fusion_idx[0]]))) \
                      + (-0.5 * tf.reduce_mean(1 + z_log_vars_shared_[fusion_idx[0]] - tf.square(z_means_shared_[fusion_idx[0]]) - tf.exp(z_log_vars_shared_[fusion_idx[0]])))
            reconstruction_loss_ = tf.reduce_mean(tf.sqrt(tf.keras.losses.mean_squared_error(x_[fusion_idx[0]], Decoders_out_[fusion_idx[0]])))
            com_spec_loss_ = com_loss_ / (spe_loss_ + epsilon)  

        "The classification loss"
        classification_loss_ = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_, predict_out_))

        "The total loss"
        total_loss_ = params[0] * reconstruction_loss_ + params[1] * kl_loss_ + params[2] * com_spec_loss_ + params[3] * classification_loss_

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
            return Model(inputs=input_layer, outputs=[mean_private, log_var_private, z_private, mean_shared, log_var_shared, z_shared], name='encoder_{}'.format(name))
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
        fusion_feature = Input((len(fusion_idx)*self.private_dim+self.shared_dim,), name='fused_input')
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
        indices = np.random.choice(remaining_indices, num_missing, replace=False)

        if len(remaining_indices) < num_missing:
            raise ValueError(f"Missing too many to meet conditions.")

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

    npv = confusion[0, 0]/(confusion[0, 0]+confusion[1, 0])

    return auc_value, accuracy, sensitivity, specificity, precision, recall, npv, f1, confusion, jaccard_index, kappa


if __name__ == '__main__':
    "Step 1: Load data of the training and test set"
    data_Flair = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Train/T2-FLAIR.csv')).values
    data_T1ce = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Train/CE-T1WI.csv')).values
    data_T1 = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Train/T1WI.csv')).values
    data_T2 = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Train/T2WI.csv')).values

    X_Flair, y_Flair = data_Flair[:, 2:111], (data_Flair[:, 1]).astype(int)
    X_T1ce, y_T1ce = data_T1ce[:, 2:111], (data_T1ce[:, 1]).astype(int)
    X_T1, y_T1 = data_T1[:, 2:111], (data_T1[:, 1]).astype(int)
    X_T2, y_T2 = data_T2[:, 2:111], (data_T2[:, 1]).astype(int)
    origin_label = [y_T1, y_T1ce, y_T2, y_Flair]

    data_Flair_test = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Test/T2-FLAIR.csv')).values  
    data_T1ce_test = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Test/CE-T1WI.csv')).values
    data_T1_test = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Test/T1WI.csv')).values
    data_T2_test = (pd.read_csv('D:/Glioma/2-Features/test_files/Grading/Test/T2WI.csv')).values
    X_Flair_test, y_Flair_test = data_Flair_test[:, 2:111], (data_Flair_test[:, 1]).astype(int)
    X_T1ce_test, y_T1ce_test = data_T1ce_test[:, 2:111], (data_T1ce_test[:, 1]).astype(int)
    X_T1_test, y_T1_test = data_T1_test[:, 2:111], (data_T1_test[:, 1]).astype(int)
    X_T2_test, y_T2_test = data_T2_test[:, 2:111], (data_T2_test[:, 1]).astype(int)
    origin_label_te = [y_T1_test, y_T1ce_test, y_T2_test, y_Flair_test]

    "Step 2: Standardized pre-processing"
    X_T1, X_T1_test = Standardization(X_T1, X_T1_test)
    X_T1ce, X_T1ce_test = Standardization(X_T1ce, X_T1ce_test)
    X_T2, X_T2_test = Standardization(X_T2, X_T2_test)
    X_Flair, X_Flair_test = Standardization(X_Flair, X_Flair_test)

    "Step 3: Define model parameters"
    num_epochs = 100
    nfolds = 5
    fusion_idx = [0, 1, 3]  # The index of each sequence, 0: T1WI, 1: T1ce, 2: T2WI, 3: T2-FLAIR
    folder_name = "+".join(str(idx) for idx in fusion_idx)
    params = [0.01, 0.00001, 0.001, 0.001]  # The weight coefficient of each loss
    feature_dim = 10  # The dimension of latent feature space
    private_dim = 4   # The dimension of sequence-specific component
    shared_dim = 6    # The dimension of sequence-neutral component
    N, d = X_T1ce.shape[0], X_T1ce.shape[1]
    N_test, d_test = X_T1ce_test.shape[0], X_T1ce_test.shape[1]
    M = len(fusion_idx)  # The number of MRI sequences
    METRICS = [tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.BinaryAccuracy(name='acc')]
    missing_ratio = '10%'
    missing_percentages = [0.1, 0]  # The first number represents the proportion of sequences with only one missing, and the second is the proportion of sequences with 2 missing at the same time
    total_best_result = []

    "Step 4: Start iterative training"
    for ite in [1, 2, 3, 4, 5]:

        "The original features and generated features"
        origin_features_total_tr = np.zeros((4, N, d))
        generate_features_total_tr = np.zeros((4, N, d))
        origin_features_total_te = np.zeros((4, N_test, d))
        generate_features_total_te = np.zeros((4, N_test, d))
        for l in range(4):
            origin_features_total_tr[l] = [X_T1, X_T1ce, X_T2, X_Flair][l]
            origin_features_total_te[l] = [X_T1_test, X_T1ce_test, X_T2_test, X_Flair_test][l]

        "The missing index of train"
        missing_idxs_tr = Generate_missing_data(N, M, missing_percentages)
        missing_datas_tr = preprocess_data([X_T1, X_T1ce, X_T2, X_Flair], fusion_idx, missing_idxs_tr)
        new_X_T1, new_X_T1ce, new_X_T2, new_X_Flair = missing_datas_tr[0], missing_datas_tr[1], missing_datas_tr[2], missing_datas_tr[3]  

        "The missing index of test"
        missing_idxs_te = Generate_missing_data(N_test, M, missing_percentages)
        missing_datas_te = preprocess_data([X_T1_test, X_T1ce_test, X_T2_test, X_Flair_test], fusion_idx, missing_idxs_te)
        new_X_T1_test, new_X_T1ce_test, new_X_T2_test, new_X_Flair_test = missing_datas_te[0], missing_datas_te[1], missing_datas_te[2],  missing_datas_te[3]

        "SMOTE data enhancement for training set"
        sm = SMOTE()
        new_X_T1_sm, y_T1_sm, new_X_T1_test, y_T1_test = apply_smote(new_X_T1, y_T1, new_X_T1_test, y_T1_test)
        new_X_T1ce_sm, y_T1ce_sm, new_X_T1ce_test, y_T1ce_test = apply_smote(new_X_T1ce, y_T1ce, new_X_T1ce_test, y_T1ce_test)
        new_X_T2_sm, y_T2_sm, new_X_T2_test, y_T2_test = apply_smote(new_X_T2, y_T2, new_X_T2_test, y_T2_test)
        new_X_Flair_sm, y_Flair_sm, new_X_Flair_test, y_Flair_test = apply_smote(new_X_Flair, y_Flair, new_X_Flair_test, y_Flair_test)

        inputs = [new_X_T1_sm, new_X_T1ce_sm, new_X_T2_sm, new_X_Flair_sm]
        inputs_test = [new_X_T1_test, new_X_T1ce_test, new_X_T2_test, new_X_Flair_test]

        "Define model"
        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=40, decay_rate=0.5, staircase=True)
        model = Completion_model(inp_shape=inputs[0].shape[1:], latent_dim=feature_dim, private_dim=private_dim, shared_dim=shared_dim)
        model.compile(optimizer=Adam(lr=initial_learning_rate), metrics=METRICS)

        "The matrix for saving training and test results"
        fusion_features_tr = np.zeros((num_epochs, len(y_T1_sm), M * private_dim + shared_dim))
        confusion_matrix_tr = np.zeros((num_epochs, 2, 2))
        y_probas_tr = np.zeros((num_epochs, len(y_T1_sm)))
        y_preds_tr = np.zeros((num_epochs, len(y_T1_sm)))
        y_trues_tr = np.zeros((num_epochs, len(y_T1_sm)))
        y_true_tr = y_T1_sm

        fusion_features_te = np.zeros((num_epochs, len(y_T1_test), M * private_dim + shared_dim))
        confusion_matrix_te = np.zeros((num_epochs, 2, 2))
        y_probas_te = np.zeros((num_epochs, len(y_T1_test)))
        y_preds_te = np.zeros((num_epochs, len(y_T1_test)))
        y_trues_te = np.zeros((num_epochs, len(y_T1_test)))
        y_true_te = y_T1_test

        "The lists for saving the results of all iterations of training and test"
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
        classification_list_ = []

        loss_not_improved_count = 0
        best_loss = float('inf')

        "Begin training and test"
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
            auc_tr, acc_tr, sen_tr, spe_tr, precision_tr, recall_tr, npv_tr, f1score_tr, confusion_tr, jaccard_tr, cohen_tr = evalution_metric(y_T1_sm, predict_out)

            print( f'Train Result - AUC: {auc_tr}, ACC: {acc_tr}, SEN:{sen_tr}, SPE:{spe_tr}, NPV:{npv_tr}, PPV:{precision_tr}, F1:{f1score_tr}, total_Loss: {total_loss}')

            "Testing"
            results_test = model.test_step(inputs_test, y_T1_test, missing_idxs_te)

            total_loss_ = results_test['loss']
            rec_loss_ = results_test['rec_loss']
            kl_loss_ = results_test['kl_loss']
            classification_loss_ = results_test['classification_loss']
            decoder_out_ = results_test['Decoders_out']
            decoder_out_fusion_ = results_test["Decoders_fusion_out"]
            fusion_code_ = results_test['fusion_code']
            predict_out_ = results_test['predict_out']
            auc_, acc_, sen_, spe_, precision_, recall_, npv_, f1score_, confusion_, jaccard_, cohen_ = evalution_metric(y_T1_test, predict_out_)

            print(f'Test Result - AUC: {auc_}, ACC: {acc_}, SEN:{sen_}, SPE:{spe_}, NPV:{npv_}, PPV:{precision_}, F1:{f1score_}, total_Loss: {total_loss_}')

            "Save values"
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
            classification_list.append(classification_loss)
            classification_list_.append(classification_loss_)

            "Calculating predictive probabilities, predictive labels, and confusion matrices"
            fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_true_tr, predict_out)
            youden_index = tpr_tr - fpr_tr
            best_threshold_tr = thresholds_tr[np.argmax(youden_index)]
            y_pred_tr = np.where(predict_out > best_threshold_tr, 1, 0)
            y_preds_tr[epoch] = np.reshape(y_pred_tr, y_pred_tr.shape[0])
            y_trues_tr[epoch] = np.reshape(y_true_tr, y_true_tr.shape[0])
            y_probas_tr[epoch] = np.reshape(predict_out, predict_out.shape[0])
            confusion_matrix_tr[epoch] = confusion_matrix(y_true_tr, y_pred_tr)
            fusion_features_tr[epoch] = fusion_code

            y_pred_te = np.where(predict_out_ > best_threshold_tr, 1, 0)
            y_preds_te[epoch] = np.reshape(y_pred_te, y_pred_te.shape[0])
            y_trues_te[epoch] = np.reshape(y_true_te, y_pred_te.shape[0])
            y_probas_te[epoch] = np.reshape(predict_out_, predict_out_.shape[0])
            confusion_matrix_te[epoch] = confusion_matrix(y_true_te, y_pred_te)
            fusion_features_te[epoch] = fusion_code_

            "Before the start of the next iteration, the original missing part is replaced by the mean of the missing " \
            "sequence's own reconstruction + joint reconstruction"
            for j in range(len(fusion_idx)):
                "Training"
                missing_idxs_train = missing_idxs_tr[:, j]
                smote_idxs_train = np.ones((inputs[fusion_idx[j]].shape[0] - len(missing_idxs_train)))
                new_idxs_train = np.hstack((missing_idxs_train, smote_idxs_train))
                inputs[fusion_idx[j]][new_idxs_train == 0] = (decoder_out[fusion_idx[j]][new_idxs_train == 0] + decoder_out_fusion[fusion_idx[j]][new_idxs_train == 0])/2

                "Test"
                missing_idxs_test = missing_idxs_te[:, j]
                inputs_test[fusion_idx[j]][missing_idxs_test == 0] = (decoder_out_[fusion_idx[j]][missing_idxs_test == 0] + decoder_out_fusion_[fusion_idx[j]][missing_idxs_test == 0])/2

                "Replacing original data with reconstructed data"
                generate_features_total_tr[fusion_idx[j]] = inputs[fusion_idx[j]][0:len(missing_idxs_train)]
                generate_features_total_te[fusion_idx[j]] = inputs_test[fusion_idx[j]]

            "Check if validation loss has improved"
            if total_loss_ < best_loss:
                best_loss = total_loss_
                loss_not_improved_count = 0
            else:
                loss_not_improved_count += 1

            print(f'Valid Result - total_Loss: {total_loss_}')

            if loss_not_improved_count >= 10:
                print('Validation loss has not improved for 10 epochs. Stopping training.')
                break

        epoch = np.argmin(total_list_)
        print(f'Best Train Result - AUC: {auc_list[epoch]}, ACC: {acc_list[epoch]}, SEN/Recall:{sen_list[epoch]}, SPE:{spe_list[epoch]},'
              f'NPV:{npv_list[epoch]}, Precision/PPV:{precision_list[epoch]}, F1-score:{f1score_list[epoch]}, total_Loss: {total_list[epoch]}')
        print(f'Best Test Result - AUC: {auc_list_[epoch]}, ACC: {acc_list_[epoch]}, SEN/Recall:{sen_list_[epoch]}, SPE:{spe_list_[epoch]},'
              f'NPV:{npv_list_[epoch]}, Precision/PPV:{precision_list_[epoch]}, F1-score:{f1score_list_[epoch]}, total_Loss: {total_list_[epoch]}')

        "Save metrics（AUC, ACC, SEN, SPE, et.al) of the best iteration"
        best_result = []
        best_result.append(auc_list[epoch])
        best_result.append(acc_list[epoch])
        best_result.append(sen_list[epoch])
        best_result.append(spe_list[epoch])
        best_result.append(npv_list[epoch])
        best_result.append(precision_list[epoch])
        best_result.append(f1score_list[epoch])
        best_result.append(auc_list_[epoch])
        best_result.append(acc_list_[epoch])
        best_result.append(sen_list_[epoch])
        best_result.append(spe_list_[epoch])
        best_result.append(npv_list_[epoch])
        best_result.append(precision_list_[epoch])
        best_result.append(f1score_list_[epoch])

        # Path = 'D:/Glioma/3-Results/Train and Test (Grading)/Incomplete/' + folder_name +'/missing(0-T1WI)/'+ missing_ratio +'/TCIA'
        # sub_Path = Path + '/Metrics/' + str(ite)
        # if not os.path.exists(sub_Path):
        #     os.makedirs(sub_Path)
        # np.save(sub_Path + '/' + 'best_result.npy', best_result)

        # best_result = np.array(best_result)
        # best_result = np.reshape(best_result, (1, len(best_result)))
        # df = pd.DataFrame(best_result)
        # df.to_excel(Path + '/Metrics/best_result.xlsx', header=False, index=False)

        "Save predictive probability and labeling, etc. of the best iteration"
        Y_probas_list_train_best = y_probas_tr[epoch]
        Y_preds_list_train_best = np.where(y_probas_tr[epoch]> 0.5, 1, 0)
        Y_trues_list_train_best = y_trues_tr[epoch]
        Fusion_feature_list_train_best = fusion_features_tr[epoch]
        Confusion_matrix_list_train_best = confusion_matrix_tr[epoch]

        Y_probas_list_test_best = y_probas_te[epoch]
        Y_preds_list_test_best = np.where(y_probas_te[epoch] > 0.5, 1, 0)
        Y_trues_list_test_best = y_trues_te[epoch]
        Fusion_feature_test_valid_best = fusion_features_te[epoch]
        Confusion_matrix_test_valid_best = confusion_matrix_te[epoch]

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
        # np.save(sub_Path2 + '/Y_probas.npy', Y_probas_list_test_best)
        # np.save(sub_Path2 + '/Y_preds.npy', Y_preds_list_test_best)
        # np.save(sub_Path2 + '/Y_trues.npy', Y_trues_list_valid_best)
        # np.save(sub_Path2 + '/fusion_features.npy', Fusion_feature_list_test_best)
        # np.save(sub_Path2 + '/confusion_matrix.npy', Confusion_matrix_list_test_best)











