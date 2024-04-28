# Leveraging the Untapped Potentials of Incomplete MRI Sequences for Glioma Grading and IDH Mutation Status Prediction: A Large-Scale Multicenter Cohort Study
A disentangled-learning based incomplete sequence completion enhanced robust network (DISCERN) is developed to impute and integrate incomplete MRI sequences for glioma grading and IDH mutation status prediction.
![1](https://github.com/xinzhen-lab/GBM-Prediction-on-Missing-MRI-Sequences/assets/131331281/5c958256-b2f3-4866-bba3-80165840833d)


# DISCERN
The DISCERN is capable of handling different clinical scenarios involving the missing of any one or several MRI sequences. Radiomics features were respectively extracted from various MRI sequences and seamlessly fed to train the DISCERN model, with missing feature values in the absent sequences initialized with zeros. For each MRI sequence, an encoder network is built to disentangle and learn hidden representations capturing both the sequence-specific and sequence-neutral components. These representations were then passed through a decoder network to impute missing values in patient samples with the absent sequence. This mechanism for completing missing sequences facilitated the acquisition of more representative representations. Subsequently, the sequence-specific and sequence-neutral components from all MRI sequences were integrated to produce the fused representation, which was finally input into a prediction network dedicated to the downstream tasks of glioma grading or IDH status prediction.  In general, DISCERN exhibited tolerance to varying missing rates of sequences ranging from 10% to 30%, yielding satisfactory prediction performance in both GBM grading (AUC:0.8843 to 0.9246) and IDH prediction (AUC: 0.8181 to 0.8655).

<p align="center">
  ![2](https://github.com/xinzhen-lab/GBM-Prediction-on-Missing-MRI-Sequences/assets/131331281/8c7316c9-c539-47af-8d93-5d38a2666416)
</p>

<p align="center">
  <img width="460" height="300" src="https://picsum.photos/460/300">
</p>

# Model training & validation
The DISCERN is trained and validated on datasets collected from four institutions, including the First Affiliated Hospital of Zhengzhou University (FAHZZU), the Nanfang Hospital of Southern Medical University (NFHSMU), the Second Affiliated Hospital of South China University of Technology (SAHSCUT), and the Beijing Tiantan Hospital (BJTTH), as well as one public database: The Cancer Imaging Archive (TCIA). The training (n=1440, 80%) and internal validation (n=456, 20%) sets consisted of patients enrolled from FAHZZU, NFHSMU, and SAHSCUT in accordance with the WHO 2016 criteria. The two independent external validation sets were collected from TCIA and BJTTH based on the WHO 2021.


# Note
Other investigators are invited to share their data in order to further improve the generalization capability of the current model. The model should only be used to support clinical diagnosis by health care professionals as a complementary tool for predicting glioma grades and IDH mutation status with incomplete MRI sequences. Any responsibility for using this model and its results will rest solely by the health care professional using the model. Using it you should understand and agree that this tool is not responsible or liable for any claim, loss, or damage resulting from its use. While we try to keep the information on the tool as accurate as possible, we disclaim any warranty concerning its accuracy, timeliness, and completeness, and any other warranty, express or implied, including warranties of merchantability or fitness for a particular purpose.
