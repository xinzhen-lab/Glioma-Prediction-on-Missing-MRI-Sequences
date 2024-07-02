# Leveraging the Untapped Potentials of Incomplete MRI Sequences for Glioma Grading and IDH Mutation Status Prediction: A Large-Scale Multicenter Cohort Study
A disentangled-learning based incomplete sequence completion enhanced robust network (DISCERN) is developed to impute and integrate incomplete MRI sequences for glioma grading and IDH mutation status prediction.
![1](https://github.com/xinzhen-lab/Glioma-Prediction-on-Missing-MRI-Sequences/assets/131331281/633a6026-fd9d-4a04-a61b-265f15e95844)


# DISCERN
The DISCERN is capable of handling different clinical scenarios involving the missing of any one or several MRI sequences. Radiomics features were respectively extracted from various MRI sequences and seamlessly fed to train the DISCERN model, with missing feature values in the absent sequences initialized with zeros. For each MRI sequence, an encoder network is built to disentangle and learn hidden representations capturing both the sequence-specific and sequence-neutral components. These representations were then passed through a decoder network to impute missing values in patient samples with the absent sequence. This mechanism for completing missing sequences facilitated the acquisition of more representative representations. Subsequently, the sequence-specific and sequence-neutral components from all MRI sequences were integrated to produce the fused representation, which was finally input into a prediction network dedicated to the downstream tasks of glioma grading or IDH status prediction.  In general, DISCERN exhibited tolerance to varying missing rates of sequences ranging from 10% to 30%, yielding satisfactory prediction performance in both GBM grading (AUC:0.8843 to 0.9246) and IDH mutation status prediction (AUC: 0.8181 to 0.8655) on the training set. And the mean AUCs for glioma grading with a 10% missing rate were 0.7942 ± 0.0084, 0.8955 ± 0.0095, and 0.7759 ± 0.0063 on the internal validation set, external validation set 1, and external validation set 2. For IDH status prediction, the mean AUCs were 0.8574 ± 0.0062, 0.8477 ± 0.0091, and 0.7282 ± 0.0059, respectively.






<p align="center">
  <img src ="https://github.com/xinzhen-lab/Glioma-Prediction-on-Missing-MRI-Sequences/assets/131331281/fffbb326-1868-4e5d-8430-7ff17c1b81ec">
</p>


# Model training & validation
The DISCERN is trained and validated on datasets collected from four institutions, including the First Affiliated Hospital of Zhengzhou University (FAHZZU), the Nanfang Hospital of Southern Medical University (NFHSMU), the Second Affiliated Hospital of South China University of Technology (SAHSCUT), and the Beijing Tiantan Hospital (BJTTH), as well as one public database: The Cancer Imaging Archive (TCIA). The training (n=1440, 80%) and internal validation (n=456, 20%) sets consisted of patients enrolled from FAHZZU, NFHSMU, and SAHSCUT in accordance with the 2016 4th edition of the WHO classification of the central nervous system tumors (WHO 2016). The two independent external validation sets were collected from TCIA and BJTTH based on the 2021 5th edition of the WHO classification of the central nervous system tumors (WHO 2021).

# Requirements
Python-3.6.5  
numpy==1.19.0  
pandas==0.25.3  
scikit-learn==0.24.2  
imbalanced-learn==0.7.0  
tensorflow==2.1.0  
shap==0.39.0  
matplotlib==2.2.5  

# Usage
Make sure that all the required packages are installed, run the 'DISCERN.py' to check. if all required packages are properly installed, it will return the evaluation metrics of each iteration, including the area under the receiver operating characteristic (ROC) curve (AUC), accuracy (ACC), sensitivity (SEN), specificity (SPE), positive predictive value (PPV), negative predictive value (NPV), F1 score, and total loss. The feature files for grading and IDH mutation status used for testing 'DISCERN.py' are provided in the 'test_files' folder.
![3](https://github.com/xinzhen-lab/Glioma-Prediction-on-Missing-MRI-Sequences/assets/131331281/150e5a9f-cac4-4476-8ec9-08de50b832db)

Additionally, 'DISCERN_SHAP_interpretability.py' is built upon 'DISCERN.py' by incorporating SHAP interpretability to quantitatively assess the contribution of each radiomics feature to the model's output.
![shap](https://github.com/xinzhen-lab/Glioma-Prediction-on-Missing-MRI-Sequences/assets/131331281/95192f6b-3313-466a-a012-e9afda69afc5

# Note
Other investigators are invited to share their data in order to further improve the generalization capability of the current model. The model should only be used to support clinical diagnosis by health care professionals as a complementary tool for predicting glioma grades and IDH mutation status with incomplete MRI sequences. Any responsibility for using this model and its results will rest solely by the health care professional using the model. Using it you should understand and agree that this tool is not responsible or liable for any claim, loss, or damage resulting from its use. While we try to keep the information on the tool as accurate as possible, we disclaim any warranty concerning its accuracy, timeliness, and completeness, and any other warranty, express or implied, including warranties of merchantability or fitness for a particular purpose.
