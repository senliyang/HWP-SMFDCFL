# SNVENMDA:A Microbe–Diseases Association Identifcation Framework Based on Deep Neural Network and Interpretable Boosting Machine
A large body of research evidence suggests that the occurrence of many complex diseases in the human body is related to microbial communities. Therefore, identifying potential associations between microbes and diseases is of great significance for disease diagnosis, prognosis, and treatment. However, traditional biomedical experiments are costly, time-consuming, and labor-intensive. To address this, we propose a novel computational model, SNVENMDA, to predict potential associations between microbes and diseases. First, we integrated information from three databases and used a nonlinear iterative approach to fuse multi-source features of microbes and diseases based on known associations between microbes and diseases. Second, the model employs feature extraction using stacked graph autoencoders, non-negative matrix factorization, and variational graph autoencoders, combined with deep neural networks and Interpretable Boosting Machines for classification to predict potential microbes-disease associations. Under 5-fold cross-validation on four experimental settings (microbe, disease, microbe-disease pair, and independent microbe and disease), we compared SNVENMDA with four advanced prediction methods (GPUDMDA, DSAE_RF, VGAMF, and GCNDAEMDA) and four classical ensemble learning classifiers (Random Forest, Extra Trees, XGBoost, and LightGBM). Experimental results demonstrate that SNVENMDA outperforms the other methods significantly in all four cross-validation settings on the HMDIP and Peryton databases. Furthermore, case studies on Parkinson's disease, obesity, Crohn's disease, and colorectal cancer show that the microbial-disease associations predicted by SNVENMDA are mostly validated, further proving the reliability of SNVENMDA in predicting microbe-disease associations.SNVENMDA is publicly available at https://github.com/senliyang/SNVENMDA.
# Flowchart
![image](https://github.com/senliyang/SNVENMDA/blob/main/SNVENMDA.png)
# Requirements
Install python3.7 for running this model. And these packages should be satisfied:
tensorflow-gpu ≈2.6.0
numpy ≈ 1.19.5
pandas ≈ 1.1.5
scikit-learn ≈ 0.24.2
# Usage
Default is 5-fold cross validation from CVm to CVmid on HMDIP and Peryton databases. To run this model：python main.py
Calculate the integrated similarity between microbes and diseases  　&ensp;                  python SNF.py          
Extract the linear features of microbes and diseases             　&ensp;        python NMF.py                 
Extract the nonlinear features of microbes and diseases          　&ensp;      python SGAE.py ;python VGAE.py
