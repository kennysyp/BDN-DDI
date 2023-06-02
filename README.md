# BDN-DDI: A bilinear dual-view representation learning framework for drug-drug interaction prediction
Authors: Guoquan Ning, Yuping Sun*, Jie Ling, Jijia Chen and Jiaxi He.
*Corresponding author: syp@gdut.cn(Y. Sun)

## Introduction
Drug-drug interactions (DDIs) refer to the potential effects of two or more drugs interacting with each other when used simultaneously, which may lead to adverse reactions or reduced drug efficacy. Accurate prediction of DDIs is a significant concern in recent years. Currently, the drug chemical substructure-based learning method has substantially improved DDIs prediction. However, we notice that most related works ignore the detailed interaction among atoms when extracting the substructure information of drugs. This problem results in incomplete information extraction and may limit the model's predictive ability. In this work, we proposed a novel framework named BDN-DDI (a bilinear dual-view representation learning framework for drug-drug interaction prediction) to infer potential DDIs. In the proposed framework, the encoder consists of six stacked BDN blocks, each of which extracts the feature representation of drug molecules through a bilinear representation extraction layer. The extracted feature is then used to learn embeddings of drug substructures from the single drug learning layer (intra-layer) and the drug-pair learning layer (inter-layer). Finally, the learned embeddings are fed into a decoder to predict DDI events. Based on our experiments, BDN-DDI has an AUROC value of over 99\% for the warm-start task. Additionally, it outperformed the state-of-the-art methods by an average of 3.4\% for the cold-start tasks. Finally, our method's effectiveness is further validated by visualizing several case studies.

## Flowchart
<img src="[image_url](https://github.com/kennysyp/BDN-DDI/blob/main/flow_A.png?raw=true)" alt="Alt Text"/>
<img src="[image_url](https://github.com/kennysyp/BDN-DDI/blob/main/flow_B.png?raw=true)" alt="Alt Text"/>

## Requirement
To run the code, you need the following dependencies:
* Python == 3.8
* pytorch == 1.12.0
* PyTorch Geometry == 2.3.0
* rdkit == 2020.09.2

