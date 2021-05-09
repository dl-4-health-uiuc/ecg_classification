# ECG Heartbeat Classification using Convolutional Neural Network (CNN) and Application of Transfer Learning for MI Classification

Deep Learning (DL) has recently become a research subject in a variety of fields, including healthcare, where timely identification of anomalies on an electrocardiogram (ECG) can be critical in patient monitoring. This technology has recently shown remarkable progress in a variety of tasks, and there are high hopes for its potential to transform clinical practice. Rather than studying and applying transferable information through tasks, most studies have focused on classifying a collection of conditions on a dataset annotated for that task. In this project, we implemented a model based on deep convolutional neural networks i.e. CNNNet which is able to accurately classify five different heartbeat arrhythmias as per the AAMI EC57 standards. Further, we proposed a method for transferring the knowledge acquired by the model in this task to myocardial infraction (MI) classification. We trained our proposed model on PhysionNet’s MIT-BIH and PTB Diagnostics datasets and evaluated its performance. According to the results, the proposed CNNNet is able to make predictions with overall F1-score of 0.9366 on arrhythmia classification, 0.9677 on MI prediction using the learned representation through transfer learning and 0.9957 on MI prediction when the model is trained from scratch.



The main network used in this code is:

![image-20210506100736395](images/cnet)

 

If a transfer model is given for MI prediction then the last 2 fully connected layers are modified based on the model type. 



## Instructions

- The dataset i.e. 4 input csv files should be present inside an input folder in the root path. It can be downloaded from https://www.kaggle.com/shayanfazeli/heartbeat

- The code will use GPU if present.

- Use python 3 to run.

- Arguments include:

  - epochs -- number of epochs to run, default is 40.
  - model -- model name. Valid values are cnet, seq, bilstm, bigru.
  -  smote -- t if smote is required else f, default is False.
  - batch_size -- default is 256.
  - lr -- learning rate, default is 0.001
  - transfer_path -- model state dict path for pretrained models.
  - mi -- t to run mi prediction, else runs ecg classification.
  - save_model_path -- path to save the trained model.

- Example commands:

  ```python
  python main.py --epochs 80 --model cnet --smote f --save_model_path cnet_no_smote
  python main.py --epochs 600 --model cnet --mi t --smote f
  python main.py --epochs 600 --model cnet --mi t --smote f --transfer_path cnet_no_smote
  python main.py --epochs 80 --model seq --smote f
  python main.py --epochs 80 --model bigru --smote t
  python main.py --epochs 80 --model bilstm --smote t
  ```



## Results

- Results without augmentation.

| CNet        |          |          |          |          |          |          |                |          |
| ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------------- | -------- |
|             | overall  | N        | S        | V        | F        | Q        | MI(pretrained) | MI       |
| Precision   | 93.66    | 99.18    | 90.75    | 97.6     | 81.82    | 98.94    | 96.71          | 99.57    |
| Recall      | 92.08    | 99.58    | 82.91    | 95.65    | 83.33    | 98.94    | 96.57          | 99.86    |
| f1          | 92.83118 | 99.3796  | 86.65303 | 96.61516 | 82.5681  | 98.94    | 96.6399493     | 99.71479 |
|             |          |          |          |          |          |          |                |          |
| **Seq2Seq** |          |          |          |          |          |          |                |          |
|             | overall  | N        | S        | V        | F        | Q        | MI             |          |
| Precision   | 92.46    | 98.77    | 84.54    | 96.24    | 84.4     | 98.38    | 97.78          |          |
| Recall      | 88.5     | 99.34    | 77.7     | 93.65    | 73.46    | 98.38    | 98.67          |          |
| f1          | 90.37765 | 99.05418 | 80.97581 | 94.92734 | 78.55092 | 98.38    | 98.22298397    |          |
|             |          |          |          |          |          |          |                |          |
| **BiLSTM**  |          |          |          |          |          |          |                |          |
|             | overall  | N        | S        | V        | F        | Q        | MI             |          |
| Precision   | 92.16    | 98.83    | 82.57    | 95.62    | 85.11    | 98.69    | 98.67          |          |
| Recall      | 88.83    | 99.22    | 77.52    | 94.89    | 74.07    | 98.45    | 98.57          |          |
| f1          | 90.40412 | 99.02462 | 79.96535 | 95.2536  | 79.20716 | 98.56985 | 98.61997465    |          |
|             |          |          |          |          |          |          |                |          |
| **BiGRU**   |          |          |          |          |          |          |                |          |
|             | overall  | N        | S        | V        | F        | Q        | MI             |          |
| Precision   | 92.88    | 98.63    | 92.47    | 96.27    | 77.78    | 99.24    | 98.68          |          |
| Recall      | 89.32    | 99.5     | 72.84    | 94.41    | 82.1     | 97.76    | 99.29          |          |
| f1          | 90.85192 | 99.06309 | 81.4895  | 95.33093 | 79.88164 | 98.49444 | 98.98406021    |          |

- Results with augmentation.

| CNet smote        |          |          |          |          |          |          |                |       |
| ----------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------------- | ----- |
|                   | overall  | N        | S        | V        | F        | Q        | MI(pretrained) | MI    |
| Precision         | 92.92    | 99.15    | 84.24    | 96.6     | 84.87    | 99.75    | 96.77          | 99.57 |
| Recall            | 91.44    | 99.38    | 83.63    | 96.2     | 79.63    | 98.38    | 96.72          | 99.38 |
| f1                | 92.16503 | 99.26487 | 83.93389 | 96.39959 | 82.16654 | 99.06026 | 96.74499354    | 99.06 |
|                   |          |          |          |          |          |          |                |       |
| **Seq2Seq smote** |          |          |          |          |          |          |                |       |
|                   | overall  | N        | S        | V        | F        | Q        | MI             |       |
| Precision         | 89.87    | 98.94    | 78.42    | 94.89    | 79.01    | 98.07    | 97.86          |       |
| Recall            | 90.28    | 98.87    | 81.65    | 94.11    | 78.53    | 98.26    | 97.95          |       |
| f1                | 90.06799 | 98.90499 | 80.00241 | 94.49839 | 78.76927 | 98.16491 | 97.91          |       |
|                   |          |          |          |          |          |          |                |       |
| **BiLSTM smote**  |          |          |          |          |          |          |                |       |
|                   | overall  | N        | S        | V        | F        | Q        | MI             |       |
| Precision         | 87       | 99.1     | 76.62    | 93.21    | 67.33    | 98.75    | 97.19          |       |
| Recall            | 91.27    | 98.63    | 80.76    | 94.82    | 83.95    | 98.2     | 97.42          |       |
| f1                | 88.94187 | 98.86444 | 78.63555 | 94.00811 | 74.72704 | 98.47423 | 97.30486409    |       |
|                   |          |          |          |          |          |          |                |       |
| **BiGRU smote**   |          |          |          |          |          |          |                |       |
|                   | overall  | N        | S        | V        | F        | Q        | MI             |       |
| Precision         | 87.91    | 98.99    | 76.82    | 95.46    | 69.35    | 98.93    | 97.05          |       |
| Recall            | 91.26    | 98.85    | 79.86    | 94.48    | 85.19    | 97.95    | 98.5           |       |
| f1                | 89.41874 | 98.91995 | 78.31051 | 94.96747 | 76.45822 | 98.43756 | 97.76962414    |       |


## Contribution

This project was implemented as the Final Course Project for the CS 598 Deep Learning in Healthcare class in Spring 2021 by anantas2, akshayn4, aviralj2 and sp30.

## References

[1] E.J. Benjamin, S.S. Virani, C.W. Callaway, A.M. Chamberlain, A.R. Chang, S. Cheng, ..., R. Deo, et al. Heart disease and stroke statistics-2018 update: A report from the american heart association. Circulation, 137 (12) (2018), p. e67
[2] ACM H. Society, “Heart diseases and disorders,” 2018. [Online] Available:https://www.hrsonline.org/Patient-Resources/Heart-Diseases-Disorders
[3] Miotto R, Wang F, Wang S, Jiang X, Dudley JT. Deep learning for healthcare: review, opportunities and challenges. Brief Bioinform.2 018;19(6):1236-1246. doi:10.1093/bib/bbx044
[4] M. Oquab, L. Bottou, I. Laptev, and J. Sivic, “Learning and transferring mid-level image representations using convolutional neural networks,” 2014 IEEE Conference on Computer Vision and Pattern Recognition, pp. 1717–1724, 2014. 
[5] Goodfellow, Ian, et al. Deep learning. Vol. 1. No. 2. Cambridge: MIT press, 2016.
[6] Andreotti, Fernando, et al. "Comparing feature-based classifiers and convolutional neural networks to detect arrhythmia from short segments of ECG." 2017 Computing in Cardiology (CinC). IEEE, 2017.
[7] U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, M. Adam, A. Gertych, and R. San Tan, “A deep convolutional neural network model to classify heartbeats,” Computers in biology and medicine, vol. 89, pp. 389–396, 2017.
[8] S. Mousavi and F. Afghah, "Inter- and Intra- Patient ECG Heartbeat Classification for Arrhythmia Detection: A Sequence to Sequence Deep Learning Approach," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, UK, 2019, pp. 1308-1312, doi: 10.1109/ICASSP.2019.8683140.
[9] A. L. Goldberger, L. A. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “Physiobank, physiotoolkit, and physionet,” Circulation, vol. 101, no. 23, pp. e215–e220, 2000.
[10]  G. B. Moody and R. G. Mark, “The impact of the mit-bih arrhythmia database,” IEEE Engineering in Medicine and Biology Magazine, vol. 20, no. 3, pp. 45–50, 2001.
[11]  R. Bousseljot, D. Kreiseler, and A. Schnabel, “Nutzung der ekgsignaldatenbank cardiodat der ptb uber das internet,” ¨ Biomedizinische Technik/Biomedical Engineering, vol. 40, no. s1, pp. 317–318, 1995.
[12]  Preprocessed data available in Kaggle datasets: https://www.kaggle.com/shayanfazeli/heartbeat
[13]  Kachuee, Mohammad and Fazeli, Shayan and Sarrafzadeh, Majid, “ECG Heartbeat Classification: A Deep Transferable Representation”, 2018 IEEE International Conference on Healthcare Informatics (ICHI)
[14]  Chawla NV et al (2002) SMOTE: synthetic minority over-sampling technique. J Artif Intell Res 16:321–357
[15]  K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778. 
[16]  V. Nairand G. E. Hinton, “Rectified linear units improve restricted boltz-mann machines,” in Proceedings of the 27th international conference on machine learning (ICML-10), 2010, pp. 807–814. 
[17]  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
[18]  D. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014. 
[19]  Maaten LV, Hinton G. Visualizing data using t-SNE. Journal of machine learning research. 2008;9(Nov):2579-605.
[20]  Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to sequence learning with neural networks. In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'14). MIT Press, Cambridge, MA, USA, 3104–3112.
[21]  Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory. Neural Comput. 9, 8 (November 15, 1997), 1735–1780. DOI:https://doi.org/10.1162/neco.1997.9.8.1735
[22]  Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. In NIPS 2014 Workshop on Deep Learning, December 2014



