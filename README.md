### File organization 

The files in the submission are organized as follows
1. Data_process.ipynb 
```
Code for preprocessing the data to a required format by all other codes. Converts the `categories` feature into a one hot vector, converts and normalizes the date attribute, fills the missing sentiment score value with mean value and concatenates all the meta attributes to a single vector dense_feats. Generates 2 folders - binary_data: data for binary classification (1,2,3 stars - negative, and 4,5 - positive) and processed_data: data for 5-star classfication.
```
2. BasicMLModels.ipynb
```
Code for all the basic ML models presented in the report with all 6 modes of classification binary: meta, text, meta+text, and 5-star: meta, text, meta+text. Hyper pameters. We use the following hyperparameter configurations for the the ML baselines tested in this work: Decision Trees - {gini entropy, no maximum depth, RF - 100 estimators, gini entropy, no max depth}, SVM - {rbf kernel with C value 1.0}, Gradient boosting - {learning rate 0.1, max depth of 5, max features of 0.5}, KNN - {K=10}, MLP - {2 hidden layers with 256, 128 units, learning rate 0.001, adam optimizer}. 
```
3. KimModel.ipynb 
```
Code for the CNN model explained in the report. Input is the text through the glove 100 dimensional embedding format. Depending on the input data folder and num_classes variable, can be altered between binary and 5-star mode classifications.
```
4. KimModel-MT-Joint.ipynb 
```
Code for the CNN model with Multitasking and joint input representation explained in the report. Input is the text in form or glove 100 dimensional embedding format or text + meta. Depending on the input data folder and num_classes variable, can be altered between binary and 5-star mode classifications. Layers of MT need to be commented to alter between MT and normal modes. 
```
5. HAN.ipynb 
```
Code for the HAN model explained in the report. Input is the text through the glove 100 dimensional embedding format. Depending on the input data folder and num_classes variable, can be altered between binary and 5-star mode classifications.
```
6. HAN-MT-Joint.ipynb 
```
Code for the HAN model with Multitasking and joint input representation explained in the report. Input is the text in form or glove 100 dimensional embedding format or text 100d + meta. Depending on the input data folder and num_classes variable, can be altered between binary and 5-star mode classifications. Layers of MT need to be commented to alter between MT and normal modes. 
```
7. BERT_Text_Classification.ipynb
```
Code for the simple BERT model explained in the report. Input is the text of the review. Depending on the input data folder and num_classes, labels variables in the code, can be altered between binary and 5-star mode classifications.
```
8. BERT-Joint.ipynb 
```
Code for the BERT model with review text + meta features as the input. Depending on the input data folder and num_classes, labels variables in the code, can be altered between binary and 5-star mode classifications.
```
9. BERT-MT.py
```
Code for the BERT model with Multitasking for star rating prediction and sentiment score estimation. Input is the text of the review. Depending on the input data folder and and num_classes, labels variables in the code, can be altered between binary and 5-star mode classifications.
```
10. BERT-MT-Joint.py
```
Code for the final BERT model variant with Multitasking for star rating prediction and sentiment score estimation with review text and meta features as the Input. Depending on the input data folder and and num_classes, labels variables in the code, can be altered between binary and 5-star mode classifications.
```
11. bias_loss_curves.ipynb
```
Code for plotting the training bias graph and the loss curve with training and validation losses for the final BERT-MT-Joint model. 
```
12. BERT_visualization.ipynb
```
Code for plotting the qualitative visualization of attention weights for a given input to the model. Currently in the notebook, the BERT-MT variant is used for plotting on a out-of-dataset example (rare example in the report). The path needs to be changed to  BERT-MT-Joint variant for the outputs reported in Figure 5. 
```
13. Hyperparameter configurations for deep models
```
For deep models CNN and HAN, we use 100-dimensional GloVe embeddings for input words. 
CNN encoder - 256 1D filters of size 3 and stride 1. 
HAN model - hidden size of word and sentence encoder LSTM is 50. 
HAN model - Input to the sentence encoder is 50 dimensional. 
BERT encoder - base-uncased variant with 12 hidden layers, 12 attention heads and hidden size 768 (embedding dim). 

Meta encoder in Joint models - Single hidden layer FNN with 512 units followed by a dropout layer. 
Concatenation module in Joint models -  Single layer FNN with 256 units. 
Classification and Regression heads - Dense layers have number of hidden units equal to the outputs i.e 2 (binary) or 5 (5-star) for star rating prediction and 1 for sentiment score estimation. 

Loss $\mathcal{L}_1$ - cross-entropy.
loss $\mathcal{L}_2$ - mean squared error loss. 

For all models -
Activation functions - RELU
batch size - 32

maximum sequence length for CNN and HAN - 50
maximum sequence length for BERT - 128 
optimizer - Adam for CNN and HAN with learning rate 1e-4
optmizer - AdamW for BERT with learning rate 5e-5 and epsilon 1e-8
Num epochs - 5 for CNN and HAN and 
Num fine-tune epochs - 3 for BERT
```