# background
In kaggle, Quora Insincere Questions Classification aims at labeling insincere questions, which are founded upon false premises or intend to make a statement rather than look for helpful answers. It is a sentence-level classification task.

In the competition, 1306122 training data and 56370 test data are provided. In training set, 6.2% of the training data are insincere questions and the rest are sincere questions. We divide training set into two parts, 90% for the and training 10% for the validation. We program from the aspects of sentence preprocess, transmission from word to vector, training model and prediction. We apply different models and compare the results of different models, including Convolutional Neural Networks (CNN), Long Short-Term Memory Networks (LSTM), Gated Recurrent Unit (GRU), and then we add the Bidirectional way in Long Short-Term Memory Networks (LSTM) and Gated Recurrent Unit (GRU). We also blend different models together with the methods of Linear Regression and feedforward neural network. Besides, in order to improve accuracy, we introduce Attention Mechanism and combine four embeddings in the process of transforming world to vector. We selected the best model and got the public F1 score 0:695 with the highest score of 0:711. Finally, we draw some conclusions about comparison of different models and effectiveness of attention mechanism and embeddings combination.

# Result and discussions  
## Different models  
[CNN] CNN utilizes layers with convolving filters that
are applied to capture local features. Originally invented
for computer vision, CNN models have subsequently been
shown to be effective for NLP and have achieved excel-
lent results in many traditional NLP tasks. Yoon Kims
paper successfully applied CNN to Sentence Classification
[1]. We follow the CNN architecture of Kim. We connect
two Convolutional layers with and 32 feature maps, one
Max-over-time pooling layer and two fully connected lay-
ers with dropout. We test our model on testing set and
get public score of 0.640.
[LSTM] LSTM is an efficient gradient-based method
with the application of multiplicative gate units [2]. We
train a LSTM based model, including two 64-units LSTM
layers with dropout and two fully connected layers with
dropout. The public score based on this model is 0.655.
[GRU] Proposed by K.Cho, GRU only has two gates and
has fewer parameters than LSTM [3], and thus it is eas-
ier to train. The GRU based model includes two 64-units
GRU layers with dropout and two fully connected layers
with dropout. The public score based on this model is
0.652.
[Bi-LSTM] Bidirectional Long Short-Term Memory
Networks(Bi-LSTM) can be trained without the limita-
tion of using input information just up to a preset future
frame by training it simultaneously in positive and nega-
tive time direction [4].We train a Bi-LSTM based model,
including two 64-units LSTM layers with dropout and two
fully connected layers with dropout, and get the public
score of 0.660.
[Bi-GRU] The principle of Bidirectional GRU(Bi-GRU)
is the same as Bi-LSTM. We subsitute the LSTM layers
in Bi-LSTM based model with GRU layers and get the
public score of 0.655.
## Attention Mechanism  

## Different embeddings and their combination  

## Blending different models  


# Summary and conclusions  
