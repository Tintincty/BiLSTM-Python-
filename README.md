# BiLSTM-Python-
BiLSTM algorithm
For Sina Weibo text sentiment classification, 
the BiLSTM algorithm is used to build a text sentiment analysis model to analyze the impact of different batch data, 
different parameters, and different learning rates on the accuracy of the model, 
and establish an sentiment classification prediction model to provide support for related decision-making.
(1) A sentiment classification model based on BiLSTM is constructed using the PyTorch framework.
Compared with traditional machine learning algorithms, 
the BiLSTM-based model can automatically extract the sequence features in the text data,
and can learn more deeply the information contained in the text context.
The performance on the Weibo text data test set is significantly higher than that of Pu Subei The three benchmark classifiers of Yesh, 
support vector machine and multi-layer perceptron network have higher prediction performance in the text sentiment analysis task of public opinion evaluation.
(2) Using the model to build an online sentiment analysis system based on the Flask framework.
Using the trained BiLSTM model combined with the Flask framework to build an online sentiment analysis system,
the user inputs the text to be analyzed into the system, 
and the system calls the model to return the sentiment analysis results. 
The system can be applied to actual public opinion analysis tasks to assist relevant personnel in making decisions.
