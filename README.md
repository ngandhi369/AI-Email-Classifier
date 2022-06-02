# AI-Email-Classifier

* Mail classifier made using flask that classifies unread mails from authorized Gmail account by IMAP4 protocol.
* Training datasets contains only two columns: subject & category and having 20 different categories like junkfile, online trading, management, corporate, etc.
* Used SGDClassifier for trainning the dataset.
* Model Accuracy: 76.6%
* Used Tfidf Transformer & CountVectorizer for feature extraction in ML.
* Used Gridsearchcsv for model selection.
