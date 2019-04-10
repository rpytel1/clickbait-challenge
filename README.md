# Clickbait challenge #
Project is partly reproducing a [paper](https://arxiv.org/pdf/1710.06699.pdf) and also extending with a lot of new features.
Originally project aims to detect clickbait phrases and was mostly motivated by [Clickbait Challenge](https://www.clickbait-challenge.org/)

## Structure of the modules ##
### data_reading ###
**data_reading** contains how data is read from file provided from challenge page, preprocessing related content and also an example how to read data from file after feature extraction

### feature_extraction ###
**feature_extraction** module contains code for extracting features. It is divided into few services: 

- *article_service* extracts data from structure of article e.g. number of paragraphs
- *behaviour_service* extracts data like retweets or hashtags
- *clickbait_words_service* check if data contain clickbait specific phrases
-  *common_words_service* extracts data overlapping between keywords and each section of article/post
- *cosine_similarity_service* calculates cosine similarity between each parts of the post/article
- *dependencies_service* calculates different number of POS per each part of article/post
- *formality_service* looks for formal and informal word in article and post
- *image_service* extracts features from image
- *ngrams_service* extracts all ngram related features
- *patternPOS_service* looks for patterns in POS
- *readability_service* calculates readability features
- *sentiment_analysis_service* extracts all sentiment related features
- *slang_service* checks for occurrence of clickbait specific phrases
- *time_service* extracts features about time of posting the article
- *word_service* extracts features related to words and characters

### feature_selection ###
This module contains code for selecting best features (by name), calculating the
statistics and creation of the ranking. In addition this module also contains selected features for 78, 79, 81 and 246 features (both data and labels of the features selected). 
Code in this module also ranks the features according to their usefulness to improve regression and classification problem. 

### figures ###
This module contains some figures related to the performance of each Machine Learning algorithm. 


### ml ###
**ml** module contains all classification/regression code, mostly scripts performing tests on data. We can distinguish following packages related to certain algorithms: 

- *AdaBoost*, code related to regression using AdaBoost
- *GTBoost*, code related to regression using GTBoost
- *linear_regression*, code related to regression using simple linear regression
- *random_forest*, code related to regression using random forest
- *ridge_regression*, code related to regression using Ridge
- *svr*, code related to regression using SVM
- *utils* module containing helper functions used among ml module


## Instalation related info ##
Dependencies:
```python3
pip3 install numpy nltk pytesseract pickle sklearn xgboost textstat
```
Also might need to install tesseract.
```
brew install tesseract
```
For POS extraction it is crucial to have [POS taggers](https://nlp.stanford.edu/software/tagger.shtml) and [NER taggers](https://nlp.stanford.edu/software/CRF-NER.html) from Stanford at project root.

For sentiment analysis it is crucial to have Stanford's Core NLP server running in the background on port 9000.
 Instructions how to run and download can be found [here](https://stackoverflow.com/questions/32879532/stanford-nlp-for-python?fbclid=IwAR2rECAGbOzMyez-Q6DcbqRbNr1ZZfK0jklHa9joU9PTzA6Uwpi6ocRQphM).
 Without server running features will not be extracted. 