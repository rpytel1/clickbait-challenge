# Clickbait challenge #
Project is reproducing a [paper](https://arxiv.org/pdf/1710.06699.pdf) for Clickbait challenge

## Structure of the modules ##
**data_reading** contains how data is read from file provided from challenge page but also an example how to read data from file after feature extraction

**feature_extraction** module contains code for extracting features. It is divided into few services: 

- *article_service* extracts data from structure of article e.g. number of paragraphs
- *behaviour_service* extracts data like retweets or hashtags
-  *common_words_service* extracts data overlapping between keywords and each section of article/post
- *formality_service* looks for formal and informal word in article and post
- *image_service* extracts features from image
- *time_service* extracts features about time of posting the article
- *word_service* extracts features related to words and characters

**ml** module will contain all content related to Machine Learning to perform classification

Dependencies:
```python3
pip3 install numpy nltk pytesseract json-pickle PyDictionary
```
Also might need to install tesseract.
```
brew install tesseract
```