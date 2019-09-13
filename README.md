# Exploration of Sentiment Analysis using Lexicon and Machine-Learning Based Methods 

This repo provides the submission entry for an in-class NLP sentiment analysis competition held at Microsoft AI Singapore group using techniques learned in class to classify text in identifying positive or negative sentiment.

Data for this in-class competition comes from the Sentiment140 dataset where the training and test data consists of randomly sampled 10% and 5% of the Sentiment140 dataset.


![jpg](images/inclass-competition.jpg)


View the codes in Google Colab [here](https://drive.google.com/open?id=1d_po5AQDFRovk4livi2kvv1hhjPLxqAC) or download the SentimentAnalysis.ipynb


# Walk-through of the submission entry:


## 1. Adding imports & installing neccessay packages ##


```python
### run this if using google colab to mount google drive as local storage

from google.colab import drive
import os
drive.mount('/content/gdrive')

repo_path = '/content/gdrive/My Drive/colab/NLP-Bootcamp/'
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive



```python
import pandas as pd
import collections
%matplotlib inline

# Import modules to calculate accuracy and confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
```

## 2. Loading Data ##


```python
### run below 2 lines of code for setting train & test data path on google colab
'''
trainData = os.path.join(repo_path, 'data/sentiment140_160k_tweets_train.csv')
testData = os.path.join(repo_path, 'data/sentiment140_test.csv')
'''

### run below 3 lines of code for setting train & test data path on local machine
DATA = './data/'
trainData = DATA + 'sentiment140_160k_tweets_train.csv'
testData =  DATA + 'sentiment140_test.csv'

train = pd.read_csv(trainData)
test = pd.read_csv(testData)

train.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>ids</th>
      <th>user</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>1978186076</td>
      <td>ceruleanbreeze</td>
      <td>@nocturnalie Anyway, and now Abby and I share ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>p</td>
      <td>1994697891</td>
      <td>enthusiasticjen</td>
      <td>@JoeGigantino Few times I'm trying to leave co...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p</td>
      <td>2191885992</td>
      <td>LifeRemixed</td>
      <td>@AngieGriffin Good Morning Angie  I'll be in t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>1753662211</td>
      <td>lovemandy</td>
      <td>had a good day driving up mountains, visiting ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>p</td>
      <td>2177442789</td>
      <td>_LOVELYmanu</td>
      <td>downloading some songs  i love lady GaGa.</td>
    </tr>
  </tbody>
</table>
</div>



Looking at distribution of *'positives'* & *'negatives'* samples in train dataset 


```python
collections.Counter(train['target'])
```




    Counter({'n': 79985, 'p': 80000})




```python
train.groupby('target').size().plot(kind='bar')
```



![png](images/output_7_1.png)


We will find that it is a relatively well-balanced dataset

## 3. Data (Text) Preprocessing ##


```python
### mapping a dictionary of apostrophe words

appos = {
"aren't" : "are not",
"can't" : "cannot",
"cant" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"im" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"gg" : "going"
}
```


```python
import re

def preprocess_text(sentence):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', sentence['text'])
    text = re.sub('@[^\s]+','', text)
    text = text.lower().split()
    reformed = [appos[word] if word in appos else word for word in text]
    reformed = " ".join(reformed) 
    text = re.sub('&[^\s]+;', '', reformed)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    #text = re.sub(' [\w] ', ' ', text)
    return text.strip()

preprocess = train
preprocess['ugc'] = preprocess.apply(preprocess_text, axis=1)

preprocess.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>ids</th>
      <th>user</th>
      <th>text</th>
      <th>ugc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>1978186076</td>
      <td>ceruleanbreeze</td>
      <td>@nocturnalie Anyway, and now Abby and I share ...</td>
      <td>anyway and now abby and i share all our crops ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>p</td>
      <td>1994697891</td>
      <td>enthusiasticjen</td>
      <td>@JoeGigantino Few times I'm trying to leave co...</td>
      <td>few times I am trying to leave comments in you...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p</td>
      <td>2191885992</td>
      <td>LifeRemixed</td>
      <td>@AngieGriffin Good Morning Angie  I'll be in t...</td>
      <td>good morning angie I will be in the atl july 8...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>1753662211</td>
      <td>lovemandy</td>
      <td>had a good day driving up mountains, visiting ...</td>
      <td>had a good day driving up mountains visiting k...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>p</td>
      <td>2177442789</td>
      <td>_LOVELYmanu</td>
      <td>downloading some songs  i love lady GaGa.</td>
      <td>downloading some songs i love lady gaga</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Sentiment Analysis using Lexicon-based Method

There are two types of lexicon-based sentiment analyzing approcaches - _Polarity_ and _Valence_ based.

_VADER_ is a _VALENCE_ based sentiment analyzer.

*Valence*-based approach taken into consideration the "intensity" of a word as opposed to only the polarity (+ve or -ve). For example, "Great" is treated as more +ve as opposed to "Good".

References:
http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf

Scale for the classification model used base on compound value:

1. Positive = >=0
2. Negative = <0



```python
!pip install vaderSentiment
```

    Collecting vaderSentiment
    [?25l  Downloading https://files.pythonhosted.org/packages/86/9e/c53e1fc61aac5ee490a6ac5e21b1ac04e55a7c2aba647bb8411c9aadf24e/vaderSentiment-3.2.1-py2.py3-none-any.whl (125kB)
    [K     |████████████████████████████████| 133kB 1.4MB/s 
    [?25hInstalling collected packages: vaderSentiment
    Successfully installed vaderSentiment-3.2.1



```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
```


```python
def print_sentiment_scores(ugc):
    snt = analyzer.polarity_scores(ugc['ugc'])  # Calling the polarity analyzer
    return snt['compound']
```


```python
compound = train
compound['VADER']=compound.apply(print_sentiment_scores, axis=1)

compound.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>ids</th>
      <th>user</th>
      <th>text</th>
      <th>ugc</th>
      <th>VADER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>1978186076</td>
      <td>ceruleanbreeze</td>
      <td>@nocturnalie Anyway, and now Abby and I share ...</td>
      <td>anyway and now abby and i share all our crops ...</td>
      <td>0.6361</td>
    </tr>
    <tr>
      <th>1</th>
      <td>p</td>
      <td>1994697891</td>
      <td>enthusiasticjen</td>
      <td>@JoeGigantino Few times I'm trying to leave co...</td>
      <td>few times I am trying to leave comments in you...</td>
      <td>-0.0258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p</td>
      <td>2191885992</td>
      <td>LifeRemixed</td>
      <td>@AngieGriffin Good Morning Angie  I'll be in t...</td>
      <td>good morning angie I will be in the atl july 8...</td>
      <td>0.4404</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>1753662211</td>
      <td>lovemandy</td>
      <td>had a good day driving up mountains, visiting ...</td>
      <td>had a good day driving up mountains visiting k...</td>
      <td>0.7717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>p</td>
      <td>2177442789</td>
      <td>_LOVELYmanu</td>
      <td>downloading some songs  i love lady GaGa.</td>
      <td>downloading some songs i love lady gaga</td>
      <td>0.6369</td>
    </tr>
  </tbody>
</table>
</div>




```python
confusion_matrix(compound['target'], compound['predict'])
accuracy_score(compound['target'], compound['predict'])
```




    0.6673063099665594




```python
def custom_predict(ugc):
    snt = analyzer.polarity_scores(ugc['ugc'])  # Calling the polarity analyzer
    if snt['neg'] > snt['pos']:
        return 'n'
    elif snt['pos'] > snt['neg']:
        return 'p'
    else:
        return 'p'

vader = train
vader['predict']=vader.apply(custom_predict, axis=1)
```


```python
confusion_matrix(vader['target'], vader['predict'])
accuracy_score(vader['target'], vader['predict'])
```




    0.6673063099665594



## 5. Sentiment Analysis using Machine Learning-based Method: Naive Bayes


```python
#Import feature engineering modules and test_train_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

#Import classification algorithm
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

#Import modules to calculate accuracy and confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
```

Naive Bayes with TF-IDF on original text data


```python
tv = TfidfVectorizer(ngram_range=(1,3),max_features=20000,stop_words='english') 
X = tv.fit_transform(train['text'])

Xtrain, Xtest, ytrain, ytest = train_test_split(X, train['target'],
                                               test_size = 0.2, shuffle=True)

nb = MultinomialNB(alpha=6.5, fit_prior=False)
nb.fit(Xtrain,ytrain)
pred = nb.predict(Xtest)

print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(classification_report(ytest,pred))
```

    0.753820670687877
    [[12287  3768]
     [ 4109 11833]]
                  precision    recall  f1-score   support
    
               n       0.75      0.77      0.76     16055
               p       0.76      0.74      0.75     15942
    
        accuracy                           0.75     31997
       macro avg       0.75      0.75      0.75     31997
    weighted avg       0.75      0.75      0.75     31997
    


Naive Bayes with TF-IDF on pre-processed text data - achieved very minimal accuracy improvement


```python
tv = TfidfVectorizer(ngram_range=(1,3),max_features=20000,stop_words='english') 
X = tv.fit_transform(preprocess['ugc'])

Xtrain, Xtest, ytrain, ytest = train_test_split(X, preprocess['target'],
                                               test_size = 0.2, shuffle=True)

nb = MultinomialNB(alpha=6.5, fit_prior=False)
nb.fit(Xtrain,ytrain)
pred = nb.predict(Xtest)

print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(classification_report(ytest,pred))
```

    0.7545707410069694
    [[12184  3730]
     [ 4123 11960]]
                  precision    recall  f1-score   support
    
               n       0.75      0.77      0.76     15914
               p       0.76      0.74      0.75     16083
    
        accuracy                           0.75     31997
       macro avg       0.75      0.75      0.75     31997
    weighted avg       0.75      0.75      0.75     31997
    


Naive Bayes with Grid Search Hyperparameter Tuning & 10-Fold Cross Validation - achieving higher accuracy over the mdoel without hyperparameter tuning 


```python
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}
```


```python
x_train, x_test, y_train, y_test = train_test_split(train['text'], train['target'],
                                               test_size = 0.2, shuffle=True)
```


```python
from sklearn.metrics import classification_report
clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))
print(accuracy_score(y_test, clf.predict(x_test)))
print(confusion_matrix(y_test, clf.predict(x_test)))
```

                  precision    recall  f1-score   support
    
               n     0.7525    0.8386    0.7932     15876
               p     0.8209    0.7284    0.7719     16121
    
        accuracy                         0.7831     31997
       macro avg     0.7867    0.7835    0.7825     31997
    weighted avg     0.7870    0.7831    0.7825     31997
    
    0.7830734131324811
    [[13314  2562]
     [ 4379 11742]]



```python
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(preprocess['ugc'], preprocess['target'],
                                               test_size = 0.2, shuffle=True)
```


```python
from sklearn.metrics import classification_report
clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))
print(accuracy_score(y_test, clf.predict(x_test)))
print(confusion_matrix(y_test, clf.predict(x_test)))
```

                  precision    recall  f1-score   support
    
               n     0.7571    0.8380    0.7955     16035
               p     0.8177    0.7299    0.7713     15962
    
        accuracy                         0.7841     31997
       macro avg     0.7874    0.7840    0.7834     31997
    weighted avg     0.7873    0.7841    0.7834     31997
    
    0.7840735068912711
    [[13438  2597]
     [ 4312 11650]]



```python
print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)
```

    Best Score:  0.7837531643591586
    Best Params:  {'clf__alpha': 0.1, 'tfidf__norm': 'l1', 'tfidf__use_idf': False, 'vect__ngram_range': (1, 2)}



```python
from google.colab import files ### remove this line of code if not using colab

test['ugc'] = test.apply(preprocess_text, axis=1)
y_kaggle = clf.predict((test['ugc']))
test['target'] = pd.DataFrame(y_kaggle.tolist())
test[['target', 'ids']].to_csv("nb_submission.csv", index=False)

files.download('nb_submission.csv') ### remove this line of code if not using colab

```


```python
from sklearn.metrics import classification_report
clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))
print(accuracy_score(y_test, clf.predict(x_test)))
print(confusion_matrix(y_test, clf.predict(x_test)))
```

                  precision    recall  f1-score   support
    
               n     0.7734    0.8187    0.7954     16002
               p     0.8073    0.7600    0.7829     15995
    
        accuracy                         0.7894     31997
       macro avg     0.7904    0.7893    0.7892     31997
    weighted avg     0.7904    0.7894    0.7892     31997
    
    0.7893552520548801
    [[13101  2901]
     [ 3839 12156]]


## 6. Sentiment Analysis using Machine Learning-based Method: Linear SVM ##
with Grid Search Hyperparameter Tuning & 10-Fold Cross Validation


```python
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC())])
tuned_parameters = {
    'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
    'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__tol': [1, 1e-1, 1e-2, 1e-3]
}
```


```python
x_train, x_test, y_train, y_test = train_test_split(preprocess['ugc'], preprocess['target'],
                                               test_size = 0.2, shuffle=True)
```


```python
clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))
print(accuracy_score(y_test, clf.predict(x_test)))
print(confusion_matrix(y_test, clf.predict(x_test)))

print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)
```

                  precision    recall  f1-score   support
    
               n     0.7939    0.8293    0.8112     15870
               p     0.8243    0.7882    0.8058     16127
    
        accuracy                         0.8086     31997
       macro avg     0.8091    0.8087    0.8085     31997
    weighted avg     0.8092    0.8086    0.8085     31997
    
    0.8085758039816233
    [[13161  2709]
     [ 3416 12711]]
    Best Score:  0.8010751007906991
    Best Params:  {'clf__tol': 0.1, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 4)}



```python
from google.colab import files ### remove this line of code if not using colab

test['ugc'] = test.apply(preprocess_text, axis=1)
y_kaggle = clf.predict((test['ugc']))
test['target'] = pd.DataFrame(y_kaggle.tolist())
test[['target', 'ids']].to_csv("l_svm_submission.csv", index=False)

files.download('l_svm_submission.csv') ### remove this line of code if not using colab
```

## 7. Sentiment Analysis using Machine Learning-based Method: XGBoost


```python
!pip install xgboost
```

    Requirement already satisfied: xgboost in /usr/local/lib/python3.6/dist-packages (0.90)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from xgboost) (1.3.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from xgboost) (1.16.4)



```python
tv = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english', min_df=.0025, max_df=0.25) 
X = tv.fit_transform(preprocess['ugc'])

x_train, x_test, y_train, y_test = train_test_split(X, preprocess['target'],
                                               test_size = 0.2, shuffle=True)
```


```python
xgb = XGBClassifier(max_depth=10, n_estimators=400, learning_rate=0.3, objective='binary:logistic')
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
```


```python
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

    0.7501015720223771
    [[11302  4860]
     [ 3136 12699]]
                  precision    recall  f1-score   support
    
               n       0.78      0.70      0.74     16162
               p       0.72      0.80      0.76     15835
    
        accuracy                           0.75     31997
       macro avg       0.75      0.75      0.75     31997
    weighted avg       0.75      0.75      0.75     31997
    


## 8. Sentiment Analysis using Machine Learning-based Method: Decision Tree


```python
tv = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english') 
X = tv.fit_transform(preprocess['ugc'])

x_train, x_test, y_train, y_test = train_test_split(X, preprocess['target'],
                                               test_size = 0.2, shuffle=True)
```


```python
dt = DecisionTreeClassifier()
dt.fit(Xtrain,ytrain)
pred = dt.predict(Xtest)
```


```python
print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(classification_report(ytest,pred))
```

    0.6914398224833578
    [[10944  5018]
     [ 4855 11180]]
                  precision    recall  f1-score   support
    
               n       0.69      0.69      0.69     15962
               p       0.69      0.70      0.69     16035
    
        accuracy                           0.69     31997
       macro avg       0.69      0.69      0.69     31997
    weighted avg       0.69      0.69      0.69     31997
    


## 9. Sentiment Analysis using Machine Learning-based Method: Random Forest


```python
rf = RandomForestClassifier()
rf.fit(Xtrain,ytrain)
pred = rf.predict(Xtest)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)



```python
print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(classification_report(ytest,pred))
```

## 10. Sentiment Analysis using Machine Learning-based Method: Extra Trees


```python
etc=ExtraTreesClassifier()
etc.fit(Xtrain,ytrain)
pred=etc.predict(Xtest)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)



```python
print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(classification_report(ytest,pred))
```

    0.7307872613057474
    [[11874  4088]
     [ 4526 11509]]
                  precision    recall  f1-score   support
    
               n       0.72      0.74      0.73     15962
               p       0.74      0.72      0.73     16035
    
        accuracy                           0.73     31997
       macro avg       0.73      0.73      0.73     31997
    weighted avg       0.73      0.73      0.73     31997
    


## 11. Sentiment Analysis using Machine Learning-based Method: SVC ##

_Warning - approximately 3hrs of processing_ 


```python
#Import feature engineering modules and test_train_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

tv = TfidfVectorizer(ngram_range=(1,3)) 
X = tv.fit_transform(preprocess['ugc'])

Xtrain, Xtest, ytrain, ytest = train_test_split(X, preprocess['target'],
                                               test_size = 0.2, shuffle=True)
```


```python
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(Xtrain,ytrain)
pred = svm.predict(Xtest)

print(accuracy_score(ytest,pred))
print(confusion_matrix(ytest,pred))
print(classification_report(ytest,pred))
```

    0.805137981685783
    [[12754  3333]
     [ 2902 13008]]
                  precision    recall  f1-score   support
    
               n       0.81      0.79      0.80     16087
               p       0.80      0.82      0.81     15910
    
        accuracy                           0.81     31997
       macro avg       0.81      0.81      0.81     31997
    weighted avg       0.81      0.81      0.81     31997
    



```python
# Uncomment and run below line of code if using google colab
# from google.colab import files

test['ugc'] = test.apply(preprocess_text, axis=1)
y_kaggle = svm.predict(tv.transform(test['ugc']))
test['target'] = pd.DataFrame(y_kaggle.tolist())
test[['target', 'ids']].to_csv("svc_submission.csv", index=False)

# Uncommon and run below line of code if using google colab 
# files.download('svc_submission.csv')
```

## Text Pre-processing Steps - References ##

https://www.topbots.com/text-preprocessing-for-machine-learning-nlp/

## Further - Text Preprocessing: Porter Stemmer ##


```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

#nltk.download('punkt') 

#create an object of class PorterStemmer
porter = PorterStemmer()
lancaster=LancasterStemmer()

```


```python
def stemSentence(sentence):
    token_words = word_tokenize(sentence['ugc'])
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
  
preprocess['stem'] = preprocess.apply(stemSentence, axis=1)
preprocess.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>ids</th>
      <th>user</th>
      <th>text</th>
      <th>ugc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>1978186076</td>
      <td>ceruleanbreeze</td>
      <td>@nocturnalie Anyway, and now Abby and I share ...</td>
      <td>anyway and now abbi and i share all our crop w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>p</td>
      <td>1994697891</td>
      <td>enthusiasticjen</td>
      <td>@JoeGigantino Few times I'm trying to leave co...</td>
      <td>few time i m tri to leav comment in your blog ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>p</td>
      <td>2191885992</td>
      <td>LifeRemixed</td>
      <td>@AngieGriffin Good Morning Angie  I'll be in t...</td>
      <td>good morn angi i ll be in the atl juli 8th 1 t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>1753662211</td>
      <td>lovemandy</td>
      <td>had a good day driving up mountains, visiting ...</td>
      <td>had a good day drive up mountain visit kati ea...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>p</td>
      <td>2177442789</td>
      <td>_LOVELYmanu</td>
      <td>downloading some songs  i love lady GaGa.</td>
      <td>download some song i love ladi gaga</td>
    </tr>
  </tbody>
</table>
</div>




