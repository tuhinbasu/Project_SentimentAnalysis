### Sentiment_Analysis<br>
This is the working video - [here](https://www.linkedin.com/posts/tuhinbasu_project-wordcloud-nlp-activity-6855188109936640000-b895) <br>
<h4> Business Objective and Constraints </h4> <br>
Business Problem: Build a sentiment analytic model where the user will give data in the format of PDF or text and it will reveal the sentiment of the document in granular level.<br>
  
Business Objective: Maximize detection of sentiment; minimize the time taken. <br>

  
Business Constraints: Storage capacity; one page at a time; Sarcasm detection; new lingo; typo error. <br>

### This was a hobby project for fun.
### Segementation of code
Libraries used –
1.	Nltk
2.	Pypdf2
3.	Re
4.	Matplotlib
5.	Sklearn for count vectorizer
6.	Gensim
<br>

We used a user defined function that will take PDF as input and a loop that will take page number and copy all the text and convert it into string format.
* After that we used Pre processing techniques like removing special characters, numbers and words less than 3 characters. Then we remove all the extra space and convert it into lower case.
* We broke them into tokens.
* We had 3 text files containing Stop words, Positive words and Negative words and stored them into 3 different variables in a list format. Then we removed stop words, positive and negative words each word comparing with the token word. After that we add the tokens into string format. And then stored the string into .txt file in static folder.
* We performed Polarity analysis. <br>

```
''' Polarity analysis '''
def sentiment(OT_):
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(OT_)
    del polarity["compound"] # removing compound as we don’t need for our objective
# Plot the polarity
    #y_val = list(polarity.values())
    br = plt.bar(range(len(polarity)), list(polarity.values()), align='center', color='crimson', width = 0.7)
    plt.xticks(range(len(polarity)), list(polarity.keys()))
    plt.grid(axis = "y")
    # access the bar attributes to place the text in the appropriate location
    for br in br:
        yval = br.get_height()
        plt.text(br.get_x()+.25, yval + .01, yval)
    plt.savefig("E:\Sentiment_analysis\polarity.png")
    plt.close()
sentiment_ = sentiment(original_text)
```

We plotted Bigram word cloud of general, positive and negative words. Bigram is very useful when we want context of a word.
We performed LDA for topic modeling.
We took the data or the string and converted that into a dataframe.<br>
```
d1 = {'Variables':['document'], 'text':[info]}
data = pd.DataFrame(d1)

```
We removed adjectives, nouns from the text. We used count vectorizer and Document term matrix.
We created gensim corpus and vocabulary dictionary. We used model.Lda from gensim to generate topics.
<br>
```
ldana = models.LdaModel(corpus=corpusna, num_topics=1, id2word=id2wordna, passes=10)
```
# Code - <br>
```
# Libraries
import nltk
import PyPDF2
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction import text
from nltk import word_tokenize, pos_tag
from gensim import matutils, models
import scipy.sparse
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

''' PDF '''
def input_pdf():
    # creating a pdf file object
    pdfFileObj = open('E:\Sentiment_analysis\Sample2.pdf', 'rb')
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # printing number of pages in pdf file
    page = pdfReader.numPages # Page numbers
    ''' Creating a string where our doc will be stored '''
    text_str = ""
    ''' this loop will keep running multiple times based on the ppage numbers '''
    for i in range(page):
    # creating a page object
        pageObj = pdfReader.getPage(i)
    # extracting text from page
        text_str += pageObj.extractText() # Counts each characters in a text
    return text_str

# =================================================
info = input_pdf() # user
# ==================================================
''' Removing Punctuations, special chars '''
info = re.sub('[^A-Za-z0-9]+',' ',info)
 
''' Removing numbers from text data '''
info = re.sub(r'[0-9]+',' ',info)

''' removing words less than 3 characters '''
info = re.sub(r'\b\w{1,3}\b','',info)

''' Removed excessive spaces replaced with one space between two words '''
info = " ".join(info.split()).lower()
# ==================================================
# Stop words
with open(r"E:\Sentiment_analysis\stop.txt","r") as sw:
    stop_words = sw.read().split("\n")
stop_words.extend(["like"])
 # positive words # Choose the path for +ve words stored in system
with open(r"E:\Sentiment_analysis\positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")
# negative words Choose path for -ve words stored in system
with open(r"E:\Sentiment_analysis\negative-words.txt", "r") as neg:
    negwords = neg.read().split("\n")
# Creating tokens
info_words = info.split(" ")
# ==================================================
# Checking stopwords, positive and negative words with tokens
info_stpwrd = [w for w in info_words if not w in stop_words]
pos_txt =[w for w in info_words if w in poswords]
neg_txt =[w for w in info_words if w in negwords]

# Orginal text after removing stopwords
original_text =" ".join(info_stpwrd) # token to string
input_f = open("E:\Sentiment_analysis\data.txt", "w")
print(original_text, file=input_f) # used in topic modellig

''' Polarity analysis '''
def sentiment(OT_):
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(OT_)
    del polarity["compound"]
# Plot the polarity
    #y_val = list(polarity.values())
    br = plt.bar(range(len(polarity)), list(polarity.values()), align='center', color='crimson', width = 0.7)
    plt.xticks(range(len(polarity)), list(polarity.keys()))
    plt.grid(axis = "y")
    # access the bar attributes to place the text in the appropriate location
    for br in br:
        yval = br.get_height()
        plt.text(br.get_x()+.25, yval + .01, yval)
    plt.savefig("E:\Sentiment_analysis\polarity.png")
    plt.close()
sentiment_ = sentiment(original_text)

def bigram_(info_):
    bigrams_list = list(nltk.bigrams(info_))
    dictionary2 = [' '.join(tup) for tup in bigrams_list]
    # Using count vectoriser to view the frequency of bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    bag_of_words = vectorizer.fit_transform(dictionary2)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq[:50])
    return words_dict
words_gen = bigram_(info_stpwrd)
words_pos = bigram_(pos_txt)
words_neg = bigram_(neg_txt)

def wordcloud(words_,name_): # info_stopwords

    WC_height = 1000
    WC_width = 1800
    #WC_max_words = 100
    wordCloud = WordCloud(background_color="white",collocations = False, height=WC_height, width=WC_width)
    wordCloud.generate_from_frequencies(words_)
    plt.title('Most frequently occurring bigrams')
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    plt.savefig(name_)
    plt.close()
name = ["E:/Sentiment_analysis/gen.png","E:/Sentiment_analysis/pos.png","E:/Sentiment_analysis/neg.png"]
words = [words_gen,words_pos,words_neg]

for i in range(len(name)):
    wordcloud(words[i-1],name[i-1])
    
import random
text_data = []
text_sum = ""
with open('E:/Sentiment_analysis/data.txt') as f:
    for line in f:
        for token_ in info_words:
            tokens = token_
            if random.random() > .99:
                text_data.append(tokens)
Summary = ' '.join(text_data)            
path_ = r"E:/SA_flask/Upload/data.pdf"

info = input_pdf()

d1 = {'Variables':['doccument'], 'text':[info]}
data = pd.DataFrame(d1)  

def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)

data_nouns_adj = pd.DataFrame(data.text.apply(nouns_adj))
data_nouns_adj

# Re-add the additional stop words since we are recreating the document-term matrix

stop_words = text.ENGLISH_STOP_WORDS
def summary(sw,dna):
    # creating a document-term matrix with nouns and adjectives
    cvn = CountVectorizer(stop_words=stop_words)
    data_cvn = cvn.fit_transform(dna.text)
    data_dtmna = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
    data_dtmna.index = data_nouns_adj.index
    data_dtmna
    # Create the gensim corpus
    corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))
    # Create the vocabulary dictionary
    id2wordna = dict((v, k) for k, v in cvn.vocabulary_.items())
    # Let's start with 2 topics
    ldana = models.LdaModel(corpus=corpusna, num_topics=1, id2word=id2wordna, passes=10)
    topic_1 = ldana.print_topics()
    
    topic_words= topic_1[0][1]
    topic_words = re.sub('[^A-Za-z0-9]+',' ',topic_words)
    topic_words = re.sub(r'[0-9]+',' ',topic_words)
    tw = topic_words.strip()

    tw.replace('     ',', ')
    topic_words = re.sub(r'\b\w{1,3}\b','',topic_words)
    
    d2 = {'Variables':['doccument'], 'text':[topic_words]}
    data2 = pd.DataFrame(d2)  
    data2
    cvn = CountVectorizer(stop_words=sw)
    data2_cvn = cvn.fit_transform(data2.text)
    data2_dtmna = pd.DataFrame(data2_cvn.toarray(), columns=cvn.get_feature_names())
    data2_tdm = data2_dtmna.transpose()
    data2_tdm=data2_tdm.reset_index()
    data2_tdm_ = data2_tdm['index'].values

    return tw
x = summary(stop_words,data_nouns_adj)

#############################################################

    ```




