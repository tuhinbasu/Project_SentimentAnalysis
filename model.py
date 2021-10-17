# Libraries
import nltk
import re
import PyPDF2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction import text
from nltk import word_tokenize, pos_tag
from gensim import matutils, models
import pandas as pd
import scipy.sparse
import warnings
warnings.filterwarnings("ignore")

''' PDF '''
def input_pdf(p):
    # creating a pdf file object
    pdfFileObj = open(p, 'rb')
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
''' User Input '''
#n = 1 # PDf
#n = 2 # text file
#n = 3 # text
##def choice_input(x):#
    #if x == 1:
        #str_ = input_pdf()
    #elif x == 2:
        #str_ = input_txtf()
    #else:
        #str_ = input_txt()
    #return str_

# ==================================================
def pre_pros(info):
    ''' Removing Punctuations, special chars '''
    info = re.sub('[^A-Za-z0-9]+',' ',info)
 
    ''' Removing numbers from text data '''
    info = re.sub(r'[0-9]+',' ',info)

    ''' removing words less than 3 characters '''
    info = re.sub(r'\b\w{1,3}\b','',info)

    ''' Removed excessive spaces replaced with one space between two words '''
    info = " ".join(info.split()).lower()
    return info
# ==================================================

# ==================================================
# Checking stopwords, positive and negative words with tokens


# Orginal text after removing stopwords


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
    plt.savefig("E:\SA_flask\static\polarity.png")
    



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


def wordcloud(words_,name_): # info_stopwords

    WC_height = 1000
    WC_width = 1800
    #WC_max_words = 100
    wordCloud = WordCloud(background_color="white",collocations = False, height=WC_height, width=WC_width)
    wordCloud.generate_from_frequencies(words_)
    plt.title('Most frequently occurring bigrams')
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(name_)
''' Topic Summary'''
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)

def summary(sw,dna):
    # creating a document-term matrix with nouns and adjectives
    cvn = CountVectorizer(stop_words=sw)
    data_cvn = cvn.fit_transform(dna.text)
    data_dtmna = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
    data_dtmna.index = dna.index
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
    t = tw.replace('     ',', ')
    return t


    
      

#############################################################












