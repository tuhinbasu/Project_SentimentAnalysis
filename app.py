from flask import Flask,render_template,request
import glob
import os
import pandas as pd
from sklearn.feature_extraction import text
from model import input_pdf,pre_pros,sentiment,bigram_,wordcloud, summary,nouns_adj
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
 
@app.route("/output", methods = ["GET","POST"])
def output():
        if request.method == "POST":
            str = request.files["file"]
            if os.path.exists("E:/SA_flask/Upload/data.pdf"):
                os.remove("E:/SA_flask/Upload/data.pdf")
            str.save(os.path.join('Upload', str.filename))
            for f in glob.glob('E:/SA_flask/Upload/*'):
                name = os.path.split(f)[-1]
            os.rename(r'E:/SA_flask/Upload/'+ name, r'E:/SA_flask/Upload/data.pdf')
            path_ = r"E:/SA_flask/Upload/data.pdf"
            info = input_pdf(path_)
            info =  pre_pros(info)
            # Stop words
            with open(r"E:\SA_flask\static\stop.txt","r") as sw:
                stop_words = sw.read().split("\n")
            stop_words.extend(["like"])
            # positive words # Choose the path for +ve words stored in system
            with open(r"E:\SA_flask\static\positive-words.txt","r") as pos:
                poswords = pos.read().split("\n")
            # negative words Choose path for -ve words stored in system
            with open(r"E:\SA_flask\static\negative-words.txt", "r") as neg:
                negwords = neg.read().split("\n")
            info_words = info.split(" ")
            # Checking stopwords, positive and negative words with tokens
            info_stpwrd = [w for w in info_words if not w in stop_words]
            pos_txt =[w for w in info_words if w in poswords]
            neg_txt =[w for w in info_words if w in negwords]

            # Orginal text after removing stopwords
            original_text =" ".join(info_stpwrd) # token to string
            input_f = open("E:\SA_flask\static\data.txt", "w")
            print(original_text, file=input_f) # used in topic modelling

            sentiment(original_text)
            words_gen = bigram_(info_stpwrd)
            words_pos = bigram_(pos_txt)
            words_neg = bigram_(neg_txt)

            name = ["E:/SA_flask/static/gen.png","E:/SA_flask/static/pos.png","E:/SA_flask/static/neg.png"]
            words = [words_gen,words_pos,words_neg]
            for i in range(len(name)):
                wordcloud(words[i-1],name[i-1])
            d1 = {'Variables':['doccument'], 'text':[info]}
            data = pd.DataFrame(d1)
            data_nouns_adj = pd.DataFrame(data.text.apply(nouns_adj))
            stop_words = text.ENGLISH_STOP_WORDS
            sum = summary(stop_words,data_nouns_adj)
        return render_template("output.html", msg = sum, positvie_words = words_pos, negetive_words = words_neg)   

if __name__ == "__main__":
    app.run(debug=True)
