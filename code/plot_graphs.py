import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import os

def plot_title_distribution():
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    event="voter_fraud"
    labelled_folder=os.path.join("..", "data", event, "labelled")
    datafile = os.path.join(labelled_folder, "distance_pseudo_complete.csv")
    data=pd.read_csv(datafile)
    data["topic"] = ['Promote Voterfraud' if label == 0 else 'US Elections' for label in data["label"]]
    ax = data["topic"].value_counts().plot(kind = 'bar', figsize=(12,6), fontsize=12, color = '#087E8B', rot=0)
    ax.set_title('Counts of Promote Voter Fraud and US Elections Titles')
    ax.set_ylabel('Count', fontsize = 14)



    for i in ax.patches:
        ax.text(i.get_x() + 0.19, i.get_height() + 100, str(round(i.get_height(), 2)), fontsize=13)

    plt.savefig("topic.png")

def plot_results():
    pass



def plot_wordcloud():
    train = "train"
    event = "voter_fraud" 
    label_folder = os.path.join("..", "data", event, "label")
    train_path = os.path.join(label_folder, train+".csv")
    train_data = pd.read_csv(train_path)

    #Reading VoterFraud 
    text = train_data[train_data.labels == 0].text.tolist()
    text = ' '.join(text).lower()
    wordcloud = WordCloud(stopwords = STOPWORDS, collocations = True).generate(text)

    plt.imshow(wordcloud, interpolation = 'bilInear')
    plt.axis('off')
    plt.savefig('voterfraud.png')
    
    text_dictionary = wordcloud.process_text(text)
    word_freq = {k : v for k, v in sorted(text_dictionary.items(), reverse = True, key = lambda item: item[1])}
    rel_freq = wordcloud.words_ 

    #print results
    print(list(word_freq.items())[:5])
    print(list(rel_freq.items())[:5])

    text = train_data[train_data.labels == 1].text.tolist()
    text = ' '.join(text).lower()
    stop_words = ["trump", "election", "voter fraud", "fraud", "biden", "donald"] + list(STOPWORDS)
    wordcloud = WordCloud(stopwords = stop_words, collocations = True , min_word_length = 4).generate(text)

        
    text_dictionary = wordcloud.process_text(text)
    word_freq = {k : v for k, v in sorted(text_dictionary.items(), reverse = True, key = lambda item: item[1])}
    rel_freq = wordcloud.words_ 

    #print results
    print(list(word_freq.items())[:5])
    print(list(rel_freq.items())[:5])
    plt.imshow(wordcloud, interpolation = 'bilInear')
    plt.axis('off')
    plt.savefig('elections.png')



if __name__ == "__main__":
    plot_title_distribution()
