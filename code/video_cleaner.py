import os, json
import pandas as pd
import numpy as np
from experiment_settings import event, filter_words


# filter data.
def filter_data(r):
    r["text"] = r["video_title"]
    r[event] = false
    for word in filter_words[event]:
        if word in r["text"].lower():
            r[event] = true
            break
    return r


# get tweet-based pseudo-labels.
def get_community_labels(r):
    engagement = []
    for i in range(5):  # 4 communities.
        engagement.append(0)
        for col in {"tweet_count_by_community_{}".format(i),
                    "retweet_count_by_community_{}".format(i),
                    "quote_count_by_community_{}".format(i)}:
            print(col)
            engagement[i] += r[col]
        print("print_engagement\n")
        print(np.argmax(engagement))
    return np.argmax(engagement)


# read data.
pd.set_option('display.max_colwidth', None)
data_path = os.path.join("..", "data", event, "raw.csv")
data = pd.read_csv(data_path)
title = data['video_title']
description = data['video_description']

print( title[35])
print('description')
print( description[35])
'''
# filter data.
data = data.fillna("").sort_values("tweet_count", ascending=false)
data = data.apply(filter_data, axis=1)
data = data[data[event]].drop_duplicates(subset={"text"})
print("data filtered.")


# get tweet-based pseudo-labels.
data["community_label"] = data.apply(get_community_labels, axis=1)
print("community labeled.")

# save data.
data = data[["video_id", "text", "community_label"]]
data_path = os.path.join("..", "data", event, "clean.csv")
data.to_csv(data_path, index=false)
print("data exported:\n", data)
'''
