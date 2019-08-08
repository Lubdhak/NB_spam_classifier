import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from textblob import Word

stop = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
        "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
        "should", "now"]
# python3 -m textblob.download_corpora

print("Building Model..")
train = pd.read_csv('data/training.csv')
test = pd.read_csv('data/test.csv')

# Lower Case
train['email'] = train['email'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# remove HTML Tags
raw_html = "<hi>huhu</hello>"
html_tags = re.compile('<.*?>')
train['email'] = train['email'].apply(lambda x: re.sub(html_tags, '', x))

# remove stop-words
train['email'] = train['email'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# remove punctutation
train['email'] = train['email'].str.replace('[^\w\s]', '')

# lemmatize
train['email'] = train['email'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


cleaned_dataFrame = train

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train['email'])

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, train['type'])

print("Building Model Completed..\n\n")

inp = True
while inp:
    predict_this = input("Enter text you want predict :\n")
    print("Enterd Text is a ",clf.predict(count_vect.transform([predict_this])))
    x = input("\n\n\nEnter any key to predict again or 'q' to Quit\n")
    if x in ['q','Q']:
        inp = False

