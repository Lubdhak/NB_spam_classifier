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


predict_this = "But wondering is thing its and devil disaster flown thy evilprophet not and tapping myself back the muttered. More heaven nevermore but nevermore god bore. Some of came some shore upon is. Leave of had perched my then sorrow of more doubtless no bird. Divining fancy and while within door its no chamber wheeled rapping seeming explore was my not. Into the your methought tinkled. Longer ominous whispered wondering my of as out the only his. Shadows much stillness stock upstarting god chamber distant hath my thou fowl thy i me and or reclining. The pallid thrilled crest stayed the whispered to this. And door unbrokenquit evilprophet stopped. Had still implore my the stock raven bird terrors the stock let radiant truly i i perched this decorum. Name rapping i oer then in devil oer each from as adore have thy stood door of still. Nevermore i when before with nodded still the but i implore disaster whispered ancient craven rustling tapping crest that. And thy then in my hath swung of so disaster. No quaff lining plume the till hath sought. This whose many and me name stillness sure beating land fact the spoke my came nepenthe token nevermore whispered. Before came this lie a stepped raven on he.</p><p>Window flung parting nepenthe sorrowsorrow countenance evilprophet of have lonely explore and. It lenore and hope farther my pallid a above. This nevermore the grim chamber chamber no cushions nights what i raven wretch i my. Saintly the at lamplight. Streaming nevermore shaven on theeby pallas long while lady hope lining me seat peering merely the on. Of clasp on though remember soul feather soul stepped. Grim this prophet said i or. The relevancy bust velvet from ungainly fiery fluttered seeming word upon curtain by to. But seraphim upstarting both its doubtless sitting whispered dreaming ever my quaint burden by thing entreating some tossed. Open these and quaff raven thou before the of of though smiling. Sought more nights lenore whose and kind mefilled name opened flitting decorum morrow name borrow all faintly my door. Name this entrance window demons wrought from raven.</p><p>That in if rare thy the before core meaninglittle the. Thou i friends. Not within him visiter. The soul said i morrow burning beguiling entrance fantastic but lonely i raven nearly melancholy the what this. Nodded that placid of the window i of his napping my. Marvelled leave of one nothing radiant memories forgotten this i on reply if bore heard out. From and whether thing ever while ghost the lady bird nevermore bird above thee pallas placid implore. My bird placid. Shutter thereis tell if made shall raven. A rare though. Sat name evilprophet. And that velvet now of into the more burning the came in some the something stepped stayed gaunt obeisance. Sad wheeled nevermore door this let tis eyes or open so little to a morrow rapping the of tis. Of muttered i his still nevermore there on horror thrilled now many my shall by"
print(clf.predict(count_vect.transform([predict_this])))