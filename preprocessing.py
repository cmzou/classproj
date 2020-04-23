# Functions to help create word embeddings
# and reading in data and stuff
import pandas as pd
import numpy as np 
from nltk.corpus import stopwords # get stopwords to remove
import re # regular expression
from gensim.models import doc2vec, Word2Vec # for word embeddings
from gensim.utils import simple_preprocess # to tokenize automatically
from sklearn.preprocessing import MultiLabelBinarizer # to convert to a format that can do multi-label classification
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, f1_score, recall_score # for metrics
import random
import collections

# Ensure reproducibility
seed = 561
np.random.seed(seed)
random.seed(seed)

# Reads and cleans and merges datasets
def load_mea():
	raw_text = pd.read_csv('./data/clean_mea_text.csv') # this holds the raw text
	reasons = pd.read_csv('./data/mea_reasons_filtered.csv') # these are our target classifications

	# There is a bit of data mismatch, so filter both dfs for text that appears in both
	bothdocs = set(raw_text.docket_num.values).intersection(reasons.docket_num.values)
	raw_text = raw_text[raw_text.docket_num.isin(bothdocs)]
	reasons = reasons[reasons.docket_num.isin(bothdocs)]

	# Also need to convert datatypes to prevent type mismatch
	raw_text['text'] = raw_text['text'].astype(str)

	# Remove irrelevant/unhelpful labels
	toremove = ['Unknown', 
			   'Other', 
			   'Did not appear for the Hearing', 
			   'Other failure to disclose', 
			   'Unknown failure to disclose', 
			   'Allowed falsification and misrepresentation of loans',
			   'Circumvented the requirements that a branch manager of a licensed mortgage broker have at least three years experience',
			   "Did not verify or make a reasonable effort to verify the borrower's information",
			   'Employed simultaneously by more than one affiliated mortgage banker or mortgage broker',
			   'Engaged in fraud',
			   'Failure to disclose charges',
			   'Violated NC Securities Act',
			   'Withdrew appeal',
			   'Unsatisfactory credit']
	reasons = reasons[~reasons.reason.isin(toremove)]

	# Since we want to do multi-label classification, binarize outputs
	# First, need to aggregate reason by docket_num
	reasonsls = reasons.groupby('docket_num')['reason'].apply(set).reset_index(name='reason')

	mlb = MultiLabelBinarizer()
	classesbin = mlb.fit_transform(reasonsls.reason.values)
	classesbin = pd.DataFrame(classesbin)
	classesbin.columns = mlb.classes_
	print('{:d} unique classes found'.format(len(mlb.classes_))) # print dim of y

	reasonsls = pd.concat([reasonsls, classesbin], axis=1)

	# Let's combine the input and output datasets for easier handling
	merged = raw_text.merge(reasonsls)

	# Remove extra space
	merged['text'] = merged['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

	return merged

def tokenize(df, col, tokens_only=False):
	"""
	Given a DataFrame and a column, tokenizes the words in that column

	Parameters
	----------
	df: DataFrame
		dataframe with column to be tokenized
	col: str
		column name of text to be tokenized
	tokens_only: bool
		to train the doc2vec model, we’ll need to 
		associate a tag/number with each document of the training corpus. 
		tokens_only=True means don't associate anything
	Returns
	----------
	list
		tokenized words
	"""
	tokens = df[col].apply(lambda x: simple_preprocess(x, deacc=True, max_len=20)) # max_len=20 just in case there are important words 15 chars long)
	if tokens_only:
		return tokens
	else:
		# For training data, add tags -- notice it is just an index number
		return [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]

# Params:
#    w2v - dictionary of words to vectors
#    text - list of tokenized words to convert to vector
# Returns a vector resulting from the average of all present words in text
def average_vectors(w2v, text):
	num_count = collections.Counter(text) # words and their counts
	return np.mean([w2v[word]*count for word, count in num_count.items()], axis=0)

def create_tokens(data, col, vec_dim, type):
	"""
	Given a DataFrame and a column, creates document embeddings

	Parameters
	----------
	data: DataFrame
		dataframe to 
	col: str
		column name
	vec_dim: int
		dimension of the output document embedding
	type: str
		the type of document embedding to create
		accepted values: "doc" for doc2vec
			"word" for averaged word2vec
	Returns
	----------
	DataFrame
		document embedding
	"""

	corpus = tokenize(data, col)
	if type == 'doc':
		# Use doc2vec
		docmodel = doc2vec.Doc2Vec(vector_size=vec_dim, min_count=1, epochs=40) # min_count=1 because we're not sure if relevant words occur multiple times
		docmodel.build_vocab(corpus)
		# Train
		docmodel.train(corpus, total_examples=docmodel.corpus_count, epochs=docmodel.epochs)
		
		docvecs = [docmodel.infer_vector(corpus[doc_id].words) for doc_id in range(len(corpus))]
		return docvecs
	elif type == 'word':
		textls = tokenize(data, col, tokens_only=True)
		wordmodel = Word2Vec(textls, min_count=1)
		w2v = dict(zip(wordmodel.wv.index2word, wordmodel.wv.syn0))
		docvecs_avg = [average_vectors(w2v, corpus[i].words) for i in range(len(corpus))]
		return docvecs_avg
	else:
		# Invalid input
		print('type must be either "doc" or "word"')
		return

# Function that returns the hamming score
# correctly predicted / number of labels
# Effectively acts as an accuracy metric in multilabel classification
# Could eventually try to make it completely compatible with sklearn's metrics
def hamming_score(y_true, y_pred):
	return (y_pred == y_true).mean()

# Function that given predicted probabilities 
# and the true labels,
# calculates a bunch of metrics and returns them
def calc_metrics(y_true, y_prob):
	y_pred = np.copy(y_prob) # classes
	y_pred[y_pred>=0.5] = 1
	y_pred[y_pred<0.5] = 0

	# Metrics
	# average='micro' because we care a little more about global statistics
	# If adding/removing metrics, change num_metrics
	ham_score = hamming_score(y_true, y_pred) # accuracy
	emr = accuracy_score(y_true, y_pred) # exact match ratio
	f1 = f1_score(y_true, y_pred, average='micro') # f1 -- care about false positives and false negatives
	prec = precision_score(y_true, y_pred, average='micro') # tp / (tp + fp) # care about false positives slightly more let's look at precision instead of both
	rec = recall_score(y_true, y_pred, average='micro') # just for interpretation
	auc = roc_auc_score(y_true, y_prob, average='micro') # for ease, pretend the threshold should be the same for all classes
	metrics = [ham_score, emr, f1, prec, rec, auc]
	
	return metrics