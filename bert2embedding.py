from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from flair.data import Dictionary, Sentence
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

from pathlib import Path
import numpy
import os
import subprocess
import json
import sys
import shutil
import unicodedata

import passages2topics

KEYWORD = 'my_keyword'

def defineCorpus(publishers, passages, documents, output, mode='w'):
	ignore_list = set()
	for publisher, document in zip(publishers, documents):
		corpusDir = os.path.join(output, 'corpus', publisher)
		if not os.path.exists(corpusDir):
			os.makedirs(corpusDir)
		document = numpy.array(sent_tokenize(document.lower()))
		
		if len(document) < 10:
			print(f"\n{publisher} is ignored")
			ignore_list.add(publisher)
			continue
		
		# train : 80%. valid : 10%. test : 10%.
		x_train, x_test_n_valid = train_test_split(document, test_size=0.2)
		x_valid, x_test = train_test_split(x_test_n_valid, test_size=0.5)
		
		merged_x_document = {'train.txt':'[SEP]'.join(x_train),
							'valid.txt':'[SEP]'.join(x_valid),
							'test.txt':'[SEP]'.join(x_test)}
		
		# write files
		for outFile, doc in merged_x_document.items():
			with open(os.path.join(corpusDir,outFile), mode) as items:
				items.write(doc)
		
		with open(os.path.join(corpusDir,'total.json'), mode) as total_file:
			json.dump(document.tolist(), total_file)
	return ignore_list

def fineTuneEmbedding(publishers, output):
	for publisher in publishers:
		corpusDir = os.path.join(output, 'corpus', publisher)
		_train_path = os.path.join(corpusDir, 'train.txt')
		_test_path = os.path.join(corpusDir, 'test.txt')
		_output_path = os.path.join(output, 'finetune', publisher)
		subprocess.run(["python3", "run_language_modeling.py",
						f"--output_dir={_output_path}",
						"--model_type=bert",
						"--model_name_or_path=bert-large-uncased",
						"--do_train",
						f"--train_data_file={_train_path}",
						"--do_eval",
						f"--eval_data_file={_test_path}",
						"--overwrite_output_dir",
						"--mlm"])
	
def generateFineTuneAnalysis(publishers, output):
	publishersPerplexity = dict()
	for publisher in publishers:
		_output_path = os.path.join(output, 'finetune', publisher)
		eval_file = os.path.join(_output_path, 'eval_results_lm.txt')
		publishersPerplexity[publisher] = open(eval_file, 'r').readline().strip()
	
	_eval_output_path = os.path.join(output, 'total_eval.txt')
	with open(_eval_output_path, 'w') as fp:
		json.dump(publishersPerplexity, fp)
	
def combineModelFlair(publishers, documents, ignores, output):
	embeddingVectors = dict()

	# init Flair embeddings
	flair_forward_embedding = FlairEmbeddings('multi-forward')
	flair_backward_embedding = FlairEmbeddings('multi-backward')

	for publisher, document in zip(publishers, documents):

		print(f"\nEmbedding sentences for {publisher}...")

		embeddingVectors[publisher] = dict()

		_bert_path = os.path.join(output, 'finetune', publisher)
		
		# check if the finetuning is done correctly
		try:
			bert_embedding = TransformerWordEmbeddings(_bert_path)
			
			# delete finetuned model for extra space
			# shutil.rmtree(_bert_path)
		except:
			ignores.add(publisher)
			bert_embedding = TransformerWordEmbeddings('bert-large-uncased')

		stacked_embeddings = StackedEmbeddings(embeddings=[flair_forward_embedding, flair_backward_embedding, bert_embedding])

		corpusDir = os.path.join(output, 'corpus', publisher)

		for sep in numpy.array(sent_tokenize(document.lower())):
			sentence = Sentence(sep)
			try:
				stacked_embeddings.embed(sentence)
			except:
				continue

			# now check out the embedded tokens.
			for token in sentence:
				if token in embeddingVectors[publisher]:
					embeddingVectors[publisher][token.text].append(token.embedding.tolist())
				else:
					embeddingVectors[publisher][token.text] = [token.embedding.tolist()]

		# save embedding
		Path(os.path.join(output, 'embedding', publisher)).mkdir(parents=True, exist_ok=True)

		with open(os.path.join(output, 'embedding', publisher, 'embeddingVector.json'), 'w+') as fp:
			json.dump(embeddingVectors[publisher], fp)
		
		print(f"{publisher} embedding vector is saved")
	
	with open(os.path.join(output, 'embedding', 'embeddingVectors.json'), 'w+') as fp:
		json.dump(embeddingVectors, fp)

	return ignores		

# Only call saveEmbedding when enough memory is available
def saveEmbedding(embeddingVectors, output):
	print(f"Loading embedding vectors for `{KEYWORD}`...")
	wordEmbeddingVectors = dict()
	for publisher, vectors in embeddingVectors.items():
		if KEYWORD in embeddingVectors:
			wordEmbeddingVectors[publisher] = embeddingVectors[KEYWORD]
	
	with open(os.path.join(output, 'embedding', f'{KEYWORD}_EmbeddingVectors.json'), 'w+') as fp:
		json.dump(wordEmbeddingVectors, fp)
	print(f"Embedding vectors saved")

# Call loadEmbedding when memory is limited
def loadEmbedding(publishers, output):
	keyEmbeddings = dict()
	
	for publisher in publishers:
		embedding_path = os.path.join(output, 'embedding', publisher, 'embeddingVector.json')
		embedding = json.load(open(embedding_path))
		print(f'loading embedding of {publisher}...')
		if KEYWORD in embedding:
			print(f'Keyword {KEYWORD} found')
			keyEmbeddings[publisher] = embedding[KEYWORD]
		else:
			keyEmbeddings[publisher] = None
	
	with open(os.path.join(output, 'embedding', f'{token}_EmbeddingVectors.json'), 'w+') as fp:
			json.dump(keyEmbeddings, fp)


def trainModel(data, output):
	
	# unpack data
	publishers, passages, documents = data

	# define corpus directory structure
	print('\ndefining corpus...')
	ignores = defineCorpus(publishers, passages, documents, output)
	
	# fine-tuning bert model based on the corpus
	fineTuneEmbedding(publishers, output)

	# generate evaluation for fine-tuning results
#	generateFineTuneAnalysis(publishers, output)

	# train embedding to produce embedding vectors
	ignores = combineModelFlair(publishers, documents, ignores, output)
		
	# save embeddings for certain word
#	saveEmbedding(embeddingVectors, output)
	
	# load embeddings for keyword
	loadEmbedding(publishers, output)

	return ignores

def main():
	path = '/my/path/to/data/'
	output = '/my/path/to/output/'
	components = 30

	nltk.download('punkt')

	# load text and publisher data
	publishers, passages, documents = passages2topics.processPassages(path)
	merged_publishers, merged_passages, merged_documents = passages2topics.mergeDocuments(publishers, passages, documents)
	
	merged_data = (merged_publishers, merged_passages, merged_documents)
	
	# train the model with bert
	ignores = trainModel(merged_data, output)

	# print ignored publishers
	for ignore in ignores:
		print(ignore)
							
if __name__ == "__main__": main()