from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import ttest_ind_from_stats

import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
from scipy import stats

import sys
from pprint import pprint
from tabulate import tabulate
import os
import json
import statistics
import itertools
import functools

import my_script_to_load_data

print = functools.partial(print, flush=True)

KEYWORD = 'marriage'
	
def loadEmbedding(publishers, output):
	if publishers == None:
		os.path.join(output, 'clusters', f"./{KEYWORD}_EmbeddingVectors.json")
		return json.load(open(os.path.join(output, 'clusters', f"./{KEYWORD}_EmbeddingVectors.json"), 'r'))
	
	embed_dir = os.path.join(output, 'embedding')
	
	publishers = [x for x in os.listdir(embed_dir)]
	count = len(publishers)
	for publisher in publishers:
		embedding_path = os.path.join(embed_dir, publisher, 'embeddingVector.json')
		print(f"loading embeddings of {publisher}...")
		try:
			embedding = json.load(open(embedding_path))
		except:
			print(f'embedding of {publisher} not found')
			continue
		if KEYWORD in embedding:
			print(f'Keyword {KEYWORD} found')
			_path = os.path.join(output, 'clusters', KEYWORD, publisher)
			Path(_path).mkdir(parents=True, exist_ok=True)
			json.dump(embedding[KEYWORD], open(os.path.join(_path, "embedding.json"), 'w+'))
			print(f'{len(embedding[KEYWORD])} of embeddings found')
		else:
			continue
		count = count - 1
		print(f"{count} publishers' embeddings to go...")
	
	keyEmbeddings = dict()
	pprint("\nloading embeddings of marriage...")
	for publisher in publishers:
		embedding_path = os.path.join(output, publisher, 'clusters', KEYWORD, publisher, "embedding.json")
		try:
			keyEmbeddings[publisher] = json.load(open(embedding_path))
		except:
			continue
	
	json.dump(keyEmbeddings, open(os.path.join(output, 'clusters', KEYWORD, f"{KEYWORD}_EmbeddingVectors.json"), 'w+'))
	print(f'{KEYWORD} embedding saved')
	
	return keyEmbeddings

def t_test_clustering(publishers, output, keyEmbedding):
	def t_test_score(a, b):

		# get M_a
		sims_a = cosine_similarity(a)

		sims_a = np.array(sims_a)

		M_a = sims_a.mean()
		S_a = sims_a.std()
		N_a = sims_a.size
		
		# get M_b
		sims_b = cosine_similarity(b)
		
		sims_b = np.array(sims_b)

		M_b = sims_b.mean()
		S_b = sims_b.std()
		N_b = sims_b.size
		
		# get M_ab
		sims_ab = cosine_similarity(a, b)

		sims_ab = np.array(sims_ab)

		M_ab = sims_ab.mean()
		S_ab = sims_ab.std()
		N_ab = sims_ab.size
		
		# tt_test evaulation
		t_stat_a, p_value_a = ttest_ind_from_stats(M_a, S_a, N_a, M_ab, S_ab, N_ab, equal_var = False)
		t_stat_b, p_value_b = ttest_ind_from_stats(M_b, S_b, N_b, M_ab, S_ab, N_ab, equal_var = False)		

		return p_value_a, p_value_b
	
	# skip p-value calculation if the result json exists
	p_score_results = []
	if not os.path.isfile(os.path.join(output, 'clusters', f"{KEYWORD}_p_score_results.json")):
		validEmbeddings = {k: v for k, v in keyEmbedding.items() if v != None and len(v) > 1}
		
		print(f"\n{len(validEmbeddings)} total publishers found")

		for pair in itertools.permutations(validEmbeddings.keys(), r=2):
			embed_a = keyEmbedding[pair[0]]
			embed_b = keyEmbedding[pair[1]]
			
			p_score_a, p_score_b = t_test_score(embed_a, embed_b)
			print(f'p value of {(pair[0]+" - "+pair[1]):>40} and {pair[0]:>20} is {p_score_a:>30} and p value of {(pair[0]+" - "+pair[1]):>40} and {pair[1]:>20} is {p_score_b:>30}')

			p_score_results.append(((pair[0], p_score_a), (pair[1], p_score_b)))

		json.dump(p_score_results, open(os.path.join(output, 'clusters', f"{KEYWORD}_p_score_results.json"), 'w+'))
	else:
		p_score_results = json.load(open(os.path.join(output, 'clusters', f"{KEYWORD}_p_score_results.json"), 'r'))
	
	clusters = []
	threshold = 1e-20
	while len(clusters) != 7:
		if threshold >= 1.0 or threshold == 0:
			print(f"\nDidn't find a suitable threshold. Clustering failed.")
			break

		if len(clusters) > 7:
			print(f"\n{len(clusters)} clusters complete, lifting threshold...")
			threshold = threshold * 10
		else:
			print(f"\n{len(clusters)} clusters complete, lowering threshold...")
			threshold = threshold / 10
		
		clusters = []
		num = len(clusters)
		clustered = set()
		for result in p_score_results:
			pub_a, pub_b = result
			
			_num = len(clusters)
			if _num > num:
				pprint(clusters)
				num = _num

			# if they are close enough to be clustered together
			if pub_a[1] > threshold and pub_b[1] > threshold:

				isUnclustered = True
				for cluster in clusters:
					if pub_a[0] in cluster or pub_b[0] in cluster:
						isUnclustered = False
						if not pub_a[0] in clustered:
							cluster.add(pub_a[0])
							clustered.add(pub_a[0])
						if not pub_b[0] in clustered:
							cluster.add(pub_b[0])
							clustered.add(pub_b[0])
				if isUnclustered:
					clusters.append({pub_a[0], pub_b[0]})
					clustered.add(pub_a[0])
					clustered.add(pub_b[0])
			else:
				# cluster pub_a
				isUnclustered = True
				for cluster in clusters:
					if pub_a[0] in cluster:
						isUnclustered = False
				if isUnclustered:
					clusters.append({pub_a[0]})
					clustered.add(pub_a[0])
				# cluster pub_b
				isUnclustered = True
				for cluster in clusters:
					if pub_b[0] in cluster:
						isUnclustered = False
				if isUnclustered:
					clusters.append({pub_b[0]})
					clustered.add(pub_b[0])

		pprint(clusters)
		print(f"\nclustered at a threshold of {threshold}")
		json.dump(list(clusters), open(os.path.join(output, 'clusters', f"cluster_by_{KEYWORD}.json"), 'w+'))
	
	return clusters

def loadAnnotations(path):
	return json.load(open(path, 'r'))

def FMI_score(clusters, annotations):	
	_clusters = []
	for cluster in clusters:
		_cluster = []
		for publisher in cluster:
			if publisher in annotations:
				_cluster.append(publisher)
		_clusters.append(_cluster)

	_annotations = dict()
	for publisher, cluster in annotations.items():
		if cluster in _annotations:
			_annotations[cluster].append(publisher)
		else:
			_annotations[cluster] = [publisher]
	
	_annotations = list(_annotations.values())
	
	_fmi_score = fowlkes_mallows_score(_clusters, _annotations)
	
	print(_fmi_score)

def visualize(clusters):
	['A', 'B', 'C', 'D', 'F', 'G']
	G = nx.Graph()
	for cluster in clusters:
		for pair in itertools.permutations(cluster, r=2):
			G.add_edge(pair[0], pair[1], weight=1)

	nx.draw(G)
	plt.show()
	
def analysis(data, output):
	publishers, passages, documents = data
	
	keyEmbedding = loadEmbedding(publishers, output)

	clusters = t_test_clustering(publishers, output, keyEmbedding)

#	annotations = loadAnnotations("/path/to/my/annotations")

#	FMI_score(clusters, annotations)

#	visualize(clusters)

def main():
	path = '/path/to/my/data/'
	output = '/path/to/my/output'
	
	publishers, passages, documents = my_script_to_load_data.processPassages(path)
	merged_publishers, merged_passages, merged_documents = my_script_to_load_data.mergeDocuments(publishers, passages, documents)
	
	merged_data = (merged_publishers, merged_passages, merged_documents)
	
	analysis(merged_data, output)

if __name__ == "__main__": main()