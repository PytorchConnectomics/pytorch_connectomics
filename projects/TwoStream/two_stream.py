# The two-stream query suggestion algorithm for incorporating unsupervised image features in 
# active learning as describe in "Two-Stream Active Query Suggestion for Active Learning in 
# Connectomics (ECCV 2020, https://zudi-lin.github.io/projects/#two_stream_active)".

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
from sklearn.cluster import KMeans

class TwoStream(object):
    """Two-stream query suggestion.
    """
    name = 'two_stream'

    def __init__(self, X, embedding):
        self.X = X
        self.embedding = embedding
        
    def select_batch_(self, already_selected, N1, N2):
        budget = N1 * N2
        iteration = len(already_selected) // budget
        print('Budget %d, Already Selected %d and Iteration %d' % (budget, len(already_selected), iteration))
        feature_copy = self.X.copy()

        indices = np.arange(self.X.shape[0])
        indices_copy = indices.copy()
        indices_copy = np.delete(indices_copy, already_selected, axis=0)
        
        embedding_copy = self.embedding.copy()
        embedding_copy = np.delete(embedding_copy, already_selected, axis=0)
        embedding_selected = self.embedding[already_selected]

        print('Start Clustering ==>')
        kmeans_instance = KMeans(n_clusters=N1, random_state=0).fit(feature_copy)
        level1_label = kmeans_instance.labels_
        level1_label_copy = level1_label.copy()
        level1_label_copy = np.delete(level1_label_copy, already_selected, axis=0)
        level1_label_selected = level1_label[already_selected]
        
        SELECTED = []
        LEVEL1_CLUSTERS = []
        NUM_CLUSTERS2 = []
        NUM_IMAGES = []
        LEVEL1_LABEL = []

        # Calculate the number sub-clusters in hierarchical clustering.
        for i in range(N1):
            if (level1_label_copy==i).astype(int).sum() > 0: # non-empty after removing already selected
                LEVEL1_LABEL.append(i)
                LEVEL1_CLUSTERS.append(embedding_copy[np.where(level1_label_copy==i)])
                num_features = len(embedding_copy[np.where(level1_label_copy==i)])
                num_selected = (level1_label_selected==i).astype(int).sum()

                num_clusters2 = N2
                if num_features < num_clusters2:
                    num_clusters2 = num_features
                NUM_CLUSTERS2.append(num_clusters2)
                NUM_IMAGES.append(num_features)

        # Make sure the annotation budget is used up.
        NUM_CLUSTERS2 = np.array(NUM_CLUSTERS2)
        extra = N1 * N2 - NUM_CLUSTERS2.sum()
        num_clusters2 = N2
        while extra > 0:
            add_num = (np.array(NUM_IMAGES) > num_clusters2).sum()
            add_num = min(add_num, extra)
            add_idx = np.array(NUM_IMAGES).argsort()[::-1][:add_num]
            NUM_CLUSTERS2[add_idx] += 1
            num_clusters2 += 1
            extra -= add_num

        assert NUM_CLUSTERS2.sum() == N1 * N2
        print('Number of candidate samples: ', np.array(NUM_IMAGES).sum())

        # Run second-round of clustering with unsupervised features.
        for i in range(len(LEVEL1_CLUSTERS)):
            print('progress %03d/%d..' % (i+1, len(LEVEL1_CLUSTERS)), end=' ')
            print('number of samples: %04d' % (len(LEVEL1_CLUSTERS[i])), end=' ')
            print('selected samples %04d' % (level1_label_selected==LEVEL1_LABEL[i]).sum(), end=' ')
            num_selected = (level1_label_selected==LEVEL1_LABEL[i]).sum()

            # determine number of clusters
            num_clusters2 = NUM_CLUSTERS2[i]
            print('number of clusters: ', num_clusters2)
            level1_index = indices_copy[np.where(level1_label_copy==LEVEL1_LABEL[i])]
            subcluster_embedding = np.array(LEVEL1_CLUSTERS[i])

            # filter out close samples in the unsupervised feature space
            if num_selected > iteration * N2:
                tail_portion = 0.2
                subcluster_selected = embedding_selected[np.where(level1_label_selected==LEVEL1_LABEL[i])]
                distances = subcluster_embedding[:,None,:] - subcluster_selected[None,:,:]
                distances = (distances**2).sum(2).min(1)
                end_index = int(len(LEVEL1_CLUSTERS[i]) * tail_portion)
                end_index = max(end_index, num_clusters2)
                condidate = distances.argsort()[-end_index:]
                subcluster_embedding = subcluster_embedding[condidate]
                level1_index = level1_index[condidate]

            # second-round of clustering
            kmeans_instance = KMeans(n_clusters=num_clusters2, random_state=0, n_init=10, max_iter=200).fit(subcluster_embedding)
            level2_label = kmeans_instance.labels_

            # select the sample closest to the cluster center as suggested query
            LEVEL2_CLUSTERS = []
            for j in range(num_clusters2):
                LEVEL2_CLUSTERS.append(LEVEL1_CLUSTERS[i][np.where(level2_label==j)])
                dist = np.array(LEVEL2_CLUSTERS[j]) - kmeans_instance.cluster_centers_[j][None,:]
                dist = (dist**2).sum(1)

                idx = np.argmin(dist)
                level2_index = level1_index[np.where(level2_label==j)] 
                SELECTED.append(level2_index[idx])

        return SELECTED

def run_softmax(x):
    return np.exp(x)/np.sum(np.exp(x), 1, keepdims=True)

def main(feature='feature_vector', already_selected='init_1000', 
         embedding_file='embedding', softmax=False, N1=200, N2=5):

    data = pickle.load(open(feature, 'rb'))
    if softmax: 
        data = run_softmax(data)

    selected_index = pickle.load(open(already_selected, 'rb'))
    embedding = pickle.load(open(embedding_file, 'rb'))

    solver = TwoStream(X=data, embedding=embedding)
    new_batch = solver.select_batch_(already_selected=selected_index, N1=N1, N2=N2)
    assert True not in [x in new_batch for x in selected_index]
    print('Query suggestion finished. Save data...')

    new_pool = list(selected_index) + list(new_batch)
    print('Number of training images: ', len(new_pool))
    dir_path = os.path.dirname(os.path.realpath(already_selected))
    outfile = open(dir_path+'/new_pool_'+solver.name+'_'+str(len(new_pool)), 'wb')
    pickle.dump(new_pool, outfile)
    outfile.close()

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='K-Center Greedy for Active Learning')
    parser.add_argument('--feature', type=str, help='feature vector address')
    parser.add_argument('--indices', type=str, help='already selected indices')
    parser.add_argument('--embedding', type=str, help='embedding of images')
    parser.add_argument('--softmax', action='store_true', help='softmax')
    parser.add_argument('--N1', default=100, type=int, help='number of level 1 clusters')
    parser.add_argument('--N2', default=10, type=int, help='number of level 2 clusters')
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = get_args()
    main(feature=args.feature,
         already_selected=args.indices,
         embedding_file=args.embedding,
         softmax=args.softmax,
         N1=args.N1, 
         N2=args.N2)
