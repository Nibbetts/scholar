import numpy as np
import cPickle as pkl
import math

import matplotlib.pyplot as plt


class Cluster:

    def __init__(self, scholar):
        self.s = scholar
        self.clusters = {}

    def new_cluster(self, cluster_name, word_list):
        center = sum(map(self.s.get_vector,word_list))/len(word_list)
        dispersion = sum([euclidean_distance(center, self.s.get_vector(word))
                          for word in word_list]) / len(word_list)
        self.clusters[cluster_name] = (center, dispersion)

    def get_center(self, cluster_name):
        return self.clusters[cluster_name][0]

    def get_dispersion(self, cluster_name):
        return self.clusters[cluster_name][1]

    def euclidean_distance(self, array1, array2):
        return math.sqrt(sum((array2-array1)**2))

    def measure_center(self, word_list):
        return sum(map(self.s.get_vector,word_list))/len(word_list)

    def measure_dispersion(self, word_list):
        c = self.measure_center(word_list)
        return sum([distance(c, self.s.get_vector(word))
                    for word in word_list]) / len(word_list)

    #def cbrt(self, x):
    #    return (x**(1.0/3.0) if x >= 0 else -(-x)**(1.0/3.0))

    #def scale_bimodal(self, theta):
    #    deg = theta*180/np.pi
    #    return 0.5 + (self.cbrt((deg-90)) / (2*self.cbrt(90)))

    def rescale(self, theta, alpha=15, power=0.5):
        ''' Rescales based on observed distribution of angles between words.
            Accepts theta in radians.'''
        return (0.5 + (math.atan((theta*180/np.pi - 90)/alpha)
                         / (2*math.atan(90/alpha))))**power


    def test_distances(self, n, alpha=15, power=0.5):
        dist = [self.rescale(self.s.angle(
                    self.s.get_vector(self.s.model.vocab[int(x)]),
                    self.s.get_vector(self.s.model.vocab[int(2*x)])),
                    alpha, power)
                for x in (np.random.random(n)*len(self.s.model.vocab)/2.0)]
        plt.hist(dist, 90)
        plt.show()

    def cluster_analogy(self, A, B, C, AC_clustername, B_clustername,
                        num_words=1, exclude=True):
        ''' Follows form: A:B::C:D.
            Assumes that we know which cluster each word comes from.'''
        dist = self.s.get_angle(A, B)
        A_tighter = (self.clusters[AC_clustername][1]
                     <= self.clusters[B_clustername][1]
        C_vec = self.s.get_vector(C)
        dir_vec = self.clusters[AC_clustername][0] - C_vec
        if A_tighter: dir_vec = -dir_vec
        D_vec = self.s.yarax(C_vec, dir_vec, dist)
        D_vec /= np.linalg.norm(D_vec)

        if exclude:
            if self.s.slim == True: # This branch other part of patch:
                results = self.s.wordify(
                    self.s.model.get_closest_words(D_vec, num_words+3))
                trimmed = ([word for word in results[0]
                            if word not in [A, B, C]],
                           [results[1][i] for i in range(len(results[1]))
                            if results[0][i] not in [A, B, C]])
                return (np.array(trimmed[0][:num_words:]),
                        np.array(trimmed[1][:num_words:]))
            else: # This branch is the original return:
                return self.s.wordify(self.s.model.get_closest_words_excluding(
                    D_vec, [self.s.get_vector(A), self.s.get_vector(B), C_vec],
                    num_words))
        else: # The real original return...
            return self.s.wordify(
                self.s.model.get_closest_words(D_vec, num_words))

    def divergence_analogy(self, A, B, C):
        ''' Automatically tries to find clusters around A and B,
            and then does a cluster analogy.'''
        raise NotImplementedError("Function not implemented.")
