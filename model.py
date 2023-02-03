import matplotlib.pyplot as plt
import numpy as np
import time
import logging
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from counterfactuals import PatternsCounterfactualExplainer
from scipy.special import softmax
from sklearn.metrics import log_loss


import pickle
from itertools import product

from sksurv.compare import compare_survival


class Loggable:
    @staticmethod
    def _configure_logger(results_folder, prefix_name, timestamp):
        name = f'Exp_{prefix_name}'
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(message)s')
        file_handler = logging.FileHandler(f'{results_folder}/{prefix_name}_Experiment_{timestamp}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


class XSurv(Loggable):
    def __init__(self, prefix_name, results_folder='Results', max_k=30, patience=3, z_explained_variance_ratio_threshold=0.99, curves_diff_significance_level=0.05, verbose=True):
        self.results_folder = results_folder + '/' + prefix_name
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        self.prefix_name = prefix_name
        self.pca_mdl = None  # initilized in fit
        self.clustering_mdl = None  # initilized in fit
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.logger = self._configure_logger(results_folder=self.results_folder, prefix_name=self.prefix_name, timestamp=self.timestamp)
        self.max_k = max_k
        self.patience = patience
        self.z_explained_variance_ratio_threshold=z_explained_variance_ratio_threshold
        self.curves_diff_significance_level = curves_diff_significance_level
        self.verbose = verbose

    def fit(self, xte_data, survival_curves, event_times, survival_mdl=None, pretrained_clustering_model=None, k=None):

        # Find Survival Patterns: z=h(s), c=g(z)

        # data prepatation x, s
        (self.x_train, self.y_train, self.e_train,
         self.x_val, self.y_val, self.e_val,
         self.x_test, self.y_test, self.e_test) = xte_data

        # special for sksurv
        dt = np.dtype('bool,float')
        self.ey_train_surv = np.array([(bool(e), y) for e, y in zip(self.e_train, self.y_train)], dtype=dt)
        self.ey_val_surv = np.array([(bool(e), y) for e, y in zip(self.e_val, self.y_val)], dtype=dt)
        self.ey_test_surv = np.array([(bool(e), y) for e, y in zip(self.e_test, self.y_test)], dtype=dt)
        self.survival_curves_train, self.survival_curves_val, self.survival_curves_test = survival_curves

        self.event_times = event_times
        self.survival_mdl = survival_mdl

        # transformation to lower dimentions (s -> z), z=h(s)
        self.pca_mdl, self.z_train, self.z_val, self.z_test = self._get_z()

        # clustering (z -> c), c=g(z)
        if k is None:
            self.optimal_k = self._find_optimal_clusters_number()
        else:
            self.optimal_k = k

        self.labels_train, self.labels_val, self.labels_test = self._cluster_in_z(pretrained_model=pretrained_clustering_model)

        if self.verbose:
            self._plot_z(self.z_train, self.labels_train, label='train')
            self._plot_z(self.z_val, self.labels_val, label='val')
            self._plot_z(self.z_test, self.labels_test, label='test')
            self._plot_curves(curves=self.survival_curves_train, clusters=self.labels_train, event_times=self.event_times, fig_name='Concepts')
            self._plot_context(curves=self.survival_curves_train, clusters=self.labels_train, event_times=self.event_times)

    def predict_survival_function(self, x):
        s = self.survival_mdl.predict_survival_function(x, return_array=True)
        return s

    def predict(self, x):
        """predicts the patterns to which x belong.
           this will be used to check pattern of the counterfactual example

        Arguments:
            x - features vector

        Returns:
            the pattern to which x belong , int
        """

        s = self.predict_survival_function(x)
        z = self.pca_mdl.transform(s)
        c = self.clustering_mdl.predict(z)
        return c

    def get_centers_distences(self, x):
        s = self.predict_survival_function(x)
        z = self.pca_mdl.transform(s)
        dists = []
        for c in self.clustering_mdl.cluster_centers_:
            dist_to_c = np.sqrt(((c - z) ** 2).sum(axis=1))
            dists.append(dist_to_c[:,np.newaxis])

        dist_to_cs = np.concatenate(dists, axis=1)
        return dist_to_cs

    def get_cross_entropy(self, x, target_pattern):
        """Computes the cross-entropy between the predicted pattern and the target pattern

        Arguments:
            x - features vector
            target_pattern - the target pattern

        Returns:
            the value of the cross-entropy , float
        """
        s = self.predict_survival_function(x)
        z = self.pca_mdl.transform(s)
        distances = np.sqrt(((self.clustering_mdl.cluster_centers_ - z) ** 2).sum(axis=1))
        logits = softmax(-distances)
        labels = list(range(len(self.clustering_mdl.cluster_centers_)))
        return log_loss(target_pattern, [logits], labels=labels)

    def _get_z(self):
        """
        finds the best number of components for the pca model and predicts the pca values of the survival curves
        :return:
            pca - the model model
            z_train - the pca values of train survival curves
            z_val - the pca values of val survival curves
            z_test - the pca values of test survival curves
        """
        pca = PCA(n_components=1)
        print("shape:",self.survival_curves_train.shape[1])
        for i in range(1, self.survival_curves_train.shape[1]+1):
            pca = PCA(n_components=i)
            pca.fit(self.survival_curves_train)
            if (pca.explained_variance_ratio_.sum() >= self.z_explained_variance_ratio_threshold) or (pca.explained_variance_ratio_.sum() >= 0.999):
                self.logger.info(f'Z Space Dimensions: {i}')
                break

        z_train = pca.transform(self.survival_curves_train)
        z_val = pca.transform(self.survival_curves_val)
        z_test = pca.transform(self.survival_curves_test)
        if self.verbose:
            if z_train.shape[1] >= 2:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].scatter(z_train[:, 0], z_train[:, 1], c=self._get_colors(self.e_train.astype(int)), alpha=0.1)
                ax[1].scatter(z_val[:, 0], z_val[:, 1], c=self._get_colors(self.e_val.astype(int)), alpha=0.1)
                ax[2].scatter(z_test[:, 0], z_test[:, 1], c=self._get_colors(self.e_test.astype(int)), alpha=0.1)
            else:
                plt.figure()
                plt.hist(z_train[:, 0], bins=100)
            plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_z.pdf', format='pdf', bbox_inches='tight')
            plt.show()
        return pca, z_train, z_val, z_test

    def _cluster_in_z(self, pretrained_model=None):
        """
        clusters the survival curves in z space
        :param pretrained_model: the clustering model to be used if provided
        :return:
        labels_train - the clustering labels of the train survival curves
        labels_val - the clustering labels of the val survival curves
        labels_test - the clustering labels of the test survival curves
        """
        if pretrained_model is None:
            self.clustering_mdl = self._base_clustering_model(number_clusters=self.optimal_k).fit(self.z_train)
            pickle.dump(self.clustering_mdl,
                        open(f'{self.results_folder}/{self.prefix_name}_clustering_model_{self.optimal_k}_{self.timestamp}.mdl', 'wb'))
        else:
            self.clustering_mdl = pickle.load(open(pretrained_model, 'rb'))

        labels_train = self.clustering_mdl.predict(self.z_train)
        labels_val = self.clustering_mdl.predict(self.z_val)
        labels_test = self.clustering_mdl.predict(self.z_test)

        return labels_train, labels_val, labels_test

    def _classify(self):
        pass

    def _find_optimal_clusters_number(self):
        diffs = []
        diffs_stds = []
        diffs_means = []
        ntrials = self.patience + 1
        for k in range(2, self.max_k):
            if ntrials == 0:
                break
            diffs_temp = []
            for i in range(10):
                d = self._count_logrank_diffs(self.z_train, ye=self.ey_train_surv, e=self.e_train, number_clusters=k, random_state=i)
                diffs_temp.append(d)
            diffs_means.append(np.mean(diffs_temp))
            diffs.append(np.mean(diffs_temp))
            diffs_stds.append(np.std(diffs_temp))
            if diffs[-1] == 1:
                ntrials = self.patience
            else:
                ntrials -= 1


        diffs = list(diffs)
        max_k_with_max_diffs = len(diffs) - diffs[::-1].index(max(diffs)) + 1

        diffs = np.array(diffs)
        diffs_stds = np.array(diffs_stds)
        ks = list(range(2, len(diffs)+2))

        if self.verbose:
            plt.figure(figsize=(4, 4))
            plt.plot(ks, diffs)
            highs = diffs + diffs_stds
            highs = [1 if x>1 else x for x in highs]
            lows = diffs - diffs_stds
            plt.fill_between(ks, lows, highs, alpha=0.2)
            plt.vlines(max_k_with_max_diffs, ymin=min(diffs), ymax=1, color='C1', linestyle='--', label='Last Max')

            plt.xticks(ks, rotation=90)
            plt.ylabel('% of Different Patterns')
            plt.xlabel('Number of Clusters')
            plt.legend()
            plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_clusters_number.pdf', format='pdf', bbox_inches='tight')
            plt.show()

        return max_k_with_max_diffs

    def _count_logrank_diffs(self, data, ye, e, number_clusters, random_state=None):
        mdl = self._base_clustering_model(number_clusters=number_clusters, random_state=random_state).fit(data)
        labels = mdl.predict(data)
        total_diffs = 0
        total_comps = 0
        for i, j in product(range(number_clusters), range(number_clusters)):
            if i < j:
                is_diff = self._is_different(i=i, j=j, ye=ye, e=e, group=labels, sig_threshold=self.curves_diff_significance_level)
                total_diffs += int(is_diff)
                total_comps += 1

        return total_diffs / total_comps

    def _def_clustering_model(self, number_clusters):
        pass

    @staticmethod
    def _is_different(i, j, ye, e, group, sig_threshold=0.05):
        f = (group == i) | (group == j)
        ye_sub = ye[f]
        g_sub = group[f]
        e_sub = e[f]

        if (e_sub == 1).any():  # there should be at least one event
            # logrank_test
            try:
                _, p = compare_survival(y=ye_sub, group_indicator=g_sub)
            except:
                p = 1
        else:
            p = 1
        return p <= sig_threshold

    def _plot_z(self, z, labels, label='train'):
        plt.figure(figsize=(5, 5))
        if self.z_train.shape[1] >= 2:
            plt.figure(figsize=(5, 5))
            plt.scatter(z[:, 0], z[:, 1], c=self._get_colors(labels), alpha=0.2)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        else:
            for i in set(labels):
                f = (labels == i)
                plt.hist(z[f, 0], bins=100, color=f'C{i}')

        plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_z_clusters_{label}.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    def _plot_curves(self, curves, clusters, event_times, title='', alpha=0.02, sameplot=False, figsize=None, fig_name='curves'):
        if figsize is None:
            figsize = (3, 2)
        else:
            figsize = figsize

        if sameplot:
            n = 1
        else:
            n = len(set(clusters))

        fig, ax = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
        if n > 1:
            for i, s in enumerate(zip(curves, clusters)):
                ax[s[1]].step(event_times, s[0], where="post", c='C' + str(s[1]), alpha=alpha)
                ax[s[1]].set_xlabel('Time')
                ax[s[1]].set_ylabel('S(t)')
                ax[s[1]].set_ylim(0, 1)
                ax[s[1]].set_title(f'Pattern {s[1]}')
                ax[s[1]].grid(True)
        else:
            for i, s in enumerate(zip(curves, clusters)):
                ax.step(event_times, s[0], where="post", c='C' + str(s[1]), alpha=alpha, label=f'Pattern {i}')
                ax.set_xlabel('Time')
                ax.set_ylabel('S(t)')
                ax.set_ylim(0, 1)
                ax.grid(True)
            lgnd = plt.legend(loc=(1, 0))
            for handle in lgnd.legendHandles:
                handle.set_alpha(1)

        if (not sameplot) and (n > 1):
            for i in set(clusters):
                x_avg = curves[clusters == i].mean(axis=0)
                ax[i].plot(event_times, x_avg, c='k', linestyle='dashed', alpha=1)

        fig.suptitle(title)
        plt.grid(True)
        plt.savefig(f'{self.results_folder}/{self.prefix_name}_{self.timestamp}_{fig_name}_s.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    def _plot_context(self, curves, clusters, event_times):
        fig_name = 'Context'
        means = []
        ccolors = []
        for i in range(self.optimal_k):
            avg_ci = curves[clusters == i].mean(axis=0)
            means.append(avg_ci)
            ccolors.append(i)
        patterns = np.array(means)
        self._plot_curves(curves=patterns, clusters=ccolors, event_times=event_times, sameplot=True, alpha=0.5, figsize=(5, 3), fig_name=fig_name)

    @staticmethod
    def _get_colors(labels):
        return ['C' + str(i) for i in labels]


class SurvCounterfactual(XSurv):
    def __init__(self, max_depth=10, *args, **kwargs):
        self.max_depth = max_depth
        super().__init__(*args, **kwargs)

    def _base_clustering_model(self, number_clusters, random_state=10):
        return KMeans(n_clusters=number_clusters, n_init=10, random_state=random_state)

    def explain(self, x, target_pattern, features_names_list, feature_types=None, ohe_features=None, mask=None, anomaly_model=None, anomaly_threshold=0, norm=2, n_particles=50, n_iterations=100, patience=10, options=None, loss_target_weight=1e5, loss_distance_weight=1, loss_mutual_exclusions_weight=1e5, loss_anomaly_weight=1, verbose=False):
        """
        Finds the counterfactual example of x
        :param x: the input sample for which the counterfactual example is generated
        :param target_pattern: the target survival pattern of the counterfactual
        :param features_names_list: list of the names of the features
        :param mask: list of ones and zeros to indicate the features that are allowed to change
        :param anomaly_model: the autoencoder model to be used as anomaly detector
        :param norm: the L-norm used to compute distance between points, default: 2
        :param n_particles: number of particles in the swarm of the pso algorithm
        :param n_iterations: maximum number of iterations of the pso algorithm
        :param patience: number of iterations with no improvements.
        :param loss_target_weight: weight of the target loss
        :param loss_distance_weight: weight of the distance loss
        :param loss_mutual_exclusions_weight: weight of the mutual exclusions loss
        :param loss_anomaly_weight: weight of the anomaly loss
        :param verbose: if True additional information is printed.
        :return: the generated counterfactual example
        """

        if feature_types is None:
            feature_types = ['float']*self.x_train.shape[1]
        if mask is None:
            mask = [1] * self.x_train.shape[1]
        self.optimizer = PatternsCounterfactualExplainer(model=self,
                                                         X=self.x_train,
                                                         feature_types=feature_types,
                                                         ohe_features=ohe_features,
                                                         mask=mask,
                                                         anomaly_model=anomaly_model,
                                                         anomaly_threshold=anomaly_threshold,
                                                         norm=norm,
                                                         loss_target_weight=loss_target_weight,
                                                         loss_distance_weight=loss_distance_weight,
                                                         loss_mutual_exclusions_weight=loss_mutual_exclusions_weight,
                                                         loss_anomaly_weight=loss_anomaly_weight
                                                         )

        x_exp = self.optimizer.compute_explanation_cat_pso(x=x,
                                                           target_pattern=target_pattern,
                                                           n_particles=n_particles,
                                                           n_iterations=n_iterations,
                                                           patience=patience,
                                                           options=options,
                                                           verbose=verbose)

        self.optimizer.visualize_explanation(x, x_exp, features=features_names_list)
        return x_exp

