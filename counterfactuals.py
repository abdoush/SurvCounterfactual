from sklearn.preprocessing import MinMaxScaler
from optimization import SimulatedAnnealing
import numpy as np
import copy
import matplotlib.pyplot as plt
import pyswarms as ps


class CounterfactualExplainer:
    """A basic class implementing the counterfactual explanations.
    It computes explanations for regression models. It can be also used for binary classification
    by applying y_min_change = 1 or y_min_change = -1 (depending on the label of original class).
    Not yet adapted to classification problem (TODO)
    
    Properties:
        _model - stores the instance of the model, which must implement predict() method
        _norm - the L-norm used to compute distance between points, default: 2
        _weight_distance - weight for the distance metric in loss function
        _weight_target - weight for the target in loss function
        _weight_anomlay - weight for the anomaly metric in loss function
        _X - base dataset, which is used to define feature boundaries and perform scaling
        _feature_types - list containing the type of each feature. Currently supports 
                         'bool' for boolean and 'float' for numerical data. If other type
                         is passed it is automatically treated like 'float'
    """
    
    def __init__(self, 
                 model, 
                 X, 
                 feature_types, 
                 anomaly_model=None, 
                 norm_distance=1, 
                 norm_target=2, 
                 weight_distance=1,
                 weight_target=1,
                 weight_anomaly=1, 
                 lambda_distance=1,
                 lambda_anomaly=1,
                 anomaly_threshold=None):
        self._model = model
        self._norm_distance = norm_distance
        self._norm_target = norm_target
        self._weight_distance = weight_distance
        self._weight_target = weight_target
        self._weight_anomaly = weight_anomaly
        self._X = X
        self._feature_types = feature_types
        self._anomaly_model = anomaly_model
        self._anomaly_threshold = anomaly_threshold
            
            
    def _distance_features(self, x1, x2):
        """Computes the distance between two points based on the provided L-norm.
        
        Arguments:
            x1 - fisrt point
            x2 - second point
            
        Returns:
            distance between points, float
        """
        
        x1 = x1.astype(float)
        x2 = x2.astype(float)
        
        
        return np.linalg.norm(x1 - x2, self._norm_distance, axis=1)
    
    def _distance_target(self, y1, y2):
        """Computes the distance between two points based on the provided L-norm.
        
        Arguments:
            y1 - fisrt point
            y2 - second point
            
        Returns:
            distance between targets, float
        """
        
        y1 = y1.astype(float)
        y2 = y2.astype(float)
        
        return np.linalg.norm(y1 - y2, self._norm_target, axis=0)
        
        
    def _loss(self, x, x_original, y_change, anomaly_threshold=None, verbose=False):
        """Calculates the value of loss function, which is used to find the best counterfactual explanation
        
        Arguments:
            x - the array of values, which is optimized - a candidate for counterfactual
            x_original - the original point for which counterfactual is generated
            y_change - required change of the target value as fraction (e.g. 0.2 = 20% increase)
            verbose - flag for debugging (prints additional output)
        
        Returns:
            Value of loss for a counterfactual candidate
        """
        if x.ndim == 1:
            x = x.reshape((1, -1))
        if x_original.ndim == 1:
            x_original = x_original.reshape((1, -1))
        
        y_original = self._model.predict(x_original)
        y = self._model.predict(x)
        
        if y.ndim == 1:
            y = y.reshape((1, -1))
        if y_original.ndim == 1:
            y_original = y_original.reshape((1, -1))
            
        y_target = y_original * (1 + y_change)
        
        loss_distance = self._weight_distance * self._distance_features(x, x_original)
        loss_target = self._weight_target * self._distance_target(y, y_target)
        
        if self._anomaly_model is not None:
            loss_anomaly = self._weight_anomaly * self._anomaly_model.anomaly_score(x, anomaly_threshold=self._anomaly_threshold)
        else:
            loss_anomaly = 0
        
        total_loss = loss_distance + loss_target + loss_anomaly
        
        if verbose:
            print("Loss Target: %.3g, Loss Distance: %.3g, Loss Anomaly: %.3g, Total: %.3g" % (loss_target, loss_distance, loss_anomaly, total_loss))
        
        return total_loss
           
    
    def compute_explanation(self, optimizer, x_original, x_exp_0, y_change, verbose=False, step=0.001):
        if x_original.ndim == 1:
            x_original = np.expand_dims(x_original, 0)
        
        x_original_flat = x_original.flatten()
        y_pred_original = self._model.predict(x_original)
        
        bounds = []

        for i, cat in enumerate(self._feature_types):
            bounds.append((0, 1))
        #     if  cat == "bool":
        #         steps.append(1)
        #     else:
        #         steps.append(step)
        
        # x0 = copy.deepcopy(x_original_flat) # we start optimization from the original point
        x_explanation = optimizer.minimize(func=self._loss, args=(x_original_flat, y_change), x0=x_exp_0, step=step, bounds=bounds)
               
        if verbose:
            print("Original Point:", x_original)
            print("Original prediction:", y_pred_original)
            print("Counterfactual Point:", x_explanation)
            print("Counterfactual prediction:", self._model.predict(x_explanation))
 
        return x_explanation.flatten()

    def compute_explanation_pso(self, optimizer, x_original, x_exp_0, y_change, verbose=False, step=0.001):
        if x_original.ndim == 1:
            x_original = np.expand_dims(x_original, 0)
        
        x_original_flat = x_original.flatten()
        y_pred_original = self._model.predict(x_original)
        
        bounds = []

        for i, cat in enumerate(self._feature_types):
            bounds.append((0, 1))
        
        x_explanation = optimizer.minimize(func=self._loss, bounds=bounds, **dict(x_original=x_original_flat, y_change=y_change), verbose=False)
               
        if verbose:
            print("Original Point:", x_original)
            print("Original prediction:", y_pred_original)
            print("Counterfactual Point:", x_explanation)
            print("Counterfactual prediction:", self._model.predict(x_explanation))
 
        return x_explanation.flatten()

    
    def print_individual_losses(self, x, x_exp, y_change):
        print("Weight Target: %.3g, Weight DistanceL %.3g, Weight Anomaly: %.3g" % (self._weight_target, self._weight_distance, self._weight_anomaly))
        result = self._loss(x_exp, x, y_change, verbose=True)
        return
    

    def visualize_explanation(self, x, x_exp, features=None):
        """Visualizes the counterfactual explanation as the difference between the counterfactual and original point
        
        Arguments:
            x - original point
            x_exp - counterfactual explanation
            features - names of the features
        
        Returns:
            Barplot with explanation
        """
        
        plt.figure(figsize=(7, 6))
        plt.barh(y=range(x.shape[-1]), width=x_exp-x)
        if features is not None:
            plt.yticks(range(len(features)), features)
        plt.show()

class PatternsCounterfactualExplainer:
    """A class to search for a conterfactual example

    Properties:
        _model - stores the instance of the model, which must implement predict() method
        _X - base dataset, which is used to define feature boundaries and perform scaling
         _feature_types - list containing the type of each feature. Currently supports
                         'bool' for boolean and 'float' for numerical data. If other type
                         is passed it is automatically treated like 'float'
        _ohe_features - list of lists where each list contains the indicies of the features representing a one-hot-encoded feature ex: [[3, 4, 5], [6, 7, 8, 9]]
        _mask - list of zeros and ones of length _X.shape[1] to indicate the features that are allowed to change
        _anomaly_model - the autoencoder model used as anomaly detection model
        _norm - the L-norm used to compute distance between points, default: 2
        _loss_target_weight - weight of the target loss
        _loss_mutual_exclusions_weight - weight of the mutual exclusions loss
        _loss_distance_weight - weight of the distance loss
        _loss_anomaly_weight - weight of the anomaly loss
        _scaler - scaler used for data normalization, MinMaxScaler is currently hardcoded
        history - contains the loss values during iterations
    """

    def __init__(self, model, X, feature_types, ohe_features, mask=None, anomaly_model=None, anomaly_threshold=0, norm=2, loss_target_weight=1e5, loss_distance_weight=1, loss_mutual_exclusions_weight=1e5, loss_anomaly_weight=1):
        self._model = model
        self._X = X
        self._feature_types = feature_types
        self._ohe_features = ohe_features
        if mask is None:
            self._mask = np.ones(len(feature_types))
        else:
            self._mask = np.array(mask)

        self._anomaly_model = anomaly_model
        self._anomaly_threshold=anomaly_threshold
        self._norm = norm

        self._loss_target_weight = loss_target_weight
        self._loss_mutual_exclusions_weight = loss_mutual_exclusions_weight
        self._loss_distance_weight = loss_distance_weight
        self._loss_anomaly_weight = loss_anomaly_weight

        self._scaler = MinMaxScaler()
        self._scaler.fit(self._X)

        self.history = {'loss_distance': [], 'loss_target': [], 'loss_mutual_exclusions': [],'loss_anomaly':[], 'total_loss': []}

    def _distance(self, x1, x2):
        """Computes the distance between two points based on the provided L-norm.

        Arguments:
            x1 - fisrt point
            x2 - second point

        Returns:
            distance between points, float
        """

        x1 = x1.astype(float)
        x2 = x2.astype(float)
        dist = np.sum(np.abs(x1 - x2)**self._norm, axis=1)**(1/self._norm)
        return dist # np.linalg.norm(x1 - x2, self._norm)

    def _loss_cat_pso(self, params, x, target_pattern):
        """Calculates the value of loss function, which is used to find the best counterfactual explanation using pso optimization

        Arguments:
            params - the array of values, which is optimized - candidates for counterfactuals of shape: (number of particles, number of features)
            x - the original point for which counterfactual is generated
            target_pattern - the target survival pattern

        Returns:
            Value of loss for a counterfactual candidate
        """
        x_new = np.zeros_like(params)
        for i, cat in enumerate(self._feature_types):
            if cat == "bool":
                x_new[:,i] = (params[:,i]>0.5).astype(int)
            else:
                x_new[:,i] = params[:,i]

        x_new = np.where(self._mask, x_new, x)

        # reshape to ensure correct dimensions of the array
        if x_new.ndim == 1:
            x_new = np.expand_dims(x_new, 0)

        # calculate if the output of the model matches desired target pattern
        dist_to_patterns = self._model.get_centers_distences(x_new)
        dist_to_target_pattern = dist_to_patterns[:,target_pattern]
        y_new_pred = self._model.predict(x_new)
        is_target_not_reached = y_new_pred != target_pattern
        loss_target = is_target_not_reached.astype(int) * dist_to_target_pattern
        loss_target = loss_target

        # scale the points
        x_scaled = self._scaler.transform(x)
        x_new_scaled = self._scaler.transform(x_new)
        # Make sure the generated point is as close to the original point as possible
        loss_distance = self._distance(x_scaled, x_new_scaled)

        # to make sure the generated example have a correct set of one-hot-enccoded features
        if self._ohe_features:
            conditions = []
            for ohe_feature in self._ohe_features:
                is_not_only_one_bit = x_new[:,ohe_feature].sum(axis=1)!=1
                conditions.append(is_not_only_one_bit)
            loss_mutual_exclusions = (np.sum(conditions, axis=0)>0).astype(int)
        else:
            loss_mutual_exclusions = np.zeros(x_new.shape[0])

        # Make sure the generated point is from the same disbtribution of the dataset
        if self._anomaly_model is not None:
            anomaly_score = self._anomaly_model.anomaly_score_multi(x_new)
            loss_anomaly = np.maximum(0, anomaly_score - self._anomaly_threshold)
        else:
            loss_anomaly = np.zeros(x_new.shape[0])

        total_loss = self._loss_distance_weight * loss_distance + \
                     self._loss_target_weight * loss_target + \
                     self._loss_mutual_exclusions_weight * loss_mutual_exclusions + \
                     self._loss_anomaly_weight * loss_anomaly

        self.history['loss_mutual_exclusions'].append(loss_mutual_exclusions[total_loss==np.min(total_loss)])
        self.history['loss_distance'].append(loss_distance[total_loss==np.min(total_loss)])
        self.history['loss_target'].append(loss_target[total_loss==np.min(total_loss)])
        self.history['loss_anomaly'].append(loss_anomaly[total_loss==np.min(total_loss)])
        self.history['total_loss'].append(np.min(total_loss))

        return total_loss

    def compute_explanation_cat_pso(self, x, target_pattern, n_particles=50, n_iterations=100, patience=10, options=None, verbose=False):
        """ Generates a counterfactual explanation using swarm optimization

        Arguments:
            x - the original point for which counterfactual is generated.
            target_patterns - the target survival pattern.
            n_particles - number of particles in the swarm.
            n_iterations - maximum number of iterations.
            patience - number of iterations with no improvements.
            verbose - if True additional information is printed.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, 0)

        y_pred = self._model.predict(x)

        # define bounds for the pso optimization algorithm
        bounds_min = []
        bounds_max = []
        for i, cat in enumerate(self._feature_types):
            if cat == "bool":
                bounds_min.append(0)
                bounds_max.append(1)
            else:
                bounds_min.append(self._X.min(axis=0)[i])
                bounds_max.append(self._X.max(axis=0)[i])

        """
        hyperparameters for the pso: 
            c1: the weight of the congnitive component of velocity (personal best). 
            c2: the weight of the social component of velocity (global best).
            w: the weight of the random component of velocity.
        """
        if options is None:
            #options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
            options = {'c1': 1.49618, 'c2': 1.49618, 'w': 0.7298}
        bounds = (np.array(bounds_min), np.array(bounds_max))

        # run the pso minimization
        self.pso_opt = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=x.shape[1], bounds=bounds, options=options, ftol_iter=patience, ftol=1e-3)
        best_cost, best = self.pso_opt.optimize(self._loss_cat_pso, x=x, target_pattern=target_pattern, iters=n_iterations, verbose=True)

        x_exp = np.zeros_like(best)
        for i, cat in enumerate(self._feature_types):
            if cat == "bool":
                x_exp[i] = (best[i] > 0.5).astype(int)
            else:
                x_exp[i] = best[i]

        x_exp = np.where(self._mask, x_exp, x)
        if x_exp.ndim == 1:
            x_exp = np.expand_dims(x_exp, 0)

        if verbose:
            print("Original Point:", x)
            print("Original value:", y_pred)
            print("Counterfactual Point:", x_exp)
            print("Counterfactual value:", self._model.predict(x_exp))

        return x_exp.flatten()

    def visualize_explanation(self, x, x_exp, features=None):
        """Visualizes the counterfactual explanation as the difference between the counterfactual and original point

        Arguments:
            x - original point
            x_exp - counterfactual explanation
            features - names of the features

        Returns:
            Barplot with explanation
        """

        plt.figure(figsize=(7, (len(features))))
        plt.barh(y=range(x.shape[-1]), width=x_exp - x)
        if features is not None:
            plt.yticks(range(len(features)), features)
        plt.show()
