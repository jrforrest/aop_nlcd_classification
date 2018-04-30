#!/usr/bin/env python3.6

import os
import re
import pandas

from ipdb import set_trace as st;

from sklearn import model_selection, svm, ensemble, metrics, naive_bayes, neural_network, preprocessing
from matplotlib import pyplot

NLCD_CODES = ["DF", "DHI", "EF", "OW", "PH", "SS"]

class Visualizer:
    def __init__(self, data_set):
        self.data_set = data_set

    def save_figures(self):
        scaled_features = self.data_set.scaled().without_empty_rows().features_by_class()
        unscaled_features = self.data_set.without_empty_rows().features_by_class()

        plots = {
              'unscaled_mean': unscaled_features.mean(),
              'unscaled_median': unscaled_features.median(),
              'scaled_mean': scaled_features.mean(),
              'scaled_median': scaled_features.median()
        }

        for name, data in plots.items():
            data.plot.bar(title = name, figsize = (15, 10))
            pyplot.savefig("/tmp/" + name + ".png")

class DataSet:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def get_classes(self):
        return self.data_frame['NLCD'].cat.codes

    def get_features(self):
        return self.data_frame[self.band_keys()]

    def features_for_class(self, nlcd_code):
        return self.data_frame[self.data_frame['NLCD'] == nlcd_code][self.band_keys()]

    def band_keys(self):
        return [key for key in list(self.data_frame) if re.search("B\d+", key)]

    def scaled(self):
        scaler = preprocessing.MinMaxScaler()
        new = self.data_frame.copy()
        new[self.band_keys()] = scaler.fit_transform(self.get_features())
        return DataSet(new)

    def features_by_class(self):
        return self.data_frame.groupby('NLCD')[self.band_keys()]

    def non_feature_columns(self):
        return self.data_frame.columns.difference(self.band_keys())

    def split_samplings(self, test_size):
        train, test = model_selection.train_test_split(
            self.data_frame,
            test_size = test_size)
        return DataSet(train), DataSet(test)

    def without_empty_rows(self):
        return DataSet(self.data_frame.dropna(axis = 'index', how = 'any'))

class Trainer:
    def __init__(self, data_set):
        self.data_set = data_set
        self.training_set, self.test_set = data_set.split_samplings(0.2)
        self.scaler = preprocessing.MinMaxScaler()
        self.models = {
            'svc': svm.SVC(),
            'rf': ensemble.RandomForestClassifier(
                n_estimators = 50,
                min_samples_split = 0.25,
                max_features = "sqrt"),
            'gnb': naive_bayes.GaussianNB(),
            'mlp': neural_network.MLPClassifier(max_iter = 2000, hidden_layer_sizes = [100])
        }
        self.is_fit = False

    def fit(self):
        if not self.is_fit:
            self.scaler.fit(self.data_set.get_features())

            scaled_training_features = self.scaler.transform(self.training_set.get_features())

            for model in self.models.values():
                model.fit(scaled_training_features, self.training_set.get_classes())

            self.is_fit = True

    def accuracies(self):
        return {k: self.accuracy_for_model(k) for k in self.models.keys()}

    def accuracies_on_trained(self):
        return {k: self.accuracy_for_model(k, self.training_set) for k in self.models.keys()}

    def accuracy_for_model(self, name, test_set = None):
        test_set = test_set or self.test_set

        return metrics.accuracy_score(
            test_set.get_classes(),
            self.results_for_model(name, test_set = test_set))

    def results_for_model(self, name, test_set = None):
        test_set = test_set or self.test_set
        scaled_test_features = self.scaler.transform(test_set.get_features())

        return self.models[name].predict(scaled_test_features)

def load_data_class(name):
    current_dir = os.path.dirname(__file__)
    data_frame = pandas.read_csv(os.path.join(current_dir, (f"{name}_training.csv")))
    return data_frame

def drop_rows_with_empty_bands_from(data_frame):
    return data_frame.dropna(axis = 'index', how = 'any')

def load_data_set():
    data_frame = pandas.DataFrame()

    for code in NLCD_CODES:
         data_frame = data_frame.append(load_data_class(code))

    data_frame = drop_rows_with_empty_bands_from(data_frame)
    data_frame['NLCD'] = data_frame['NLCD'].astype('category')

    return DataSet(data_frame)

def render_histograms():
    Visualizer(load_data_set()).render_histogram('DF')

def build_trainer():
    return Trainer(load_data_set())

def main():
    trainer = build_trainer()
    #trainer.fit()
    #print(trainer.accuracies())
    #print(trainer.accuracies_on_trained())
    Visualizer(trainer.data_set).save_figures()

if __name__ == "__main__":
    main()
