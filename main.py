#!/usr/bin/env python3.6

import os
import re
import pandas

from sklearn import model_selection
from sklearn import svm
from sklearn import ensemble
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import neural_network

NLCD_CODES = ["DF", "DHI", "EF", "OW", "PH", "SS"]

class DataSet:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def get_classes(self):
        return self.data_frame['NLCD'].cat.codes

    def get_features(self):
        band_keys = [key for key in list(self.data_frame) if re.search("B\d+", key)]

        return self.data_frame[band_keys]

    def split_samplings(self, test_size):
        train, test = model_selection.train_test_split(
            self.data_frame,
            test_size=test_size)
        return DataSet(train), DataSet(test)

class Trainer:
    def __init__(self, data_set):
        self.training_set, self.test_set = data_set.split_samplings(0.2)
        self.models = {
            'svc': svm.SVC(),
            'rf': ensemble.RandomForestClassifier(),
            'gnb': naive_bayes.GaussianNB(),
            'mlp': neural_network.MLPClassifier()
        }
        self.is_fit = False

    def fit(self):
        if not self.is_fit:
            for model in self.models.values():
                model.fit(
                    self.training_set.get_features(),
                    self.training_set.get_classes())
            self.is_fit = True

    def accuracies(self):
        return {k: self.accuracy_for_model(k) for k in self.models.keys()}

    def accuracy_for_model(self, name):
        return metrics.accuracy_score(
            self.test_set.get_classes(),
            self.results_for_model(name))

    def results_for_model(self, name):
        return self.models[name].predict(self.test_set.get_features())

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

def build_trainer():
    return Trainer(load_data_set())

def main():
    trainer = build_trainer()
    trainer.fit()
    print(trainer.accuracies())

if __name__ == "__main__":
    main()
