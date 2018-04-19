#!/usr/bin/env python3.6

import os
import re
import pandas

from sklearn import model_selection, svm, ensemble, metrics, naive_bayes, neural_network, preprocessing

NLCD_CODES = ["DF", "DHI", "EF", "OW", "PH", "SS"]

class DataSet:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def get_classes(self):
        return self.data_frame['NLCD'].cat.codes

    def get_features(self):
        band_keys = [key for key in list(self.data_frame) if re.search("B\d+", key)]
        return self.data_frame[band_keys]

    def get_scaled_features(self):
        scaler = preprocessing.StandardScaler()
        return scaler.fit_transform(self.get_features())

    def scale_features():
        scaler = preprocessing.StandardScaler

    def split_samplings(self, test_size):
        train, test = model_selection.train_test_split(
            self.data_frame,
            test_size=test_size)
        return DataSet(train), DataSet(test)

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

            import ipdb; ipdb.set_trace()

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

def build_trainer():
    return Trainer(load_data_set())

def main():
    trainer = build_trainer()
    trainer.fit()
    print(trainer.accuracies())
    print(trainer.accuracies_on_trained())

if __name__ == "__main__":
    main()
