import datasets
import config
import numpy as np
import image_features
import utils

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


class Trainer:
    def __init__(self):
        self.dataset = datasets.DefaultDatasetLoader(config.Pipeline.DEBUG)
        self.classifier = None

    def look_for_feature_params(self):
        i = 0
        color_spaces = ['LUV']
        hog_chanels = ['ALL']
        hog_orients = [8]
        spatial_sizes = [(16, 16)]
        hist_bins = [64]
        hog_fs = [True]
        spatial_fs = [False]
        hist_fs = [True]

        for color_space in color_spaces:
            for hog_chanel in hog_chanels:
                for orient in hog_orients:
                    for spatial in spatial_sizes:
                        for hist in hist_bins:
                            for hog_f in hog_fs:
                                for spatial_f in spatial_fs:
                                    for hist_f in hist_fs:
                                        if not (hist_f or spatial_f or hog_f):
                                            continue
                                        i = i + 1
                                        print(color_space, " hog_f=", hog_f, " spatial_f=", spatial_f, " hist_f=", hist_f)
                                        self.extract_training_data(color_space=color_space, hog_channel=hog_chanel, orient=orient, spatial_size=spatial, hist_bins=hist, hog_f=hog_f, spatial_f=spatial_f, hist_f=hist_f)
                                        self.train()


    def look_for_classifier_type(self):
        self.extract_training_data()

        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree':[2, 3, 4], 'kernel': ['poly']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']},
        ]
        svr = SVC()
        clf = GridSearchCV(estimator=svr, param_grid=param_grid)
        clf.fit(self.X_train, self.y_train)

        print(clf.best_params_, clf.best_score_, clf.cv_results_)
        # on 1000 samples {'kernel': 'rbf', 'gamma': 0.0001, 'C': 10}
        # trained on full DS = 0.9955


    def extract_training_data(self, color_space=config.Classifier.COLOR_SPACE,
                              orient=config.Classifier.ORIENT,
                              pix_per_cell=config.Classifier.PIX_PER_CELL,
                              cell_per_block=config.Classifier.CELL_PER_BLOCK,
                              hog_channel=config.Classifier.HOG_CHANNEL,
                              spatial_size=config.Classifier.SPATIAL_SIZE,
                              hist_bins=config.Classifier.HIST_BINS,
                              spatial_f=config.Classifier.SPATIAL_F,
                              hist_f=config.Classifier.HIST_F,
                              hog_f=config.Classifier.HOG_F):

        n_samples = config.Trainer.TRAIN_ON_SUBSET

        if n_samples > 0:
            random_idxs = np.random.randint(0, len(self.dataset.vehicles), n_samples)
            test_cars = np.array(self.dataset.vehicles)[random_idxs]
            test_noncars = np.array(self.dataset.non_vehicles)[random_idxs]
        else:
            test_cars = self.dataset.vehicles
            test_noncars = self.dataset.non_vehicles

        car_features = image_features.extract_features(test_cars,
                                                       color_space,
                                                       spatial_size,
                                                       hist_bins,
                                                       orient,
                                                       pix_per_cell,
                                                       cell_per_block,
                                                       hog_channel,
                                                       spatial_f,
                                                       hist_f,
                                                       hog_f)

        noncar_features = image_features.extract_features(test_noncars,
                                                          color_space,
                                                          spatial_size,
                                                          hist_bins,
                                                          orient,
                                                          pix_per_cell,
                                                          cell_per_block,
                                                          hog_channel,
                                                          spatial_f,
                                                          hist_f,
                                                          hog_f)

        X = np.vstack((car_features, noncar_features)).astype(np.float64)
        self.X_scaler = StandardScaler().fit(X)
        scaled_X = self.X_scaler.transform(X)

        # labels
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

        rand_state = np.random.randint(0, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)


    def train(self, cls=LinearSVC()):
        self.classifier = cls

        self.classifier.fit(self.X_train, self.y_train)

        print('test accuracy of classifier = ', round(self.classifier.score(self.X_test, self.y_test), 4))

        utils.save_classifier(self.X_scaler, config.Classifier.SCALER_FILE)
        utils.save_classifier(self.classifier, config.Classifier.CLS_FILE)
