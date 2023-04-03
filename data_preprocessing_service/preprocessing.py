import os
import numpy as np
import pandas as pd
import deepchem
from from_root import from_root
from pandas import DataFrame
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, df: DataFrame, feature:str, label: str, test_size: int, random_state: int):
        self.df = df
        self.feature = feature
        self.label = label
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessed_dataset_path = os.path.join(from_root(),"preprocessed_dataset")

    def _train_test_split(self):
        X = self.df[self.feature].values.tolist()
        y = self.df[self.label].values.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test

    def _feature_scaling(self,X_train,X_test):
        maccskeys = deepchem.feat.MACCSKeysFingerprint()
        circular = deepchem.feat.CircularFingerprint()
        rdkit = deepchem.feat.RDKitDescriptors()
        pubchem = deepchem.feat.PubChemFingerprint()

        maccskeys_train = maccskeys.featurize(X_train)
        maccskeys_test = maccskeys.featurize(X_test)

        circular_train = circular.featurize(X_train)
        circular_test = circular.featurize(X_test)

        rdkit_train = rdkit.featurize(X_train)
        rdkit_test = rdkit.featurize(X_test)

        pubchem_train = pubchem.featurize(X_train)
        pubchem_test = pubchem.featurize(X_test)

        # process pubchem empty lists
        pubchem_train = np.array([0] * 881 if len(pubchem_train[i]) == 0 else pubchem_train[i].tolist() for i in range(len(pubchem_train)))
        pubchem_test = np.array([0] * 881 if len(pubchem_test[i]) == 0 else pubchem_test[i].tolist() for i in range(len(pubchem_test)))

        # combine features
        fp_train = np.concatenate((maccskeys_train, circular_train, rdkit_train, pubchem_train), axis=1)
        fp_test = np.concatenate((maccskeys_test, circular_test, rdkit_test, pubchem_test), axis=1)

        # convert nan to 0
        X_train = np.nan_to_num(fp_train, nan=0, posinf=0)
        X_test = np.nan_to_num(fp_test, nan=0, posinf=0)

        return X_train, X_test

    def preprocess(self):
        X_train, X_test, y_train, y_test = self._train_test_split()
        X_train, X_test = self._feature_scaling(X_train, X_test)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        
        # save to npy
        np.save(open(self.preprocessed_dataset_path, "wb"), X_train)
        np.save(open(self.preprocessed_dataset_path, "wb"), X_test)
        np.save(open(self.preprocessed_dataset_path, "wb"), y_train)
        np.save(open(self.preprocessed_dataset_path, "wb"), y_test)

        return X_train, X_test, y_train, y_test
