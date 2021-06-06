import numpy as np 
from sklearn.svm import SVC
import os
import joblib

class svm_classification:
    def __init__(self, root_path, dst_path='sound_classes.pkl '):
        """
        Initialize the class configs/

        Args:
            root_path: the path of the data source.
            dst_path: the path of the pickle file containing the training data.
        """
        self.root_path = root_path
        self.dst_path = os.path.join(root_path, dst_path)

    def svm_train(self, labels, features, dst_path=None):
        if dst_path is None:
            dst_path = self.dst_path
        
        clf = SVC(C=10, tol=1e-3, probability = True)
        clf.fit(features.reshape(1, -1), labels)
        joblib.dump(clf, dst_path)
        
        print('SVM training has been finished!')


# if __name__ == '__main__':
