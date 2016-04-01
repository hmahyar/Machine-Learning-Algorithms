import numpy as np
from numpy import linalg


class data_generator(object):
    def __init__(self):
        pass

    def linear_separable(self,num_d1=100, num_d2=100,split_ratio=0.7):
        mean = np.array([0, 3])
        cov = np.array([[0.5, 0.2], [0.2, 0.5]])
        d1 = np.hstack((np.random.multivariate_normal(mean, cov, num_d1),np.ones(num_d1)))
        d2 = np.hstack((np.random.multivariate_normal(mean[::-1], cov, num_d2),np.ones(num_d2)*-1))
        print (d1,d2)

        return  X1,y1,X2,y2

    def non_linear_separable(self):
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def lin_separable_with_overlap(self):
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2



