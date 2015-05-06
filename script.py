import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.svm import SVC

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection
    """
    
    mat = loadmat('mnist_all.mat'); #loads the MAT object as a Dictionary
    
    n_feature = mat.get("train1").shape[1];
    n_sample = 0;
    for i in range(10):
        n_sample = n_sample + mat.get("train"+str(i)).shape[0];
    n_validation = 1000;
    n_train = n_sample - 10*n_validation;
    
    # Construct validation data
    validation_data = np.zeros((10*n_validation,n_feature));
    for i in range(10):
        validation_data[i*n_validation:(i+1)*n_validation,:] = mat.get("train"+str(i))[0:n_validation,:];
        
    # Construct validation label
    validation_label = np.ones((10*n_validation,1));
    for i in range(10):
        validation_label[i*n_validation:(i+1)*n_validation,:] = i*np.ones((n_validation,1));
    
    # Construct training data and label
    train_data = np.zeros((n_train,n_feature));
    train_label = np.zeros((n_train,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("train"+str(i)).shape[0];
        train_data[temp:temp+size_i-n_validation,:] = mat.get("train"+str(i))[n_validation:size_i,:];
        train_label[temp:temp+size_i-n_validation,:] = i*np.ones((size_i-n_validation,1));
        temp = temp+size_i-n_validation;
        
    # Construct test data and label
    n_test = 0;
    for i in range(10):
        n_test = n_test + mat.get("test"+str(i)).shape[0];
    test_data = np.zeros((n_test,n_feature));
    test_label = np.zeros((n_test,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("test"+str(i)).shape[0];
        test_data[temp:temp+size_i,:] = mat.get("test"+str(i));
        test_label[temp:temp+size_i,:] = i*np.ones((size_i,1));
        temp = temp + size_i;
    
    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis = 0);
    index = np.array([]);
    for i in range(n_feature):
        if(sigma[i] > 0.001):
            index = np.append(index, [i]);
    train_data = train_data[:,index.astype(int)];
    validation_data = validation_data[:,index.astype(int)];
    test_data = test_data[:,index.astype(int)];

    # Scale data to 0 and 1
    train_data = train_data/255.0;
    validation_data = validation_data/255.0;
    test_data = test_data/255.0;
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z));
    
def blrObjFunction(params, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    n_data = train_data.shape[0];
    n_feature = train_data.shape[1];
    error = 0;
    error_grad = np.zeros((n_feature+1));
   
    w = params
    data, label = args
    w = w.reshape((w.shape[0],1))
    b = np.c_[np.ones(data.shape[0]),data]
    for i in range (b.shape[0]):
        a1 = np.dot(w.T,b[i])
        y = sigmoid(a1)
        error =error +((label[i]*np.log(y)+((1-label[i])*(np.log(1-y)))))
        error_grad = error_grad + (np.subtract(y,label[i])*b[i])
    error = -1 * error
    error_grad = error_grad.reshape(n_feature+1,1)
    error_grad = np.squeeze(np.asarray(error_grad))
    print(error)
    print(error_grad.shape)
    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0],1));

    matsize = len(data)
    data = np.c_[data,np.ones((matsize, 1), dtype = float)]
    thing = sigmoid(np.dot(W.T,data.T))
    max = np.argmax(thing, axis=0)
    label = max
    label = label.reshape((label.shape[0],1))

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

# number of classes
n_class = 10;

# number of training samples
n_train = train_data.shape[0];

# number of features
n_feature = train_data.shape[1];

T = np.zeros((n_train, n_class));
for i in range(n_class):
    T[:,i] = (train_label == i).astype(int).ravel();
    
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature+1, n_class));
initialWeights = np.zeros((n_feature+1,1));
opts = {'maxiter' : 50};
for i in range(n_class):
    labeli = T[:,i].reshape(n_train,1);
    args = (train_data, labeli);
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
    W[:,i] = nn_params.x.reshape((n_feature+1,));

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')

train_label = train_label.reshape(train_label.shape[0])
test_label = test_label.reshape(test_label.shape[0])
validation_label = validation_label.reshape(validation_label.shape[0])

clf = SVC( kernel='linear')
clf.fit(train_data, train_label)
print('\n Accuracy of linear kernel and all other parameters as default\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value 1 and all other parameters as default\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf')
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and all other parameters as default\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf',C=1)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and c=1 and all other parameters as default\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 10)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=10\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 20)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=20\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 30)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=30\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 40)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=40\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 50)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=50\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 60)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=60\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 70)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=70\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 80)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=80\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 90)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=90\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')

clf = SVC( kernel='rbf', C = 100)
clf.fit(train_data, train_label)
print('\n Accuracy of rbf kernel and gamma value default and C value=100\n')
print('\n Training set Accuracy: ' +str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' +str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' +str(clf.score(test_data, test_label)*100) + '%')
