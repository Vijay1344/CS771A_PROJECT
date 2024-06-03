import numpy as np
import sklearn
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0


	# Mapping the training data
	obs = X_train.shape[0]
	feat = my_map(X_train)

	# Fitting the model to feat
	from sklearn.linear_model import LogisticRegression
	log_reg = LogisticRegression(max_iter = 20000, C = 75)
	log_reg.fit(feat, y_train)

	# Getting the weight vector
	n_dim = feat.shape[1]
	w = log_reg.coef_.reshape(n_dim)
	b = log_reg.intercept_

	# return weight = w and bias = b
	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points

	# n_dim = dimension of a single observation
    n_dim = X.shape[1]
    obs = X.shape[0]

    # Temporary variables for choosing columns from khatri-rao product
    upper_triangular_indices = np.array(np.triu_indices(n = n_dim, m = n_dim, k = 1))
    col_indices = upper_triangular_indices[0][:] * n_dim + upper_triangular_indices[1][:]

    # Preparing the map

    # Cross-product terms
    X = np.cumprod( np.flip(1-2*X, axis = 1), axis = 1 )
    feat = khatri_rao(np.transpose(X), np.transpose(X))
    feat = np.transpose(feat)
    feat = feat[:, col_indices]

    # Appending x0, x1, ..., x31
    feat = np.append(feat, X, axis = 1)

    return feat