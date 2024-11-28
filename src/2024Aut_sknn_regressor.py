#%%
import os
import math
import numpy as np
import pandas as pd
#pip install rfit
import rfit 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier
#
import warnings
warnings.filterwarnings("ignore")
print("\nReady to continue.")

#%%[markdown]
# Project tasks and goals:
# 
# 1. Use this Housing Price dataset. 
# - Use SalePrice as target for K-NN regression. 
# - For features that are *ORDINAL*, recode them as 0,1,2,... 
# - Drop features that are purely categorical.
# 2. Modify the sknn class to perform K-NN regression.
# 3. Modify the sknn class as you see fit to improve the algorithm performance, logic, or presentations.
# 3. Find optimized scaling factors for the features for the best model score.
# 4. Modify the sknn class to save some results (such as scores, scaling factors, gradients, etc, at various points, like every 100 epoch).
# 5. Compare the results of the optimized scaling factors to Feature Importance from other models, such as Tree regressor for example.
# 
# Please ask me anything about this project. You can either work individually or team with one other student
# to complete this project.
# 
# You/your team need to create a github repo (private) and add myself (physicsland) as a collaborator. 
# Please setup an appropriate .gitignore as the first thing when you create a repo. 
# 
# 
#%%
def preclean(data):

    # Ordinal Mapping, higher is better
    mappings = {
    'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},  
    'ExterQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, 
    'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 
    'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}, 
    'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},  
    'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'CentralAir': {'N': 0, 'Y': 1},  
    'KitchenQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},  
    'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
    'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
    'PoolQC': {'NA': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, 
    'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},  
    }

    # Drop categorical columns
    data = data.drop(columns=['Id','MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','Electrical','Functional','GarageType','MiscFeature','SaleType','SaleCondition'],axis=1)


    for column, mapping in mappings.items():
        if column in data.columns:
            data[column] = data[column].map(mapping).fillna(0).astype(int)
    
    
    # Remove NAs
    data = data.dropna()

    return data

df = pd.read_csv(f'..{os.sep}data{os.sep}HousePricesAdv{os.sep}train.csv', header=0)
df_test = pd.read_csv(f'..{os.sep}data{os.sep}HousePricesAdv{os.sep}test.csv', header=0)

df_test = preclean(df_test)
df_test_x = df_test.iloc[:,:53]

df = preclean(df)
df_x = df.iloc[:,:53]
df_y = df.iloc[:,-1]

#%%
print("\nReady to continue.")

#%%

class sknn:
    '''
    Scaling k-NN model
    v2
    Using gradient to find max
    '''
    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor

    # contructor and properties
    def __init__(self, 
                 data_x, 
                 data_y, 
                 resFilePfx='results', 
                 classifier=True, 
                 k=7, 
                 kmax=33, 
                 zscale=True, 
                 caleExpos_init = (), 
                 scales_init = (), 
                 ttsplit=0.5, 
                 max_iter = 100, 
                 seed=1, 
                 scoredigits = 6, 
                 learning_rate_init = 0.1, 
                 atol = 1e-8 ) :
        """
        Scaling kNN model, using scaling parameter for each feature to infer feature importance and other info about the manifold of the feature space.

        Args:
            data_x (numpy ndarray or pandas DataFrame): x-data
            data_y (numpy ndarray or pandas Series or DataFrame): y-data
            resFilePfx (str, optional): result file prefix. Defaults to 'scores'.
            classifier (bool, optional): classifier vs regressor. Defaults to True.
            k (int, optional): k-value for k-N. Defaults to 7.
            kmax (int, optional): max k-value. Defaults to 33.
            zscale (bool, optional): start with standardized z-score. Defaults to True.
            probeExpos (tuple, optional): Tuple of the exponents for scaling factors. Defaults to ().
            scaleExpos (tuple, optional): Tuple of the scaling factors. Defaults to ().
            ttsplit (float, optional): train-test-split ratio. Defaults to 0.5.
            max_iter (int, optional): maximum iteration. Defaults to 100.
            seed (int, optional): seed value. Defaults to 1.
            scoredigits (int, optional): number of digitis to show/compare in results. Defaults to 6.
            learning_rate_init (float, optional): learning rate, (0,1]. Defaults to 0.01.
            tol (_type_, optional): tolerance. Defaults to 1e-4.
        """
        self.__classifierTF = classifier  # will extend to regression later
        self.k = k
        self.__kmax = kmax
        self.__iter = 0 # the number of trials/iterations
        self.max_iter = max_iter
        self.__seed = seed
        self.__scoredigits = scoredigits
        # self.__resFilePfx = resFilePfx
        self.__learning_rate_init = abs(learning_rate_init)
        self.learning_rate = abs(learning_rate_init)
        self.__atol = atol
        
        # prep data
        self.data_x = data_x
        self.data_xz = data_x # if not to be z-scaled, same as original
        self.zscaleTF = zscale
        # transform z-score 
        if (self.zscaleTF): self.zXform() # will (re-)set self.data_xz
        self.data_y = data_y
        # train-test split
        self.__ttsplit = ttsplit if (ttsplit >=0 and ttsplit <= 1) else 0.5 # train-test split ratio
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.__xdim = 0  # dimension of feature space
        self.traintestsplit() # will set X_train, X_test, y_train, y_test, __xdim
        # set x data column names
        # self.__Xcolnames = (); self.__setXcolnames()
        self.__vector0 = np.zeros(self.__xdim)
        # self.__vector1 = np.ones(self.__xdim)
        
        # set exponents and scaling factors 
        self.__scaleExpos = [] # tuple or list. length set by number of features. Because of invariance under universal scaling (by all features with same factor), we can restrict total sum of exponents to zero.
        # self.__scaleExpos_init = [] # tuple or list. length set by number of features
        self.__scaleFactors = None # numpy array. always calculate from self.__setExpos2Scales
        self.__setExpos2Scales([]) # will set the initial self.scaleExpos and self.__scaleFactors
        # self.__gradients = [] # partial y/partial exponents (instead of partial scaling factors)
        
        # set sklearn knnmodel objects, train, and get benchmark scores on test data
        self.__knnmodels = [np.nan, np.nan] # matching index value as k value
        for i in range(2,self.__kmax +1): 
            if (self.__classifierTF): 
                self.__knnmodels.append( KNeighborsClassifier(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train ) )
            else: 
                self.__knnmodels.append( KNeighborsRegressor(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train ) ) # TODO
        self.benchmarkScores = [np.nan, np.nan] +  [ round(x.score(self.X_test, self.y_test ), self.__scoredigits) for x in self.__knnmodels[2:] ]
        print(f'These are the basic k-NN scores for different k-values: {repr(self.benchmarkScores)}, where no individual feature scaling is performed.') 
        
        # set pandas df to save some results
        # self.__resultsDF = None
        
    # END constructor
    
    def zXform(self):
        '''
        standardize all the features (if zscale=True). Should standardize/scale before train-test split
        :return: None
        '''
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.data_xz = scaler.fit_transform(self.data_x)  # data_x can be ndarray or pandas df, data_xz will be ndarray
        return
    
    def traintestsplit(self):
        '''
        train-test split, 50-50 as default
        :return: None
        '''
        # train-test split
        from sklearn.model_selection import train_test_split
        # data_y can be pd series here, or 
        dy = self.data_y.values if (isinstance(self.data_y, pd.core.series.Series) or isinstance(self.data_y, pd.core.frame.DataFrame)) else self.data_y # if (isinstance(data_y, np.ndarray)) # the default
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_xz, dy, test_size=self.__ttsplit, random_state = self.__seed)
        # these four sets should be all numpy ndarrays.

        nrows_Xtest, self.__xdim = self.X_test.shape  # total rows and columns in X_test. # not needed for nrows
        # notice that 
        # self.__xdim == self.X_test.shape[1]   # True
        # self.__xdim is self.X_test.shape[1]   # True
        # nrows_Xtest == self.X_test.shape[0]   # True
        # nrows_Xtest is self.X_test.shape[0]   # False
        return

    def __setExpos2Scales(self, expos=[]):
        """
        set Scaling Exponents, a tuple or list
        Should make sure expos is centered (using __shiftCenter)

        Args:
            expos (list, optional): _description_. Defaults to [], should match number of features in data_x
        """

        # Can add more checks to ensure expos is numeric list/tuple
        if (len(expos) != self.__xdim):
            self.__scaleExpos = np.zeros(self.__xdim) # tuple, exp(tuple) gives the scaling factors.
            if self.__xdim >1: 
                self.__scaleExpos[0] = 0
                self.__scaleExpos[1] = 0
        else:
            self.__scaleExpos =  expos
        self.__scaleFactors = np.array( [ math.exp(i) for i in self.__scaleExpos ] ) # numpy array
        return
    
    def __shiftCenter(self, expos = []):
        """
        Enforce sum of exponents or any vectors like gradient = 0 (for xdim > 1)

        Args:
            expos (np array, optional): array of scaling exponents. Defaults to [].
        """
        return expos.copy() - expos.sum()/len(expos) if len(expos) > 1 else expos.copy()
        
    
    def __evalGradients(self, learning_rate=0, use = 'test'):
        """
        evaluate Gradients/partial derivatives with respect to exponential factors (not scaling factor)
        Args:
            learning_rate (float, optional): learning_rate. Defaults to 0.
            use (str, optional): use 'test' (default) or 'train' dataset to score. 
        """
        # set learning_rate
        grad = np.array( [ self.__eval1Gradient(i, learning_rate, use=use) for i in range(self.__xdim) ] )
        # normalize grad here?
        # self.__gradients = grad.copy()
        # return
        return grad # gradient as numpy array
    
    def __eval1Gradient(self, i, learning_rate=0, use='test'):
        """
        evaluate a single Gradient/partial derivative with respect to the exponential factor (not scaling factor)

        Args:
            i (int): the column/feature index.
            learning_rate (float, optional): learning_rate. Defaults to 0.
            use (str, optional): use 'test' (default) or 'train' dataset to score. 
        """
        thescale = self.__scaleExpos[i]
        thestep = max(learning_rate, self.learning_rate, abs(thescale)*self.learning_rate ) # modify step value appropriately if needed.
        # maxexpo = thescale + thestep/2
        # minexpo = thescale - thestep/2
        maxexpos = self.__scaleExpos.copy()
        maxexpos[i] += thestep/2
        minexpos = self.__scaleExpos.copy()
        minexpos[i] -= thestep/2
        slope = ( self.scorethis(scaleExpos=maxexpos, use=use) - self.scorethis(scaleExpos=minexpos, use=use) ) / thestep
        return slope
    
    def __setNewExposFromGrad(self, grad=() ):
        """
        setting new scaling exponents, from the gradient info
        steps: 
        1. center grad (will take care of both grad = 0 and grad = (1,1,...,1) cases)
        2. normalize grad (with learning rate as well)
        3. add to original expos

        Args:
            grad (tuple, optional): the gradient calculated. Defaults to empty tuple ().
        """
        grad = self.__shiftCenter(grad)
        if np.allclose(grad, self.__vector0, atol=self.__atol): 
            print(f"Gradient is zero or trivial: {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            return False
        norm = np.sqrt( np.dot(grad,grad) )
        deltaexpos = grad / norm * self.learning_rate
        self.__scaleExpos += deltaexpos
        self.__setExpos2Scales(self.__scaleExpos)
        return True
    
    def optimize(self, scaleExpos_init = (), maxiter = 0, learning_rate=0):
        """
        Optimizing scaling exponents and scaling factors

        Args:
            scaleExpos_init (np array, optional): initial search vector. Defaults to empty.
            maxiter (int, optional): max iteration. Defaults to 1e5.
            learning_rate (float, optional): learning_rate. Defaults to 0 or self.learning_rate
        """
        history = []
        maxi = max( self.max_iter, maxiter)
        skip_n = 5 # rule of thumb math.floor(1/learning_rate)
        expos = scaleExpos_init 
        if (len(scaleExpos_init) == self.__xdim): self.__scaleExpos = scaleExpos_init # assumes the new input is the desired region.
        print(f"Begin: \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}, \nmaxi= {maxi}, k={self.k}, learning_rate={self.learning_rate}\n")
        for i in range(maxi):
            grad = self.__evalGradients(learning_rate, use='train')
            
            grad_norm = np.linalg.norm(grad)  # Compute gradient magnitude
            # Stop conditions: grad parallel to (1,1,...,1)
            if np.allclose(grad / grad_norm, np.ones(self.__xdim) / np.sqrt(self.__xdim)):  
                print(f"Stopping at iteration {i}: gradient is parallel to (1,1,...,1).")
                break

            result = self.__setNewExposFromGrad(grad)
            if (i<10 or i%skip_n==0 ): 
                history.append([i,round(np.dot(grad,grad),self.__scoredigits),self.scorethis(use='train'),self.scorethis(use='test')])
                print(f"i: {i}, |grad|^2={round(np.dot(grad,grad),self.__scoredigits)}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            if not result: 
                break
            
        if i==maxi-1: print(f"max iter reached. Current |grad|^2={np.dot(grad,grad)}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
        scaling_factor = self.__scaleFactors
        return history, scaling_factor
    
    def scorethis(self, scaleExpos = [], scaleFactors = [], use = 'test'):
        if len(scaleExpos)==self.__xdim :
            self.__setExpos2Scales( self.__shiftCenter(scaleExpos) )
        # elif len(scaleFactors)==self.__xdim:
        #     self.__scaleFactors = np.array(scaleFactors)
        #     self.__scaleExpos = [ round(math.log(x), 2 ) for x in scaleFactors ]
        else:
            # self.__setExpos2Scales(np.zeros(self.__xdim))
            if (len(scaleExpos)>0 or len(scaleFactors)>0) : print('Scale factors set to default values of unit (all ones). If this is not anticipated, please check your input, making sure the length of the list matches the number of features in the dataset.')
        
        sfactors = self.__scaleFactors.copy() # always start from the pre-set factors, whatever it might be
        self.__knnmodels[self.k].fit(sfactors*self.X_train, self.y_train)
        # For optimizing/tuning the scaling factors, use the train set to tune. 
        newscore = self.__knnmodels[self.k].score(sfactors*self.X_train, self.y_train) if use=='train' else self.__knnmodels[self.k].score(sfactors*self.X_test, self.y_test)
        return round(newscore,self.__scoredigits)

###### END class sknn

#%%
diabetes = sknn(data_x=df_x, data_y=df_y, k=13, classifier=False, learning_rate_init=0.4)
history,knn_scaling_factors = diabetes.optimize()
df_history = pd.DataFrame(columns=['iteration','grad_square','train_score','test_score'])
for row in history:
    df_history.loc[len(df_history)] = row

#%%
df_history

# %%
# Compare with other models
# Tree Regressor
performance = {}
from sklearn.model_selection import train_test_split
dy = df_y.values if (isinstance(df_y, pd.core.series.Series) or isinstance(df_y, pd.core.frame.DataFrame)) else df_y # if (isinstance(data_y, np.ndarray)) # the default
df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x, dy, test_size=0.5, random_state = 1)

# Normalize importance and treat negative/positive equivalently
from sklearn.tree import DecisionTreeRegressor
def normalize_coef(coefs):
    abs_sum = np.sum(np.abs(coefs))
    normalized_coefficients = coefs / abs_sum
    return np.abs(normalized_coefficients)

# Fit the model
tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(df_train_x, df_train_y)
performance['tree'] = tree_model.score(df_test_x,df_test_y)
# Feature Importance
tree_feature_importance = tree_model.feature_importances_
tree_feature_importance = normalize_coef(tree_feature_importance)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train_x)
X_test_scaled = scaler.fit_transform(df_test_x)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression(random_state=1)
logit_model.fit(X_train_scaled, df_train_y)
performance['logit'] = logit_model.score(X_test_scaled,df_test_y)
# Extract feature coefficients (importance)
logit_coefficients = logit_model.coef_[0]
logit_coefficients = normalize_coef(logit_coefficients)

# SVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# Train an SVM with a linear kernel
svm_model = SVC(kernel='linear', random_state=1)
svm_model.fit(X_train_scaled, df_train_y)
performance['svm'] = svm_model.score(X_test_scaled,df_test_y)
# Extract feature coefficients (importance)
svm_coefficients = svm_model.coef_[0]
svm_coefficients = normalize_coef(svm_coefficients)

# Lasso Regression
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1, random_state=1)
lasso_model.fit(X_train_scaled, df_train_y)
performance['lasso'] = lasso_model.score(X_test_scaled,df_test_y)
# Extract feature coefficients (importance)
lasso_coefficients = lasso_model.coef_
lasso_coefficients = normalize_coef(lasso_coefficients)

# Neural Network
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=500)
model.fit(df_train_x,df_train_y)
performance['neural network'] = model.score(df_test_x,df_test_y)


#Normalize knn_scaling_factors
knn_scaling_factors = normalize_coef(knn_scaling_factors)

# %%
performance
#%%
# Display feature importance
importance_df = pd.DataFrame({
    'Feature': df_train_x.columns,
    'KNN': knn_scaling_factors,
    'Tree': tree_feature_importance,
    'SVM' : svm_coefficients,
    'Logit': logit_coefficients,
    'Lasso' : lasso_coefficients
}).sort_values(by='KNN',ascending=False)
importance_df.head(10)

# %%
tree_importance_df = pd.DataFrame({
    'Feature': df_train_x.columns,
    'Tree': tree_feature_importance
}).sort_values(by='Tree',ascending=False)
tree_importance_df.head(10)

#%%
lasso_importance_df = pd.DataFrame({
    'Feature': df_train_x.columns,
    'Lasso' : lasso_coefficients
}).sort_values(by='Lasso',ascending=False)
lasso_importance_df.head(10)
# %%
