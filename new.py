# Introduction to Data Science with Python
# MTA - Spring 2021-2022.
# Final Home Exercise.

# ID of  student:
# First and Last Names of student:


# In this exercise you should implement a classification pipeline which aim at predicting the amount of hours
# a worker will be absent from work based on the worker characteristics and the work day missed.
# Download the dataset from the course website, which is provided as a .csv file. The target label is 'TimeOff'.
# You are free to use as many loops as you like, and any library functions from numpy, pandas and sklearn, etc...
#
# You should implement the body of the functions below. The main two points of entry to your code are DataPreprocessor class and
# the train_model function. In the '__main__' section you are provided with an example of how your submission will be evaluated. 
# You are free to change the body of the functions and classes as you like - as long as it adheres to the provided input & output structure.
# In all methods and functions the input structure and the required returned variables are explicitly stated.
# Note that in order to evaluate the generalization error, you'll need to run cross validation as we demonstrated in classs, 
# However!!! In the final sunbmission we your file needs to contain only the methods of DataPreprocessor and the train_model function. Your 
# submision will be retrained on all the train dataset. 
# You are encouraged to run gridsearch to find the best model and hyper parameters as demonstrated in the previous exercise and class.
#
# To make thigs clear: you need to experiment with the preprocessing stage and the final model that will be used to fit. To get the 
# sense of how your model performs, you'll need to apply the CV approach and, quite possibly, do a grid search of the meta parameters. 
# In the end, when you think that you've achieved your best, you should make a clean - and runnable!!! - version of your insights, 
# which must adhere to the api provided below. Needless to say, it's better to work with the API from the get-go (from the start), to avoid 
# unnecessary bugs. In the evaluation stage, your code will be run once for training on *all* the train data, and then run once on the test data. 
#
# You are expoected to get results between 50% and 100% accuracy on the test set.
# Of course, the test set is not provided to you. Hhowever, as previously mentioned, running cross validation
# (with enough folds) will give you a good estimation of the accuracy.
#
# Important: obtaining accuracy less than 50%, will grant you 65 points for this exercise.
# Obtaining accuracy score above 50% will grant you 75 points minimum, however, your final score 
# will be according to the distribution of all submissions. Therefore, due to the competition nature of this exercise, 
# you may use any method or library that will grant you the highest score, even if not learned in class.
#
# Identical or equivalent submissions will give rise to a suspicion of plagiarism.
#
# Final submission instruction: in addition to stating your names and ID numbers in the body of this file, name the file in the following way:
#
# ex4_FirstName_LastName.py
#
# where FirstName, ... stand, naturally, for your name(s)

from sklearn import *
from cmath import nan
from re import U
import numpy as np
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB


def load_dataset(train_csv_path):
    data = pd.read_csv(train_csv_path, sep=',')
    return data


class DataPreprocessor(object):

    """ 
    This class is a mandatory API. More about its structure - few lines below.

    The purpose of this class is to unify data preprocessing step between the training and the testing stages. 
    This may include, but not limited to, the following transformations:
    1. Filling missing (NA / nan) values
    2. Dropping non descriptive columns
    3 ...

    The test data is unavailable when building the ML pipeline, thus it is necessary to determine the 
    preprocessing steps and values on the train set and apply them on the test set.


    *** Mandatory structure ***
    The ***fields*** are ***not*** mandatory
    The ***methods***  - "fit" and "transform" - are ***required***.

    You're more than welcome to use sklearn.pipeline for the "heavy lifting" of the preprocessing tasks, but it is not an obligation. 
    Any class that implements the methods "fit" and "transform", with the required inputs & outps will be accepted. 
    Even if "fit" performs no taksks at all.
    """

    def __init__(self):
      self.transformer:Pipeline = None

    def fit(self, dataset_df):


        self.filling_vals = {}
        
        median_cols = ['Transportation expense', 'Residence Distance', 'Service time', 'Weight', 'Height', 'Son', 'Pet']
        freq_cols = ['Reason','Month','Day','Education','Season','Age Group','Smoker','Drinker']
        
        for col in median_cols:
            self.filling_vals[col] = dataset_df[col].median()
            
        for col in freq_cols:
            self.filling_vals[col] = dataset_df[col].value_counts().idxmax()    
        """
        Input:
        dataset_df: the training data loaded from the csv file as a dataframe containing only the features
        (not the target - see the main function).

        Output:
        None

        Functionality:
        Based on all the provided training data, this method learns with which values to fill the NA's, 
        how to scale the features, how to encode categorical variables etc.

        *** This method will be called exactly once during evaluation. See the main section for details ***


        Note that implementation below is a boilerplate code which performs very basic categorical and numerical fields
        preprocessing.

        """
        return
    
    #-------------------------------------------------------------------------------------------------------------------------
    
    def add_log1p(self,df):
    
      for col in ['Service time']:
          df['log1p_' + col] = np.log1p(df[col])

      return df
  
    def ordinal_transformer(self, col):
        unique_vals = pd.Series(col.unique()) 
        
        def get_corr_num (val):
            
            if(val not in unique_vals.values):
                return val
            else:
                return unique_vals[unique_vals == val].index[0]

        res = col.apply(get_corr_num)
        return res
            
    def one_hot_transformer(self, df, colName):
        unique_vals = pd.Series(df[colName].unique()).dropna()

        one_hot_dict = {'Reason':28, 'Month':12,'Day':7,'Education':4,'Season':4,'Age Group':4}        
        
        for x in range(one_hot_dict[colName]):
            df[colName + "_" + str(x+1)] = 0
            df[colName + "_" + str(x+1)] = df[colName + "_" + str(x+1)].where(df[colName] != x,1)
        df.drop(labels=[colName],axis=1,inplace=True)
        return        
 #------------------------------------------------------------------------------------------------------------------------------------------ 
    def transform(self, df):
        
        df_fill = df.drop(labels=["ID"],axis=1)
        one_hot_cols = ['Reason','Month','Day','Education','Season','Age Group']
        ordinal_cols = ['Education', 'Smoker', 'Drinker', 'Age Group']
        
        median_cols = ['Transportation expense', 'Residence Distance', 'Service time', 'Weight', 'Height', 'Son', 'Pet']
        freq_cols = ['Reason','Month','Day','Education','Season','Age Group','Smoker','Drinker']
           
        
        for col in median_cols:
            df_fill[col].fillna(self.filling_vals[col],inplace=True)
        for col in freq_cols:
            df_fill[col].fillna(self.filling_vals[col],inplace=True)
              
        for col in ordinal_cols:
            df_fill[col] = self.ordinal_transformer(df_fill[col])
            
            
        for col in one_hot_cols:            
            self.one_hot_transformer(df=df_fill,colName=col)
        
        self.add_log1p(df_fill)
        
        
        important_atr = ['Transportation expense', 'Smoker', 'Pet', 'Reason_10', 'Reason_13', 'Reason_19', 'Month_10', 'Education_2', 'log1p_Service time']
        df_fill.drop(labels = list(set(df_fill.columns) - set(important_atr)), inplace=True, axis=1)
        """
        Input:
        df:  *any* data similarly structured to the train data (dataset_df input of "fit")

        Output: 
        A processed dataframe or ndarray containing only the input features (X).

        It should maintain the same row order as the input.
        Note that the labels vector (y) should not exist in the returned ndarray object or dataframe.


        Functionality:
        Based on the information learned in the "fit" method, apply the required transformations to the passed data (df)

        """
        return df_fill
        # think about if you would like to add additional computed columns.



def train_model(processed_X, y):
    """
    This function gets the data after the pre-processing stage  - after running DataPreprocessor.transform on it, 
    a vector of labels, and returns a trained model. 

    Input:
    processed_X (ndarray or dataframe): the data after the pre-processing stage
    y: a vector of labels

    Output:
    model: an object with a "predict" method, which accepts the ***pre-processed*** data and outputs the prediction


    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score 
    from sklearn.metrics import f1_score 

    para_grid = {'C' : [0.001, 0.01, 0.1, 1, 10 ,50], # internal regularization parameter of LogisticRegression
                 'solver' : ['sag', 'saga']}

    Logit1 = GridSearchCV(LogisticRegression(penalty='l2' ,random_state=1), para_grid, cv = 5)
    Logit1.fit(processed_X, y)
    
    model = GaussianNB()
    model.fit(processed_X, y)

    return Logit1

def warn(*args, **kwargs):
      pass

def split_data(df):
    from sklearn.model_selection import train_test_split

    Y = df['TimeOff']
    X = df.drop(['TimeOff'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=1, stratify = Y)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    import warnings
    warnings.warn = warn 
    
    preprocessor = DataPreprocessor()
    train_csv_path = 'time_off_data_train.csv'
    train_dataset_df = load_dataset(train_csv_path)

    
    X_train, X_test, y_train, y_test = split_data(train_dataset_df)    
    #X_train = train_dataset_df.iloc[:, :-1]
    #y_train = train_dataset_df['TimeOff']
    preprocessor.fit(X_train)
    processed_X_train = preprocessor.transform(X_train)
    #'''
    finall_df = processed_X_train[:]
    finall_df['TimeOff'] = preprocessor.ordinal_transformer(y_train)
    corr = finall_df.corr()['TimeOff']
    impotant_attr = list(corr.index[ abs(corr.values) > 0.1])
    #print(impotant_attr)
    #'''
    model = train_model(processed_X_train, y_train)


    ### Evaluation Section ####
    # test_csv_path = 'time_off_data_test.csv'
    # Obviously, this will be different during evaluation. For now, you can keep it to validate proper execution
    test_csv_path = train_csv_path
    test_dataset_df = load_dataset(test_csv_path)

    #X_test = test_dataset_df.iloc[:, :-1]
    #y_test = test_dataset_df['TimeOff']

    processed_X_test = preprocessor.transform(X_test)
    print(processed_X_test.shape)
    
    predictions = model.predict(processed_X_test)
    test_score = accuracy_score(y_test, predictions)
    print("test:", test_score)

    predictions = model.predict(preprocessor.transform(X_train))
    test_score = accuracy_score(y_train, predictions)
    print('train:', test_score)
