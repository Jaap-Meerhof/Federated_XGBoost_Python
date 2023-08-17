import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd

dataset = 'purchase-10' 
dataset_list = ['purchase-10', 'purchase-20', 'purchase-50', 'purchase-100', 'texas', 'MNIST', 'synthetic', 'Census', 'DNA']

def check_mul_paths(filename, paths):
    import pickle
    for path in paths:
        try:
            with open(path + filename, 'rb') as file:
                obj = pickle.load(file)
                return obj
        except FileNotFoundError:
            continue
    raise FileNotFoundError("File not found in all paths :(")

def check_mul_paths_csv(filename, paths):
    for path in paths:
        # print(f"testing {path+ filename + '.csv'}")
        if os.path.exists(path+ filename + '.csv'):
            return pd.read_csv(path + filename + ".csv")
    raise FileNotFoundError("File not found in all paths :(")

def take_and_remove_items(arr, size): #sshoutout to Chat-gpt
    indices = np.random.choice(len(arr), size,replace=False )
    selected_items = np.take(arr, indices, axis=0)
    arr = np.delete(arr, indices, axis=0)
    return selected_items, arr

def makeOneHot(y):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(y)
    y = encoder.transform(y).toarray()
    return y

def getPurchase(num, paths):
    #first try local
    # 
    # logger.warning(f"getting purchase {num} dataset!")

    train_size = 20_000
    test_size = 10_000
    random_state = 690
    shadow_size = 30_000 # take in mind that this shadow_set is devided in 3 sets

    def returnfunc():
        X = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_features.p', paths)
        y = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_' + str(num) + '_labels.p', paths)
        total_size = shadow_size + test_size + train_size
        if not total_size < len(X) : raise Exception(f"your don't have enough data for these settings. your original X is of size {len(X)} ")
        
        X_shadow, X = take_and_remove_items(X, shadow_size)
        y = y.reshape(-1, 1)
        y = makeOneHot(y) 
        y_shadow, y = take_and_remove_items(y, shadow_size)       
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state)
        
        fName = []
        for i in range(600):
            fName.append(str(i))
        # logger.warning(f"got purchase {num} dataset!")

        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
        
    return returnfunc

def getTexas(paths):
    def returnfunc():
        # logger.warning("getting Texas database!")
        train_size = 50_000
        test_size = 30_000
        random_state = 69
        shadow_size = 30_000    

        X = check_mul_paths('texas/' + 'texas_100_v2_features.p', paths)
        X = np.array(X)
        y = check_mul_paths('texas/' + 'texas_100_v2_labels.p', paths)
        # fName = check_mul_paths('texas/' + 'texas_100_v2_feature_desc.p', paths)
        fName = ['THCIC_ID', 'SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', \
             'LENGTH_OF_STAY', 'PAT_AGE', 'PAT_STATUS', 'RACE', 'ETHNICITY', \
                'TOTAL_CHARGES', 'ADMITTING_DIAGNOSIS']
        total_size = shadow_size + test_size + train_size
        if not total_size < len(X) : raise Exception(f"your don't have enough data for these settings. your original X is of size {len(X)} ")
        
        X_shadow, X = take_and_remove_items(X, shadow_size)
        y = y.reshape(-1, 1)
        y = makeOneHot(y)
        y_shadow, y = take_and_remove_items(y, shadow_size)       
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state)
        # logger.warning("got Texas database!")
    
        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
    return returnfunc
POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
# getTexas(POSSIBLE_PATHS)

def getMNIST(paths):
    return

def getSynthetic():
    train_size = 50_000
    test_size = 30_000
    random_state = 420
    shadow_size = 30_000    
    
    def returnfunc():
        """returns ndarrays with all X types and y where y is One Hot encoded

        Returns:
            _type_: _description_
        """
        from sklearn.datasets import make_classification
        n_features = 8
        X, y = make_classification(n_samples=train_size+test_size+shadow_size, n_features=n_features, n_informative=5, n_redundant=0, n_clusters_per_class=1, class_sep=1.0, n_classes=4, random_state=random_state)
        y = y.reshape(-1, 1)
        y = makeOneHot(y)
        fName = [str(i) for i in range(0, n_features)]
        X_train, X_test_shadow, y_train, y_test_shadow = train_test_split(X, y, train_size=train_size, test_size=test_size+shadow_size, random_state=random_state)
        X_shadow, X_test, y_shadow, y_test = train_test_split(X_test_shadow, y_test_shadow, train_size=shadow_size, test_size=test_size, random_state=random_state)
        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
    return returnfunc

def getCensus(paths):
    return

def getDNA(paths):
    return

def getHealthcare(paths, federated=False): # https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii
    """retireves the Healtcare dataset from kaggle https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii
    general information:
    nFeatures = 16 
    nClasses = 11
    nUsers = ~318k
    I'm not using test_data.csv and sample_sub.cvs as there it is only testing the classification of 0-10 days y/n?

    for a federated data retrieval 
    Args:
        paths (_type_): _description_
        federated (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    train_size = 30_000
    test_size = 10_000
    random_state = 420
    shadow_size = train_size*2

    def returnfunc():
        train = check_mul_paths_csv("AV_HealthcareAnalyticsII/train_data", paths)
        test = check_mul_paths_csv("AV_HealthcareAnalyticsII/test_data", paths)
        dict = check_mul_paths_csv("AV_HealthcareAnalyticsII/train_data_dictionary", paths)
        sample = check_mul_paths_csv("AV_HealthcareAnalyticsII/sample_sub", paths)
        non_continuous = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age', 'Stay']
        train = train.dropna()

        for featureName in non_continuous:
            train[featureName] = train[featureName].factorize()[0]

        # train[strings] = train[strings].apply(lambda x: pd.factorize(x)[0])
        # train = train.apply(lambda x: pd.factorize(x)[0])
        fName = train.columns.tolist()[1:17]
        X = train.values[:, 1:17]
        y = makeOneHot(y = train.values[:, 17].reshape(-1,1))
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        # X_test = test.values[:test_size, 1:]
        # y_test = sample.values[:test_size, 1]
        X_test = X[train_size:train_size+test_size]
        y_test = y[train_size:train_size+test_size]

        if federated:
            size = X.shape[0]
            begin = train_size+test_size
            X_shadow = [X[i:i+shadow_size] for i in range(begin, size-shadow_size, shadow_size)]
            y_shadow = [y[i:i+shadow_size] for i in range(begin, size-shadow_size, shadow_size)]
        else:
            X_shadow = X[train_size+test_size:train_size+test_size+shadow_size]
            y_shadow = y[train_size+test_size:train_size+test_size+shadow_size]
        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow

    # data = np.genfromtxt(paths + "AV_HealthcareAnalyticsII/train_data.csv")
    return returnfunc

POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
# getHealthcare(POSSIBLE_PATHS)()

def getDataBase(dataBaseName, paths, federated=False):
    """After setting the database in the config, this will retrieve the database
    """
    get_databasefunc = {'purchase-10': getPurchase(10, paths, federated), 'purchase-20':getPurchase(20, paths, federated), 
                    'purchase-50':getPurchase(50, paths, federated), 'purchase-100':getPurchase(100, paths, federated), 
                    'texas':getTexas(paths, federated), 'healthcare':getHealthcare(paths, federated), 'MNIST':getMNIST(paths, federated), 'synthetic':getSynthetic(federated), 
                    'Census':getCensus(paths, federated), 'DNA':getDNA(paths, federated)
                   }[dataBaseName]
    return get_databasefunc

def getConfigParams(dataBaseName): # retreive n_classes, n_features
    """shamefully hardcoded nClasses and nFeatures retriever. 
    

    Args:
        dataBaseName (str): name of the datset

    Returns:
        tuple(int, int): tuple of (nClasses, nFeatures)
    """
    get_databasefunc = {'purchase-10': (10, 600), # nClasses, nFeatures
                        'purchase-20': (20, 600), 
                        'purchase-50': (50, 600), 
                        'purchase-100':(100, 600), 
                        'texas':(100, 11), 
                        'healthcare':(11, 16),
                        'MNIST':(-1, -1), 
                        'synthetic':(4, 8), 
                        'Census':(-1, -1), 
                        'DNA':(-1, -1)
                   }[dataBaseName]
    return get_databasefunc[0], get_databasefunc[1]