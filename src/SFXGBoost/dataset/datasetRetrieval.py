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

def take_and_remove_items(arr, size, seed=0): #sshoutout to Chat-gpt
    np.random.seed(seed)
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

def getPurchase(num, paths, federated=False):
    #first try local
    # 
    # logger.warning(f"getting purchase {num} dataset!")

    train_size = 20_000
    test_size = 10_000
    random_state = 690
    shadow_size = 30_000 # take in mind that this shadow_set is devided in 3 sets
    n_shadows = 10
    def returnfunc():
        X = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_features.p', paths)
        y = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_' + str(num) + '_labels.p', paths)
        y = y.reshape(-1, 1)
        y = makeOneHot(y) 
        fName = []
        for i in range(600):
            fName.append(str(i))
        return split_D((X, y), federated, train_size, n_shadows, fName)        
    return returnfunc

def getTexas(paths, federated=False):
    """
    size = 925_128
    features = 11
    n_classes = 100


    Args:
        paths (_type_): _description_
        federated (bool, optional): _description_. Defaults to False.
    """
    def returnfunc():
        # logger.warning("getting Texas database!")
        train_size = 50_000
        n_shadows = 10

        
        X = check_mul_paths('texas/' + 'texas_100_v2_features_2006.p', paths)
        X = np.array(X)
        shape = np.shape(X) # 2516991, 11 2006 = 925128, 11
        y = check_mul_paths('texas/' + 'texas_100_v2_labels_2006.p', paths) 
        # fName = check_mul_paths('texas/' + 'texas_100_v2_feature_desc.p', paths)
        fName = ['THCIC_ID', 'SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', \
             'LENGTH_OF_STAY', 'PAT_AGE', 'PAT_STATUS', 'RACE', 'ETHNICITY', \
                'TOTAL_CHARGES', 'ADMITTING_DIAGNOSIS']
        X = X[100_000:]
        y = y[100_000:]
        y = y.reshape(-1, 1)
        # print(np.unique(y[:train_size]))
        y = makeOneHot(y)
        
        return split_D((X,y), federated, train_size, n_shadows, fName)
    return returnfunc


POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
# X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getTexas(POSSIBLE_PATHS)()


def getMNIST(paths, federated=False):
    return

def getSynthetic(federated=False):
    train_size = 2_000
    test_size = 2_000
    random_state = 420
    shadow_size = train_size*2   
    n_shadows = 10

    if federated:
        shadow_size = train_size //2
    else:
        shadow_size = train_size

    def returnfunc():
        """returns ndarrays with all X types and y where y is One Hot encoded

        Returns:
            _type_: _description_
        """
        from sklearn.datasets import make_classification
        n_features = 8
        X, y = make_classification(n_samples=train_size+test_size+(shadow_size * n_shadows), n_features=n_features, n_informative=5, n_redundant=0, n_clusters_per_class=1, class_sep=1.0, n_classes=4, random_state=random_state)
        y = y.reshape(-1, 1)
        y = makeOneHot(y)
        fName = [str(i) for i in range(0, n_features)]

        return split_D((X,y), federated, train_size, n_shadows, fName)
    return returnfunc
# X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getSynthetic(False)()
# x = 1
def getCensus(paths, federated=False): # binary issue
    return

def getDNA(paths, federated=False):
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
    
    train_size = 2_000
    test_size = 2_000
    n_shadows = 10
    random_state = 420
    shadow_size=0
    if federated:
        shadow_size = train_size//2 # * n_shadows + test_size 
    else: 
        shadow_size = train_size
    # A MINIMUM OF 4 SHADOWS ARE NEEDED!!!
    def returnfunc():
        train = check_mul_paths_csv("AV_HealthcareAnalyticsII/train_data", paths)
        test = check_mul_paths_csv("AV_HealthcareAnalyticsII/test_data", paths)
        dict = check_mul_paths_csv("AV_HealthcareAnalyticsII/train_data_dictionary", paths)
        sample = check_mul_paths_csv("AV_HealthcareAnalyticsII/sample_sub", paths)
        non_continuous = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age', 'Stay']
        train = train.dropna()

        for featureName in non_continuous:
            train[featureName] = train[featureName].factorize()[0]  # string to int. 

        # train[strings] = train[strings].apply(lambda x: pd.factorize(x)[0])
        # train = train.apply(lambda x: pd.factorize(x)[0])
        
        fName = train.columns.tolist()[1:17]
        X = train.values[:, 1:17]
        y = makeOneHot(y = train.values[:, 17].reshape(-1,1))

        return split_D((X,y), federated, train_size, n_shadows, fName)

    # data = np.genfromtxt(paths + "AV_HealthcareAnalyticsII/train_data.csv")
    return returnfunc

def split_D(D, federated, train_size, n_shadows, fName):

    X = D[0]
    y = D[1]
    total_size = X.shape[0]

    if federated:
        assert (train_size*2) + ((train_size//2) * n_shadows) <= total_size
    else:
        assert train_size*4 <= total_size

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:train_size*2]
    y_test = y[train_size:train_size*2]
    if federated:
        begin = train_size*2
        shadow_size = train_size//2
        X_shadow = [X[i:i+shadow_size] for i in range(begin, begin+(shadow_size*n_shadows), train_size//2)]
        y_shadow = [y[i:i+shadow_size] for i in range(begin, begin+(shadow_size*n_shadows), train_size//2)]
    else:
        X_shadow = X[train_size*2:train_size*4]
        y_shadow = y[train_size*2:train_size*4]


    return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
# X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getHealthcare(POSSIBLE_PATHS, True)()
pass
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