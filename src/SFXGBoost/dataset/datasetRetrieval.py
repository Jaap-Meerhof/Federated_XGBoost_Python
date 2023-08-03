from SFXGBoost.config import rank
import numpy as np

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

    train_size = 10_000
    test_size = 10_000
    random_state = 69
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
        from sklearn.model_selection import train_test_split
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
        train_size = 30_000
        test_size = 20_000
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
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=random_state)
        # logger.warning("got Texas database!")
    
        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
    return returnfunc
POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
getTexas(POSSIBLE_PATHS)
def getMNIST(paths):
    return

def getSynthetic(paths):
    return

def getCensus(paths):
    return

def getDNA(paths):
    return


def getDataBase(dataBaseName, paths):
    """After setting the database in the config, this will retrieve the database
    """
    get_databasefunc = {'purchase-10getTexas': getPurchase(10, paths), 'purchase-20':getPurchase(20, paths), 
                    'purchase-50':getPurchase(50, paths), 'purchase-100':getPurchase(100, paths), 
                    'texas':getTexas(paths), 'MNIST':getMNIST(paths), 'synthetic':getSynthetic(paths), 
                    'Census':getCensus(paths), 'DNA':getDNA(paths)
                   }[dataBaseName]
    return get_databasefunc