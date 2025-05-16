import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
# Using Intel distribution to improve speed
patch_sklearn()
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mutual_info_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from scipy.stats import pearsonr
import openml
from sklearn.datasets import fetch_openml
from collections import Counter
import time
import csv

def load_dataset(name):
    """
    Returns data from chosen dataset as NumPy arrays.

    Args:
        name (str): Name of dataset.

    Returns:
        X: 2D mxn NumPy array; m data points and n features.
        y: 1D m NumPy array with corresponding class labels of data points in X.
    """
    try:
        sets = openml.datasets.list_datasets(output_format="dataframe")
        id = sets[sets["name"] == name].iloc[0]["did"]
        dataset = openml.datasets.get_dataset(id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    except:
        X, y = fetch_openml(name=name, version=1, return_X_y=True, as_frame=True)

    # Convert y to numeric value
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X.to_numpy(), y
    
def conditional_MI(X_binned, y, i, j):
    """
    Returns MI(f_i, y | f_j) using discretization. Calculates
    \sum_{x,y}p(x,y|z)\log\left(\frac{p(x,y|z)}{p(x|z)p(y|z)}\right)
    """
    xi = X_binned[:, i] # feature i
    xj = X_binned[:, j] # feature j
    # Total number of samples
    n = len(y)

    cmi = 0.
    xj_vals = np.unique(xj) # unique feature j values

    for val in xj_vals:
        mask = xj == val
        n_j = np.sum(mask) # number of samples with f_j = val
        
        mi = mutual_info_score(X_binned[mask, i], y[mask])

        # filter rows to only include rows where x_j = val
        xi_sub = xi[mask]
        y_sub = y[mask]
        joint = Counter(zip(xi_sub, y_sub)) # counts how often (f_i, y) appear in dataset and returns a dict with (f_i, y):count
        px = Counter(xi_sub) # counts how often each f_i appears in dataset
        py = Counter(y_sub) # counts how often each y appears in dataset

        for (xi_val, y_val), joint_count in joint.items():
            prob_xi_y = joint_count / n_j # Calculating p(f_i,y|f_j)
            p_xi = px[xi_val] / n_j # Calculating p(f_i|f_j)
            p_y = py[y_val] / n_j # Calculating p(y|f_j)

            if prob_xi_y > 0 and p_xi > 0 and p_y > 0:
                cmi += (n_j/n) * prob_xi_y * np.log(prob_xi_y/(p_xi * p_y))

    return cmi

def MI_matrix(X, y, alpha, beta):
    """
    Returns Q matrix with entries representing MI (Mutual Information) of features given dataset and hyperparameters
    alpha and k. Operation is currently O(n^2 * m) where we have n features and m datapoints (Very Slow).
    
    Args:
        X (np.ndarray[np.ndarray[float]]):
        y (np.ndarray[int]):
        alpha (float): Bias of importance and independence parts (between 0 and 1).
        beta (float): Scaling constant for importance.

    Returns:
        Nxn Q matrix.
    """
    start = time.time()
    
    n_features = X.shape[1]
    Q = np.zeros((n_features, n_features))

    # Discretize features since MI estimators require discrete inputs
    bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_binned = bins.fit_transform(X)

    # Compute feature importance vector using sklearn: MI(f_i, y)
    mi_f_y = mutual_info_classif(X, y)

    np.fill_diagonal(Q, -beta*alpha*mi_f_y) # Fill diagonal of Q matrix with importance vector

    # Compute feature correlation in non-diagonal elements: MI(f_i, f_j)
    # Note that MI(f_i, f_j) = MI(f_j, f_i) which is why we can set Q[i, j] = Q[j, i]
    for i in range(n_features):
        for j in range(i+1, n_features):
            # Ensuring symmetry
            Q[i, j] = Q[j, i] = (1 - alpha) * mutual_info_score(X_binned[:, i], X_binned[:, j])

    runtime = time.time() - start
    print(f"Time Elapsed: {runtime}")
    return Q

def pearson_matrix(X, y, alpha, beta):
    """
    Returns Pearson matrix with entries representing the Pearson coefficient of features given dataset and hyperparameters
    alpha and k.
    Args:
        X (np.ndarray[np.ndarray[]]):
        y (np.ndarray[int]):
        alpha (float): Bias of importance and independence parts (between 0 and 1).
        k (int): Number of features to select.
    
    Returns:
        Nxn Q matrix.
    """

    start = time.time()
    
    n_features = X.shape[1] # Number of features
    Q = np.zeros((n_features, n_features))
    no_use = []

    for i in range(n_features):
        if np.std(X[:, i]) == 0: # Features that do not contribute
            no_use.append(i) # Features that do not contribute
            Q[i, i] = 0
        else:
            corr, _ = pearsonr(X[:, i], y)
            Q[i, i] = - alpha* beta * corr

    corr_matrix = np.corrcoef(X.T)
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            if i in no_use or j in no_use:
                Q[i, j] = Q[j, i] = 0
            else:
                Q[i, j] = (1 - alpha) * corr_matrix[i,j]
                Q[j, i] = Q[i, j]

    runtime = time.time() - start
    print(f"Time Elapsed: {runtime}")
    return Q

def preprocess_data(feature_classes, df_train, df_test):
    """
    Returns updated feature class indices, training dataset, and testing dataset after
    preprocessing (dropping NaN values, removing constant features, One-Hot Encoding category variables, etc.).

    Args:
        feature_classes (dictof[Str, List[int, int]]): Dictionary of feature classes and their corresponding
            start and end column indices of features.
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.

    Returns:
        Updated feature_classes, df_train, and df_test after data preprocessing.
    """
    print("Dropping duplicates and nan values...")
    # Dropping rows with NaN values
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    # Dropping duplicate rows
    df_train = df_train.drop_duplicates()
    df_test = df_test.drop_duplicates()

    # Define class label mappings
    attack_mapping = {
        'normal': 0,  # Normal Access
        'dos': 1,  # Denial of Service Attack
        'u2r': 2,  # User to Root Attack
        'r2l': 3,  # Remote to Local Attack
        'probe': 4   # Probing Attack
    }

    def map_attack_category(label):
        """
        Map labels to label classes.
        """
        if label == "normal":
            return attack_mapping["normal"] # Normal
        elif label in ["back", "land", "neptune", "pod", "smurf", "teardrop"]:
            return attack_mapping["dos"]  # DoS
        elif label in ["buffer_overflow", "loadmodule", "perl", "rootkit"]:
            return attack_mapping["u2r"]  # U2R
        elif label in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"]:
            return attack_mapping["r2l"]  # R2L
        else:
            return attack_mapping["probe"] # Probe
        
    # Apply the label class mapping
    df_train["label"] = df_train["label"].apply(map_attack_category)
    df_test["label"] = df_test["label"].apply(map_attack_category)

    # Getting rid of features with constant values
    to_keep = [col for col in df_train.columns if df_train[col].nunique() > 1]

    dropped = set(df_train.columns) - set(to_keep)
    print(f"Dropping constant feature(s): {dropped}")

    feature_classes_updated = feature_classes.copy()

    offset = 0
    # Adjusting feature class indices
    for ind, [name, (start, end)] in feature_classes_updated.items():
        # Features dropped within class
        dropped_within_class = [col for col in dropped if start <= df_train.columns.get_loc(col) < end]
        count = len(dropped_within_class)
        
        new_start = start - offset
        new_end = end - count - offset
        feature_classes_updated[ind] = [name, (new_start, new_end)]
        offset += count

    df_train = df_train[to_keep]
    df_test = df_test[to_keep]

    df_test = df_test.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)

    # Applying One-Hot Encoding to features of categorical type
    # Detecting categorical features
    categorical_features = df_train.select_dtypes(include=["object", "category"]).columns
    categorical_features = categorical_features[categorical_features != "label"]

    encoder = OneHotEncoder(dtype="int32")
    feat_indices = {}

    print(f"One-hot encoding: {list(categorical_features)}")

    # Updating feature class indices after One-Hot Encoding
    for feat in categorical_features:
        # Original location of feature
        index = df_train.columns.get_loc(feat)

        encoded_train = encoder.fit_transform(df_train[[feat]]).toarray()
        encoded_test = encoder.transform(df_test[[feat]]).toarray()

        # New One-Hot Encoded features
        feature_names = encoder.get_feature_names_out([feat])

        # Convert to DataFrame
        encoded_train_df = pd.DataFrame(encoded_train, columns=feature_names, index = df_train.index)
        encoded_test_df = pd.DataFrame(encoded_test, columns=feature_names, index = df_test.index)

        # Drop original categorical column
        df_train = df_train.drop(columns=[feat])
        df_test = df_test.drop(columns=[feat])

        # Add new one-hot encoded features in place
        df_train = pd.concat([df_train.iloc[:, :index], encoded_train_df, df_train.iloc[:, index:]], axis=1)
        df_test = pd.concat([df_test.iloc[:, :index], encoded_test_df, df_test.iloc[:, index:]], axis=1)

        num_encoded = len(feature_names) - 1
        print(f"Num features added to {feat}:", num_encoded)

        feat_indices[feat] = (index, index + num_encoded + 1)

        # Updating feature class indices
        for ind, [name, (start, end)] in feature_classes_updated.items():
            if start > index:
                feature_classes_updated[ind] = [name, (start + num_encoded, end + num_encoded)]
            elif start <= index < end:
                feature_classes_updated[ind] = [name, (start, end + num_encoded)]

    return feature_classes_updated, feat_indices, df_train, df_test

def get_selected_features(x, cols, feat_indices, feature_classes):
    """
    Returns and prints features used in solution x, taking into account one-hot encoded
    features.
    """
    print("**Selected Features**")
    x_collapsed = []

    # Mapping x to x_collapsed by collapsing one-hot encoded features
    for ind in range(len(cols)):
        if ind >= len(cols):
            break
        collapsed = False
        for value in feat_indices.values():
            start, end = value
            if start == ind:
                collapsed = True
                bin = x[ind]
                ind = end - 1
                break
        if not collapsed:
            bin = x[ind]
            x_collapsed.append(bin)

    for ind, [name, (start, end)] in feature_classes.items():
        count = int(sum(x_collapsed[start:end]))
        print(f"{name} ({ind}) class: {count} original features selected")
    
    num_selected = sum(x)
    print(f"Total one-hot encoded features selected: {sum(x)}")
    print(f"Total original features selected: {sum(x_collapsed)}")
    print(f"{(len(x) - num_selected) / len(x) * 100}% total feature reduction\n")

    return x_collapsed

def randomforest_small(X_train, X_test, y_train, y_test):
    """
    Returns accuracy score of Random Forest classifier on given dataset. Used for small exxample.
    """
    start = time.time()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    runtime = time.time() - start

    y_pred = clf.predict(X_test)
    print(f"Training time: {runtime} s")

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted', zero_division=0), recall_score(y_test, y_pred, average='weighted')

def classify(param_grid, X_train, X_test, y_train, y_test, classifier):
    """
    Returns accuracy score of Random Forest classifier on given dataset. Used for Network Intrusion example.
    """
    skf = StratifiedKFold(shuffle=True, random_state=42)
    
    if classifier == "rf":
        c = RandomForestClassifier(random_state=42, class_weight="balanced") # RF classifier, taking into account class frequencies to not prefer certain classes
    elif classifier == "svc":
        c = SVC()
    else:
        print("Invalid Classifier!")
        return

    # Scales numerical features in pipeline
    pipeline = Pipeline([
        ('scalar', StandardScaler()),
        (classifier, c)
    ])

    # Searching for the best hyperparameters
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=20, cv=skf, scoring="accuracy", n_jobs=-1, verbose=1, random_state=42)
    start = time.time()
    random_search.fit(X_train, y_train)
    runtime = time.time() - start

    # printing best model parameters
    print("Best Params:", random_search.best_params_)
    print("Best CV Score:", random_search.best_score_)

    # Evaluate on test set
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test Precision:", precision_score(y_test, y_pred, average="weighted"))
    print("Test Recall:", recall_score(y_test, y_pred, average="weighted"), "\n")

    print(f"Total Training Time: {runtime} s\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy_score(y_test, y_pred)

def load_hyperparameters(dataset:str, alpha:float, K:int) -> tuple[float, float, float, float, int, int, bool] | None: 
    """
    Returns optimal hyperparameters for the TitanQ solver for specified instances. 

    Args:
        dataset (str): name of the dataset
        alpha (float): Weighting of importance vs independence when selecting
        K (int): number of features to select

    Returns:
        float: Minimum temperature hyperparameter used for setting up beta values
        float: Maximum temperature hyperparameter used for setting up beta values
        float: Coupling multiplier hyperparameter used for configuring TitanQ solver
        float: Penalty scaling hyperparameter used for configuring TitanQ solver
        int:   Number of chains hyperparameter for configuring TitanQ solver 
        int:   Number of engines hyperparameter for configuring TitanQ solver 
        bool:  Whether solver uses equality constraint on number of features for TitanQ solver
    """
    with open('dataset_hyperparameters.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None) #Skip header row

        hyperparameters_dict = {}
        for row in reader:
            hyperparameters_dict[(str(row[0]), float(row[1]), int(row[2]))] = row[3:]

    if (dataset, alpha, K) in hyperparameters_dict:
        T_min = float(hyperparameters_dict[(dataset, alpha, K)][0])
        T_max = float(hyperparameters_dict[(dataset, alpha, K)][1])
        coupling_mult = float(hyperparameters_dict[(dataset, alpha, K)][2])
        penalty_scaling = float(hyperparameters_dict[(dataset, alpha, K)][3])
        num_chains = int(hyperparameters_dict[(dataset, alpha, K)][4])
        num_engines = int(hyperparameters_dict[(dataset, alpha, K)][5])
        timeout = int(hyperparameters_dict[(dataset, alpha, K)][6])
        uses_equality = bool(int(hyperparameters_dict[(dataset, alpha, K)][7]))
        return T_min, T_max, coupling_mult, penalty_scaling, num_chains, num_engines, timeout, uses_equality
    else:
        return None