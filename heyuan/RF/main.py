import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(44)

print("Numpy version: ", np.__version__)
print("Pandas version: ", pd.__version__)
print("Scikit-learn version: ", sklearn.__version__)


# box-plot
train_data = pd.read_csv("../ARIMA/train_FD001.txt", sep= "\s+", header = None)
plt.figure(figsize = (16, 21))
for i in range(21):
    temp_data = train_data.iloc[:,i+5]
    plt.subplot(7,3,i+1)
    plt.boxplot(temp_data)
plt.show()



def process_targets(data_length, early_rul = None):
    """
    Takes datalength and earlyrul as input and
    creates target rul.
    """
    if early_rul == None:
        return np.arange(data_length-1, -1, -1)
    else:
        early_rul_duration = data_length - early_rul
        if early_rul_duration <= 0:
            return np.arange(data_length-1, -1, -1)
        else:
            return np.append(early_rul*np.ones(shape = (early_rul_duration,)), np.arange(early_rul-1, -1, -1))


def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    """Depending on values of window_length and shift, this function generates batchs of data and targets
    from input_data and target_data.

    Number of batches = np.floor((len(input_data) - window_length)/shift) + 1

    **We don't check input dimensions uisng exception handling. So readers should be careful while using these
    functions. If input data are not of desired dimension, either error occurs or something undesirable is
    produced as output.**

    Arguments:
        input_data: input data to function (Must be 2 dimensional)
        target_data: input rul values (Must be 1D array)s
        window_length: window length of data
        shift: Distance by which the window moves for next batch. This is closely related to overlap
               between data. For example, if window length is 30 and shift is 1, there is an overlap of
               29 data points between two consecutive batches.

    """
    num_batches = np.int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(num_batches,
                                                                                                window_length,
                                                                                                num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets


def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    """ This function takes test data for an engine as first input. The next two inputs
    window_length and shift are same as other functins.

    Finally it takes num_test_windows as the last input. num_test_windows sets how many examplles we
    want from test data (from last). By default it extracts only the last example.

    The function return last examples and number of last examples (a scaler) as output.
    We need the second output later. If we are extracting more than 1 last examples, we have to
    average their prediction results. The second scaler halps us do just that.
    """
    max_num_test_batches = np.int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, num_test_windows


test_data = pd.read_csv("../ARIMA/test_FD001.txt", sep="\s+", header=None)
true_rul = pd.read_csv('../ARIMA/RUL_FD001.txt', sep='\s+', header=None)

window_length = 1
shift = 1
early_rul = 125
processed_train_data = []
processed_train_targets = []

# How many test windows to take for each engine. If set to 1 (this is the default), only last window of test data for
# each engine are taken. If set to a different number, that many windows from last are taken.
# Final output is the average of output of all windows.
num_test_windows = 5
processed_test_data = []
num_test_windows_list = []

columns_to_be_dropped = [0, 1, 2, 3, 4, 5, 9, 10, 14, 20, 22, 23]

num_machines = np.min([len(train_data[0].unique()), len(test_data[0].unique())])

for i in np.arange(1, num_machines + 1):

    temp_train_data = train_data[train_data[0] == i].drop(columns=columns_to_be_dropped).values
    temp_test_data = test_data[test_data[0] == i].drop(columns=columns_to_be_dropped).values

    # Verify if data of given window length can be extracted from both training and test data
    if (len(temp_test_data) < window_length):
        print("Test engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")
    elif (len(temp_train_data) < window_length):
        print("Train engine {} doesn't have enough data for window_length of {}".format(i, window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    temp_train_targets = process_targets(data_length=temp_train_data.shape[0], early_rul=early_rul)
    data_for_a_machine, targets_for_a_machine = process_input_data_with_targets(temp_train_data, temp_train_targets,
                                                                                window_length=window_length,
                                                                                shift=shift)

    # Prepare test data
    test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length=window_length, shift=shift,
                                                             num_test_windows=num_test_windows)

    processed_train_data.append(data_for_a_machine)
    processed_train_targets.append(targets_for_a_machine)

    processed_test_data.append(test_data_for_an_engine)
    num_test_windows_list.append(num_windows)

processed_train_data = np.concatenate(processed_train_data)
processed_train_targets = np.concatenate(processed_train_targets)
processed_test_data = np.concatenate(processed_test_data)
true_rul = true_rul[0].values

# Shuffle data
index = np.random.permutation(len(processed_train_targets))
processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index]

print("Processed trianing data shape: ", processed_train_data.shape)
print("Processed training ruls shape: ", processed_train_targets.shape)
print("Processed test data shape: ", processed_test_data.shape)
print("True RUL shape: ", true_rul.shape)


processed_train_data = processed_train_data.reshape(-1, processed_train_data.shape[2])
processed_test_data = processed_test_data.reshape(-1, processed_test_data.shape[2])
print("Processed train data shape: ", processed_train_data.shape)
print("Processed test data shape: ", processed_test_data.shape)


# 初始化随机森林
rf_model = RandomForestRegressor(n_estimators= 300, max_features = "sqrt",
                                 n_jobs = -1, random_state = 38)
rf_model.fit(processed_train_data, processed_train_targets)
rul_pred = rf_model.predict(processed_test_data)


# First split predictions according to number of windows of each engine
preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows))
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)]
RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))
print("RMSE: ", RMSE)


# number setting
totalloss = []
totalpara = []
loss = []
para = []
for i in range(100,300,10):
    rf_model = RandomForestRegressor(n_estimators=i, max_features="sqrt",
                                     n_jobs=-1, random_state=38)
    rf_model.fit(processed_train_data, processed_train_targets)
    rul_pred = rf_model.predict(processed_test_data)

    # First split predictions according to number of windows of each engine
    preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
    mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights=np.repeat(1 / num_windows, num_windows))
                                 for ruls_for_each_engine, num_windows in
                                 zip(preds_for_each_engine, num_test_windows_list)]
    RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))
    print("RMSE: ", RMSE)
    loss.append(RMSE)
    para.append(i)

totalloss.append(loss.copy())
totalpara.append(para.copy())
loss.clear()
para.clear()

# 叶节点最小样本
for i in range(1,6):
    rf_model = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                     n_jobs=-1, random_state=38,min_samples_leaf=i)
    rf_model.fit(processed_train_data, processed_train_targets)
    rul_pred = rf_model.predict(processed_test_data)

    # First split predictions according to number of windows of each engine
    preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
    mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights=np.repeat(1 / num_windows, num_windows))
                                 for ruls_for_each_engine, num_windows in
                                 zip(preds_for_each_engine, num_test_windows_list)]
    RMSE = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine))
    print("RMSE: ", RMSE)
    loss.append(RMSE)
    para.append(i)

totalloss.append(loss.copy())
totalpara.append(para.copy())





# param_grid = {"n_estimators": [100, 200, 250, 300, 350, 400],
#               "max_features": ["auto", "sqrt", "log2"]}
param_grid = {"n_estimators": [300],
              "max_features": ["sqrt"]}
grid = GridSearchCV(RandomForestRegressor(), param_grid = param_grid,scoring = "neg_root_mean_squared_error",
                    n_jobs = -1, cv = 5,verbose=2)
grid.fit(processed_train_data, processed_train_targets)

best_rf_model = grid.best_estimator_
rul_pred_tuned = best_rf_model.predict(processed_test_data)

preds_for_each_engine_tuned = np.split(rul_pred_tuned, np.cumsum(num_test_windows_list)[:-1])
mean_pred_for_each_engine_tuned = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows))
                                   for ruls_for_each_engine, num_windows in zip(preds_for_each_engine_tuned,
                                                                                num_test_windows_list)]
RMSE_tuned = np.sqrt(mean_squared_error(true_rul, mean_pred_for_each_engine_tuned))
print("RMSE after hyperparameter tuning: ", RMSE_tuned)


indices_of_last_examples = np.cumsum(num_test_windows_list) - 1
preds_for_last_example = np.concatenate(preds_for_each_engine_tuned)[indices_of_last_examples]

RMSE_new = np.sqrt(mean_squared_error(true_rul, preds_for_last_example))
print("RMSE (Taking only last examples): ", RMSE_new)

def compute_s_score(rul_true, rul_pred):
    """
    Both rul_true and rul_pred should be 1D numpy arrays.
    """
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))

s_score = compute_s_score(true_rul, preds_for_last_example)
print("S-score: ", s_score)


# Plot true and predicted RUL values
plt.plot(true_rul, label = "True RUL", color = "red")
plt.plot(preds_for_last_example, label = "Pred RUL", color = "blue")
plt.xlabel('unit number')
plt.ylabel('Remaining useful life')
plt.legend(loc = 1)
plt.savefig(fname="RF趋势预测.svg",format="svg")
plt.show()



# fg,ax =plt.subplots()
# plt.plot(preds_for_last_example[np.argsort(true_rul)],'.-',label = 'pred_value')
# plt.xlabel('unit number')
# plt.ylabel('Remaining useful life')
# plt.plot(np.sort(true_rul),'.-',label = 'true_value')
# plt.legend(loc = 1)
# plt.savefig(fname="RF趋势预测2.svg",format="svg")
# plt.show()