import pickle
import torch

def model_saved(model_name,model,input_data = None):


        # 1、pkl类型  适用于统计和机器学习
    if model_name in ['ARIMA','ExponentialSmoothing','SVR','RandomForestRegressor']:
        with open("../saved_models/" + model_name + '.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        # 2、pth类型
        torch.save(model, "../saved_models/" + model_name + '.pth')

    # 3、h5类型
    # torch.save(model, 'model.h5')

    # 4、pt类型
    # torch.save(model, 'model_name' + '.pt')


    # 5、onnx类型
    # assert input_data != None
    # torch.onnx.export(model, input_data, model_name + '.onnx')



def model_loaded(file_path,model_name):



    if model_name in ['ARIMA', 'ExponentialSmoothing', 'SVR', 'RandomForestRegressor']:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
    else:
        # 2、pth类型
        model = torch.load(file_path)

    return model







