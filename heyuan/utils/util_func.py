import io
import sys
from torchsummary import summary


def ModelSummary(model,input_size,decive='cpu'):
    """
    打印模型
    :param model: 要打印的模型
    :param input_size: 输入网络
    :param decive:  cuda/cpu
    """

    output_buffer = io.StringIO()
    # 重定向标准输出
    sys.stdout = output_buffer

    summary(model, input_size=input_size, device=decive)

    # 恢复标准输出
    sys.stdout = sys.__stdout__

    # 从缓存中获取摘要信息的字符串
    summary_str = output_buffer.getvalue()

    return summary_str


def ModelSummary_MachineLearing(model):
    """
    打印机器学习模型
    :param model: 要打印的模型
    :param input_size: 输入网络
    """

    if hasattr(model, 'rf_model'):
        summary_str = model.rf_model.feature_importances_
    elif hasattr(model, 'svm_model'):
        summary_str = model.svm_model.get_params()



    return summary_str



def ModelSummary_StatisticalLearning(model):
    """
    打印机器学习模型
    :param model: 要打印的模型
    :param input_size: 输入网络
    """

    if hasattr(model, 'arima_model'):
        summary_str = model.arima_model.summary()
    elif hasattr(model, 'ES_model'):
        summary_str = model.ES_model.summary()



    return summary_str