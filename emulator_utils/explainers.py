"""
explainers.py
============
something

"""

import alibi
from alibi.explainers import KernelShap
from alibi.explainers import IntegratedGradients
from alibi.explainers import ALE
from alibi.explainers import plot_ale
import matplotlib.pylab as plt

import shap

shap.initjs()


__all__ = ("shap_estimate", "plot_shap_summary_single", "plot_shap_summary_multiple", "plot_shap_force_single", "plot_shap_force_multiple", "global_explainer", )

#### local interpreters ######


def shap_estimate(model, training_data, test_data, input_names, output_names):  
    predictor = model.predict
    explainer = shap.KernelExplainer(predictor, training_data, features = input_names, out_names = output_names)
    shap_values = explainer.shap_values(test_data)
    expected_values = explainer.expected_value

    return explainer, shap_values, expected_values

def plot_shap_summary_single(shap_values_single, test_data, input_names, plot_type):   
    p1 = shap.summary_plot(shap_values_single, test_data, feature_names = input_names, plot_type=plot_type)

    
def plot_shap_summary_multiple(shap_values, test_data, input_names, plot_type):
    p2 = shap.summary_plot(shap_values, test_data, feature_names = input_names, plot_type=plot_type)


def plot_shap_force_single(expected_values, shap_values, input_names, output_names, out_id, test_id):
    # predictor = model.predict
    # explainer = shap.KernelExplainer(predictor, training_data, features = input_names, out_names = output_names)
    p3 = shap.force_plot(expected_values[out_id], shap_values[out_id][test_id], feature_names = input_names, out_names = output_names[out_id])
    return p3

def plot_shap_force_multiple(expected_values, shap_values, input_names, output_names, out_id):
    # predictor = model.predict
    # explainer = shap.KernelExplainer(predictor, training_data, features = input_names, out_names = output_names)
    # out_id = 0                                                                                                                                   
    p4 = shap.force_plot(expected_values[out_id], shap_values[out_id], feature_names = input_names, out_names = output_names[out_id])
    return p4




##### global interpreters #####

def global_explainer(model, training_data, test_data, input_names, output_names):
    """
    global explaination using Accumulated Local explainer model

    Parameters
    ----------
    model: float
        explain
    training_data: float
        explain
    test_data: float
        explain
    input_names: float
        explain
    output_names: float
        explain

    Returns
    -------
    p1: float
        explain

    """
    predictor = model.predict
    ale = ALE(predictor, feature_names=input_names, target_names=output_names)
    explainer = ale.explain(training_data)


    if (len(input_names)> 1):

        fig, ax = plt.subplots(1, len(input_names), figsize=(4*len(input_names), 3.5), sharey=True)
        
        for axis_ind in range( len(input_names)):
            ax[axis_ind].set_xlabel(input_names[axis_ind], fontsize=18)
        
        ax[axis_ind].set_ylabel(ALE, fontsize=18)

        p5 = plot_ale(explainer, features=input_names, ax = ax, sharey=True, constant=False)


    else:

        p5 = plot_ale(explainer, features=input_names, constant=False)

    return ale, p5



    



