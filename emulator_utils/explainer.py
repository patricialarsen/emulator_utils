from alibi.explainers import KernelShap
from alibi.explainers import IntegratedGradients
from alibi.explainers import ALE
from alibi.explainers import plot_ale

import shap


#shap.initjs()

def local_explainer(model, training_data, test_data, input_names, output_names):

    predictor = model.predict
    
    ex = shap.KernelExplainer( predictor, X_train, features = input_names)
    shap_values = ex.shap_values(X_test[0:nb_samples, :])



    explainer = shap.KernelExplainer(predictor, X_train, features = input_names, out_names = target_names)
    shap_valuesKE = explainer.shap_values(X_test)


    p1 = shap.summary_plot(shap_valuesKE[0], X_test, feature_names = input_names, plot_type='violin')
    p2 = shap.summary_plot(shap_valuesKE[1], X_test, feature_names = input_names, plot_type='violin')

    p3 = shap.force_plot(ex.expected_value[out_id], shap_values[out_id][4], feature_names = input_names, out_names = target_names[out_id])
    shap.force_plot(ex.expected_value[out_id], shap_values[out_id], feature_names = input_names, out_names = target_names[out_id])



    return p1, p2, p3


def global_explainer(model, training_data, test_data, input_names, output_names):

    ale = ALE(predictor, feature_names=input_names, target_names= target_names)
    ex = ale.explain(X_train)

    p1 = plot_ale(ex, features=input_names, ax = ax, sharey=True, constant=False)

    return p1



    



