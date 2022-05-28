import shap
import numpy as np
import matplotlib.pyplot as plt


def shap_tensor(model, X, val_labels, classes, fig_dir):

    explainer = shap.DeepExplainer(model, X)
    shap_values, indexes = explainer.shap_values(X, ranked_outputs=1, output_rank_order = 'max')

    shap_numpy = [np.swapaxes(np.swapaxes(s,1,-1),1,2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(X.cpu().numpy(),1,-1),1,2)

    # get names for the classes
    index_names = np.vectorize(lambda x: classes[str(x)])(indexes)
    val_labels_names = np.vectorize(lambda x: classes[str(x)])(val_labels)

    index_names = index_names.astype('U50')
    for idx in range(len(index_names)):
        index_names[idx] = ['Actual: ' + val_labels_names[idx] + ' , Pred: ' + index_names[idx][0]]

    shap.initjs()

    for i in range(len(shap_numpy)):
        for j in range(len(shap_numpy[i])):
            image = shap.image_plot(shap_numpy[i][j], test_numpy[j], show = False)
            plt.title(index_names[j][0], x=0, y=1)
            plt.savefig(fig_dir + str(j)+'_shap.jpeg')
            plt.close(image)

    return


def shap_numpy(model, X, val_labels, indexes, classes, fig_dir):

    explainer = shap.DeepExplainer(model, X)
    
    if indexes == []:
        shap_values, indexes = explainer.shap_values(X, ranked_outputs=1, output_rank_order = 'max')
    else: shap_values = explainer.shap_values(X)

    # get names for the classes
    index_names = np.vectorize(lambda x: classes[str(x)])(indexes)
    val_labels_names = np.vectorize(lambda x: classes[str(x)])(val_labels)

    index_names = index_names.astype('U50')
    for idx in range(len(index_names)):
        index_names[idx] = ['Actual: ' + val_labels_names[idx] + ' , Pred: ' + index_names[idx][0]]

    shap.initjs()
    for i in range(len(shap_values)):
            for j in range(len(shap_values[i])):
                image = shap.image_plot(shap_values[i][j], X[j], show = False)
                plt.title(index_names[j][0], x=0, y=1)
                plt.savefig(fig_dir + str(j)+'_shap.jpeg')
                plt.close(image)

    return






