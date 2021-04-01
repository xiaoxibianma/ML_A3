from process import process_mushroom_dataset, process_titanic_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as ppccaa
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics.cluster import v_measure_score
import matplotlib.pyplot as plt
import os
from sklearn.random_projection import SparseRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import FactorAnalysis


def plot_scatter(data, k, x, seed):
    pca = ppccaa(n_components=k, random_state=seed)
    x_transformed = pca.fit_transform(x)
    plt.figure(figsize=(8, 6))
    km_dic = {"n_clusters": 2, "init": "k-means++", "n_init": 1, "max_iter": 100}
    res = KMeans(**km_dic).fit_predict(x_transformed)
    plt.figure()
    plt.scatter(x_transformed[:, 0], x_transformed[:, 1], c=res, alpha=0.5)
    plt.title("{} on KM for {} dataset(tyang358)".format("CPA", data))


    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} KM {} plot".format("PCA", data))
    plt.savefig(image_path, dpi=100)


def draw_reduction_param_curve(data, param_range, vals, param, al):
    plt.close()

    plt.figure()
    plt.title("{} parameters in {} Algorithm for {} dataset (tyang358)".format(param, al, data), size=9)
    plt.xlabel("number of component")
    plt.ylabel(param)

    plt.grid()
    plt.plot(param_range, vals, label=param, color='b', marker='o', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} data for {} {}".format(data, al, param))
    plt.savefig(image_path, dpi=100)
    return


def draw_ssd_reduction_curve(reduction, method, data, cluster_range, sse, sse_post):

    plt.close()
    plt.figure()
    plt.title("{} {} SSD for {} Data after reduction (tyang358)".format(reduction, method, data))
    plt.xlabel("clusters number ")
    plt.ylabel("total squared distance")

    # Draw lines
    plt.grid()
    plt.plot(cluster_range, sse, label="sse", color='red', marker='o', markersize=5)
    plt.plot(cluster_range, sse_post, '--', label="sse_post", color='blue', marker='s', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} data {} {} SSD".format(data, reduction, method))
    plt.savefig(image_path, dpi=100)
    return


def draw_km_reduction_curve(reduction, method, data, p_range, homogeneity_before, completeness_before,
                                 silhouette_before,
                                 homogeneity_after, completeness_after, silhouette_after):

    plt.close()

    # Create plot
    plt.figure()
    plt.title("Evaluation for {} {} in {} Data (tyang358)".format(reduction, method, data))
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")

    # Draw lines
    plt.grid()
    plt.plot(p_range, homogeneity_before, label="homogeneity_before", color='g', marker='o', markersize=5)
    plt.plot(p_range, homogeneity_after, '--', label="homogeneity_after", color='g', marker='s', markersize=5)
    plt.plot(p_range, completeness_before, label="completeness_before", color='b', marker='p', markersize=5)
    plt.plot(p_range, completeness_after, '--', label="completeness_after", color='b', marker='P', markersize=5)
    plt.plot(p_range, silhouette_before, label="silhouette_before", color='c', marker='d', markersize=5)
    plt.plot(p_range, silhouette_after, '--', label="silhouette_after", color='c', marker='D', markersize=5)
    plt.legend(loc="best", fontsize=6)
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "Evaluation for {} data {} {}".format(data, reduction, method))
    plt.savefig(image_path, dpi=100)
    return


def PCA(data, x, seed):
    pca = ppccaa(random_state=seed)
    pca.fit_transform(x)
    variance = pca.explained_variance_ratio_.cumsum()
    variance_range = np.arange(0, len(variance), 1)
    draw_reduction_param_curve(data, variance_range, variance, "%Variance", "PCA")
    return


def ICA(data, x, parameters_range, seed):
    kurtosis = []
    for p in parameters_range:
        ica = FastICA(n_components=p, random_state=seed)
        ex = ica.fit_transform(x)
        transformed = pd.DataFrame(ex)
        kurt = transformed.kurt(axis=0)
        kurtosis.append(kurt.abs().mean())

    draw_reduction_param_curve(data, parameters_range, kurtosis, "Kurtosis", "ICA")


def Randomized_Projections(data, x, y, parameters_range, seed):
    cv_accuracy = []
    for p in parameters_range:
        sp = SparseRandomProjection(n_components=p, random_state=seed)
        x_transformed = sp.fit_transform(x)
        clf = MLPClassifier(activation='relu', max_iter=500, alpha=0.0001, hidden_layer_sizes=(200, ),
                            early_stopping=True, random_state=10)
        clf.fit(x_transformed, y)
        cv_accuracy.append(accuracy_score(clf.predict(x_transformed), y))

    draw_reduction_param_curve(data, parameters_range, cv_accuracy, "CV_accuracy","Randomized Projections")


def compare_KM_(data, x, y, p_range, seed):


    for al in ["PCA", "ICA", "Randomized_Projections", "Factor Analysis"]:

        homogeneity_before = []
        homogeneity_after = []
        completeness_before = []
        completeness_after = []
        silhouette_before = []
        silhouette_after = []
        likelihood, likelihood_post = [], []
        ssd_before = []
        sse_after = []

        for p in p_range:

            new_x = None

            if al == "PCA":
                pca = ppccaa(n_components=p, random_state=seed)
                new_x = pca.fit_transform(x)
            elif al == "ICA":
                ica = FastICA(n_components=p, random_state=seed)
                new_x = ica.fit_transform(x)
            elif al == "Randomized_Projections":
                rp = SparseRandomProjection(n_components=p, random_state=seed)
                new_x = rp.fit_transform(x)
            elif al == "Factor Analysis":
                fa = FactorAnalysis(n_components=p, random_state=seed)
                new_x = fa.fit_transform(x)

            #       PRE
            km_before = KMeans(n_clusters=p, random_state=seed)
            preds_pre = km_before.fit_predict(x)
            km_before.fit(x)

            ssd_before.append(km_before.inertia_)
            silhouette_before.append(silhouette_score(x, preds_pre, metric='euclidean'))
            homogeneity_before.append(homogeneity_score(y, preds_pre))
            completeness_before.append(completeness_score(y, preds_pre))

            #        AFTER
            km_post = KMeans(n_clusters=p, random_state=seed)
            preds_post = km_post.fit_predict(new_x)
            km_post.fit(new_x)

            sse_after.append(km_post.inertia_)
            silhouette_after.append(silhouette_score(new_x, preds_post, metric='euclidean'))
            homogeneity_after.append(homogeneity_score(y, preds_post))
            completeness_after.append(completeness_score(y, preds_post))

        draw_ssd_reduction_curve(al, "KMeans", data, p_range, ssd_before, sse_after)

        draw_km_reduction_curve(al, "KMeans", data, p_range, homogeneity_before, completeness_before,
                                silhouette_before,
                                homogeneity_after, completeness_after, silhouette_after)


def compare_EM_(data, x, y, p_range, seed):

    for al in ["PCA", "ICA", "Randomized_Projections", "Factor Analysis"]:

        homogeneity_before = []
        homogeneity_after = []
        completeness_before = []
        completeness_after = []
        silhouette_before = []
        silhouette_after = []
        ssd_before = []
        sse_after = []
        v_measure_before = []
        v_measure_after = []

        for p in p_range:

            new_x = None

            if al == "PCA":
                pca = ppccaa(n_components=p, random_state=seed)
                new_x = pca.fit_transform(x)
            elif al == "ICA":
                ica = FastICA(n_components=p, random_state=seed)
                new_x = ica.fit_transform(x)
            elif al == "Randomized_Projections":
                rp = SparseRandomProjection(n_components=p, random_state=seed)
                new_x = rp.fit_transform(x)
            elif al == "Factor Analysis":
                fa = FactorAnalysis(n_components=p, random_state=seed)
                new_x = fa.fit_transform(x)

            #       PRE
            gmm_before = GaussianMixture(n_components=p, random_state=seed)
            preds_pre = gmm_before.fit_predict(x)
            gmm_before.fit(x)

            silhouette_before.append(silhouette_score(x, preds_pre, metric='euclidean'))
            homogeneity_before.append(homogeneity_score(y, preds_pre))
            completeness_before.append(completeness_score(y, preds_pre))
            v_measure_before.append(v_measure_score(y, preds_pre))

            #        AFTER
            gmm_after = GaussianMixture(n_components=p, random_state=seed)
            pred_after = gmm_after.fit_predict(new_x)
            gmm_after.fit(new_x)

            silhouette_after.append(silhouette_score(new_x, pred_after, metric='euclidean'))
            homogeneity_after.append(homogeneity_score(y, pred_after))
            completeness_after.append(completeness_score(y, pred_after))
            v_measure_after.append(v_measure_score(y, pred_after))

        draw_km_reduction_curve(al, "Expectation Maximization", data, p_range, homogeneity_before, completeness_before,
                                silhouette_before,
                                homogeneity_after, completeness_after, silhouette_after)



if __name__ == "__main__":

    random_seed = 40

    cluster_range = [2,3,4,5]
    for i in range(10,100,5):
        cluster_range.append(i)

    data_T, data_T_x, data_T_y = process_titanic_dataset()
    data_M, data_M_x, data_M_y = process_mushroom_dataset()

    data_titanic = StandardScaler().fit_transform(data_T_x)
    data_mushroom = StandardScaler().fit_transform(data_M_x)

    pr4t = np.arange(2, data_titanic.shape[1] + 1, 1)
    pr4m = np.arange(2, data_mushroom.shape[1] + 1, 1)

    pr = [2,3,4,5,6,7,8,9]
    for i in range(10,50,5):
        cluster_range.append(i)

    # PCA
    # ICA
    # Randomized Projections
    # Any other feature selection algorithm you desire

    print("PCA in")
    PCA("Titanic",data_titanic, random_seed)
    PCA("Mushroom",data_mushroom, random_seed)
    print("PCA OUT")

    print("ICA in")
    ICA("Titanic",data_titanic,pr4t,random_seed)
    ICA("Mushroom",data_mushroom,pr4m,random_seed)
    print("ICA OUT")

    print("Randomized Projections in")
    Randomized_Projections("Titanic", data_titanic, data_T_y, pr4t, random_seed)
    Randomized_Projections("Mushroom", data_mushroom, data_M_y, pr4m, random_seed)
    print("Randomized Projections out")

    print("Compare reduction al Start : ")
    print("1 in ")
    compare_KM_("Titanic",data_titanic, data_T_y, pr4t, random_seed)
    print("1 out")
    print("2 in ")
    compare_KM_("Mushroom",data_mushroom, data_M_y, pr4m, random_seed)
    print("2 out ")
    print("3 in ")
    compare_EM_("Titanic",data_titanic, data_T_y, pr4t, random_seed)
    print("3 out ")
    print("4 in ")
    compare_EM_("Mushroom",data_mushroom, data_M_y, pr4m, random_seed)
    print("4 out ")
    print("Compare reduction al Done ### ")

    plot_scatter("Titanic", 6, data_titanic, random_seed)
    plot_scatter("Mushroom", 15, data_mushroom, random_seed)

