import numpy as np
import pandas as pd
import time
from process import process_mushroom_dataset, process_titanic_dataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os


def plot_NN_reduction(data_name, components, vals, param):
    plt.close()

    # Create plot
    plt.figure()
    plt.title("Neural Network {} with Dimension Reduction for {} Data (tyang358)".format(param, data_name), size=9)
    plt.xlabel("Number of components")
    plt.ylabel(param)

    plt.grid()
    plt.plot(components, vals[0], label="No_Reduction", color='b', marker='o', markersize=5)
    plt.plot(components, vals[1], label="PCA", color='g', marker='s', markersize=5)
    plt.plot(components, vals[2], label="ICA", color='m', marker='D', markersize=5)
    plt.plot(components, vals[3], label="RP", color='c', marker='d', markersize=5)
    plt.plot(components, vals[4], label="FA", color='r', marker='P', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "NN Reduction_{} for {}".format(param, data_name))
    plt.savefig(image_path, dpi=100)
    return


def plot_NN_reduction_cluster(cluster, data_name, components, vals, param):
    plt.close()

    # Create plot
    plt.figure()
    plt.title("Neural Network {} with Dimension Reduction and {} Clustering for {} Data (tyang358)".format(param, cluster, data_name), size=9)
    plt.xlabel("Number of components")
    plt.ylabel(param)

    # KM: ["No_Reduction", "PCA", "ICA", "RP", "FA"]
    # EM: ["No_Reduction", "PCA", "ICA", "RP", "FA"]
    # Draw lines
    plt.grid()
    plt.plot(components, vals[0], label="No_Reduction_Cluster", color='b', marker='o', markersize=5)
    plt.plot(components, vals[1], label="PCA", color='g', marker='s', markersize=5)
    plt.plot(components, vals[2], label="ICA", color='m', marker='D', markersize=5)
    plt.plot(components, vals[3], label="RP", color='c', marker='d', markersize=5)
    plt.plot(components, vals[4], label="FA", color='r', marker='P', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "NN {} Reduction {} for {} Dataset".format(cluster, param, data_name))
    plt.savefig(image_path, dpi=100)
    return


def nn_compare(data_name, x, y, components):
    accuracy_list = []
    running_time_list = []
    max_accuracy_list = []
    max_running_time_list = []
    max_accuracy_idx = []

    for reduction in ["No_Reduction", "PCA", "ICA", "RP", "FA"]:
        accuracy = []
        running_time = []
        for k in components:
            start_time = time.time()
            if reduction == "PCA":
                pca = PCA(n_components=k, random_state=42)
                x_transformed_pca = pca.fit_transform(x)
                clf_pca = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                    early_stopping=True, random_state=10)
                clf_pca.fit(x_transformed_pca, y)
                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_pca.predict(x_transformed_pca), y_true=y))

            elif reduction == "ICA":
                ica = FastICA(n_components=k, random_state=42)
                x_transformed_ica = ica.fit_transform(x)
                clf_ica = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                        early_stopping=True, random_state=10)
                clf_ica.fit(x_transformed_ica, y)
                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_ica.predict(x_transformed_ica), y_true=y))

            elif reduction == "RP":
                rp = SparseRandomProjection(n_components=k, random_state=42)
                x_transformed_rp = rp.fit_transform(x)
                clf_rp = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                        early_stopping=True, random_state=10)
                clf_rp.fit(x_transformed_rp, y)
                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_rp.predict(x_transformed_rp), y_true=y))

            elif reduction == "FA":
                fa = FactorAnalysis(n_components=k, random_state=42)
                x_transformed_fa = fa.fit_transform(x)
                clf_fa = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                        early_stopping=True, random_state=10)
                clf_fa.fit(x_transformed_fa, y)
                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_fa.predict(x_transformed_fa), y_true=y))

            elif reduction == "No_Reduction":
                clf = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                    early_stopping=True, random_state=10)
                clf.fit(x, y)
                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf.predict(x), y_true=y))

        accuracy_list.append(accuracy)
        running_time_list.append(running_time)
        max_accuracy_list.append(max(accuracy))
        idx = accuracy.index(max(accuracy))
        max_running_time_list.append(running_time[idx])
        max_accuracy_idx.append(components[idx])


    plot_NN_reduction(data_name, components, accuracy_list, "Accuracy")
    plot_NN_reduction(data_name, components, running_time_list, "Running Time")
    return accuracy_list[0], accuracy_list[1:], running_time_list[0], running_time_list[1:]


def compare_nn_km_with_reduction(data_name, x, y, components):
    accuracy_list = []
    running_time_list = []
    max_accuracy_list = []
    max_running_time_list = []
    max_accuracy_idx = []

    for reduction in ["No_Reduction", "PCA", "ICA", "RP", "FA"]:
        accuracy = []
        running_time = []
        if reduction == "PCA":
            for k in components:
                start_time = time.time()
                # pca reduction
                pca = PCA(n_components=k, random_state=42)
                x_transformed_pca = pca.fit_transform(x)
                # km clustering
                clf_km = KMeans(n_clusters=k, random_state=42)
                x_transformed_pca = clf_km.fit_transform(x_transformed_pca)
                # NN reduction clustering
                clf_pca = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                        early_stopping=True, random_state=10)
                clf_pca.fit(x_transformed_pca, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_pca.predict(x_transformed_pca), y_true=y))

        elif reduction == "ICA":
            for k in components:
                start_time = time.time()
                # ica reduction
                ica = FastICA(n_components=k, random_state=42)
                x_transformed_ica = ica.fit_transform(x)
                # km clustering
                clf_km = KMeans(n_clusters=k, random_state=42)
                x_transformed_ica = clf_km.fit_transform(x_transformed_ica)
                # NN reduction clustering
                clf_ica = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001,
                                        hidden_layer_sizes=(200,),
                                        early_stopping=True, random_state=10)
                clf_ica.fit(x_transformed_ica, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_ica.predict(x_transformed_ica), y_true=y))

        elif reduction == "RP":
            for k in components:
                start_time = time.time()
                # RP reduction
                rp = SparseRandomProjection(n_components=k, random_state=42)
                x_transformed_rp = rp.fit_transform(x)
                # km clustering
                clf_km = KMeans(n_clusters=k, random_state=42)
                x_transformed_rp = clf_km.fit_transform(x_transformed_rp)
                # NN reduction clustering
                clf_rp = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001,
                                       hidden_layer_sizes=(200,),
                                       early_stopping=True, random_state=10)
                clf_rp.fit(x_transformed_rp, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_rp.predict(x_transformed_rp), y_true=y))

        elif reduction == "FA":
            for k in components:
                start_time = time.time()
                # FA reduction
                fa = FactorAnalysis(n_components=k, random_state=42)
                x_transformed_fa = fa.fit_transform(x)
                # km clustering
                clf_km = KMeans(n_clusters=k, random_state=42)
                x_transformed_fa = clf_km.fit_transform(x_transformed_fa)
                # NN reduction clustering
                clf_fa = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001,
                                       hidden_layer_sizes=(200,),
                                       early_stopping=True, random_state=10)
                clf_fa.fit(x_transformed_fa, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_fa.predict(x_transformed_fa), y_true=y))

        elif reduction == "No_Reduction":
            for k in components:
                start_time = time.time()
                clf = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                    early_stopping=True, random_state=10)
                clf.fit(x, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf.predict(x), y_true=y))

        accuracy_list.append(accuracy)
        running_time_list.append(running_time)
        max_accuracy_list.append(max(accuracy))
        idx = accuracy.index(max(accuracy))
        max_running_time_list.append(running_time[idx])
        max_accuracy_idx.append(components[idx])

    plot_NN_reduction_cluster("KM", data_name, components, accuracy_list, "Accuracy")
    plot_NN_reduction_cluster("KM", data_name, components, running_time_list, "Running Time")

    return accuracy_list[0], accuracy_list[1:], running_time_list[0], running_time_list[1:]


def plot_NN_compare(cluster, data_name, components, vals_pre, vals_post, benchmark, param):
    plt.close()

    # Create plot
    plt.figure()
    plt.title(" Parameter {} Neural Network Dimension Reduction and {} Clustering for {} Data (tyang358)".
              format(param, cluster, data_name), size=7)
    plt.xlabel("Number of components")
    plt.ylabel(param)


    plt.grid()
    plt.plot(components, benchmark, label="No_Reduction_Cluster", color='b', marker='o', markersize=5)
    plt.plot(components, vals_pre[0], label="PCA", color='g', marker='s', markersize=5)
    plt.plot(components, vals_post[0], '--', label="PCA_{}".format(cluster), color='g', marker='P', markersize=5)
    plt.plot(components, vals_pre[1], label="ICA", color='m', marker='D', markersize=5)
    plt.plot(components, vals_post[1], '--', label="ICA_{}".format(cluster), color='m', marker='h', markersize=5)
    plt.plot(components, vals_pre[2], label="RP", color='c', marker='d', markersize=5)
    plt.plot(components, vals_post[2], '--', label="RP_{}".format(cluster), color='c', marker='H', markersize=5)
    plt.plot(components, vals_pre[3], label="FA", color='r', marker='p', markersize=5)
    plt.plot(components, vals_post[3], '--', label="FA_{}".format(cluster), color='r', marker='X', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} comparison for NN {} Cluster Reduction {}".format(param, cluster, data_name))
    plt.savefig(image_path, dpi=100)
    return


def get_gmm_cluster(k, x):
    clf_gmm = GMM(n_components=k, random_state=42)
    labels = clf_gmm.fit_predict(x)
    x['cluster'] = labels
    df_dummies = pd.get_dummies(x['cluster'], prefix='cluster')
    x = pd.concat([x, df_dummies], axis=1)
    x.drop('cluster', axis=1, inplace=True)
    return x


def compare_nn_em_with_reduction(data_name, x, y, components):
    accuracy_list = []
    running_time_list = []
    max_accuracy_list = []
    max_running_time_list = []
    max_accuracy_idx = []
    for reduction in ["No_Reduction", "PCA", "ICA", "RP", "FA"]:
        accuracy = []
        running_time = []
        if reduction == "PCA":
            for k in components:
                start_time = time.time()
                # pca reduction
                pca = PCA(n_components=k, random_state=42)
                x_transformed_pca = pca.fit_transform(x)
                # gmm clustering
                x_transformed_pca = get_gmm_cluster(k, pd.DataFrame(data=x_transformed_pca))
                # NN reduction clustering
                clf_pca = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                        early_stopping=True, random_state=10)
                clf_pca.fit(x_transformed_pca, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_pca.predict(x_transformed_pca), y_true=y))

        elif reduction == "ICA":
            for k in components:
                start_time = time.time()
                # ica reduction
                ica = FastICA(n_components=k, random_state=42)
                x_transformed_ica = ica.fit_transform(x)
                # gmm clustering
                x_transformed_ica = get_gmm_cluster(k, pd.DataFrame(data=x_transformed_ica))
                # NN reduction clustering
                clf_ica = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001,
                                        hidden_layer_sizes=(200,),
                                        early_stopping=True, random_state=10)
                clf_ica.fit(x_transformed_ica, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_ica.predict(x_transformed_ica), y_true=y))

        elif reduction == "RP":
            for k in components:
                start_time = time.time()
                # RP reduction
                rp = SparseRandomProjection(n_components=k, random_state=42)
                x_transformed_rp = rp.fit_transform(x)
                # gmm clustering
                x_transformed_rp = get_gmm_cluster(k, pd.DataFrame(data=x_transformed_rp))
                # clf_gmm = GMM(n_components=k, random_state=42)
                # x_transformed_rp = clf_gmm.fit(x_transformed_rp)

                # NN reduction clustering
                clf_rp = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001,
                                       hidden_layer_sizes=(200,),
                                       early_stopping=True, random_state=10)
                clf_rp.fit(x_transformed_rp, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_rp.predict(x_transformed_rp), y_true=y))

        elif reduction == "FA":
            for k in components:
                start_time = time.time()
                # FA reduction
                fa = FactorAnalysis(n_components=k, random_state=42)
                x_transformed_fa = fa.fit_transform(x)
                # gmm clustering
                x_transformed_fa = get_gmm_cluster(k, pd.DataFrame(data=x_transformed_fa))
                # NN reduction clustering
                clf_fa = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001,
                                       hidden_layer_sizes=(200,),
                                       early_stopping=True, random_state=10)
                clf_fa.fit(x_transformed_fa, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf_fa.predict(x_transformed_fa), y_true=y))

        elif reduction == "No_Reduction":
            for k in components:
                start_time = time.time()

                clf = MLPClassifier(activation='relu', max_iter=2000, alpha=0.0001, hidden_layer_sizes=(200,),
                                    early_stopping=True, random_state=10)
                clf.fit(x, y)

                run_time = time.time() - start_time
                running_time.append(run_time)
                accuracy.append(accuracy_score(y_pred=clf.predict(x), y_true=y))

        accuracy_list.append(accuracy)
        running_time_list.append(running_time)
        max_accuracy_list.append(max(accuracy))
        idx = accuracy.index(max(accuracy))
        max_running_time_list.append(running_time[idx])
        max_accuracy_idx.append(components[idx])


    plot_NN_reduction_cluster("GMM", data_name, components, accuracy_list, "Accuracy")
    plot_NN_reduction_cluster("GMM", data_name, components, running_time_list, "Running Time")
    return accuracy_list[0], accuracy_list[1:], running_time_list[0], running_time_list[1:]


if __name__ == "__main__":

    data_T, data_T_x, data_T_y = process_titanic_dataset()

    random_seed = 40

    cluster_range = [2,3,4,5]
    for i in range(10,100,5):
        cluster_range.append(i)

    # Standardize the data
    data_titanic = StandardScaler().fit_transform(data_T_x)

    # Parameter range
    titanic_param_range = np.arange(2, data_titanic.shape[1] + 1, 1)

    print("step 1 in")
    t_accuracy, t_accuracies, t_time, t_times = nn_compare("Titanic", data_titanic, data_T_y, titanic_param_range)

    print("step 2 in")
    t_accuracy, t_accuracies_km, t_time, t_times_km = compare_nn_km_with_reduction("Titanic", data_titanic, data_T_y, titanic_param_range)
    t_accuracy, t_accuracies_em, t_time, t_times_em = compare_nn_em_with_reduction("Titanic", data_titanic, data_T_y, titanic_param_range)


    plot_NN_compare("KM", "Titanic", titanic_param_range, t_accuracies, t_accuracies_km, t_accuracy, "Accuracy")
    plot_NN_compare("EM", "Titanic", titanic_param_range, t_accuracies, t_accuracies_em, t_accuracy, "Accuracy")

    plot_NN_compare("KM", "Titanic", titanic_param_range, t_times, t_times_km, t_time, "Running Time")
    plot_NN_compare("EM", "Titanic", titanic_param_range, t_times, t_times_em, t_time, "Running Time")