from process import process_mushroom_dataset, process_titanic_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics.cluster import v_measure_score
import matplotlib.pyplot as plt
import os



def get_evaluation_curve(method, data_name, cluster_range, homogeneity, completeness, v_measure, silhouette):

    plt.close()

    # Create plot
    plt.figure()
    plt.title("{} Evaluation on {} Data (tyang358)".format(method, data_name))
    plt.xlabel("clusters amount")
    plt.ylabel("Score")

    # Draw lines
    plt.grid()
    plt.plot(cluster_range, homogeneity, label="homogeneity", color='black', marker='o', markersize=5)
    plt.plot(cluster_range, completeness, label="completeness", color='green', marker='s', markersize=5)
    plt.plot(cluster_range, v_measure, label="v_measure", color='orange', marker='d', markersize=5)
    plt.plot(cluster_range, silhouette, label="silhouette", color='blue', marker='D', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} {} Evaluation".format(data_name, method))
    plt.savefig(image_path, dpi=100)
    return


def draw_square_distance_graph(al, data, cluster_range, square_distance):
    # Plot sse against k
    plt.close()

    # Create plot
    plt.figure()
    plt.title("{} SSD for {} (tyang358)".format(al, data))
    plt.xlabel("clusters amount")
    plt.ylabel("total squared distance")

    # Draw lines
    plt.grid()
    plt.plot(cluster_range, square_distance, label="SSD", color='b', marker='o', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} {} SSD".format(data, al))
    plt.savefig(image_path, dpi=100)


def draw_score_graph(al, data, cluster_range, score):

    plt.close()

    # Create plot
    plt.figure()
    plt.title("{} likelihood for {} (tyang358)".format(al, data))
    plt.xlabel("clusters amount")
    plt.ylabel("log_likelihood")

    # Draw lines
    plt.grid()
    plt.plot(cluster_range, score, label="likelihood", color='b', marker='o', markersize=5)
    plt.legend(loc="best")
    plt.tight_layout()

    # Save image
    plt.draw()
    image_path = os.path.join('.', "{} {} Likelihood".format(data, al))
    plt.savefig(image_path, dpi=100)
    return


def clustering_titanic_data(data, data_x, data_y, cluster_range, seed):
    square_distance = []
    score = []
    for cluster in cluster_range:

        al = KMeans(n_clusters=cluster, random_state=seed)
        gm = GaussianMixture(n_components=cluster, random_state=seed)

        al.fit(data_x)
        gm.fit(data_x)

        score.append(gm.score(data_x))
        square_distance.append(al.inertia_)

    draw_square_distance_graph("KM", "Titanic", cluster_range, square_distance)
    draw_score_graph("EM", "Titanic", cluster_range, score)


def clustering_mushroom_data(data, data_x, data_y, cluster_range, seed):
    square_distance = []
    score = []
    for cluster in cluster_range:
        al = KMeans(n_clusters=cluster, random_state=seed)
        gm = GaussianMixture(n_components=cluster, random_state=seed)

        al.fit(data_x)
        gm.fit(data_x)

        score.append(gm.score(data_x))
        square_distance.append(al.inertia_)

    draw_square_distance_graph("KM", "Mushroom", cluster_range, square_distance)
    draw_score_graph("EM", "Mushroom", cluster_range, score)


def get_titanic_algo_info(x, y, seed, cluster_range):
    km_homogeneity = []
    km_v_measure = []
    km_silhouette = []
    km_completeness = []

    em_homogeneity = []
    em_v_measure = []
    em_silhouette = []
    em_completeness = []

    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=seed)
        km_preds = km.fit_predict(x)
        km_silhouette.append(silhouette_score(x, km_preds, metric='euclidean'))
        km_homogeneity.append(homogeneity_score(y, km_preds))
        km_completeness.append(completeness_score(y, km_preds))
        km_v_measure.append(v_measure_score(y, km_preds))

    for k in cluster_range:
        gmm = GaussianMixture(n_components=k, random_state=seed)
        gm_preds = gmm.fit_predict(x)
        em_silhouette.append(silhouette_score(x, gm_preds, metric='euclidean'))
        em_homogeneity.append(homogeneity_score(y, gm_preds))
        em_completeness.append(completeness_score(y, gm_preds))
        em_v_measure.append(v_measure_score(y, gm_preds))

    get_evaluation_curve("K-Means", "Titanic", cluster_range, km_homogeneity, km_completeness, km_v_measure, km_silhouette)
    get_evaluation_curve("Expectation Maximization", "Titanic", cluster_range, em_homogeneity,
                         em_completeness, em_v_measure, em_silhouette)


def get_mushroom_algo_info(x, y, seed, cluster_range):
    km_homogeneity = []
    km_v_measure = []
    km_silhouette = []
    km_completeness = []

    em_homogeneity = []
    em_v_measure = []
    em_silhouette = []
    em_completeness = []

    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=seed)
        km_preds = km.fit_predict(x)
        km_silhouette.append(silhouette_score(x, km_preds, metric='euclidean'))
        km_homogeneity.append(homogeneity_score(y, km_preds))
        km_completeness.append(completeness_score(y, km_preds))
        km_v_measure.append(v_measure_score(y, km_preds))

    for k in cluster_range:
        gmm = GaussianMixture(n_components=k, random_state=seed)
        gm_preds = gmm.fit_predict(x)
        em_silhouette.append(silhouette_score(x, gm_preds, metric='euclidean'))
        em_homogeneity.append(homogeneity_score(y, gm_preds))
        em_completeness.append(completeness_score(y, gm_preds))
        em_v_measure.append(v_measure_score(y, gm_preds))

    get_evaluation_curve("K-Means", "Mushroom", cluster_range, km_homogeneity, km_completeness, km_v_measure, km_silhouette)
    get_evaluation_curve("Expectation Maximization", "Mushroom", cluster_range, em_homogeneity,
                         em_completeness, em_v_measure, em_silhouette)


if __name__ == "__main__":

    random_seed = 40

    cluster_range = [2,3,4,5]
    for i in range(10,100,5):
        cluster_range.append(i)

    data_T, data_T_x, data_T_y = process_titanic_dataset()
    data_M, data_M_x, data_M_y = process_mushroom_dataset()

    data_titanic = StandardScaler().fit_transform(data_T_x)
    data_mushroom = StandardScaler().fit_transform(data_M_x)

    print("Clusting in")
    clustering_titanic_data(data_T, data_titanic, data_T_y, cluster_range, random_seed)
    clustering_mushroom_data(data_M, data_mushroom, data_M_y, cluster_range, random_seed)

    print("Score in")
    get_titanic_algo_info(data_titanic, data_T_y, random_seed, cluster_range)
    get_mushroom_algo_info(data_mushroom, data_M_y, random_seed, cluster_range)
