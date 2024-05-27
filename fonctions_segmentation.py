#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# ====================================================================
# Foncions Segmentation clientèle - projet 5 Openclassrooms
# Version : 0.0.0 
# ====================================================================

# Chargement des librairies
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
import datetime as dt
import numpy as np
import pandas as pd
import sys
import time

# Librairies personnelles
import fonctions_data
import fonctions_segmentation

# Data visualisation
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

# Pré processing
from sklearn.preprocessing import PowerTransformer, StandardScaler, \
    MinMaxScaler

# Réduction dimension
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA

# Clustering
from sklearn.cluster import KMeans
from sklearn import mixture, preprocessing
import hdbscan
from kmodes.kprototypes import KPrototypes

# Clustering Visualisation
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Clustering Metrics
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score, silhouette_score, silhouette_samples


# Warnings
import warnings
warnings.filterwarnings('ignore')

# Versions
print('Version des librairies utilisées :')
print('Python                : ' + sys.version)
print('NumPy                 : ' + np.version.full_version)
print('Pandas                : ' + pd.__version__)
print('Matplotlib            : ' + mpl.__version__)
print('Seaborn               : ' + sns.__version__)

now = datetime.now().isoformat()
print('Lancé le           : ' + now)

# --------------------------------------------------------------------
# -- VERSION
# --------------------------------------------------------------------
__version__ = '0.0.0'



# ###########################################################################
# -- INDICE DE GINI Fonction
# ###########################################################################


def gini(Y, X=None):
    
    """
    Calcul des coefficients de Gini: https://en.wikipedia.org/wiki/Gini_coefficient
    paramètre Y: valeurs de Y
    paramètre X: valeurs de X
    return: (x, Gini(x))
    
    ``


    """
    n = len(Y)
    couples = np.empty((n, 2))
    if X is None:
        couples[:, 0] = 1
    else:
        couples[:, 0] = X
    couples[:, 1] = Y
    couples = np.cumsum(couples, axis=0)
    couples[:, 0] /= max(couples[n - 1, 0], 1e-7)
    couples[:, 1] /= max(couples[n - 1, 1], 1e-7)

    g = 0.
    n = couples.shape[0]

    for i in range(0, n):
        dx = couples[i, 0] - couples[i - 1, 0]
        y = couples[i - 1, 1] + couples[i, 1]
        g += dx * y

    return (1. - g) / 2


# ###########################################################################
# --Fonction Reduction dimension: AFFICHAGE TSNE
# ###########################################################################


# --------------------------------------------------------------------
# -- SCATTERPLOT DE VISUALISATION DE TRANSFORMATION T-SNE en 2D
# --------------------------------------------------------------------
def affiche_tsne(results_list, liste_param):
    '''
    Affiche les résultats de la transformation Lt-SNE
    Parameters
    ----------
    results_list : iste des résultats de la transformation t-SNE, obligatoire.
    liste_param : liste des valeurs de l'hyper paramètre perplexity testées,
                  obligatoire.
    Returns
    -------
    None.
    '''
    i = 0

    # Visualisation en 2D des différents résultats selon la perplexité
    plt.subplots(3, 2, figsize=[15, 20])

    for resultat_tsne in results_list:

        # Perplexité=5
        plt.subplot(3, 2, i + 1)
        tsne_results_i = results_list[i]
        sns.scatterplot(x=tsne_results_i[:, 0], y=tsne_results_i[:, 1],
                        alpha=.1)
        plt.title('t-SNE avec perplexité=' + str(liste_param[i]))
        plt.plot()

        i += 1

    plt.show()
   

    
# ###########################################################################
# --Fonction : VISUALISATION TSNE, ET RESULTAT CLUSTERING
# ###########################################################################


def variation_clustering(data, range_n_clusters):
    """
    Cette fonction effectue une visualisation du clustering (réduction de dimension TSNE), teste pour différents nombre de clusters,
    un clustering des données par la méthode k-means.
    La fonction affiche les métriques tels que le score de silhouette pour chacun des clusters, et autres.

    Parametres:
    data (pd.DataFrame): DataFrame avec l'ensemble des variables quantitatives, qualitatives.
    range_n_clusters: nombre de clusters testés (liste)

    Return :
    liste des résultats des métriques: silhouette, indice gini, prédictions 
    """

   
    # Temps d'entrainement
    fit_predict_time = []
    
    # Score de silhouette
    silhouette_score_avg = []
    
    # Score de gini
    ginis = []
    
    # Liste du nombre de clusters à tester
    palette = sns.color_palette("husl", max(range_n_clusters))
    colors_2 = palette.as_hex()

    for n_clusters in range_n_clusters:
        
        # Create a subplot with 1 row and 2 columns
        fig = plt.figure(1, figsize=(18, 7))
        ax1 = fig.add_subplot(121)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example
        # all lie within [-0.1, 1]
        
        ax1.set_xlim([-0.1, 1])
        
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        cls = KMeans(n_clusters=n_clusters,
                           init='k-means++',
                           random_state=10)

        # Calcul du temps de prédiction
        start_time = time.time()
        cls_lab = cls.fit_predict(data)
        time.time() - start_time

        fit_predict_time.append(time.time() - start_time)
        #wcss.append(clusterer.inertia_)
        
        # The score silhouette fournit la moyenne de tous les échantillons: elle
        # donne la densité et la séparation des clusters formés
        silhouette_score_avg.append(silhouette_score(data, cls_lab))

        # Nombre d'invididus dans chaque cluster
        counts = [list(cls_lab).count(i) for i in range(n_clusters)]
        ginis.append(gini(counts))

        # Calcul le coéfficient silhouette pour chaque échantillon
        sample_silh_val = silhouette_samples(data, cls_lab)
        
        
        # Application de l'algorithme t-SNE(technique de reduction de dimension)
        tnse = TSNE()
        x_tsne = tnse.fit_transform(data)
        x_tsne_df = pd.DataFrame(x_tsne,  columns=["comp_1",
                                                   "comp_2"])
        x_tsne_df["clusters"] = cls_lab

        y_lower = 10
        for i in range(n_clusters):
            # Aggrège les scores silhouettes pour chaque échantillon appartenant à 
            # un cluster i, et le trie.
            ith_cls_silh_val = sample_silh_val[cls_lab == i]

            ith_cls_silh_val.sort()

            size_cls_i = ith_cls_silh_val.shape[0]
            y_upper = y_lower + size_cls_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0,
                              ith_cls_silh_val,
                              alpha=0.7)

            # Labellise le graphe des silhouettes, avec leurs clusters au centre
            ax1.text(-0.05, y_lower + 0.5 * size_cls_i, str(i))

            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Graphe silhouette score pour différents clusters")
        ax1.set_xlabel("Silhouette scores")
        ax1.set_ylabel("Label cluster")

        # Ligne vertical du score moyen de la silhouette 
        ax1.axvline(x=np.mean(silhouette_score_avg), color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        dx = fig.add_subplot(122)

        for i in range(n_clusters):
            dx.scatter(x_tsne_df[x_tsne_df.clusters == i]["comp_1"],
                       x_tsne_df[x_tsne_df.clusters == i]["comp_2"],
                       c=colors_2[i],
                       label='Cluster ' + str(i+1),
                       s=50)

        # Titres des axes
        dx.set_xlabel("Comp_1")
        dx.set_ylabel("Comp_2")
        dx.set_title("Projection 2D par TSNE")

        plt.suptitle(("Anlayse silhouette for le clustering k-means"
                      " sur l'échantllon de données avec n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()

        print("Pour n_clusters =", n_clusters, "\n",
              "Le silhouette score moyen est de :",
              np.mean(silhouette_score_avg), "\n",
              "Le temps d'entrainement du modèle est de :",
              np.mean(fit_predict_time), '(s) \n',
              "Le score de Gini est de :", np.mean(ginis))

    # Inertie, Silhouette_score, ginis, time_fit_and_predict
    return silhouette_score_avg, ginis, fit_predict_time  # Inertie

    
    
    






# ###########################################################################
# -- CLUSTERING: Distribution des clients
# ###########################################################################

def typage_clients_par_clusters(clusters_lab):
    '''
    Affiche la répartitionn des clients par cluster
    Parameters
    ----------
    clusters_labels : la séries des labels des clusters, obligatoire.
    Returns
    -------
    None.
    '''

    # DataFrame de travail
    series_client_clus = pd.Series(clusters_lab).value_counts()
    nb_client = series_client_clus.sum()
    df_visu_client_clus = pd.DataFrame(
        {'Clusters': series_client_clus.index,
         'Nb_clients': series_client_clus.values})
    df_visu_client_clus['%'] = round(
        (df_visu_client_clus['Nb_clients']) * 100 / nb_client, 2)
    df_visu_client_clus = df_visu_client_clus.sort_values(by='Clusters')
    display(df_visu_client_clus.style.hide_index())

    # Barplot de la distribution
    plt.figure(figsize=(16, 8))
    plt.title('Nombre de clients par clusters', size=15)
    plot = sns.barplot(y = series_client_clus.values, x= series_client_clus.index)
    plt.xlabel("Clusters")
    plt.ylabel("Nombre de clients)")
    plt.show()  




# ###########################################################################
# -- CLUSTERING KMeans
# ###########################################################################


# --------------------------------------------------------------------
# -- CALCUL DES METRIQUES K-Means
# --------------------------------------------------------------------

def calcul_metrique_kmeans(data, dataframe_metrique, type_data,
                          random_seed, n_clusters, n_init, init):
    '''
    Calcul des métriques de KMeans en fonction de différents paramètres.
    Parameters
    ----------
    data : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_donnees : string intitulé des données, obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    n_clusters : liste du nombre de clusters, obligatoire,
    n_init : nombre de clusters à initialiser, obligatoire.
    init : type d'initialisation : 'k-means++' ou 'random'.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # dispersion, indice de davies_bouldin, indice d'information mutuelle
    silhouette = []
    dispersion = []
    davies_bouldin = []
    # Score de gini
    ginis = []
    dt = []
    temps = []

    result_clusters = []
    result_ninit = []
    result_type_init = []

    # Hyperparametre tuning
    n_clusters = n_clusters
    nbr_init = n_init
    type_init = init

    # Tester le modèle entre 2 et 12 clusters
    for num_clusters in n_clusters:
        
        # Create a subplot with 1 row and 2 columns
        fig = plt.figure(1, figsize=(18, 7))
        ax1 = fig.add_subplot(121)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example
        # all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data) + (num_clusters + 1) * 10])

        for init in nbr_init:

            for i in type_init:
                # Top début d'exécution
                time_start = time.time()

                # Initialisation de l'algorithme
                clusterer = KMeans(n_clusters=num_clusters,
                             n_init=init,
                             init=i,
                             random_state=random_seed)

                # Entraînement de l'algorithme
                clusterer.fit(data)

                # Prédictions
                cluster_labels = clusterer.predict(data)

                # Top fin d'exécution
                time_end = time.time()

                # Calcul du score de coefficient de silhouette
                silh = silhouette_score(data, cluster_labels)
                
                
                # Calcul silhouette score pour chaque échantillon
                sample_silh_val = silhouette_samples(data, cluster_labels)
                
                # Nombre d'invididus dans chaque cluster
                counts = [list(cluster_labels).count(i) for i in range(6)]
                
                # Calcul la dispersion
                disp = clusterer.inertia_
                
                # Calcul de l'indice davies-bouldin
                db = davies_bouldin_score(data, cluster_labels)
                
                # Durée d'exécution
                time_execution = time_end - time_start

                silhouette.append(silh)
                dispersion.append(disp)
                davies_bouldin.append(db)
                ginis.append(gini(counts))
                dt.append(type_data)
                temps.append(time_execution)

                result_clusters.append(num_clusters)
                result_ninit.append(init)
                result_type_init.append(i)
                
                y_lower = 10
                for i in range(num_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silh_val = sample_silh_val[cluster_labels == i]

                    ith_cluster_silh_val.sort()

                    size_cluster_i = ith_cluster_silh_val.shape[0]
                    y_upper = y_lower + size_cluster_i

                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0,
                                      ith_cluster_silh_val,
                                      alpha=0.7)

                    # Label the silhouette plots with
                    # their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("Le graphe silhouette pour différents clusters.")
                ax1.set_xlabel("Valeur coéfficient silhouette")
                ax1.set_ylabel("Label du cluster")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=np.mean(silhouette), color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': dt,
        'nbr_clusters': result_clusters,
        'n_init': result_ninit,
        'type_init': result_type_init,
        'coef_silh': silhouette,
        'dispersion': dispersion,
        'davies_bouldin': davies_bouldin,
        'indice_gini': ginis,
        'Durée (s)': temps
    }), ignore_index=True)
    
    
    print("Pour n_clusters =", num_clusters, "\n",
          "Le silhouette score moyen est de :", np.mean(silhouette), "\n",
          "Le score de Gini moyen est de :", np.mean(ginis), "\n",
          "Le temps d'entrainement du modèle est de :", np.mean(temps), '(s)')

    return dataframe_metrique




# ###########################################################################
# -- CLUSTERING: Affichage RADARPLOT par clusters
# ###########################################################################


def affiche_radarplot_par_clusters(data, var_cluster, nb_rows, nb_cols, colors, sub_plot=[3, 3], figsize=(36, 18)):
    '''
    Affiche les radars plot des différents clusters pour l'interprétation.
    Parameters
    ----------
    dataframe : dataframe des données, obligatoire.
    var_cluster : str nom de la variable représentant le label des clusters,
                  obligatoire.
    nb_rows : nombre de lignes pour afficher les radars plots, obligatoire.
    nb_cols : nombre de colonnes pour afficher les radars plots.
    Returns
    -------
    return: résultats des métriques
    None.
    '''
    # Radarplot des différents clusters
    # --------------------------------------------------------------------

    cols_interp = data.columns.to_list()
    df_plot_seg = data[cols_interp].set_index(var_cluster)
    # Standardisation
    min_max = MinMaxScaler()
    df_plot_seg = pd.DataFrame(min_max.fit_transform(df_plot_seg.values),
                               index=df_plot_seg.index,
                               columns=df_plot_seg.columns)
    plt.rc('axes', facecolor= 'ivory')

    # number of variable
    categories = list(df_plot_seg.columns)
    
    N = len(categories)
    
 # Quel sera l'angle de chaque axe ?
    # On divise un tour complet par le nombre de variables
    angles = [n / float(N) * 2 * pi for n in range(N)]

    fig = plt.figure(1, figsize= figsize)

    # Trace un radar chart pour chaque segment
    
    for i, segment in enumerate(df_plot_seg.index):

        ax = fig.add_subplot(nb_rows, nb_cols, i + 1, polar=True)

        ax.set_theta_offset(2 * pi / 3)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles, categories, size=10)
        plt.yticks(color="grey", size=2)
        values = list(df_plot_seg.iloc[i].values)
        ax.plot(angles, values, 'o-', linewidth=1, linestyle='solid')
        ax.fill(angles, values, colors[i], alpha=0.55)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_title('Cluster ' + str(segment), size=15, color=colors[i])
        ax.grid(True)
        plt.grid(True)
        plt.ylim(0, 1)

    plt.show()



    
# ###################################################################################
# -- CALCUL METRIQUES: Affichage résultats des métriques sur différents algorithmes
# ###################################################################################


def calcul_comp_clustering(nom_algo, data, preds, dataframe_comparaison,
                           temps_exec):
    '''
    Calcul des métriques pour comparer les différents algorithmes de clustering.
    Parameters
    ----------
    nom_algo : nom de l'algorithme, obligatoire.
    data : données à analyser, obligatoire.
    preds : le résultat de la prédiction, obligatoire.
    dataframe_comparaison : dataframe de sauvegarde des résultats, obligatoire.
    temps_exec : temps d'exécution, obligatoire.
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : le coefficient de silhouette,
    # dispersion, indice de davies_bouldin
    
    silhouette = []
    ginis = []
    davies_bouldin = []
    calin_harab = []
    algos = []
    temps = []
    nb_cluster = []

    nbcluster = len(set(preds))
    if nbcluster > 1:

        # Calcul du score de coefficient de silhouette
        silh = silhouette_score(data, preds)
        
        # Caclcul indice de Gini  
        g = gini(preds)
        
        # Calcul de l'indice davies-bouldin
        db = davies_bouldin_score(data, preds)


        silhouette.append(silh)
        ginis.append(g)
        davies_bouldin.append(db)

    else:

        silhouette.append(0)
        ginis.append(0)
        davies_bouldin.append(00)


    algos.append(nom_algo)
    temps.append(temps_exec)
    nb_cluster.append(nbcluster)

    dataframe_comparaison = dataframe_comparaison.append(pd.DataFrame({
        'Algos': algos,
        'Nb_clusters': nb_cluster,
        'coef_silh': silhouette,
        'indice_gini': ginis,
        'davies_bouldin': davies_bouldin,
        'Durée': temps
    }), ignore_index=True)

    return dataframe_comparaison



# ###################################################################################
# -- CALCUL METRIQUES: évaluation de la stabilité d'initialisation
# ###################################################################################


def calcul_stabilite_initialisation(nom_algo, model, data, dataframe_resultat,
                                    nb_iter=5):
    '''
    CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation.
    Inclusion des variables qualitatives (Usage algo K-Prototype)
    Parameters
    ----------
    nom_algo : str, nom de l'algorithme, obligatoire.
    model : instanciation du modèle dont on veut analyser la stabilité 
            d'initialisation, obligatoire.
    data : données d'entréedu modèle, obligatoire.
    dataframe_resultat : dataframe de sauvegarde des résultats.
    nb_iter : nombre d'itérations (5 par défaut).
    Returns
    -------
    dataframe_resultat : dataframe de sauvegarde des résultats.
    '''
    # Initialise la liste des partitions
    partitions = []
    var_cat = []

    # Boucle sur le nombre d'itération choisi (5 par défaut)
    for i in range(nb_iter):
        
        for col in data.columns:
            if data[col].dtypes == 'object':
                var_cat.append(data.columns.get_loc(col))
        var_cat
        
        if len(var_cat) != 0:
            
            # Entraînement du modèle avec les variables catégoriques
            model.fit(data, categorical=var_cat)
            
            # Labels des clusters
            partitions.append(model.labels_)
       
        else:                
            
            # Entraînement du modèle
            model.fit(data)
            
            # Labels des clusters
            partitions.append(model.labels_)
                        

    # Computing the ARI scores between partitions
    # --------------------------------------------------------

    # Initializing list of ARI scores
    ARI_scores = []

    # For each partition, except last one
    for i in range(nb_iter - 1):
        # Compute the ARI score with other partitions
        for j in range(i + 1, nb_iter):
            ARI_score = metrics.adjusted_rand_score(partitions[i], partitions[j])
            ARI_scores.append(ARI_score)

    # Compute the mean and standard deviation of ARI scores
    ARI_mean = statistics.mean(ARI_scores)
    ARI_std = statistics.stdev(ARI_scores)

    dataframe_resultat = dataframe_resultat.append(pd.DataFrame({
        'Algos': [nom_algo],
        'ARI_mean': [ARI_mean],
        'ARI_std': [ARI_std]
    }), ignore_index=True)

    return dataframe_resultat



# #####################################################################################
# -- CALCUL METRIQUES: évaluation de la stabilité d'initialisation (Algo K-Prototype)
# #####################################################################################


def calcul_stabilite_init_kproto(nom_algo, model, var_cat, data, 
                                 dataframe_resultat, nb_iter=5):
    '''
    CALCUL DES METRIQUES pour évaluer la stabilité d'initialisation.
    Parameters
    ----------
    nom_algo : str, nom de l'algorithme, obligatoire.
    model : instanciation du modèle dont on veut analyser la stabilité 
            d'initialisation, obligatoire.
    data : données d'entréedu modèle, obligatoire.
    var_cat : liste des index des variables catégorielles.
    dataframe_resultat : dataframe de sauvegarde des résultats.
    nb_iter : nombre d'itérations (5 par défaut).
    Returns
    -------
    dataframe_resultat : dataframe de sauvegarde des résultats.
    '''
    # Creating randomly initialized partitions for comparison
    # --------------------------------------------------------

    # Initializing the list of partitions
    partitions = []

    # Iterating
    for i in range(nb_iter):

        # Fitting the model
        model.fit(data, categorical=var_cat)

        # Getting the results (labels of points)
        partitions.append(model.labels_)

    # Computing the ARI scores between partitions
    # --------------------------------------------------------

    # Initializing list of ARI scores
    ARI_scores = []

    # For each partition, except last one
    for i in range(nb_iter - 1):
        # Compute the ARI score with other partitions
        for j in range(i + 1, nb_iter):
            ARI_score = metrics.adjusted_rand_score(partitions[i], partitions[j])
            ARI_scores.append(ARI_score)

    # Compute the mean and standard deviation of ARI scores
    ARI_mean = statistics.mean(ARI_scores)
    ARI_std = statistics.stdev(ARI_scores)

    # Display results
    print(
        "Evaluation of stability upon random initialization:\
        {:.1f}%  ± {:.1f}% ".format(100 * ARI_mean, 100 * ARI_std))

    dataframe_resultat = dataframe_resultat.append(pd.DataFrame({
        'Algos': [nom_algo],
        'ARI_mean': [ARI_mean],
        'ARI_std': [ARI_std]
    }), ignore_index=True)

    return dataframe_resultat



# #####################################################################################
# -- CALCUL METRIQUES: Stabilité des custers, suivi resultats ARI
# #####################################################################################



def segmentation_kmean_periode(dataframe, dataframe_resultat, date_ref, titre,
                                   nb_clusters):
        '''
        Segmentation de clientèle à partir d'une date + métrique de stabilité.
        Parameters
        ----------
        dataframe : dataframe à analyser, obligatoire.
        dataframe_resutat : dataframe de sauvegarde des scores ARI, obligatoire.
        date_ref : date de fin d'analyse de la segmentation avant la date de fin
                   d'historique, string au format 'YYYY-MM-DD HH24:MI:SS'.
        titre : titre correspondant à la période de résultat pour le dataframe de
                sauvegarde des scores ARI, obligatoire.
        nb_clusters : nombre de cluster, obligatoire.
        Returns
        -------
        dataframe_resutat : dataframe des sauvegarde des résultats ARI
        df_km : le dataframe de segmentation Kmeans sur la période historique.
        '''

        # ------------------------------------------------------------------------
        # Préparation des dataframes de travail
        # ------------------------------------------------------------------------
        dataframe['date_dernier_achat'] = \
            pd.to_datetime(dataframe['date_dernier_achat'],
                           format='%Y-%m-%d %H:%M:%S')
        df_copie = dataframe.copy()
        
        # Création des 2 tables de comparaison de la stabilité
        df_hist = \
            df_copie[df_copie['date_dernier_achat'] < date_ref]

        # On garde les clients qui étaient dans la base de données sur la période
        # historique
        df_copie_ref = dataframe.copy()
        df_ref = df_copie_ref[df_copie_ref.customer_unique_id.isin(
            df_hist.customer_unique_id)]

        # Sélection des variables numériques
        cols_num_cat = df_ref.select_dtypes(include=[np.number]).columns.to_list()
        
        # Transformation en logarithme pour avoir le même poids
        col_to_log_cat = cols_num_cat[2:17]
        df_hist[col_to_log_cat] = df_hist[col_to_log_cat].apply(np.log1p, axis=1)
        df_ref[col_to_log_cat] = df_ref[col_to_log_cat].apply(np.log1p, axis=1)

        # Standardisation StandardScaler - variable transformées en log
        # -----------------------------------------------------------------------
        # Préparation des données
        X_df_hist = df_hist[cols_num_cat].values
        features_df_hist = df_hist[cols_num_cat].columns
        
        # Standardisation avec StandardScaler (centre, réduit et rend plus la
        # distribution plus normale)
        scaler = StandardScaler()
        X_scaled_hist = scaler.fit_transform(X_df_hist)
        
        # Dataframe
        X_scaled_hist = pd.DataFrame(X_scaled_hist,
                                     index=df_hist[col_to_log_cat].index,
                                     columns=features_df_hist)

        X_df_ref = df_ref[cols_num_cat].values
        features_df_ref = df_ref[cols_num_cat].columns
        
        # Standardisation avec StandardScaler (centre, réduit et rend plus la
        # distribution plus normale)
        scaler = StandardScaler()
        X_scaled_ref = scaler.fit_transform(X_df_ref)  
        
        # Dataframe
        X_scaled_ref = pd.DataFrame(X_scaled_ref,
                                    index=df_ref[col_to_log_cat].index,
                                    columns=features_df_ref)

        # Encodage du moyen de paiement préféré
        encod_paiement = pd.get_dummies(df_hist['moyen_paiement_prefere'])
        X_scaled_hist = X_scaled_hist.join(encod_paiement)
        encod_paiement_2 = pd.get_dummies(df_ref['moyen_paiement_prefere'])
        X_scaled_ref = X_scaled_ref.join(encod_paiement_2)

        # Encodage de la catégorie préférée
        encod_cat_pref = pd.get_dummies(df_hist['cat_produit_prefere'])
        X_scaled_hist = X_scaled_hist.join(encod_cat_pref)
        encod_cat_pref_2 = pd.get_dummies(df_ref['cat_produit_prefere'])
        X_scaled_ref = X_scaled_ref.join(encod_cat_pref_2)

        # ------------------------------------------------------------------------
        # Clustering Kmeans sur les 2 périodes historique et de référence
        # ------------------------------------------------------------------------
        # Instanciation de kmeans
        kmeans = KMeans(n_clusters=nb_clusters, 
                        n_init=20,
                        init='k-means++')

        # Entaînement de kmeans sur la période historique
        kmeans.fit(X_scaled_hist)
        kmeans_labels_hist = kmeans.labels_
        X_scaled_hist['Cluster'] = kmeans.labels_
        
        # Entaînement de sur la période de référence
        kmeans.fit(X_scaled_ref)
        kmeans_labels_ref = kmeans.labels_
        X_scaled_ref['Cluster'] = kmeans.labels_

        # ------------------------------------------------------------------------
        # Scoring ARI de la stabilité
        # ------------------------------------------------------------------------
        # Calcul du score ARI
        ARI_kmeans = round(100*(metrics.adjusted_rand_score(kmeans_labels_ref,
                                                 kmeans_labels_hist)),3)
        

        # Sauvegarde de l'ARI dans le tableau de résultats
        dataframe_resultat = dataframe_resultat.append(pd.DataFrame({
                                    'Periode': [titre],
                                    'Date' : [date_ref],
                                    'ARI (%)': [ARI_kmeans]}),
                              ignore_index=True)

        return dataframe_resultat, X_scaled_hist, X_scaled_ref
    


# #####################################################################################
# -- CALCUL METRIQUES: Stabilité des clusters dans le temps, 
# -- suivi resultats ARI (Algo K-Prototype)
# #####################################################################################
    
    
  
def segmentation_kproto_periode(dataframe, dataframe_resultat, date_ref, titre,
                               nb_clusters):
    '''
    Segmentation de clientèle à partir d'une date + métrique de stabilité.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    dataframe_resutat : dataframe de sauvegarde des scores ARI, obligatoire.
    date_ref : date de fin d'analyse de la segmentation avant la date de fin
               d'historique, string au format 'YYYY-MM-DD HH24:MI:SS'.
    titre : titre correspondant à la période de résultat pour le dataframe de
            sauvegarde des scores ARI, obligatoire.
    nb_clusters : nombre de cluster, obligatoire.
    Returns
    -------
    dataframe_resutat : dataframe des sauvegarde des résultats ARI
    df_km : le dataframe de segmentation Kmeans sur la période historique.
    '''

    # ------------------------------------------------------------------------
    # Préparation des dataframes de travail
    # ------------------------------------------------------------------------
    dataframe['date_dernier_achat'] = \
        pd.to_datetime(dataframe['date_dernier_achat'],
                       format='%Y-%m-%d %H:%M:%S')
    df_copie = dataframe.copy()

    # Création des 2 tables de comparaison de la stabilité
    df_hist = \
        df_copie[df_copie['date_dernier_achat'] < date_ref]

    # On garde les clients qui étaient dans la base de données sur la période
    # historique
    df_copie_ref = dataframe.copy()
    df_ref = df_copie_ref[df_copie_ref.customer_unique_id.isin(
        df_hist.customer_unique_id)]

    # Suppression colonnes
    cols_a_suppr = ['customer_unique_id', 'date_premier_achat',
                    'date_dernier_achat']
    df_hist.drop(cols_a_suppr, axis=1, inplace=True)
    df_ref.drop(cols_a_suppr, axis=1, inplace=True)

    # Pré-processing
    scaler = StandardScaler()
    for col in df_hist.select_dtypes(exclude='object').columns:
        if col != 'geolocation_lat' :
            if col != 'geolocation_lng':
                df_hist[col] = df_hist[col].apply(np.log1p)
        df_hist[col] = scaler.fit_transform(np.array(df_hist[col])
                                              .reshape(-1, 1))
    scaler = StandardScaler()
    for col in df_ref.select_dtypes(exclude='object').columns:
        if col != 'geolocation_lat' :
            if col != 'geolocation_lng':
                df_ref[col] = df_ref[col].apply(np.log1p)
        df_ref[col] = scaler.fit_transform(np.array(df_ref[col])
                                              .reshape(-1, 1))        

    # Détermine les variables catégorielles par leurs index
    cols_categorical = [0, 14, 17] 

    # ------------------------------------------------------------------------
    # Clustering Kmeans sur les 2 périodes historique et de référence
    # ------------------------------------------------------------------------

    # Instanciation du modèle de clustering
    kproto = KPrototypes(n_clusters= 6, init='Cao', n_jobs = -1)
    kproto.fit_predict(df_hist, categorical=cols_categorical)
    kproto_labels_hist = kproto.labels_
    df_hist['Cluster'] = kproto.labels_
    kproto.fit_predict(df_ref, categorical=cols_categorical)
    kproto_labels_ref = kproto.labels_
    df_ref['Cluster'] = kproto.labels_

    # ------------------------------------------------------------------------
    # Scoring ARI de la stabilité
    # ------------------------------------------------------------------------
    # Calcul du score ARI
    ARI_kproto = round(100*(metrics.adjusted_rand_score(kproto_labels_ref,
                                             kproto_labels_hist)),3)


    # Sauvegarde de l'ARI dans le tableau de résultats
    dataframe_resutat = dataframe_resutat.append(pd.DataFrame({
        'Periode': [titre],
        'Date' : [date_ref],
        'ARI (%)': [ARI_kproto]}),
        ignore_index=True)

    return dataframe_resutat, df_hist, df_ref

    
  


 # #####################################################################################
 # -- CALCUL SEGMENTATION: fonctions de segmentations
 # #####################################################################################


def r_score(x, c, ds):

    if x <= ds[c][.2]:
        return 5
    elif x <= ds[c][.4]:
        return 4
    elif x <= ds[c][.6]:
        return 3
    elif x <= ds[c][.8]:
        return 2
    else:
        return 1
    

def fm_score(x, c, ds):
    if x <= ds[c][.2]:
        return 1
    elif x <= ds[c][.4]:
        return 2
    elif x <= ds[c][.6]:
        return 3
    elif x <= ds[c][.8]:
        return 4
    else:
        return 5    



 # #####################################################################################
 # -- CALCUL SEGMENTATIONS: fonctions d'attribution des segments
 # #####################################################################################    
    
    
def calculate_segments(rfm):
    """
    Retourne le segment associé au client en fonction de son score pour les
    variables R, F et M

    Paramètres:
    rfm(pd.DataFrame): doit contenir les colonnes R, F et M

    Return:
    str: Nom du segment
    """

    if 4 <= rfm["R"] <= 5 and 4 <= rfm["F"] <= 5 and 4 <= rfm["M"] <= 5:
        return "Champions"

    elif 3 <= rfm["R"] <= 5 and 3 <= rfm["F"] <= 5 and 2 <= rfm["M"] <= 5:
        return "Loyal Customers"

    elif 3 <= rfm["R"] <= 5 and 1 <= rfm["F"] <= 3 and 1 <= rfm["M"] <= 3:
        return "Potential Loyalist"

    elif rfm["R"] >= 4 and rfm["F"] <= 2:
        return "Recent Customers"

    elif 3 <= rfm["R"] <= 4 and rfm["F"] <= 1 and rfm["M"] <= 1:
        return "Promising"

    elif 2 <= rfm["R"] <= 3 and 2 <= rfm["F"] <= 3 and 2 <= rfm["M"] <= 3:
        return "Need Attention"

    elif 2 <= rfm["R"] <= 3 and rfm["F"] <= 2 and rfm["M"] <= 2:
        return "About to Sleep"

    elif rfm["R"] <= 2 and 4 <= rfm["F"] <= 5 and 4 <= rfm["M"] <= 5:
        return "Can't Lose Them"

    elif rfm["R"] <= 2 and 2 <= rfm["F"] <= 5 and 2 <= rfm["M"] <= 5:
        return "At Risk"

    elif rfm["R"] <= 2 and rfm["F"] <= 2 and rfm["M"] <= 2:
        return "Lost customers"

    else:
        return "Others"
    
    
    
 # #####################################################################################
 # -- CALCUL SEGMENTATIONS: Stabilité des clusters dans le temps, suivi resultats ARI sur 
 # -- l'algo RFM Clustering traditionnel
 # #####################################################################################     
    

def segmentation_rfm_periode(dataframe,  dataframe_rfm_complet, dataframe_resultat, var_recence, 
                             var_frequence, var_montant,  date_ref, titre):
         
        # ------------------------------------------------------------------------
        # Préparation des dataframes de travail
        # ------------------------------------------------------------------------
        
        # Conversion de la colonne 'order_purchase_timestamp' au format datetime
        # pour calculs ultérieurs
        dataframe[var_recence] = pd.to_datetime(dataframe[var_recence],
                                                format='%Y-%m-%d %H:%M:%S')
        
        df_copie = dataframe.copy()
        
        # Création des 2 tables de comparaison de la stabilité
        df_hist = \
            df_copie[df_copie[var_recence] <= date_ref]

        # On garde les clients qui étaient dans la base de données sur la période
        # historique
        df_copie_ref = dataframe.copy()
        df_ref = df_copie_ref[df_copie_ref.customer_unique_id.isin(
            df_hist.customer_unique_id)]
        

        # Date de référence fixée au lendemain de la dernière date enregistrée
        # dans la table ref et historique
        date_reference_hist = max(df_hist[var_recence]) + dt.timedelta(1)
        
#         print('Date de référence : {}'.format(date_reference_hist))
        
        date_reference_ref = max(df_ref[var_recence]) + dt.timedelta(1)
#         print('Date de référence : {}'.format(date_reference_ref))
        

        # Table RFM x mois avant fin historique
        df_rfm_hist = df_hist.groupby('customer_unique_id').agg({var_recence:
                                              lambda x: (date_reference_hist
                                                         - x.max()).days,
                                              var_frequence: 'count',
                                              var_montant: 'sum'}
                                             )
        # Table RFM x mois avant fin historique
        df_rfm_ref = df_ref.groupby('customer_unique_id').agg({var_recence:
                                              lambda x: (date_reference_ref
                                                         - x.max()).days,
                                              var_frequence: 'count',
                                              var_montant: 'sum'}
                                             )
        
        df_rfm_hist.rename(columns={var_recence: 'rfm_recence',
                               var_frequence: 'rfm_frequence',
                               var_montant: 'rfm_montant'}, inplace=True)
         
        df_rfm_ref.rename(columns={var_recence: 'rfm_recence',
                               var_frequence: 'rfm_frequence',
                               var_montant: 'rfm_montant'}, inplace=True)
              

        # Utilisation de la méthode des quantiles
        qtiles_ref = df_rfm_ref.quantile([.2, .4, .6, .8]).to_dict()
        qtiles_hist = df_rfm_hist.quantile([.2, .4, .6, .8]).to_dict()
        
        def r_score(x, c, ds):

            if x <= ds[c][.2]:
                return 5
            elif x <= ds[c][.4]:
                return 4
            elif x <= ds[c][.6]:
                return 3
            elif x <= ds[c][.8]:
                return 2
            else:
                return 1

        def fm_score(x, c, ds):
            if x <= ds[c][.2]:
                return 1
            elif x <= ds[c][.4]:
                return 2
            elif x <= ds[c][.6]:
                return 3
            elif x <= ds[c][.8]:
                return 4
            else:
                return 5 


        # Assignation des bins aux variables R, F, M
        # ------------------------------------------------------------------------

        df_rfm_ref['R'] = df_rfm_ref['rfm_recence'].apply(r_score, args=('rfm_recence', qtiles_ref))
        df_rfm_ref['F'] = df_rfm_ref['rfm_frequence'].apply(fm_score, args=('rfm_frequence', qtiles_ref))
        df_rfm_ref['M'] = df_rfm_ref['rfm_montant'].apply(fm_score, args=('rfm_montant', qtiles_ref))
        
        df_rfm_hist['R'] = df_rfm_hist['rfm_recence'].apply(r_score, args=('rfm_recence', qtiles_hist))
        df_rfm_hist['F'] = df_rfm_hist['rfm_frequence'].apply(fm_score, args=('rfm_frequence', qtiles_hist))
        df_rfm_hist['M'] = df_rfm_hist['rfm_montant'].apply(fm_score, args=('rfm_montant', qtiles_hist))


        # Concaténation des groupes R, F et M ==> segment du client
        # ------------------------------------------------------------------------
        df_rfm_ref['RFM_Segment'] = [str(row[0]) + str(row[1]) + str(row[2])
                                 for row in zip(df_rfm_ref['R'], df_rfm_ref['F'],
                                                df_rfm_ref['M'])]
        
        df_rfm_hist['RFM_Segment'] = [str(row[0]) + str(row[1]) + str(row[2])
                                 for row in zip(df_rfm_hist['R'], df_rfm_hist['F'],
                                                df_rfm_hist['M'])]

        # Calul du score
        # ------------------------------------------------------------------------
        df_rfm_hist['RFM_Score'] = df_rfm_hist[['R', 'F', 'M']].sum(axis=1)
        df_rfm_ref['RFM_Score'] = df_rfm_ref[['R', 'F', 'M']].sum(axis=1)
        

        def calculate_segments(df):
            """
            Retourne le segment associé au client en fonction de son score pour les
            variables R, F et M

            Paramètres:
            rfm(pd.DataFrame): doit contenir les colonnes R, F et M

            Return:
            str: Nom du segment
            """

            if 4 <= df["R"] <= 5 and 4 <= df["F"] <= 5 and 4 <= df["M"] <= 5:
                return "Champions"

            elif 3 <= df["R"] <= 5 and 3 <= df["F"] <= 5 and 2 <= df["M"] <= 5:
                return "Loyal Customers"

            elif 3 <= df["R"] <= 5 and 1 <= df["F"] <= 3 and 1 <= df["M"] <= 3:
                return "Potential Loyalist"

            elif df["R"] >= 4 and df["F"] <= 2:
                return "Recent Customers"

            elif 3 <= df["R"] <= 4 and df["F"] <= 1 and df["M"] <= 1:
                return "Promising"

            elif 2 <= df["R"] <= 3 and 2 <= df["F"] <= 3 and 2 <= df["M"] <= 3:
                return "Need Attention"

            elif 2 <= df["R"] <= 3 and df["F"] <= 2 and df["M"] <= 2:
                return "About to Sleep"

            elif df["R"] <= 1 and 4 <= df["F"] <= 5 and 4 <= df["M"] <= 5:
                return "Can't Lose Them"

            elif df["R"] <= 2 and 2 <= df["F"] <= 5 and 2 <= df["M"] <= 5:
                return "At Risk"

            elif df["R"] <= 2 and df["F"] <= 2 and df["M"] <= 2:
                return "Lost customers"

            else:
                return "Others"
    
    
        # RFM Score: application de la fonction de segmentation clients
        df_rfm_hist["Segment"] = df_rfm_hist.apply(calculate_segments, axis=1)
        df_rfm_ref["Segment"] = df_rfm_ref.apply(calculate_segments, axis=1)

        # Replace customer_unique_id comme variable et non index
        df_rfm_hist.reset_index(inplace=True)
        df_rfm_ref.reset_index(inplace=True)
        
        dataframe_rfm_complet_hist = dataframe_rfm_complet[dataframe_rfm_complet.customer_unique_id.isin(
                                      df_rfm_hist['customer_unique_id'])]
    
        df_rfm_hist = df_rfm_hist[df_rfm_hist.customer_unique_id.isin(
                                      dataframe_rfm_complet_hist['customer_unique_id'])]
        
        dataframe_rfm_complet_ref = dataframe_rfm_complet[dataframe_rfm_complet.customer_unique_id.isin(
                                      df_rfm_hist['customer_unique_id'])]
    
        df_rfm_ref = df_rfm_ref[df_rfm_ref.customer_unique_id.isin(
                                      dataframe_rfm_complet_ref['customer_unique_id'])]
    
        
        # ------------------------------------------------------------------------
        # Scoring ARI de la stabilité
        # ------------------------------------------------------------------------
        # Calcul du score ARI
        ARI_rfm_ref = round(100*(metrics.adjusted_rand_score(dataframe_rfm_complet_hist.RFM_Segment,
                                                 df_rfm_ref.RFM_Segment)),3)


        # Sauvegarde de l'ARI dans le tableau de résultats
        dataframe_resultat = dataframe_resultat.append(pd.DataFrame({
                                    'Periode': [titre],
                                    'Date' : [date_ref],
                                    'ARI (%)': [ARI_rfm_ref]}),
                              ignore_index=True)

        return dataframe_resultat, df_rfm_hist, df_rfm_ref

    
    
    
 # #####################################################################################
 # -- VISUALISATION RFM: fonction de visualisation 3d de la dispersion RFM 
 # #####################################################################################     
    

def proj_3d_seg(rfm_table, label_score, colors) :
    """
    Graphique montrant les différents clients associés à leur segment/cluster respectif
    rfmTable(pd.DataFrame): Table de résultats
    label_score(str): nom de la segmentation
    colors(list): liste de couleurs

    """
    fig = plt.figure(1, figsize=(18, 8))

    dx = fig.add_subplot(111, projection='3d')
    # Pour chaque segment, scatter plot des individus avec code couleur
    for i, segment in enumerate(rfm_table[label_score].unique()):
        dx.scatter(rfm_table[rfm_table[label_score] == segment].rfm_recence,
                   rfm_table[rfm_table[label_score] == segment].rfm_frequence,
                   rfm_table[rfm_table[label_score] == segment].rfm_montant,
                   label=segment,
                   s=50,
                   c=colors[segment])
    # Titre des axes et titre graphique
    dx.set_xlabel("Recency")
    dx.set_ylabel("Frequency")
    dx.set_zlabel("Monetary")
    plt.title("Représentation 3D des différents individus dans chaque segment")

    plt.legend()
    plt.show()    
    
    
    
 # #####################################################################################
 # -- VISUALISATION RFM: fonction de visualisation 2d de la dispersion RFM 
 # #####################################################################################    
    
    
def proj_2d_seg(rfm_table, xlabel, ylabel, label_score, colors) :
    """
    2D scatter plot
    data(pd.DataFrame): contient les 3 colonnes xlabel, ylabel et label_score
    xlabel(str): nom colonne data pour l'axe des x
    ylabel(str): nom colonnne data pour l'axe des y
    label_score: nom méthodologie de la segmentation
    """
    # Pour chaque segment, scatter plot
    for i, segment in enumerate(rfm_table[label_score].unique()):
        plt.scatter(rfm_table[rfm_table[label_score] == segment][xlabel],
                    rfm_table[rfm_table[label_score] == segment][ylabel],
                    label=segment,
                    c=colors[segment]
                   )
    # Titre des axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    
    
    

 # #####################################################################################
 # -- CALCUL SEGMENTATIONS: Stabilité des clusters dans le temps, suivi resultats ARI sur 
 # -- l'algo RFM Clustering traditionnel
 # #####################################################################################     
    
    
def segm_rfm_periode(dataframe, dataframe_rfm_complet,
                             dataframe_resutat, id_unique, var_recence,
                             var_frequence, var_montant, datetime, titre):
    '''
    Segmentation de clientèle à partir d'une date + métrique de stabilité.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    dataframe_rfm_complet : dataframe avec la segmentationRFM sur la période
                           complète d'analyse, obligatoire.
    dataframe_resutat : dataframe de sauvegarde des scores ARI, obligatoire.
    var_recence : date de dernière achat effectué (object), obligatoire.
    var_frequence : nombre d'achat, obligatoire.
    var_montant : montant d'achat, obligatoire.
    datetime : date de fin d'analyse de la segmentation avant la date de fin
               d'historique, string au format 'YYYY-MM-DD HH24:MI:SS'.
    titre : titre correspondant à la période de résultat pour le dataframe de
            sauvegarde des scores ARI, obligatoire.
    Returns
    -------
    dataframe_resutat : dataframe des sauvegarde des résultats ARI
    df_rfm : le dataframe de segmentation RFM sur la période historique.
    '''

    # Conversion de la colonne 'order_purchase_timestamp' au format datetime
    # pour calculs ultérieurs
    dataframe[var_recence] = pd.to_datetime(dataframe[var_recence],
                                            format='%Y-%m-%d %H:%M:%S')

    # Dataframe arrêté à la date transmise en paramètre avant la fin de
    # l'historique
    data = dataframe[dataframe[var_recence] <= datetime]

    # Vérification min max historique
    print('Période : Min : {}, Max : {}'.format(min(data[var_recence]),
                                                max(data[var_recence])))

    # Date de référence fixée au lendemain de la dernière date enregistrée
    # dans la table
    date_reference = max(data[var_recence]) + dt.timedelta(1)
    print('Date de référence : {}'.format(date_reference))

    # Table RFM x mois avant fin historique
    df_rfm = data.groupby(id_unique).agg({var_recence:
                                          lambda x: (date_reference
                                                     - x.max()).days,
                                          var_frequence: 'count',
                                          var_montant: 'sum'}
                                         )
    df_rfm.rename(columns={var_recence: 'rfm_recence',
                           var_frequence: 'rfm_frequence',
                           var_montant: 'rfm_montant'}, inplace=True)
    
    
    # Transformer les valeurs en logarithme pour que chaque variable aie
    # le même poids
    df_rfm_log = df_rfm.copy()
    cols_preproc = ['rfm_recence', 'rfm_frequence', 'rfm_montant']
    df_rfm_log = df_rfm_log[cols_preproc].apply(np.log, axis=1)
    
    # Encodeur par normalisation
    encoder = StandardScaler()

    # Normalisation des données
    var_rfm_transformed = encoder.fit_transform(
        df_rfm_log[["rfm_recence", "rfm_frequence", "rfm_montant"]])


    # Réduction dimmensionnelle avec 'Principal Component Analysis'
    pca = PCA(n_components=0.90)  # 90% de la variance
    rfm_pca = pca.fit_transform(var_rfm_transformed)

    # Mise sous forme de DataFrame
    rfm_pca = pd.DataFrame(rfm_pca,
                           columns=["Composante_" + str(i)
                                    for i in range(rfm_pca.shape[1])])
    
    # Clustering avec k=4 comme meilleur hyperparamètre
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300)

    # Clustering pour chacun des clients
    clusters = kmeans.fit_predict(rfm_pca)
    df_rfm["clusters"] = clusters
    
    # Renommage des différents cluters et regroupement
    df_rfm.loc[df_rfm["clusters"] ==
                 0, "clusters"] = 'Rang I'

    df_rfm.loc[df_rfm["clusters"] ==
                 1, "clusters"] = 'Rang III'

    df_rfm.loc[df_rfm["clusters"] ==
                 2, "clusters"] = 'Rang II'

    df_rfm.loc[df_rfm["clusters"] ==
                 3, "clusters"] = 'Rang IV'
     
    
    # Utilisation de la méthode des quantiles
    qtiles = df_rfm.quantile([.2, .4, .6, .8]).to_dict()


    # Assignation des bins aux variables R, F, M
    # ------------------------------------------------------------------------
     
    df_rfm['R'] = df_rfm['rfm_recence'].apply(r_score, args=('rfm_recence', qtiles))
    df_rfm['F'] = df_rfm['rfm_frequence'].apply(fm_score, args=('rfm_frequence', qtiles))
    df_rfm['M'] = df_rfm['rfm_montant'].apply(fm_score, args=('rfm_montant', qtiles))


    # Concaténation des groupes R, F et M ==> segment du client
    # ------------------------------------------------------------------------
    df_rfm['RFM_Segment'] = [str(row[0]) + str(row[1]) + str(row[2])
                             for row in zip(df_rfm['R'], df_rfm['F'],
                                            df_rfm['M'])]

    # Calul du score
    # ------------------------------------------------------------------------
    df_rfm['RFM_Score'] = df_rfm[['R', 'F', 'M']].sum(axis=1)

    def calculate_segments(rfm):
        """
        Retourne le segment associé au client en fonction de son score pour les
        variables R, F et M

        Paramètres:
        rfm(pd.DataFrame): doit contenir les colonnes R, F et M

        Return:
        str: Nom du segment
        """

        if 4 <= rfm["R"] <= 5 and 4 <= rfm["F"] <= 5 and 4 <= rfm["M"] <= 5:
            return "Champions"

        elif 3 <= rfm["R"] <= 5 and 3 <= rfm["F"] <= 5 and 2 <= rfm["M"] <= 5:
            return "Loyal Customers"

        elif 3 <= rfm["R"] <= 5 and 1 <= rfm["F"] <= 3 and 1 <= rfm["M"] <= 3:
            return "Potential Loyalist"

        elif rfm["R"] >= 4 and rfm["F"] <= 2:
            return "Recent Customers"

        elif 3 <= rfm["R"] <= 4 and rfm["F"] <= 1 and rfm["M"] <= 1:
            return "Promising"

        elif 2 <= rfm["R"] <= 3 and 2 <= rfm["F"] <= 3 and 2 <= rfm["M"] <= 3:
            return "Need Attention"

        elif 2 <= rfm["R"] <= 3 and rfm["F"] <= 2 and rfm["M"] <= 2:
            return "About to Sleep"

        elif rfm["R"] <= 1 and 4 <= rfm["F"] <= 5 and 4 <= rfm["M"] <= 5:
            return "Can't Lose Them"

        elif rfm["R"] <= 2 and 2 <= rfm["F"] <= 5 and 2 <= rfm["M"] <= 5:
            return "At Risk"

        elif rfm["R"] <= 2 and rfm["F"] <= 2 and rfm["M"] <= 2:
            return "Lost customers"

        else:
            return "Others"
    
    
    # RFM Score: application de la fonction de segmentation clients
    df_rfm["Segment"] = df_rfm.apply(calculate_segments, axis=1)

    # Replace customer_unique_id comme variable et non index
    df_rfm.reset_index(inplace=True)
    

    # On garde les clients de cette période historique qui étaient dans le jeu
    # de données de la période complète
    dataframe_rfm_complet = dataframe_rfm_complet[dataframe_rfm_complet.customer_unique_id.isin(
    df_rfm[id_unique])]
    
    df_rfm = df_rfm[df_rfm.customer_unique_id.isin(
        dataframe_rfm_complet[id_unique])]
    
    
    print(dataframe_rfm_complet.shape)
    print(df_rfm.shape)

    # Calcul ARI rfm
    ARI_rfm = round(100*(metrics.adjusted_rand_score(
        dataframe_rfm_complet.RFM_Segment, df_rfm.RFM_Segment)),3)
    print(f'Métrique ARI rfm (%) : {ARI_rfm}')
    
    # Calcul ARI clusters
    ARI_clusters = round(100*(metrics.adjusted_rand_score(
        dataframe_rfm_complet.K_Cluster, df_rfm.clusters)),3)
    print(f'Métrique ARI clusters (%) : {ARI_clusters}')    


    # Ajout ARI dans le tableau de résultats
    dataframe_resutat = dataframe_resutat.append(pd.DataFrame({
        'Periode': [titre],
        'ARI_rfm (%)': [ARI_rfm],
        'ARI_clusters (%)': [ARI_clusters]}), ignore_index=True)

    return dataframe_resutat, df_rfm    
    
    
    
    
    
    
# #####################################################################################
# -- VISUALISATION STABILITE DES FLUX: Fonction générant un dictionnaire associés aux
# -- paramètres du diagramme de Sankey 
# #####################################################################################       
    
    
def sankey_data(method, *rfm):

    """
    La fonction sankey_data retourne un dictionnaire adapté au tracé
    d'un diagramme de Sankey
    Plus d'informations : https://plot.ly/python/sankey-diagram/

    Paramètres :
    method: Nom de la colonne dont les informations sont à récupérer
    *rfm: Au moins 2 DataFrame correspondant à des segmentations
    temporellement différentes

    Return :

    dict: Dictionnaire de résultats

    """

    sources = []
    targets = []
    values = []
    sankey_data = {}

    list_segments = []
    list_segments_0 = []
    list_segments_1 = []

    sankey_data["label"] = sorted(list(rfm[0][method].unique()))
    source = 0
    target = 0
    i = 0

    while i < len(rfm)-1:

        list_segments_0 = sorted(list(rfm[i][method].unique()))
        list_segments_1 = sorted(list(rfm[i + 1][method].unique()))

        list_segments.append(list_segments_1)

        sankey_data["label"] += list_segments_1

        target += len(list_segments_0)
        for segment_0 in list_segments_0:
            for segment_1 in list_segments_1:

                sources.append(list_segments_0.index(segment_0) + source)
                targets.append(list_segments_1.index(segment_1) + target)

                flow = sum(rfm[i + 1].iloc[rfm[i][rfm[i][method] ==
                           segment_0].index][method] == segment_1)

                values.append(flow)

        source += len(list_segments_0)
        i += 1

    sankey_data["source"] = sources
    sankey_data["target"] = targets
    sankey_data["value"] = values

    return sankey_data    





