#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import missingno as msno 
import seaborn as sns
from IPython.display import display
from wordcloud import WordCloud
from statsmodels.graphics.gofplots import qqplot
import re
from matplotlib.collections import LineCollection


# Version:
__version__ = '0.0.0'

# In[ ]:

# --------------------------------------------------------------------
# -- DESCRIPTION DES VARIABLES STATISTIQUES
# --------------------------------------------------------------------

def stat_descriptives(data):
    '''
    Fonction prenant un dataframe en entrée et retourne les variables, avec ses statistiques
    
    '''

    df = pd.DataFrame(columns=['Variable name', 'Mean', 'Median', 'Skew', 'Kurtosis', \
     'Variance', 'Stdev', 'min','25%','50%','75%','max'])
    
    for column in data.columns:
        var_type = data[column].dtypes
        if var_type != 'object':       
            df = df.append(pd.DataFrame([[column, data[column].mean(),data[column].median(), \
            data[column].skew(),data[column].kurtosis(),data[column].var(ddof=0),data[column].std(ddof=0), \
            data[column].min(),data[column].quantile(0.25),data[column].quantile(0.5),data[column].quantile(0.75), \
            data[column].max()]], columns=['Variable name', 'Mean', 'Median', 'Skew', 'Kurtosis', \
            'Variance', 'Stdev', 'min','25%','50%','75%','max']))
    
    df = df.reset_index(drop=True)
    return df


#

def null_var(df, tx_seuil=50):
    null_tx = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_tx.columns = ['Variable','Taux_de_Null']
    high_null_tx = null_tx[null_tx.Taux_de_Null >= tx_seuil]
    return high_null_tx

#

def fill_var(df, tx_min, tx_max):
    fill_tx = (100 - (df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    fill_tx.columns = ['Variable','Taux_de_remplissage']
    high_fill_tx = fill_tx[(fill_tx.Taux_de_remplissage >= tx_min) & (fill_tx.Taux_de_remplissage <= tx_max)]
    return high_fill_tx

#

# --------------------------------------------------------------------
# -- DESCRIPTION DES VARIABLES
# --------------------------------------------------------------------


def  get_nutri_col(data,cols_suppr=False):
        columns_nutri = ['energy_100g',
                             'nutrition_score_fr_100g',
                             'saturated_fat_100g',
                             'sugars_100g',
                             'proteins_100g',
                             'fat_100g',
                             'carbohydrates_100g',
                             'salt_100g',
                             'fiber_100g']
        if cols_suppr:                      
            return data[columns_nutri].drop(cols_suppr,axis=1).columns.to_list()
        else:
            return data[columns_nutri].columns.to_list()
        
#

def rempl_caracteres(data, anc_car, nouv_car):
    """
    Remplacer les caractères avant par les caractères après
    dans le nom des variables du dataframe
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                car_avant : le caractère à remplacer
                car_apres : le caractère de remplacement
    @param OUT : dataframe modifié
    """
    # traces des variables à renommer
    cols_a_renom = data.columns[data.columns.str.contains(
        anc_car)]
    print(f'{len(cols_a_renom)} variables renommées \
          \'{anc_car}\' en \'{nouv_car}\' : \n\n {cols_a_renom.tolist()}')

    return data.columns.str.replace(anc_car, nouv_car)
        

# In[ ]:

def affichage_types_var(df_type, types, type_par_var, graph):
    """ Permet un aperçu du type des variables
    Parameters
    ----------
    @param IN : df_work : dataframe, obligatoire
                types : Si True lance dtypes, obligatoire
                type_par_var : Si True affiche tableau des types de
                               chaque variable, obligatoire
                graph : Si True affiche pieplot de répartition des types
    @param OUT :None.
    """

    if types:
        # 1. Type des variables
        print("-------------------------------------------------------------")
        print("Type de variable pour chacune des variables\n")
        display(df_type.dtypes)

    if type_par_var:
        # 2. Compter les types de variables
        #print("Répartition des types de variable\n")
        values = df_type.dtypes.value_counts()
        nb_tot = values.sum()
        percentage = round((100 * values / nb_tot), 2)
        table = pd.concat([values, percentage], axis=1)
        table.columns = [
            'Nombre par type de variable',
            '% des types de variable']
        display(table[table['Nombre par type de variable'] != 0]
                .sort_values('% des types de variable', ascending=False)
                .style.background_gradient('seismic'))

    if graph:
        # 3. Schéma des types de variable
        # print("\n----------------------------------------------------------")
        #print("Répartition schématique des types de variable \n")
        # Répartition des types de variables
        plt.figure(figsize=(5,5))
        df_type.dtypes.value_counts().plot.pie( autopct='%.0f%%', pctdistance=0.85, radius=1.2)
        #plt.pie(df_type.dtypes.value_counts(), labels = df_type.dtypes.unique(), autopct='%.0f%%', pctdistance=0.85, radius=1.2)
        centre_circle = plt.Circle((0, 0), 0.8, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(label="Repartiton des types de variables",loc="left", fontstyle='italic')
        plt.show()

        
#

def get_val_manq(df_type, pourcentage, affiche_val_manq):
    """Indicateurs sur les variables manquantes
       @param in : df_work dataframe obligatoire
                   pourcentage : boolean si True affiche le nombre heatmap
                   affiche_heatmap : boolean si True affiche la heatmap
       @param out : none
    """

    # 1. Nombre de valeurs manquantes totales
    nb_nan_tot = df_type.isna().sum().sum()
    nb_donnees_tot = np.product(df_type.shape)
    pourc_nan_tot = round((nb_nan_tot / nb_donnees_tot) * 100, 2)
    print(
        f'Valeurs manquantes :{nb_nan_tot} NaN pour {nb_donnees_tot} données ({pourc_nan_tot} %)')

    if pourcentage:
        print("-------------------------------------------------------------")
        print("Nombre et pourcentage de valeurs manquantes par variable\n")
        # 2. Visualisation du nombre et du pourcentage de valeurs manquantes
        # par variable
        values = df_type.isnull().sum()
        percentage = 100 * values / len(df_type)
        table = pd.concat([values, percentage.round(2)], axis=1)
        table.columns = [
            'Nombres de valeurs manquantes',
            '% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0]
                .sort_values('% de valeurs manquantes', ascending=False))

    if affiche_val_manq:
        print("-------------------------------------------------------------")
        print("Heatmap de visualisation des valeurs manquantes")
        # 3. Heatmap de visualisation des valeurs manquantes
        msno.matrix(df_type)

#

def detail_type_var(data, type_var='all'):
    """
    Retourne la description des variables qualitatives/quantitatives
    ou toutes les variables du dataframe transmis :
    type, nombre de nan, % de nan et desc
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                type_var = 'all' ==> tous les types de variables
                           'cat' ==> variables catégorielles
                           'num' ==> variables quantitative
    @param OUT : dataframe de description des variables
    """
    n_df = data.shape[0]

    if type_var == 'num':
        det_var = data.describe()
    elif type_var == 'cat':
        det_var = data.describe(exclude=[np.number])
    else:
        det_var = data.describe(include='all')
    
    det_type = pd.DataFrame(data[det_var.columns].dtypes, columns=['type']).T
    nb_nan = n_df - det_var.loc['count'].T
    pourcentage_nan = nb_nan * 100 / n_df
    det_nan = pd.DataFrame([nb_nan, pourcentage_nan], index=['nb_nan', '%_nan'])
    det_var = pd.concat([det_type, det_nan, det_var])
    
    return det_var



#
        
# --------------------------------------------------------------------
# -- SUPRESSION VARIABLES POUR UN TAUX DE NAN (%)
# --------------------------------------------------------------------
def clean_nan(data, taux_nan):
#     """
#     Supprime les variables à partir d'un taux en % de nan.
#     Affiche les variables supprimées et les variables conservées
#     ----------
#     @param IN : dataframe : DataFrame, obligatoire
#                 seuil : on conserve toutes les variables dont taux de nan <80%
#                         entre 0 et 100, integer
#     @param OUT : dataframe modifié
#     """
    qty_nan = round((data.isna().sum() / data.shape[0]) * 100, 2)
    cols = data.columns.tolist()
    
    # Conservation seulement des variables avec valeurs manquantes >= 80%
    cols_conservées = qty_nan[qty_nan.values < taux_nan].index.tolist()
    
    cols_suppr = [col for col in cols if col not in cols_conservées]

    data = data[qty_nan[qty_nan.values < taux_nan].index.tolist()]

    print(f'Liste des variables éliminées :\n{cols_suppr}\n')

    print(f'Liste des variables conservées :\n{cols_conservées}')

    return data
    

# In[ ]:

def trace_dispersion_boxplot_qqplot(dataframe, variable, titre, unite):
    """
    Suivi des dipsersions : boxplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers
                titre :titre pour les graphiques (str)
                unite : unité pour ylabel boxplot (str)
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))

    data = dataframe[variable]

    ax1 = fig.add_subplot(1, 2, 1)
    box = sns.boxplot(data=data, color='violet', ax=ax1)
    box.set(ylabel=unite)

    plt.grid(False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2 = qqplot(data,
                 line='r',
                 **{'markersize': 5,
                    'mec': 'k',
                    'color': 'violet'},
                 ax=ax2)
    plt.grid(False)

    fig.suptitle(titre, fontweight='bold', size=14)
    plt.show()

# In[ ]:

def plot_var_filling (df, tx_min, tx_max, graph, axe, col):
    
    if graph:
            filling_var = fill_var(df, tx_min, tx_max)

            font_title = {'family': 'serif',
                          'color':  '#114b98',
                          'weight': 'bold',
                          'size': 18,
                         }
            
        
            sns.set(font_scale=1.2)
            sns.barplot(ax = axe, x="Taux_de_remplissage", y="Variable", data=filling_var, color = col)

    

# In[ ]:

def plot_columns_boxplots(data, columns=[], ncols=2, color="goldenrod"):
    if len(columns) == 0:
        columns = data.columns.values
        
    if len(columns) == 1:
        plt.figure(figsize=(9,3))
        sns.boxplot(x=data[columns[0]], color=color)
        
    else:
        fig, axs = plt.subplots(figsize=(20,20), ncols=ncols, nrows=math.ceil(len(columns) / ncols))
        for index, column in enumerate(columns):
            row_index = math.floor(index / ncols)
            col_index = index % ncols
            sns.boxplot(x=data[column], ax=axs[row_index][col_index], color=color)


# In[ ]:


# --------------------------------------------------------------------
# -- HISTPLOT BOXPLOT QQPLOT
# --------------------------------------------------------------------


def trace_histplot_boxplot_qqplot(dataframe, var):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                var : colonne dont on veut voir les outliers
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    fig.suptitle('Distribution de ' + str(var), fontsize=16)

    data = dataframe[var]

    ax0 = fig.add_subplot(1, 3, 1)
    sns.histplot(data, kde=True, color='goldenrod', ax=ax0)
    plt.xticks(rotation=60)

    ax1 = fig.add_subplot(1, 3, 2)
    sns.boxplot(data=data, color='goldenrod', ax=ax1)
    plt.grid(False)

    ax2 = fig.add_subplot(1, 3, 3)
    qqplot(data,
           line='r',
           **{'markersize': 5,
              'mec': 'k',
              'color': 'orange'},
           ax=ax2)
    plt.grid(False)
    plt.show()


def trace_multi_histplot_boxplot_qqplot(dataframe, liste_var):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                liste_var : colonnes dont on veut voir les outliers
    @param OUT :None
    """
    for col in liste_var:
        trace_histplot_boxplot_qqplot(dataframe, col)


def trace_histplot(
        dataframe,
        variable,
        col,
        titre,
        xlabel,
        xlim_bas,
        xlim_haut,
        ylim_bas,
        ylim_haut,
        kde=True,
        mean_median_mode=True,
        mean_median_zoom=False):
    """
    Histplot pour les variables quantitatives général + histplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les histplot
                titre : titre du graphique (str)
                xlabel:légende des abscisses
                xlim_bas : limite du zoom supérieur bas(int)
                xlim_haut : limite du zoom supérieur haut(int)
                ylim_bas : limite du zoom inférieur bas(int)
                ylim_haut : limite du zoom inférieur haut(int)
                kde : boolean pour tracer la distribution normale
                mean_median_mode : boolean pour tracer la moyenne, médiane et mode
                mean_median_zoom : boolean pour tracer la moyenne et médiane sur le graphique zoomé
    @param OUT :None
    """
    # Distplot général + zoom
    
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(titre, fontsize=20, y=1.03)
    data = dataframe[variable]
    
    ax = fig.add_subplot(2, 1, 1)
    ax = sns.boxplot(x=data, color=col)
    ax.set_xlim(xlim_bas, xlim_haut)
    ax.set_ylim(ylim_bas, ylim_haut)
    plt.grid(False)
    plt.xticks([], [])
    

    ax = fig.add_subplot(2, 1, 2)
    ax = sns.histplot(data, kde=kde, color=col)

    if mean_median_mode:
        ax.vlines(data.mean(), *ax.get_ylim(), color='red', ls='-', lw=1.5)
        ax.vlines(
            data.median(),
            *ax.get_ylim(),
            color='green',
            ls='-.',
            lw=1.5)
        ax.vlines(
            data.mode()[0],
            *ax.get_ylim(),
            color='goldenrod',
            ls='--',
            lw=1.5)
    ax.legend(['mode', 'mean', 'median'])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Nombre de produits', fontsize=12)
    plt.grid(False)
    
      
    plt.show()        
        
        

def trace_pieplot(dataframe, variable, titre, legende, liste_colors):
    """
    Suivi des dipsersions : bosplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers (str)
                titre :titre pour les graphiques (str)
                legende : titre de la légende
                liste_colors : liste des couleurs
    @param OUT :None
    """

    plt.figure(figsize=(7, 7))
    plt.title(titre, size=16)
    nb_par_var = dataframe[variable].sort_values().value_counts()
    nb_par_var = nb_par_var.loc[sorted(nb_par_var.index)]
    explode = [0.1]
    for i in range(len(nb_par_var) - 1):
        explode.append(0)
    wedges, texts, autotexts = plt.pie(
        nb_par_var, labels=nb_par_var.index, autopct='%1.1f%%', colors=liste_colors, textprops={
            'fontsize': 16, 'color': 'black', 'backgroundcolor': 'w'}, explode=explode)
    axes = plt.gca()
    axes.legend(
        wedges,
        nb_par_var.index,
        title=legende,
        loc='center right',
        fontsize=14,
        bbox_to_anchor=(
            1,
            0,
            0.5,
            1))
    plt.show()

#    

def aff_eboulis_plot(pca):
    tx_var_exp = pca.explained_variance_ratio_
    scree = tx_var_exp * 100
    plt.bar(np.arange(len(scree)) + 1, scree, color='SteelBlue')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(scree)) + 1, scree.cumsum(), c='green', marker='o')
    ax2.set_ylabel('Taux cumulatif de l\'inertie')
    ax1.set_xlabel('Rang de l\'axe d\'inertie')
    ax1.set_ylabel('Pourcentage d\'inertie')
    for i, p in enumerate(ax1.patches):
        ax1.text(
            p.get_width() /
            5 +
            p.get_x(),
            p.get_height() +
            p.get_y() +
            0.3,
            '{:.0f}%'.format(
                tx_var_exp[i] *
                100),
            fontsize=8,
            color='k')
    plt.title('Eboulis des valeurs propres')
    plt.gcf().set_size_inches(8, 4)
    plt.grid(False)
    plt.show(block=False)

 
    
    
def affiche_cercle(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
 
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))
 
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])
 
            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:],
                   angles='xy', scale_units='xy', scale=1, color="black")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
             
            # affichage des noms des variables 
            if labels is not None: 
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='8', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
             
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)
 
            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
         
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')
 
            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
 
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

###


# --------------------------------------------------------------------
# -- AFFICHE LE PLAN FACTORIEL
# --------------------------------------------------------------------

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


# --------------------------------------------------------------------
# -- AFFICHE PLUSIEURS PLANS FACTORIELS
# --------------------------------------------------------------------            

def projeter_plans_factoriels(X_projected, pca, liste_plans_fact=[(0, 1)],
                              alpha=1):
    '''
    Projeter le résultat de PCA sur plusieurs plans factoriels
    Parameters
    ----------
    X_projected : X transformés par pca, obligatoire.
    pca : pca décomposition, obligatoire.
    liste_plans_fact : liste des tuples de plans factoriels
                       (default : [(0, 1)]).
    alpha : alpha, optionnel.
    Returns
    -------
    None.
    '''
    for liste in liste_plans_fact:
        dim1 = liste[0]
        dim2 = liste[1]

        # Transformation en DataFrame pandas
        df_PCA = pd.DataFrame({
            'Dim1': X_projected[:, dim1],
            'Dim2': X_projected[:, dim2]
        })

        plt.figure(figsize=(12, 12))
        g_pca = sns.scatterplot(x='Dim1', y='Dim2', data=df_PCA,
                                alpha=alpha, color='SteelBlue')

        titre = 'Représentation des clients sur le plan factoriel (' + str(
            dim1) + ',' + str(dim2) + ')'

        plt.title(titre, size=20)
        g_pca.set_xlabel('Dim ' + str(dim1 + 1) + ' : ' +
                         str(round(pca.explained_variance_ratio_[0] * 100, 2))
                         + ' %', fontsize=15)
        g_pca.set_ylabel('Dim ' + str(dim2 + 1) + ' : ' +
                         str(round(pca.explained_variance_ratio_[1] * 100, 2))
                         + ' %', fontsize=15)
        plt.axvline(color='gray', linestyle='--', linewidth=1)
        plt.axhline(color='gray', linestyle='--', linewidth=1)
        plt.show()            
            
    
# --------------------------------------------------------------------
# -- KDE PLOT graphe
# --------------------------------------------------------------------    
def plot_graph(df_work):
    """Graph densité pour 1 ou plusieurs colonne d'un dataframe
       @param in : df_work dataframe obligatoire
       @param out : none
    """

    plt.figure(figsize=(10, 5))
    axes = plt.axes()

    label_patches = []
    colors = ['Blue', 'SeaGreen', 'Sienna', 'DodgerBlue', 'Purple','Green']

    i = 0
    for col in df_work.columns:
        label = col
        sns.kdeplot(df_work[col], color=colors[i])
        label_patch = mpatches.Patch(
            color=colors[i],
            label=label)
        label_patches.append(label_patch)
        i += 1
    plt.xlabel('')
    plt.legend(
        handles=label_patches,
        bbox_to_anchor=(
            1.05,
            1),
        loc=2,
        borderaxespad=0.,
        facecolor='white')
    plt.grid(False)
    axes.set_facecolor('white')

    plt.show()    
    
    

####

def suppr_ponct(val):
    """
    Suppression de la ponctuation au texte transmis en paramètres.
    Parameters
    ----------
    val : texte dont on veut supprimer la ponctuation
    Returns
    -------
    Texte sans ponctuation
    """
    if isinstance(val, str):  # éviter les nan
        val = val.lower()
        val = re.compile('[éèêë]+').sub("e", val)
        val = re.compile('[àâä]+').sub("a", val)
        val = re.compile('[ùûü]+').sub("u", val)
        val = re.compile('[îï]+').sub("i", val)
        val = re.compile('[ôö]+').sub("o", val)
        return re.compile('[^A-Za-z" "]+').sub("", val)
    return val


####







        
