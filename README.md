# Segmentation des clients d’un site e-commerce

---

## Objectif du projet

Ce projet vise à fournir aux équipes marketing d'Olist, un site e-commerce brésilien, une segmentation client actionnable pour leurs campagnes de communication.

Les objectifs spécifiques sont :
- Comprendre les différents types de clients via leurs comportements, leurs données personnelles, etc.
- Proposer un contrat de maintenance pour évaluer la stabilité des segments au fil du temps.

![Objectifs du projet](images/objectif_du_projet.png)

---

## Contexte

**Olist** est une plateforme e-commerce fondée en 2016, qui connecte acheteurs et vendeurs. Elle gère les commandes, paiements, suivis de livraison, et les avis des clients.

![Contexte Olist](images/contexte.png)

---

## Données utilisées

- **9 jeux de données** : commandes, géolocalisation, paiements, produits, avis clients, etc.
- Étapes clés :
  1. **Fusion des datasets**.
  2. **Nettoyage** (gestion des valeurs manquantes, suppression des variables inutiles, etc.).
  3. **Feature engineering** pour extraire des KPI pertinents :
     - **Géographiques** : localisation, fréquence des commandes.
     - **Comportementaux** : récence, fréquence, montant moyen du panier (RFM).
     - **Psychologiques** : note de satisfaction.

![Données utilisées](images/donnees_utilisees.png)

---

## Approche méthodologique

### 1. Segmentation RFM
- Analyse basée sur 3 variables : **récence**, **fréquence**, **montant**.
- Résultats exprimés en scores (exemple : 444 pour les meilleurs clients, 111 pour les moins actifs).
- Définition de groupes : **Champions**, **Fidèles**, **Clients à risque**, etc.

![Segmentation RFM](images/segmentation_rfm.png)

### 2. Apprentissage non supervisé (Clustering)
- Algorithmes testés : **K-Means**, **K-Means avec ACP**, et **K-Prototype**.
- Techniques utilisées :
  - **Prétraitement** : standardisation, encodage, transformation logarithmique.
  - **Réduction de dimensionnalité** : ACP, TSNE.
- Critères d’évaluation : silhouette, Davies-Bouldin, indice de Gini.

![Méthodologie Clustering](images/methodologie_clustering.png)

---

## Résultats

### Modèle final
- **Algorithme retenu** : K-Means avec 6 clusters.
- **Raisons** :
  - Bonne stabilité sur 3 mois.
  - Segments homogènes et actionnables.

![Résultats du modèle](images/resultats_final.png)

### Types de clients identifiés
1. **Meilleurs clients** utilisant les facilités de paiement.
2. **Clients fidèles**.
3. **Nouveaux clients**.
4. **Clients presque perdus**.
5. **Bons clients** sur le point de devenir inactifs.

![Types de clients](images/types_clients.png)

---

## Propositions pour l'avenir

1. **Amélioration des données** :
   - Inclure des facteurs externes (profils familiaux, habitudes spécifiques).
   - Analyser des comportements sur plusieurs cycles d'achat.
2. **Segmentation plus fine** :
   - Développement de segments adaptés aux besoins spécifiques des équipes marketing.
   - Test d'autres algorithmes pour optimiser le temps d'exécution (par exemple, alternatives au K-Prototype).
3. **Contrat de maintenance** :
   - Mise à jour trimestrielle pour surveiller la stabilité des segments.
   - Actions spécifiques pour chaque groupe, comme des campagnes de réactivation ou de fidélisation.

![Propositions futures](images/propositions_futures.png)

---

## Contact

Pour toute question ou demande de collaboration, vous pouvez me contacter :
- **Email** : yannickquerin@gmail.com
- **LinkedIn** : [Yannick Quérin](https://linkedin.com/in/yannick-quérin/)
- **GitHub** : [YannickQuerin](https://github.com/YannickQuerin)

---

