# MaaS Equilibrium - Modèles d'Équilibre Économique pour Mobility-as-a-Service


## Description

Ce repository contient des modèles économiques pour analyser l'équilibre concurrentiel dans les systèmes **Mobility-as-a-Service (MaaS)**. Les modèles intègrent l'hétérogénéité des préférences utilisateurs et permettent de comparer différentes approches de résolution d'équilibre.

### Objectifs de Recherche

- **Modélisation économique** des interactions entre opérateurs MaaS et entreprises de transport
- **Analyse d'équilibre** via approches KKT (Karush-Kuhn-Tucker) et optimisation par groupes
- **Étude de l'hétérogénéité** des préférences utilisateurs et son impact sur l'équilibre
- **Comparaison des stratégies** de tarification et d'intégration intermodale

## Architecture du Modèle

### Segments de Transport Modélisés

Le modèle considère **5 segments de transport** distincts :

| Segment | Description |
|---------|-------------|
| **MM** | MaaS → MaaS |
| **ME** | MaaS → Entreprise |
| **EM** | Entreprise → MaaS |
| **EE** | Entreprise → Entreprise |
| **V** | Voiture individuelle |

### Composantes Principales

- **Fonctions de demande** : Modélisation des préférences utilisateurs
- **Matrice de corrélation** : Capture des interactions entre segments
- **Fonctions de profit** : Revenus et coûts des opérateurs
- **Externalités** : Impacts environnementaux et sociaux
- **Contraintes d'équilibre** : Résolution via conditions KKT

## Structure du Projet

```
maas-equilibrium/
├── 📄 model_complet_lisible_v1.py      # Modèle complet avec hétérogénéité
├── 📄 model_utilisateur_moyen_lisible_v1.py  # Modèle utilisateur moyen
├── 📄 requirements.txt                 # Dépendances Python
├── 📄 README.md                       # Documentation (ce fichier)
├── 📁 Autre/                          # Modèles de développement
│   ├── model V4/                      # Version 4 du modèle
│   ├── model v5.py                    # Version 5
│   ├── model v6.py                    # Version 6
│   └── model v7.py                    # Version 7
├── 📁 model_v5/                       # Résultats version 5
├── 📁 model_v8/                       # Résultats version 8
└── 📁 Models avant v4/                # Versions antérieures
```

## Installation et Utilisation

### Installation

1. **Cloner le repository**
   ```bash
   git clone https://github.com/votre-username/maas-equilibrium.git
   cd maas-equilibrium
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

### Exécution des Modèles

#### Modèle Utilisateur Moyen
```bash
python model_utilisateur_moyen_lisible_v1.py
```

#### Modèle Complet avec Hétérogénéité
```bash
python model_complet_lisible_v1.py
```

## Résultats et Visualisations

Les modèles génèrent automatiquement des graphiques montrant :

- **Évolution des prix** selon les coûts de correspondance
- **Répartition des quantités** entre segments
- **Analyse du bien-être** (welfare) total
- **Profits des opérateurs** MaaS et entreprises
- **Multiplicateurs de Lagrange** (conditions KKT)

### Exemples de Sorties

- **Prix d'équilibre** par segment de transport
- **Quantités optimales** de chaque mode
- **Analyse comparative** des approches de résolution
- **Impact des externalités** sur le bien-être social

## Méthodologie

### Approche KKT (Karush-Kuhn-Tucker)

Le modèle utilise les conditions KKT pour résoudre l'équilibre concurrentiel :

1. **Conditions de stationnarité** : Dérivées partielles nulles
2. **Conditions de primalité** : Respect des contraintes
3. **Conditions de complémentarité** : Multiplicateurs de Lagrange
4. **Conditions de dualité** : Non-négativité des multiplicateurs

### Optimisation par Groupes

Pour l'hétérogénéité des préférences :
- **Segmentation utilisateurs** selon le paramètre θ
- **Résolution par groupes** avec contraintes d'équilibre
- **Agrégation des résultats** pour l'analyse globale

## Paramètres du Modèle

### Paramètres de Base
- **A0** : Attractivité de base (10)
- **A_V** : Attractivité de la voiture (8)
- **P_V_fixe** : Prix fixe de la voiture (4)

### Coûts et Tarification
- **c_M, c_E** : Coûts marginaux MaaS/Entreprise (3, 2)
- **k_M, k_E** : Coûts généralisés (2.0, 2.5)
- **C_corr** : Coûts de correspondance (variable)

### Externalités
- **ExtM, ExtE** : Externalités positives transports collectifs (0.3, 0.5)
- **ExtV** : Externalité négative voiture (-0.5)

## Auteur

**Marius DISSLER**  
*Date : 10/07/2025*


## 🔧 Dépendances Techniques

| Package | Version | Usage |
|---------|---------|-------|
| `sympy` | ≥1.12.0 | Calcul symbolique et résolution KKT |
| `numpy` | ≥1.24.0 | Calculs numériques |
| `scipy` | ≥1.10.0 | Optimisation |
| `matplotlib` | ≥3.7.0 | Visualisation |
| `pandas` | ≥2.0.0 | Manipulation de données |
