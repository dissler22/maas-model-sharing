# =============================================================================================
# MODÈLE MaaS AVEC HÉTÉROGÉNÉITÉ DES PRÉFÉRENCES
# =============================================================================================
# 
# Ce modèle simule un système Mobility-as-a-Service (MaaS) avec :
# - 5 segments de transport : MM (Maas-Maas), ME (Maas-entreprise), EM (entreprise-Maas), 
#   EE (entreprise-entreprise), V (Voiture)
# - Hétérogénéité des préférences utilisateurs via le paramètre θ
# - Résolution d'équilibre par approche KKT (utilisateur moyen) + optimisation par groupes
# - Comparaison des deux approches pour mesurer l'impact de l'hétérogénéité
#
# Auteur : Marius DISSLER
# Date : 10/07/2025
# =============================================================================================

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import itertools

# =============================================================================================
# 1. DÉFINITION DES SYMBOLES SYMPY
# =============================================================================================

# Variables de quantité pour chaque segment de transport
Q_MM, Q_ME, Q_EM, Q_EE, Q_V = sp.symbols('Q_MM Q_ME Q_EM Q_EE Q_V')

# Multiplicateurs de Lagrange pour les contraintes de positivité
mu_MM, mu_ME, mu_EM, mu_EE, mu_V = sp.symbols('mu_MM mu_ME mu_EM mu_EE mu_V', real=True, nonnegative=True)

# Multiplicateur de Lagrange pour la contrainte d'égalité (somme des quantités = D)
lambda_ = sp.Symbol('lambda', positive=True)

# Multiplicateur pour la contrainte de prix fixe de la voiture
mu_PV = sp.Symbol('mu_PV', real=True)
P_V_fixe = sp.Symbol('P_V_fixe', real=True)

# Paramètres de coût
c_M, c_E = sp.symbols('c_M c_E', positive=True)  # Coûts marginaux
c_M_int, c_E_int = sp.symbols('c_M_int c_E_int', positive=True)  # Coûts d'intégration marginaux
C_M_int, C_E_int = sp.symbols('C_M_int C_E_int', positive=True)  # Coûts d'intégration fixes

# Primes intermodales
delta_M, delta_E = sp.symbols('delta_M delta_E', positive=True)

# Paramètres d'intégration
I_int = sp.Symbol('I_int')

# Paramètres de demande et attractivité
A0 = sp.Symbol('A0')  # Attractivité de base
A_V = sp.Symbol('A_V', positive=True)  # Attractivité de la voiture
k_M, k_E, k_V = sp.symbols('k_M k_E k_V', positive=True)  # Paramètres de coût généralisé
C_corrEM, C_corrME = sp.symbols('C_corrEM C_corrME', positive=True)  # Coûts de correspondance

# Paramètres de la matrice de corrélation
beta, gamma = sp.symbols('beta gamma', positive=True)

# Externalités (positives pour les transports collectifs, négatives pour la voiture)
ExtM, ExtE, ExtV = sp.symbols('ExtM ExtE ExtV')

# Demande totale (normalisée à 1)
D = 1


# =============================================================================================
# 2. DÉFINITION DES VARIABLES ET PARAMÈTRES
# =============================================================================================


# =============================================================================================
# 2.1 PARAMÈTRES NUMÉRIQUES DU MODÈLE
# =============================================================================================

param_values = {
    # Paramètres de base
    'D': 1,                    # Demande totale normalisée
    'A0': 10,                  # Attractivité de base
    'A_V': 8,                  # Attractivité de la voiture
    'P_V_fixe': 4,             # Prix fixe de la voiture
    
    # Paramètres de coût généralisé
    'k_M': 2.0,                # Coût généralisé Maas
    'k_E': 2.5,                # Coût généralisé entreprise
    'k_V': 0,                  # Coût généralisé voiture (nul par convention)
    
    # Paramètres de la matrice de corrélation
    'beta': 2,                 # Terme diagonal (substitution propre)
    'gamma': 0.5,              # Terme croisé (substitution entre modes)
    
    # Coûts marginaux
    'c_M': 3,                  # Coût marginal Maas
    'c_E': 2,                  # Coût marginal entreprise
    
    # Coûts d'intégration
    'c_M_int': 0.2,            # Coût marginal d'intégration Maas
    'c_E_int': 0.2,            # Coût marginal d'intégration entreprise
    'C_M_int': 0,              # Coût fixe d'intégration Maas
    'C_E_int': 0,              # Coût fixe d'intégration entreprise
    
    # Primes intermodales
    'delta_M': 0,              # Prime intermodale Maas
    'delta_E': 0,              # Prime intermodale entreprise
    
    # Externalités
    'ExtM': 0.3,               # Externalité positive Maas
    'ExtE': 0.5,               # Externalité positive entreprise
    'ExtV': -0.5,              # Externalité négative voiture
    
    # Paramètres d'intégration
    'I_int': 0,                # Niveau d'intégration
    
    # Coûts de correspondance (varient dans la simulation)
    'C_corrEM': 0,             # Coût de correspondance entreprise-Maas
    'C_corrME': 0,             # Coût de correspondance Maas-entreprise
    
    # Paramètres d'hétérogénéité
    'alpha': 0.2,              # Intensité de l'effet hétérogénéité
    'beta_het': 1.0            # Pénalisation de l'intermodalité selon hétérogénéité
}

# =============================================================================================
# 2.2 MATRICE DE CORRÉLATION ET VECTEURS
# =============================================================================================




# Matrice de corrélation 5x5 : capture les interactions entre segments
# beta = terme diagonal (substitution propre), gamma = terme croisé (substitution entre modes)
M_mat = sp.Matrix([
    [beta,  gamma, gamma, gamma, gamma],
    [gamma, beta,  gamma, gamma, gamma],
    [gamma, gamma, beta,  gamma, gamma],
    [gamma, gamma, gamma, beta,  gamma],
    [gamma, gamma, gamma, gamma, beta]
])

# Vecteur d'attractivité de base
A_MM = A0 - 2*k_M
A_EE = A0 - 2*k_E
A_ME = A0 - k_M - k_E - C_corrME
A_EM = A0 - k_M - k_E - C_corrEM
A_vec = sp.Matrix([A_MM, A_ME, A_EM, A_EE, A_V])

# Vecteur des quantités
Q_vec = sp.Matrix([Q_MM, Q_ME, Q_EM, Q_EE, Q_V])

# =============================================================================================
# 2.3 FONCTIONS DE PRIX ET DEMANDE
# =============================================================================================

# Fonction de prix inverse : P = A - M*Q
P_vec = A_vec - M_mat * Q_vec

# Prix par segment
P_MM = P_vec[0]  # Prix Maas-Maas
P_ME = P_vec[1]  # Prix Maas-entreprise
P_EM = P_vec[2]  # Prix entreprise-Maas
P_EE = P_vec[3]  # Prix entreprise-entreprise
P_V = P_vec[4]   # Prix Voiture

# Prix moyens par opérateur (pour les profits)
P_M = P_vec[0]/2  # Prix moyen Maas
P_E = P_vec[3]/2  # Prix moyen entreprise


# =============================================================================================
# 2.4 UTILITÉ TOTALE (sans Q_V dans les termes quadratiques)
# =============================================================================================

Q_vec_list = [q[0] for q in Q_vec.tolist()]
Q_sq = sum(q**2 for q in Q_vec_list) + Q_V**2
Q_cross = sum(Q_vec_list[i] * Q_vec_list[j] for i in range(4) for j in range(4) if i != j)
Q_vec_noV = sp.Matrix([Q_MM, Q_ME, Q_EM, Q_EE])  # sans Q_V
A_vec_noV = sp.Matrix([A_MM, A_ME, A_EM, A_EE])
M_mat_noV = sp.Matrix([
    [beta,  gamma, gamma, gamma],
    [gamma, beta,  gamma, gamma],
    [gamma, gamma, beta,  gamma],
    [gamma, gamma, gamma, beta]
])
U_total = A_vec_noV.dot(Q_vec_noV) + A_V * Q_V - (1/2) * (Q_vec_noV.T * M_mat_noV * Q_vec_noV)[0]


# =============================================================================================
# 2.5 FONCTIONS DE PROFIT
# =============================================================================================


"""
Pi_M = (P_M - c_M - c_M_int * I_int) * (2*Q_MM + Q_ME + Q_EM) + delta_M * (Q_ME + Q_EM) - C_M_int * I_int
Pi_E = (P_E - c_E - c_E_int * I_int) * (2*Q_EE + Q_ME + Q_EM) + delta_E * (Q_ME + Q_EM) - C_E_int * I_int
"""
Pi_M = (P_MM - c_M - c_M_int * I_int) * Q_MM + (P_ME - c_M - c_E_int * I_int) * Q_ME / 2 + (P_EM - c_E - c_M_int * I_int) * Q_EM / 2 + delta_M * (Q_ME + Q_EM) - C_M_int * I_int
Pi_E = (P_EE - c_E - c_E_int * I_int) * Q_EE + (P_ME - c_M - c_E_int * I_int) * Q_ME / 2 + (P_EM - c_E - c_M_int * I_int) * Q_EM / 2 + delta_E * (Q_ME + Q_EM) - C_E_int * I_int



# =============================================================================================
# 2.6 WELFARE TOTAL
# =============================================================================================


W_welfare = U_total + 2 * Q_V * ExtV + 2 * Q_MM * ExtE + Q_ME * (ExtE + ExtM) + Q_EM * (ExtM + ExtE) + 2 * Q_EE * ExtM + Pi_E + Pi_M







# =============================================================================================
# 3. RÉSOLUTION KKT SYMBOLIQUE
# =============================================================================================

# =============================================================================================
# 3.1 PRÉPARATION DES ÉQUATIONS KKT
# =============================================================================================


# Dictionnaire de substitution pour certains paramètres numériques pour simplifier la résolution

subs_dict = {
    beta: 2,
    gamma: 0.5,
    A_MM: param_values['A0'] - 2 * param_values['k_M'],
    A_EE: param_values['A0'] - 2 * param_values['k_E'],
    A_ME: param_values['A0'] - param_values['k_M'] - param_values['k_E'] - C_corrME,
    A_EM: param_values['A0'] - param_values['k_M'] - param_values['k_E'] - C_corrEM,
    A_V: param_values['A_V'],
#    I_int: 0,
#    delta_E: 0,
#    delta_M: 0,
#    c_M: 3,
#    c_E: 2,
#    ExtE: 0.5,
#    ExtM: 0.3,
#    ExtV: -0.5,
    P_V_fixe: param_values['P_V_fixe']  # ou la valeur que tu veux fixer
}


# Variables et contraintes

mu_vars = [mu_MM, mu_ME, mu_EM, mu_EE, mu_V]
Q_names = ['MM', 'ME', 'EM', 'EE', 'V']

Q_vars = [Q_MM, Q_ME, Q_EM, Q_EE, Q_V]


# =============================================================================================
# 3.2 LAGRANGIEN AVEC CONTRAINTES
# =============================================================================================

# Lagrangien complet avec contraintes d'inégalité
# L = Welfare - λ*(somme des Q - D) + Σ(μ_i * Q_i)
L = W_welfare \
    - lambda_ * (sum(Q_vars) - D) \
    + sum([mu_vars[i] * Q_vars[i] for i in range(5)]) \

# Application des substitutions numériques pour symplifier la résolution
L = L.subs(subs_dict)

# =============================================================================================
# 3.3 EXPLORATION DES CONFIGURATIONS KKT
# =============================================================================================

# Résultat stocké pour chaque configuration
solutions_kkt_full = []

# Exploration de toutes les 2^5 = 32 configurations de contraintes actives
# Chaque configuration représente quels Q_i sont actifs (1) ou forcés à zéro (0)

# Configuration par défaut : tous les Q_i actifs
list_config = [[1,1,1,1,1]]

for config in itertools.product([0, 1], repeat=5):
    eqs = []

    # 1. Stationnarité : dL/dQ_i = 0
    for i in range(5):
        eqs.append(sp.Eq(sp.diff(L, Q_vars[i]), 0))

    # 2. Stationnarité en lambda (contrainte d'égalité)
    #eqs.append(sp.Eq(sp.diff(L, lambda_), 0))

    # 3. Primalité : Q_i = 0 si contrainte active
    for i in range(5):
        if config[i] == 0:
            eqs.append(sp.Eq(Q_vars[i], 0))
        else :
            eqs.append(sp.Eq(mu_vars[i], 0))

    # 4. Complémentarité : mu_i * Q_i = 0
    """for i in range(5):
        eqs.append(sp.Eq(mu_vars[i] * Q_vars[i], 0))"""

    # Variables à résoudre
    all_vars = Q_vars + mu_vars + [lambda_]  # Ajoute lambda_
    eqs.append(sp.Eq(sum(Q_vars), D))   # Ajoute la contrainte

    # Résolution symbolique
    try:
        print(f"\nConfiguration active : {config}")
        sol = sp.solve(eqs, all_vars, dict=True)
        
        print("Nombre de solutions :", {len(sol)})
        #print(sol)
        print("Solutions trouvées :")
        for s in sol:
            print({str(var): s[var].evalf(n=2) for var in all_vars})
        solutions_kkt_full.append({
            'config_active': config,
            'solution': sol[0] if sol else None,
            'status': 'solved' if sol else 'no solution'
        })
    except Exception as e:
        solutions_kkt_full.append({
            'config_active': config,
            'solution': None,
            'status': f'error: {str(e)}'
        })

# =============================================================================================
# 3.4 FONCTION DE RÉSOLUTION NUMÉRIQUE
# =============================================================================================

# Mapping des symboles vers les paramètres
symbol_map = {
    'A0': A0, 'A_V': A_V, 'k_M': k_M, 'k_E': k_E, 'k_V': k_V, 'beta': beta, 'gamma': gamma,
    'c_M': c_M, 'c_E': c_E, 'c_M_int': c_M_int, 'c_E_int': c_E_int, 'C_M_int': C_M_int, 'C_E_int': C_E_int,
    'I_int': I_int, 'ExtE': ExtE, 'ExtM': ExtM, 'ExtV': ExtV, 'delta_M': delta_M, 'delta_E': delta_E,
    'C_corrEM': C_corrEM, 'C_corrME': C_corrME, 'D': D
}



# =============================================================================================
# 4. SOUS FONCTIONS
# =============================================================================================


def solve_group_qp(A_theta, M_mat_num, Q_sum_target, P_star):
    """
    Résout le problème d'optimisation quadratique pour un groupe d'utilisateurs.
    
    Cette fonction maximise l'utilité d'un groupe à prix fixés :
    max U(Q) = A_theta @ Q - 0.5 * Q @ M_mat_num @ Q - P_star @ Q
    s.c. sum(Q) = Q_sum_target, Q_i >= 0
    
    Args:
        A_theta (np.array): Vecteur d'attractivité pour ce groupe
        M_mat_num (np.array): Matrice de corrélation numérique
        Q_sum_target (float): Contrainte de somme des quantités
        P_star (np.array): Vecteur des prix fixés
    
    Returns:
        np.array: Quantités optimales pour ce groupe
    """
    n = len(A_theta)
    def obj(Q):
        return - (A_theta @ Q - 0.5 * Q @ M_mat_num @ Q - P_star @ Q)
    cons = {'type': 'eq', 'fun': lambda Q: np.sum(Q) - Q_sum_target}
    bounds = [(0, None)] * n
    Q0 = np.ones(n) * (Q_sum_target / n)
    res = minimize(obj, Q0, bounds=bounds, constraints=cons, method='SLSQP')
    return res.x if res.success else np.zeros(n)


def get_A_vec_theta(theta, C_corr, param_values):
    """
    Génère le vecteur d'attractivité pour un groupe d'utilisateurs donné.
    
    Le paramètre θ représente la préférence de l'utilisateur :
    - θ = 0 : Utilisateur "pur collectif" (préfère les transports collectifs)
    - θ = 0.5 : Utilisateur "équilibré" (point neutre)
    - θ = 1 : Utilisateur "pur individuel" (préfère la voiture)
    
    Args:
        theta (float): Paramètre de préférence [0, 1]
        C_corr (float): Coût de correspondance
        param_values (dict): Dictionnaire des paramètres numériques
    
    Returns:
        np.array: Vecteur d'attractivité [A_MM, A_ME, A_EM, A_EE, A_V]
    """
    alpha = param_values['alpha']
    beta_het = param_values['beta_het']
    A0 = param_values['A0']
    k_M = param_values['k_M']
    k_E = param_values['k_E']
    A_V = param_values['A_V']
    
    # Centrer theta autour de 0.5
    theta_centered = theta - 0.5  # Maintenant dans [-0.5, 0.5]
    
    # Effet symétrique sur tous les modes
    A_MM = (A0 - 2*k_M)*(1 + alpha * theta_centered)  # Collectif pur
    A_EE = (A0 - 2*k_E)*(1 + alpha * theta_centered)  # Collectif pur
    A_ME = (A0 - k_M - k_E - C_corr)*(1 + alpha * theta_centered)  # Intermodal
    A_EM = (A0 - k_M - k_E - C_corr)*(1 + alpha * theta_centered)  # Intermodal
    A_V_theta = (A_V)*(1 - alpha * theta_centered)  # Individuel (voiture)
    
    return np.array([A_MM, A_ME, A_EM, A_EE, A_V_theta], dtype=float)



def solve_kkt_given_parameters(solutions_kkt_full, param_values, config_tuple=(1, 1, 1, 1, 1)):
    """
    Évalue la solution KKT d'une configuration spécifique avec un jeu de paramètres donné,
    et retourne les quantités Q_i valides (Q_i >= 0 et somme = D).
    Si certains Q_i sont négatifs, relance récursivement avec les Q_i négatifs forcés à zéro.

    Args:
        solutions_kkt_full (list): liste de dictionnaires contenant les solutions symboliques KKT.
        param_values (dict): dictionnaire des valeurs numériques pour les paramètres du modèle.
        config_tuple (tuple): configuration binaire (1 = actif, 0 = forcé à zéro) pour Q_MM à Q_V.

    Returns:
        dict: quantités valides {Q_MM, Q_ME, Q_EM, Q_EE, Q_V}, ou None si aucune solution valide.
    """
    D = sp.Symbol('D', real=True)
    param_values_full = param_values.copy()
    subs_dict = {symbol_map[k]: v for k, v in param_values_full.items() if k in symbol_map}
    if 'P_V_fixe' in param_values_full:
        subs_dict[P_V_fixe] = param_values_full['P_V_fixe']
    if 'D' not in param_values_full:
        raise ValueError("Le paramètre D (demande totale) est requis.")

    # Debug: afficher le nombre de configurations disponibles
    if config_tuple == (1, 1, 1, 1, 1):  # Seulement pour la première fois
        #print(f"Nombre de configurations KKT disponibles: {len(solutions_kkt_full)}")
        #print(f"Première configuration: {solutions_kkt_full[0]['config_active'] if solutions_kkt_full else 'Aucune'}")
        pass

    # Chercher la configuration exacte dans la liste
    for config in solutions_kkt_full:
        if tuple(config['config_active']) != config_tuple:
            continue
        sol = config['solution']
        if sol is None:
            continue
        try:
            sol_eval = {
                k: sol[k].subs(subs_dict)
                for k in sol
            }
            Q_sum = sum(sol_eval.get(q, 0.0) for q in Q_vars)
            mass_balance = abs(Q_sum - param_values['D']) < 1e-6
            if True : #mass_balance:
                # Vérifier les Q_i négatifs
                Q_neg_vals = {q: sol_eval.get(q, 0.0) for q in Q_vars if sol_eval.get(q, 0.0) < -1e-6}
                if not Q_neg_vals:
                    return {q: sol_eval.get(q, 0.0) for q in Q_vars + mu_vars + [lambda_]}
                else:
                    # Trouver le Q_i le plus négatif
                    q_min = min(Q_neg_vals.items(), key=lambda item: item[1])[0]
                    i_min = Q_vars.index(q_min)
                    # Construire une nouvelle configuration avec ce Q_i forcé à 0
                    new_config = list(config_tuple)
                    new_config[i_min] = 0
                    return solve_kkt_given_parameters(solutions_kkt_full, param_values, tuple(new_config))
        except Exception as e:
            if config_tuple == (1, 1, 1, 1, 1):  # Seulement pour la première fois
                print(f"Erreur lors de l'évaluation: {e}")
            continue
    return None


# =============================================================================================
# 5. SIMULATION PRINCIPALE
# =============================================================================================

# =============================================================================================
# 5.1 PRÉPARATION DES PARAMÈTRES DE SIMULATION
# =============================================================================================


params = (
    A0, A_V, k_M, k_E, k_V, beta, gamma,
    c_M, c_E, c_M_int, c_E_int, C_M_int, C_E_int,
    I_int, ExtE, ExtM, ExtV, delta_M, delta_E, C_corrEM, C_corrME
)

# Paramètres pour l'analyse de sensibilité
x_values = np.linspace(0, 3, 100)  # 100 points de simulation

# =============================================================================================
# 5.2 LISTES DE STOCKAGE DES RÉSULTATS
# =============================================================================================


P_M_welf_list, P_E_welf_list, Pi_M_welf_list, Pi_E_welf_list, Q_ME_welf_list, Q_V_welf_list, Q_total_welf_list = [], [], [], [], [], [], []
P_V_welf_list = []


U_welf_list, Ext_welf_list, Pi_welf_list, W_welf_list = [], [], [], []
U_sommes_list = []
U_MM_welf_list, U_ME_welf_list, U_EM_welf_list, U_EE_welf_list, U_V_welf_list = [], [], [], [], []
Q_MM_welf_list, Q_ME_welf_list, Q_EM_welf_list, Q_EE_welf_list, Q_V_welf_list = [], [], [], [], []

mu_MM_list, mu_ME_list, mu_EM_list, mu_EE_list, mu_V_list = [], [], [], [], []

# Listes pour le welfare avec quantités pondérées
W_welf_groupes_list = []
U_groupes_list = []
Pi_M_groupes_list = []
Pi_E_groupes_list = []
Ext_groupes_list = []

colors = {
    'individuel': 'tab:blue',
    'alliance': 'tab:orange',
    'welfare': 'tab:red',
    'stackelberg': 'tab:green'
}


"""
"""
# =============================================================================================
# 5.3 CONFIGURATION DES GROUPES D'UTILISATEURS
# =============================================================================================

# Définition des groupes d'utilisateurs
nb_groupes = 10  # Changé de 10 à 100 pour correspondre à l'ancienne version
group_thetas = np.linspace(0, 1, nb_groupes)  # [0, 0.11, 0.22, ..., 0.89, 1.0]
poids_groupes = [1/nb_groupes for i in range(nb_groupes)]  # Poids égaux (1/100)

# Labels pour l'affichage
group_labels = [f"Groupe θ={theta}" for theta in group_thetas]



# =============================================================================================
# 5.4 STRUCTURES DE DONNÉES POUR LES RÉSULTATS
# =============================================================================================


# Dictionnaires pour stocker les résultats par groupe
results_group = [
    {
        'Q_MM': [], 'Q_ME': [], 'Q_EM': [], 'Q_EE': [], 'Q_V': [],
        'P_MM': [], 'P_ME': [], 'P_EM': [], 'P_EE': [], 'P_V': [],
        'U_total': [], 'U_MM': [], 'U_ME': [], 'U_EM': [], 'U_EE': [], 'U_V': [],
        'Q_total': []
    }
    for _ in range(nb_groupes)
]
# Pour la somme pondérée
results_total = {
    'Q_MM': [], 'Q_ME': [], 'Q_EM': [], 'Q_EE': [], 'Q_V': [],
    'Q_total': []
}

# =============================================================================================
# 5.5 BOUCLE PRINCIPALE DE SIMULATION
# =============================================================================================

for x in x_values:

    # Mise à jour des coûts de correspondance (paramètre de simulation)
    Ccorr_val = 3 - x
    param_values['C_corrME'] = Ccorr_val
    param_values['C_corrEM'] = Ccorr_val
    subs_dict = {symbol_map[k]: v for k, v in param_values.items() if k in symbol_map}

    # =============================================================================================
    # 5.5.1 RÉSOLUTION KKT POUR L'UTILISATEUR MOYEN
    # =============================================================================================
    
    # Résolution du problème d'équilibre global (approche utilisateur moyen)
    solution = solve_kkt_given_parameters(solutions_kkt_full, param_values)
    if solution is None:
        # On saute cette valeur si pas de solution
        for g in range(nb_groupes):
            for key in results_group[g]:
                results_group[g][key].append(np.nan)
        for key in results_total:
            results_total[key].append(np.nan)
        continue

    # Extraction des quantités et multiplicateurs de Lagrange
    Q_MM_welf_val = solution[Q_MM]
    Q_ME_welf_val = solution[Q_ME]
    Q_EM_welf_val = solution[Q_EM]
    Q_EE_welf_val = solution[Q_EE]
    Q_V_welf_val = solution[Q_V]

    # Calcul des prix à partir des quantités
    P_MM_welf_val = P_MM.subs({
        Q_MM: Q_MM_welf_val, 
        Q_ME: Q_ME_welf_val, 
        Q_EM: Q_EM_welf_val, 
        Q_EE: Q_EE_welf_val, 
        Q_V: Q_V_welf_val
    }).subs(subs_dict)
    P_ME_welf_val = P_ME.subs({
        Q_MM: Q_MM_welf_val, 
        Q_ME: Q_ME_welf_val, 
        Q_EM: Q_EM_welf_val, 
        Q_EE: Q_EE_welf_val, 
        Q_V: Q_V_welf_val
    }).subs(subs_dict)
    P_EM_welf_val = P_EM.subs({
        Q_MM: Q_MM_welf_val, 
        Q_ME: Q_ME_welf_val, 
        Q_EM: Q_EM_welf_val, 
        Q_EE: Q_EE_welf_val, 
        Q_V: Q_V_welf_val
    }).subs(subs_dict)
    P_EE_welf_val = P_EE.subs({
        Q_MM: Q_MM_welf_val, 
        Q_ME: Q_ME_welf_val, 
        Q_EM: Q_EM_welf_val, 
        Q_EE: Q_EE_welf_val, 
        Q_V: Q_V_welf_val
    }).subs(subs_dict)
    P_V_welf_val = P_V.subs({
        Q_MM: Q_MM_welf_val, 
        Q_ME: Q_ME_welf_val, 
        Q_EM: Q_EM_welf_val, 
        Q_EE: Q_EE_welf_val, 
        Q_V: Q_V_welf_val
    }).subs(subs_dict)

    # Calcul des profits avec expressions symboliques (comme dans l'ancienne version)
    Pi_M_welf_val = Pi_M.subs({
        Q_MM: Q_MM_welf_val, 
        Q_ME: Q_ME_welf_val, 
        Q_EM: Q_EM_welf_val, 
        Q_EE: Q_EE_welf_val, 
        Q_V: Q_V_welf_val,
        P_MM: P_MM_welf_val,
        P_ME: P_ME_welf_val,
        P_EM: P_EM_welf_val,
        P_EE: P_EE_welf_val
    }).subs(subs_dict)

    Pi_E_welf_val = Pi_E.subs({
        Q_MM: Q_MM_welf_val, 
        Q_ME: Q_ME_welf_val, 
        Q_EM: Q_EM_welf_val, 
        Q_EE: Q_EE_welf_val, 
        Q_V: Q_V_welf_val
    }).subs(subs_dict)

    # Calcul du Welfare 
    Q_total_welf_val = Q_MM_welf_val + Q_ME_welf_val + Q_EM_welf_val + Q_EE_welf_val + Q_V_welf_val    

    U_welf_val = U_total.subs({
        Q_MM: Q_MM_welf_val,
        Q_ME: Q_ME_welf_val,
        Q_EM: Q_EM_welf_val,
        Q_EE: Q_EE_welf_val,
        Q_V: Q_V_welf_val
    }).subs(subs_dict)

    W_welf_val = W_welfare.subs({
        Q_MM: Q_MM_welf_val,
        Q_ME: Q_ME_welf_val,
        Q_EM: Q_EM_welf_val,
        Q_EE: Q_EE_welf_val,
        Q_V: Q_V_welf_val
    }).subs(subs_dict)

    P_M_welf_list.append(P_MM_welf_val/2)
    P_E_welf_list.append(P_EE_welf_val/2)
    P_V_welf_list.append(P_V_welf_val/2)
    Pi_M_welf_list.append(Pi_M_welf_val)
    Pi_E_welf_list.append(Pi_E_welf_val)
    Q_ME_welf_list.append(Q_ME_welf_val)
    Q_MM_welf_list.append(Q_MM_welf_val)
    Q_EM_welf_list.append(Q_EM_welf_val)
    Q_EE_welf_list.append(Q_EE_welf_val)
    Q_V_welf_list.append(Q_V_welf_val)
    Q_total_welf_list.append(Q_total_welf_val)
    W_welf_list.append(W_welf_val)

    # Approfondissement du welfare ==========================================================
    subs_dict_corr = {**{p: param_values[str(p)] for p in params[:-2]}, C_corrEM: Ccorr_val, C_corrME: Ccorr_val}

    U_MM_welf_val = Q_MM_welf_val * (A_MM.subs(subs_dict_corr) - 2*P_MM_welf_val/2)
    U_ME_welf_val = Q_ME_welf_val * (A_ME.subs(subs_dict_corr) - (P_MM_welf_val/2 + P_EE_welf_val/2))
    U_EM_welf_val = Q_EM_welf_val * (A_EM.subs(subs_dict_corr) - (P_MM_welf_val/2 + P_EE_welf_val/2))
    U_EE_welf_val = Q_EE_welf_val * (A_EE.subs(subs_dict_corr) - 2*P_EE_welf_val/2)
    U_V_welf_val  = Q_V_welf_val  * (A_V.subs(subs_dict_corr) - P_V_welf_val/2)
    U_sommes_welf_val = U_MM_welf_val + U_ME_welf_val + U_EM_welf_val + U_EE_welf_val + U_V_welf_val

    Ext_val = (
        2 * Q_V_welf_val * param_values['ExtV']
        + 2 * Q_MM_welf_val * param_values['ExtE']
        + Q_ME_welf_val * (param_values['ExtE'] + param_values['ExtM'])
        + Q_EM_welf_val * (param_values['ExtM'] + param_values['ExtE'])
        + 2 * Q_EE_welf_val * param_values['ExtM']
    )
    U_MM_welf_list.append(U_MM_welf_val)
    U_ME_welf_list.append(U_ME_welf_val)
    U_EM_welf_list.append(U_EM_welf_val)
    U_EE_welf_list.append(U_EE_welf_val)
    U_V_welf_list.append(U_V_welf_val)
    U_sommes_list.append(U_sommes_welf_val)
    U_welf_list.append(U_welf_val)
    Ext_welf_list.append(Ext_val)
    Pi_welf_list.append(Pi_M_welf_val + Pi_E_welf_val)

    # Extraction des multiplicateurs de Lagrange
    mu_MM_val = solution.get(mu_MM, 0)
    mu_ME_val = solution.get(mu_ME, 0)
    mu_EM_val = solution.get(mu_EM, 0)
    mu_EE_val = solution.get(mu_EE, 0)
    mu_V_val  = solution.get(mu_V, 0)

    mu_MM_list.append(mu_MM_val)
    mu_ME_list.append(mu_ME_val)
    mu_EM_list.append(mu_EM_val)
    mu_EE_list.append(mu_EE_val)
    mu_V_list.append(mu_V_val)

    # =============================================================================================
    # 5.5.2 RÉSOLUTION PAR GROUPES D'UTILISATEURS
    # =============================================================================================
    
    M_mat_num = np.array([
        [param_values['beta'],  param_values['gamma'], param_values['gamma'], param_values['gamma'], param_values['gamma']],
        [param_values['gamma'], param_values['beta'],  param_values['gamma'], param_values['gamma'], param_values['gamma']],
        [param_values['gamma'], param_values['gamma'], param_values['beta'],  param_values['gamma'], param_values['gamma']],
        [param_values['gamma'], param_values['gamma'], param_values['gamma'], param_values['beta'],  param_values['gamma']],
        [param_values['gamma'], param_values['gamma'], param_values['gamma'], param_values['gamma'], param_values['beta']]
    ])
    # Prix fixés (ceux de l'utilisateur moyen)
    P_star = np.array([P_MM_welf_val, P_ME_welf_val, P_EM_welf_val, P_EE_welf_val, P_V_welf_val])
    Q_groupes = []
    Q_sum_target = 1.0 

    # Résolution pour chaque groupe
    for g, (theta, poids) in enumerate(zip(group_thetas, poids_groupes)):
        # Vecteur d'attractivité pour ce groupe
        A_theta = get_A_vec_theta(theta, Ccorr_val, param_values)
        
        # Résolution du problème d'optimisation pour ce groupe
        Q_theta = solve_group_qp(A_theta, M_mat_num, Q_sum_target, P_star)

        # Stockage des quantités
        Q_groupes.append(Q_theta)
        results_group[g]['Q_MM'].append(Q_theta[0])
        results_group[g]['Q_ME'].append(Q_theta[1])
        results_group[g]['Q_EM'].append(Q_theta[2])
        results_group[g]['Q_EE'].append(Q_theta[3])
        results_group[g]['Q_V'].append(Q_theta[4])
        results_group[g]['Q_total'].append(np.sum(Q_theta))
        # Stockage des prix (identiques pour tous les groupes)
        results_group[g]['P_MM'].append(P_MM_welf_val)
        results_group[g]['P_ME'].append(P_ME_welf_val)
        results_group[g]['P_EM'].append(P_EM_welf_val)
        results_group[g]['P_EE'].append(P_EE_welf_val)
        results_group[g]['P_V'].append(P_V_welf_val)
        # Utilités individuelles (exemple)
        results_group[g]['U_total'].append(float(A_theta @ Q_theta - 0.5 * Q_theta @ M_mat_num @ Q_theta))
        results_group[g]['U_MM'].append(Q_theta[0] * (A_theta[0] - 2*P_MM_welf_val/2))
        results_group[g]['U_ME'].append(Q_theta[1] * (A_theta[1] - (P_MM_welf_val/2 + P_EE_welf_val/2)))
        results_group[g]['U_EM'].append(Q_theta[2] * (A_theta[2] - (P_MM_welf_val/2 + P_EE_welf_val/2)))
        results_group[g]['U_EE'].append(Q_theta[3] * (A_theta[3] - 2*P_EE_welf_val/2))
        results_group[g]['U_V'].append(Q_theta[4] * (A_theta[4] - P_V_welf_val/2))
    # Quantités totales (somme pondérée)
    Q_total = np.zeros_like(Q_groupes[0])
    for p, Q in zip(poids_groupes, Q_groupes):
        Q_total += p * Q
    results_total['Q_MM'].append(Q_total[0])
    results_total['Q_ME'].append(Q_total[1])
    results_total['Q_EM'].append(Q_total[2])
    results_total['Q_EE'].append(Q_total[3])
    results_total['Q_V'].append(Q_total[4])
    results_total['Q_total'].append(np.sum(Q_total))
    
    # === CALCUL DU WELFARE AVEC QUANTITÉS PONDÉRÉES ===
    
    # Welfare total avec quantités pondérées
    W_welfare_groupes = W_welfare.subs({
        Q_MM: Q_total[0],
        Q_ME: Q_total[1], 
        Q_EM: Q_total[2],
        Q_EE: Q_total[3],
        Q_V: Q_total[4]
    }).subs(subs_dict)
    
    # Utilité avec quantités pondérées
    U_groupes = U_total.subs({
        Q_MM: Q_total[0],
        Q_ME: Q_total[1], 
        Q_EM: Q_total[2],
        Q_EE: Q_total[3],
        Q_V: Q_total[4]
    }).subs(subs_dict)
    
    # Profits avec quantités pondérées
    Pi_M_groupes = Pi_M.subs({
        Q_MM: Q_total[0],
        Q_ME: Q_total[1],
        Q_EM: Q_total[2], 
        Q_EE: Q_total[3],
        Q_V: Q_total[4],
        P_MM: P_MM_welf_val,
        P_ME: P_ME_welf_val,
        P_EM: P_EM_welf_val,
        P_EE: P_EE_welf_val
    }).subs(subs_dict)
    
    Pi_E_groupes = Pi_E.subs({
        Q_MM: Q_total[0],
        Q_ME: Q_total[1],
        Q_EM: Q_total[2],
        Q_EE: Q_total[3], 
        Q_V: Q_total[4]
    }).subs(subs_dict)
    
    # Externalités avec quantités pondérées
    Ext_groupes = (
        2 * Q_total[4] * param_values['ExtV'] +
        2 * Q_total[0] * param_values['ExtE'] +
        Q_total[1] * (param_values['ExtE'] + param_values['ExtM']) +
        Q_total[2] * (param_values['ExtM'] + param_values['ExtE']) +
        2 * Q_total[3] * param_values['ExtM']
    )
    
    # Stockage des résultats
    W_welf_groupes_list.append(W_welfare_groupes)
    U_groupes_list.append(U_groupes)
    Pi_M_groupes_list.append(Pi_M_groupes)
    Pi_E_groupes_list.append(Pi_E_groupes)
    Ext_groupes_list.append(Ext_groupes)




# =============================================================================================
# 6. VISUALISATION DES RÉSULTATS
# =============================================================================================

# PRÉPARATION DES DONNÉES POUR L'ANALYSE

U0 = U_welf_list[0]
Usommes0 = U_sommes_list[0]
Ext0 = Ext_welf_list[0]
Pi0 = Pi_welf_list[0]
print("Pi0:", Pi0)
PiE0 = Pi_E_welf_list[0]
PiM0 = Pi_M_welf_list[0]
W0 = W_welf_list[0]
print("W0:", W0)

U_welf_MM0 = U_MM_welf_list[0]
print("U_welf_MM0:", U_welf_MM0)
U_welf_ME0 = U_ME_welf_list[0]
print("U_welf_ME0:", U_welf_ME0)
U_welf_EM0 = U_EM_welf_list[0]
print("U_welf_EM0:", U_welf_EM0)
U_welf_EE0 = U_EE_welf_list[0]
print("U_welf_EE0:", U_welf_EE0)
U_welf_V0  = U_V_welf_list[0]
print("U_welf_V0:", U_welf_V0)

U_MM_welf_list_centered = [u - U_welf_MM0 for u in U_MM_welf_list]
U_ME_welf_list_centered = [u - U_welf_ME0 for u in U_ME_welf_list]
U_EM_welf_list_centered = [u - U_welf_EM0 for u in U_EM_welf_list]
U_EE_welf_list_centered = [u - U_welf_EE0 for u in U_EE_welf_list]
U_V_welf_list_centered  = [u - U_welf_V0  for u in U_V_welf_list]

U_welf_list_centered = [u - U0 for u in U_welf_list]
U_sommes_list_centered = [u - Usommes0 for u in U_sommes_list]
Ext_welf_list_centered = [e - Ext0 for e in Ext_welf_list]
Pi_welf_list_centered = [p - Pi0 for p in Pi_welf_list]
Pi_M_welf_list_centered = [p - PiM0 for p in Pi_M_welf_list]
Pi_E_welf_list_centered = [p - PiE0 for p in Pi_E_welf_list]
W_welf_list_centered = [w - W0 for w in W_welf_list]



# Configuration du style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11

# Couleurs cohérentes
colors = {
    'utilisateur_moyen': '#1f77b4',
    'groupes': '#ff7f0e', 
    'prix': '#2ca02c',
    'welfare': '#d62728',
    'diagnostic': '#9467bd'
}

print("=" * 60)
print("RÉSULTATS DU MODÈLE MaaS - ANALYSE COMPARATIVE")
print("=" * 60)

# === 1. GRAPHIQUE PRINCIPAL 2x2 ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Comparaison Utilisateur Moyen vs Modèle Multi-Groupes', fontsize=16, fontweight='bold')

# 1.1 Q_MM
axes[0, 0].plot(x_values, Q_MM_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[0, 0].plot(x_values, results_total['Q_MM'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[0, 0].set_xlabel('Économie sur les coûts de correspondance (3 - C_corr)')
axes[0, 0].set_ylabel('Quantité Q_MM')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Évolution de Q_MM')

# 1.2 Q_Total
axes[0, 1].plot(x_values, Q_total_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[0, 1].plot(x_values, results_total['Q_total'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[0, 1].set_xlabel('Économie sur les coûts de correspondance (3 - C_corr)')
axes[0, 1].set_ylabel('Quantité totale')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Évolution de la quantité totale')

# 1.3 Prix
axes[1, 0].plot(x_values, [P/2 for P in P_M_welf_list], label='P_M', 
                color=colors['prix'], linewidth=2)
axes[1, 0].plot(x_values, [P/2 for P in P_E_welf_list], label='P_E', 
                color=colors['prix'], linewidth=2, linestyle='--')
axes[1, 0].plot(x_values, [P/2 for P in P_V_welf_list], label='P_V', 
                color=colors['prix'], linewidth=2, linestyle=':')
axes[1, 0].set_xlabel('Économie sur les coûts de correspondance (3 - C_corr)')
axes[1, 0].set_ylabel('Prix')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Évolution des prix')

# 1.4 Welfare
axes[1, 1].plot(x_values, W_welf_list, label='Welfare (utilisateur moyen)', 
                color=colors['welfare'], linewidth=3)
axes[1, 1].plot(x_values, W_welf_groupes_list, label='Welfare (groupes pondérés)', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[1, 1].set_xlabel('Économie sur les coûts de correspondance (3 - C_corr)')
axes[1, 1].set_ylabel('Welfare')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Évolution du welfare')

plt.tight_layout()
plt.show()

# === 2. COMPARAISON DE TOUS LES SEGMENTS ===
plt.figure(figsize=(15, 10))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparaison de tous les segments: Utilisateur Moyen vs Modèle Multi-Groupes', fontsize=16, fontweight='bold')

# 2.1 Q_MM
axes[0, 0].plot(x_values, Q_MM_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[0, 0].plot(x_values, results_total['Q_MM'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[0, 0].set_xlabel('Économie sur C_corr')
axes[0, 0].set_ylabel('Quantité Q_MM')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Q_MM')

# 2.2 Q_ME
axes[0, 1].plot(x_values, Q_ME_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[0, 1].plot(x_values, results_total['Q_ME'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[0, 1].set_xlabel('Économie sur C_corr')
axes[0, 1].set_ylabel('Quantité Q_ME')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Q_ME')

# 2.3 Q_EM
axes[0, 2].plot(x_values, Q_EM_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[0, 2].plot(x_values, results_total['Q_EM'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[0, 2].set_xlabel('Économie sur C_corr')
axes[0, 2].set_ylabel('Quantité Q_EM')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_title('Q_EM')

# 2.4 Q_EE
axes[1, 0].plot(x_values, Q_EE_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[1, 0].plot(x_values, results_total['Q_EE'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[1, 0].set_xlabel('Économie sur C_corr')
axes[1, 0].set_ylabel('Quantité Q_EE')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Q_EE')

# 2.5 Q_V
axes[1, 1].plot(x_values, Q_V_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[1, 1].plot(x_values, results_total['Q_V'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[1, 1].set_xlabel('Économie sur C_corr')
axes[1, 1].set_ylabel('Quantité Q_V')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Q_V')

# 2.6 Q_Total (gardé pour référence)
axes[1, 2].plot(x_values, Q_total_welf_list, label='Utilisateur moyen', 
                color=colors['utilisateur_moyen'], linewidth=3)
axes[1, 2].plot(x_values, results_total['Q_total'], label='Somme pondérée groupes', 
                color=colors['groupes'], linewidth=3, linestyle='--')
axes[1, 2].set_xlabel('Économie sur C_corr')
axes[1, 2].set_ylabel('Quantité totale')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].set_title('Q_Total')

plt.tight_layout()
plt.show()

# === 3. TABLEAU DE SYNTHÈSE ===
print("\n" + "=" * 60)
print("TABLEAU DE SYNTHÈSE DES RÉSULTATS")
print("=" * 60)

# Créer un DataFrame avec les résultats clés
df_results = pd.DataFrame({
    'C_corr': 3 - x_values,
    'Q_MM_moyen': Q_MM_welf_list,
    'Q_total_moyen': Q_total_welf_list,
    'Q_MM_groupes': results_total['Q_MM'],
    'Q_total_groupes': results_total['Q_total'],
    'P_M': [P/2 for P in P_M_welf_list],
    'P_E': [P/2 for P in P_E_welf_list],
    'P_V': [P/2 for P in P_V_welf_list],
    'Welfare_moyen': W_welf_list,
    'Welfare_groupes': W_welf_groupes_list
})

# Afficher les premières lignes
print("\nPremières lignes du tableau de résultats:")
print(df_results.head().round(4))

# Statistiques de comparaison
print(f"\nStatistiques de comparaison:")
print(f"Différence moyenne Q_MM: {np.mean(np.abs(df_results['Q_MM_moyen'] - df_results['Q_MM_groupes'])):.4f}")
print(f"Différence moyenne Q_total: {np.mean(np.abs(df_results['Q_total_moyen'] - df_results['Q_total_groupes'])):.4f}")

# Conversion en numpy arrays pour éviter les erreurs avec les objets SymPy
q_mm_moyen = np.array([float(x) for x in df_results['Q_MM_moyen']])
q_mm_groupes = np.array([float(x) for x in df_results['Q_MM_groupes']])
q_total_moyen = np.array([float(x) for x in df_results['Q_total_moyen']])
q_total_groupes = np.array([float(x) for x in df_results['Q_total_groupes']])
welfare_moyen = np.array([float(x) for x in df_results['Welfare_moyen']])
welfare_groupes = np.array([float(x) for x in df_results['Welfare_groupes']])

print(f"Corrélation Q_MM: {np.corrcoef(q_mm_moyen, q_mm_groupes)[0,1]:.4f}")
print(f"Corrélation Q_total: {np.corrcoef(q_total_moyen, q_total_groupes)[0,1]:.4f}")
print(f"Corrélation Welfare: {np.corrcoef(welfare_moyen, welfare_groupes)[0,1]:.4f}")

# Analyse du coût de l'hétérogénéité
diff_welfare = np.abs(welfare_moyen - welfare_groupes)
print(f"Coût moyen de l'hétérogénéité (différence welfare): {np.mean(diff_welfare):.4f}")
print(f"Coût max de l'hétérogénéité: {np.max(diff_welfare):.4f}")
print(f"Efficacité relative (welfare_groupes/welfare_moyen): {np.mean(welfare_groupes/welfare_moyen):.4f}")

# === 4. RÉSUMÉ STATISTIQUE ===
print("\n" + "=" * 60)
print("RÉSUMÉ STATISTIQUE DU MODÈLE")
print("=" * 60)

print(f"Nombre de groupes: {nb_groupes}")
print(f"Thétas: {[f'{theta:.1f}' for theta in group_thetas]}")
print(f"Poids: {[f'{poids:.1f}' for poids in poids_groupes]}")
print(f"Paramètres C_corr: de {3-x_values[0]:.1f} à {3-x_values[-1]:.1f}")
print(f"Nombre de points de simulation: {len(x_values)}")

# Valeurs aux extrémités
print(f"\nValeurs aux extrémités (C_corr = {3-x_values[0]:.1f}):")
print(f"  Q_MM (moyen): {Q_MM_welf_list[0]:.4f}")
print(f"  Q_MM (groupes): {results_total['Q_MM'][0]:.4f}")
print(f"  Welfare (moyen): {W_welf_list[0]:.4f}")
print(f"  Welfare (groupes): {W_welf_groupes_list[0]:.4f}")

print(f"\nValeurs aux extrémités (C_corr = {3-x_values[-1]:.1f}):")
print(f"  Q_MM (moyen): {Q_MM_welf_list[-1]:.4f}")
print(f"  Q_MM (groupes): {results_total['Q_MM'][-1]:.4f}")
print(f"  Welfare (moyen): {W_welf_list[-1]:.4f}")
print(f"  Welfare (groupes): {W_welf_groupes_list[-1]:.4f}")

print("\n" + "=" * 60)
print("FIN DE L'ANALYSE")
print("=" * 60)



# =============================================================================================
# 7. AUTRES AFFICHAGES DÉSUÉS
# =============================================================================================


"""# === Affichage des résultats ===

# 1️⃣ Quantités par segment et par groupe
plt.figure()
for g, label in enumerate(group_labels):
    plt.plot(x_values, results_group[g]['Q_MM'], label=f'Q_MM {label}', linestyle='--')
    plt.plot(x_values, results_group[g]['Q_ME'], label=f'Q_ME {label}', linestyle='-.')
    plt.plot(x_values, results_group[g]['Q_EM'], label=f'Q_EM {label}', linestyle=':')
    plt.plot(x_values, results_group[g]['Q_EE'], label=f'Q_EE {label}', linestyle='-')
    plt.plot(x_values, results_group[g]['Q_V'], label=f'Q_V {label}', linestyle='-')
plt.plot(x_values, results_total['Q_MM'], label='Q_MM total', color='black', linestyle='--', linewidth=2)
plt.plot(x_values, results_total['Q_ME'], label='Q_ME total', color='black', linestyle='-.', linewidth=2)
plt.plot(x_values, results_total['Q_EM'], label='Q_EM total', color='black', linestyle=':', linewidth=2)
plt.plot(x_values, results_total['Q_EE'], label='Q_EE total', color='black', linestyle='-', linewidth=2)
plt.plot(x_values, results_total['Q_V'], label='Q_V total', color='black', linestyle='-', linewidth=2)
plt.xlabel('Economie sur les coups de correspondance (3 - C_corr)')
plt.ylabel('Quantités')
plt.legend()
plt.title('Quantités par segment et par groupe')
plt.grid(True)
plt.show()

# 2️⃣ Utilité totale par groupe
plt.figure()
for g, label in enumerate(group_labels):
    plt.plot(x_values, results_group[g]['U_total'], label=f'U_total {label}')
plt.xlabel('Economie sur les coups de correspondance (3 - C_corr)')
plt.ylabel('Utilité totale')
plt.legend()
plt.title('Utilité totale par groupe')
plt.grid(True)
plt.show()

# 3️⃣ Quantité totale (somme pondérée)
plt.figure()
plt.plot(x_values, results_total['Q_total'], label='Q_total (somme pondérée)', color='black', linewidth=2)
plt.xlabel('Economie sur les coups de correspondance (3 - C_corr)')
plt.ylabel('Quantité totale')
plt.legend()
plt.title('Quantité totale (somme pondérée des groupes)')
plt.grid(True)
plt.show()

"""


"""
plt.figure()
plt.plot(x_values, U_MM_welf_list_centered, label='U_MM', linestyle='--', color='tab:blue')
plt.plot(x_values, U_ME_welf_list_centered, label='U_ME', linestyle='--', color='tab:orange')
plt.plot(x_values, U_EM_welf_list_centered, label='U_EM', linestyle='--', color='tab:green')
plt.plot(x_values, U_EE_welf_list_centered, label='U_EE', linestyle='--', color='tab:red')
plt.plot(x_values, U_V_welf_list_centered, label='U_V', linestyle='-', color='tab:purple')
plt.plot(x_values, W_welf_list_centered, label='Welfare total', color='black', linewidth=2)
plt.plot(x_values, U_welf_list_centered, label='Utilité', linestyle='-', color='tab:blue')
plt.plot(x_values, U_sommes_list_centered, label='Somme des utilités', linestyle='-.', color='tab:orange')
plt.plot(x_values, Pi_welf_list_centered, label='Profit total', linestyle='-.', color='tab:red')
plt.plot(x_values, Ext_welf_list_centered, label='Externalités', linestyle='-', color='tab:green')
plt.ylabel('Évolution (Δ) vs C_corr=0')
plt.xlabel('Economie sur les coups de correpondance (3 - C_corr)')
plt.legend()
plt.title('Décomposition du bien-être (individuel)')
plt.grid(True)
plt.show()



# 1️⃣ Prix
plt.figure()
#plt.plot(x_values, P_M_list, label='P_M individuel', linestyle='-', color=colors['individuel'])
#plt.plot(x_values, P_E_list, label='P_E individuel', linestyle='--', color=colors['individuel'])
#plt.plot(x_values, P_M_coop_list, label='P_M alliance', linestyle='-', color=colors['alliance'])
#plt.plot(x_values, P_E_coop_list, label='P_E alliance', linestyle='--', color=colors['alliance'])
plt.plot(x_values, P_M_welf_list, label='P_M welfare', linestyle='-', color=colors['welfare'])
plt.plot(x_values, P_E_welf_list, label='P_E welfare', linestyle='--', color=colors['welfare'])
plt.plot(x_values, P_V_welf_list, label='P_V welfare', linestyle=':', color=colors['welfare'])
plt.plot(x_values, Pi_E_welf_list_centered, label='Pi_E welfare (profit)', linestyle='-.', color= 'green')
plt.plot(x_values, Pi_M_welf_list_centered, label='Pi_M welfare (profit)', linestyle=':', color= 'blue')  
#plt.plot(x_values, P_M_stackelberg_list, label='P_M Stackelberg', linestyle='-', color=colors['stackelberg'])
#plt.plot(x_values, P_E_stackelberg_list, label='P_E Stackelberg', linestyle='--', color=colors['stackelberg'])
plt.ylabel('Prix')
plt.xlabel('Economie sur les coups de correpondance (3 - C_corr)')
plt.legend()
plt.title('Prix')
plt.grid(True)
plt.show()

# 2️⃣ Quantités (Q_V et Q_total)
plt.figure()
#plt.plot(x_values, Q_V_list, label='Q_V individuel', linestyle='-', color=colors['individuel'])
#plt.plot(x_values, Q_V_coop_list, label='Q_V alliance', linestyle='-', color=colors['alliance'])
plt.plot(x_values, Q_V_welf_list, label='Q_V welfare', linestyle='-', color=colors['welfare'])
plt.plot(x_values, Q_ME_welf_list, label='Q_ME welfare', linestyle='-', color=colors['stackelberg'])
plt.plot(x_values, Q_MM_welf_list, label='Q_MM welfare', linestyle='--', color=colors['stackelberg'])
plt.plot(x_values, Q_EM_welf_list, label='Q_EM welfare', linestyle='-.', color=colors['stackelberg'])
plt.plot(x_values, Q_EE_welf_list, label='Q_EE welfare', linestyle=':', color=colors['stackelberg'])
#plt.plot(x_values, Q_V_stackelberg_list, label='Q_V Stackelberg', linestyle='-', color=colors['stackelberg'])

#plt.plot(x_values, Q_total_list, label='Q_total individuel', linestyle='--', color=colors['individuel'])
#plt.plot(x_values, Q_total_coop_list, label='Q_total alliance', linestyle='--', color=colors['alliance'])
plt.plot(x_values, Q_total_welf_list, label='Q_total welfare', linestyle='--', color=colors['welfare'])
#plt.plot(x_values, Q_total_stackelberg_list, label='Q_total Stackelberg', linestyle='--', color=colors['stackelberg'])

plt.ylabel('Quantités')
plt.xlabel('Economie sur les coups de correpondance (3 - C_corr)')
plt.legend()
plt.title('Quantités')
plt.grid(True)
plt.show()


plt.figure()
plt.plot(x_values, mu_MM_list, label='μ_MM')
plt.plot(x_values, mu_ME_list, label='μ_ME')
plt.plot(x_values, mu_EM_list, label='μ_EM')
plt.plot(x_values, mu_EE_list, label='μ_EE')
plt.plot(x_values, mu_V_list, label='μ_V')
plt.xlabel('Economie sur les coups de correspondance (3 - C_corr)')
plt.ylabel('Multiplicateurs de Lagrange (μ)')
plt.legend()
plt.title('Évolution des multiplicateurs de Lagrange (KKT)')
plt.grid(True)
plt.show()"""


"""
# ==========================================================================================
# 11 Graphe 2D des configurations KKT selon C_corr et delta_E
# ==========================================================================================

# Paramètres pour le graphe 2D
C_corr_range = np.linspace(0, 3, 30)
delta_E_range = np.linspace(0, 3, 30)
C_corr_mesh, delta_E_mesh = np.meshgrid(C_corr_range, delta_E_range)

# Matrice pour stocker les configurations
config_matrix = np.zeros((len(delta_E_range), len(C_corr_range)), dtype=int)

# Dictionnaire pour mapper les configurations à des couleurs
config_to_color = {
    (True, True, True, True, True): 1,  # Vert : tous actifs (nominal)
    (True, True, False, True, True): 2,  # Bleu : Q_EM = 0
    (True, False, True, True, True): 3,  # Orange : Q_ME = 0
    (True, True, True, False, True): 4,  # Rouge : Q_EE = 0
    (False, True, True, True, True): 5,  # Violet : Q_MM = 0
    (True, True, True, True, False): 6,  # Cyan : Q_V = 0
    (True, False, False, True, True): 7,  # Marron : Q_ME = Q_EM = 0
    (False, False, False, False, True): 8,  # Rose : Q_V seul actif
    (True, False, False, False, False): 9,  # Gris : Q_MM seul actif
    (False, False, False, False, False): 10, # Noir : aucun actif (impossible)
}

# Noms des configurations pour la légende (simplifiés)
config_names = {
    1: "Tous actifs",
    2: "Q_EM = 0", 
    3: "Q_ME = 0",
    4: "Q_EE = 0",
    5: "Q_MM = 0",
    6: "Q_V = 0",
    7: "Q_ME = Q_EM = 0",
    8: "Q_V seul",
    9: "Q_MM seul",
    10: "Aucun actif"
}

print("Calcul des configurations KKT...")

# Set pour collecter toutes les configurations uniques trouvées
configurations_trouvees = set()

# Parcours de la grille
for i, delta_E_val in enumerate(delta_E_range):
    for j, C_corr_val in enumerate(C_corr_range):
        # Mise à jour des paramètres
        param_values_temp = param_values.copy()
        param_values_temp['C_corrME'] = C_corr_val
        param_values_temp['C_corrEM'] = C_corr_val
        param_values_temp['delta_E'] = delta_E_val
        
        # Résolution KKT
        solution = solve_kkt_given_parameters(solutions_kkt_full, param_values_temp)
        
        if solution is not None:
            # Déterminer quelle configuration est active
            Q_MM_active = solution[Q_MM] > 1e-6
            Q_ME_active = solution[Q_ME] > 1e-6
            Q_EM_active = solution[Q_EM] > 1e-6
            Q_EE_active = solution[Q_EE] > 1e-6
            Q_V_active = solution[Q_V] > 1e-6
            
            config_tuple = (Q_MM_active, Q_ME_active, Q_EM_active, Q_EE_active, Q_V_active)
            configurations_trouvees.add(config_tuple)
            
            # Debug: afficher la configuration trouvée
            #print(f"Configuration trouvée: {config_tuple}")
            #print(f"Q_MM: {solution[Q_MM]:.6f}, Q_ME: {solution[Q_ME]:.6f}, Q_EM: {solution[Q_EM]:.6f}, Q_EE: {solution[Q_EE]:.6f}, Q_V: {solution[Q_V]:.6f}")
            
            # Assigner une couleur
            if config_tuple in config_to_color:
                config_matrix[i, j] = config_to_color[config_tuple]
                #print(f"Configuration: {config_tuple} -> Couleur: {config_to_color[config_tuple]}")
            else:
                config_matrix[i, j] = 0  # Configuration non répertoriée
                #print(f"Configuration NON répertoriée: {config_tuple}")
        else:
            config_matrix[i, j] = 0  # Pas de solution

# Debug: afficher toutes les configurations trouvées
print(f"\nConfigurations trouvées: {configurations_trouvees}")
print(f"Configurations prédéfinies: {set(config_to_color.keys())}")
print(f"Configurations manquantes: {configurations_trouvees - set(config_to_color.keys())}")

# Création du graphe
plt.figure(figsize=(12, 8))

# Création d'une colormap personnalisée
colors_list = ['white', 'green', 'blue', 'orange', 'red', 'purple', 'cyan', 'brown', 'pink', 'gray', 'black']
cmap = ListedColormap(colors_list)
bounds = np.arange(-0.5, len(colors_list) + 0.5, 1)
norm = BoundaryNorm(bounds, cmap.N)

# Affichage avec pcolormesh
im = plt.pcolormesh(C_corr_mesh, delta_E_mesh, config_matrix, cmap=cmap, norm=norm, shading='auto')

# Configuration du graphe
plt.xlabel('Coût de correspondance (C_corr)')
plt.ylabel('Prime intermodale E (δ_E)')
plt.title('Configurations KKT selon C_corr et δ_E')
# plt.colorbar(im, ticks=range(len(colors_list)), label='Configuration')  # Supprimé car redondant avec la légende

# Légende simplifiée - seulement les configurations qui apparaissent
configs_presentes = np.unique(config_matrix)
configs_presentes = configs_presentes[configs_presentes > 0]  # Exclure le blanc (0)

legend_elements = [Rectangle((0,0),1,1, facecolor=colors_list[i], label=config_names[i]) 
                   for i in configs_presentes if i in config_names]
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), title="Configurations trouvées")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Statistiques des configurations
print("\nStatistiques des configurations trouvées :")
unique_configs, counts = np.unique(config_matrix, return_counts=True)
for config_id, count in zip(unique_configs, counts):
    if config_id in config_names:
        percentage = (count / config_matrix.size) * 100
        print(f"{config_names[config_id]}: {count} points ({percentage:.1f}%)")

# ==========================================================================================
# 12 Calcul analytique des frontières KKT
# ==========================================================================================

def calculer_frontieres_analytiques():
    #Calcule analytiquement les frontières entre configurations KKT.
    #Les frontières sont définies par Q_i = 0 et μ_i > 0.

    print("Calcul des frontières analytiques...")
    
    # Symboles pour les paramètres
    C_corr_sym = sp.Symbol('C_corr', real=True)
    delta_E_sym = sp.Symbol('delta_E', real=True)
    
    # Substitution pour les paramètres variables
    subs_dict_frontieres = {
        beta: 2,
        gamma: 0.5,
        A_MM: param_values['A0'] - 2 * param_values['k_M'],
        A_EE: param_values['A0'] - 2 * param_values['k_E'],
        A_ME: param_values['A0'] - param_values['k_M'] - param_values['k_E'] - C_corr_sym,
        A_EM: param_values['A0'] - param_values['k_M'] - param_values['k_E'] - C_corr_sym,
        A_V: param_values['A_V'],
        P_V_fixe: param_values['P_V_fixe'],
        C_corrEM: C_corr_sym,
        C_corrME: C_corr_sym,
        delta_E: delta_E_sym
    }
    
    # Lagrangien avec substitution symbolique
    L_symbolique = W_welfare \
        - 0 * lambda_ * (sum(Q_vars) - D) \
        + sum([mu_vars[i] * Q_vars[i] for i in range(5)]) \
        + mu_PV * (P_V - P_V_fixe)
    
    L_symbolique = L_symbolique.subs(subs_dict_frontieres)
    
    # Calcul des dérivées partielles
    dL_dQ = [sp.diff(L_symbolique, Q_vars[i]) for i in range(5)]
    
    # Frontière "Tous actifs" → "Q_EM = 0"
    # Condition : Q_EM = 0 et μ_EM > 0
    print("\nFrontière 'Tous actifs' → 'Q_EM = 0':")
    
    # Système d'équations pour la configuration (1,1,0,1,1)
    eqs_frontiere_EM = []
    
    # Stationnarité pour Q_MM, Q_ME, Q_EE, Q_V (actifs)
    eqs_frontiere_EM.extend([sp.Eq(dL_dQ[i], 0) for i in [0, 1, 3, 4]])
    
    # Q_EM = 0 (inactif)
    eqs_frontiere_EM.append(sp.Eq(Q_EM, 0))
    
    # μ_MM = μ_ME = μ_EE = μ_V = 0 (actifs)
    eqs_frontiere_EM.extend([sp.Eq(mu_vars[i], 0) for i in [0, 1, 3, 4]])
    
    # Contrainte P_V = P_V_fixe
    eqs_frontiere_EM.append(sp.Eq(P_V - P_V_fixe, 0))
    
    # Variables à résoudre
    vars_frontiere = [Q_MM, Q_ME, Q_EE, Q_V, mu_EM, mu_PV]
    
    try:
        sol_frontiere = sp.solve(eqs_frontiere_EM, vars_frontiere, dict=True)
        if sol_frontiere:
            print("Solution trouvée pour la frontière Q_EM = 0:")
            for sol in sol_frontiere:
                print(f"Q_MM = {sol[Q_MM]}")
                print(f"Q_ME = {sol[Q_ME]}")
                print(f"Q_EE = {sol[Q_EE]}")
                print(f"Q_V = {sol[Q_V]}")
                print(f"μ_EM = {sol[mu_EM]}")
                print(f"μ_PV = {sol[mu_PV]}")
                
                # Condition μ_EM > 0 pour la frontière
                mu_EM_expr = sol[mu_EM]
                print(f"Condition μ_EM > 0: {mu_EM_expr} > 0")
        else:
            print("Aucune solution trouvée pour cette frontière")
    except Exception as e:
        print(f"Erreur lors du calcul de la frontière: {e}")
    
    return L_symbolique, dL_dQ

# Appel de la fonction (optionnel)
# L_symbolique, dL_dQ = calculer_frontieres_analytiques()


"""
