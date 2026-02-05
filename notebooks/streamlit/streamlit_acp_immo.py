from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
#  DICTIONNAIRE INTERPRÉTATION DES COMPOSANTES
# =============================================================================

INTERPRETATIONS_PC = {
    "PC1": "1: (-) Rural isolé Vs (+) Urbain dense et équipé",
    "PC2": "2: (-) Populaire ancien   Vs  (+) Résidentiel récent aisé",
    "PC3": "3: (-) Familial défavorisé   Vs   (+) CSP ++ centres anciens",
    "PC4": "4: (-) Actifs modestes   Vs  (+) Seniors aisés (mer...)",
    "PC5": "5: (-) Résidentiel ancré   Vs  (+) Récent en croissance",
    "PC6": "6: (-) Social 'aisé'  Vs  (+) Délinquence carrefour d'affluence (hub, tourisme)",
    "PC7": "7: (-) Technopole Univ.  Vs  (+) STUP quartiers sensibles",
    "PC8": "8: (-) ZAC récentes  Vs  (+) Habitat trad. contraint",
    "PC9": "9: (-) Stup evenement.  Vs (+) Vols zones transit",
    "PC10": "10: (-) Petites communes dynamiques  Vs  (+) déclinantes",
    "PC11": "11: (-) Urbain ancien  Vs  (+) Agricole en développement",
    "PC12": "12: (-) Commune dépendante Agglomération Vs (+) Commune autonome",
    "PC13": "13: (++) Communes du litoral, commerçantes",
    "PC14": "14: (-) Infrastructures rurales Vs  (+) Village touris. ancien"
}


# =============================================================================
#   OUTLIERS EXCLUS GRAPHIQUES
# =============================================================================

OUTLIERS_A_EXCLURE = [
    '33120000', '955270000', '772530000', '211920000', '503530000',
    '423130000', '32480000', '211110000', '525190000', '595930000',
    '623690000', '523840000', '54480000', '214950000', '763440000',
    '544800000', '884400000'
]


# =============================================================================
#  CHARGEMENT NETTOYAGE 
# =============================================================================

chemin_fichier = "/Users/christinejaloux/Documents/StreamlitACP/df_enrich4_immo.csv"

# Chargement  
def acp_load_data (input_path,filename) :

    chemin_fichier = input_path / filename

    df_cor = pd.read_csv(chemin_fichier.as_posix(), sep=';', dtype=str)
    if 'CODE_IRIS' in df_cor.columns:
        df_cor = df_cor.set_index('CODE_IRIS')

    # Colonnes textuelles
    cols_textuelles = ['INSEE_COM', 'REG', 'DEP', 'LIBCOM', 'LIB_IRIS', 'CATAEU2010', 'P_NP5CLA']

    # Nettoyage et conversion numérique
    for col in df_cor.columns:
        if col not in cols_textuelles:
            df_cor[col] = df_cor[col].astype(str).str.replace(' ', '').str.replace(',', '.')
            df_cor[col] = pd.to_numeric(df_cor[col], errors='coerce')

    # 
    outliers_a_supprimer = []
    df_cor = df_cor.drop(index=outliers_a_supprimer, errors='ignore')

    return df_cor

def acp_preprocess_data (df_cor) :
    # Sélection Variables Actives 

    # Colonnes textuelles
    cols_textuelles = ['INSEE_COM', 'REG', 'DEP', 'LIBCOM', 'LIB_IRIS', 'CATAEU2010', 'P_NP5CLA']
    # Colonnes illustratives pour l'ACP 
    illustratives_cols = cols_textuelles + ['prix_m2_appa', 'prix_m2_mais']

    variables_actives = [
        col for col in df_cor.columns
        if col not in illustratives_cols
        and pd.api.types.is_numeric_dtype(df_cor[col])
        and df_cor[col].notna().any()
    ]

    X_num = df_cor[variables_actives]
    X_num = X_num.fillna(X_num.mean()) 

    features = X_num.columns.tolist()

    return X_num, features


# =============================================================================
# ACP
# =============================================================================

def acp_compute_components(X_num,df_cor,features) :
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    n_ind, n_var = X_scaled.shape

    pca = PCA(n_components=min(14, n_var))
    principal_components = pca.fit_transform(X_scaled)

    n_components = principal_components.shape[1]
    pca_df = pd.DataFrame(
        data=principal_components,
        index=df_cor.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )

    eig_vals = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_


# =============================================================================
# Aides a l interpretation
# =============================================================================

    loadings = pca.components_.T * np.sqrt(eig_vals)
    cos2_var = loadings**2
    contrib_var = cos2_var / cos2_var.sum(axis=0) * 100

    var_table = pd.DataFrame(
        index=features,
        columns=pd.MultiIndex.from_product(
            [[f'PC{i+1}' for i in range(n_components)], ['Correlation', 'Cos2', 'Contribution_%']]
        )
    )
    for k in range(n_components):
        axe = f'PC{k+1}'
        var_table[(axe, 'Correlation')] = loadings[:, k]
        var_table[(axe, 'Cos2')] = cos2_var[:, k]
        var_table[(axe, 'Contribution_%')] = contrib_var[:, k]

    coords_ind = pca_df.values
    cos2_ind = (coords_ind**2) / (coords_ind**2).sum(axis=1, keepdims=True)
    contrib_ind = (coords_ind**2) / (n_ind * eig_vals) * 100

    ind_table = pd.DataFrame(
        index=df_cor.index,
        columns=pd.MultiIndex.from_product(
            [[f'PC{i+1}' for i in range(n_components)], ['Coord', 'Cos2', 'Contribution_%']]
        )
    )
    for k in range(n_components):
        axe = f'PC{k+1}'
        ind_table[(axe, 'Coord')] = coords_ind[:, k]
        ind_table[(axe, 'Cos2')] = cos2_ind[:, k]
        ind_table[(axe, 'Contribution_%')] = contrib_ind[:, k]

    return n_components,ind_table,var_table,explained_variance_ratio, pca_df

# =============================================================================
#  LABELS 
# =============================================================================

labels = {
    "INSEE_COM": "Code commune", "LIB_IRIS": "Nom IRIS", "LIBCOM": "Nom commune",
    "REG": "Région", "DEP": "Département", "P21_POP": "Pop. 2021",
    "prix_m2_mais": "Prix m² maisons", "prix_m2_appa": "Prix m² apparts",
    "TX_RP": "Part RP", "TX_SEC": "Part RS", "TX_VAC": "Part vacants",
    "PART_MAIS": "Part maisons", "PART_APP": "Part apparts",
    "PART_RP_1&2P": "Part RP 1-2 p.", "PART_RP_3&4P": "Part RP 3-4 p.",
    "PART_RP_5PP": "Part RP +5 p.", "PART_M30M2": "Part RP <30m²",
    "PART_3060M2": "Part RP 30-60m²", "PART_60100M2": "Part RP 60-100m²",
    "PART_P100M2": "Part RP >100m²", "PART_RP_AVT_1945": "Part RP <1945",
    "PART_RP_1946_1990": "Part RP 1946-90", "PART_RP_2006_2018": "Part RP 2006-18",
    "TX_PROP_RP": "Part Proprio RP", "TX_LOC_RP": "Part Locataire RP",
    "TX_LOCHLM_RP": "Part HLM RP", "PART_MEN_M2ans": "Ménages -2 ans",
    "PART_MEN_P10ans": "Ménages +10 ans", "PART_MAIS_AV1945": "Maisons <1945",
    "PART_MAIS_AP2006": "Maisons >2006", "PART_APP_AV1945": "Apparts <1945",
    "PART_APP_AP2006": "Apparts >2006", "PART_MEN_VOIT": "Mén. avec voiture",
    "PART_MEN_PARK": "Mén. avec parking", "PART_MEN_VOIT_SSPARK": "Voiture sans parking",
    "PART_SUROCCUP": "Suroccupation", "PART_ETUDIANTS": "Part étudiants",
    "PART_SSDIPLOME": "Part sans diplôme", "PART_SECONDAIRE": "Part secondaire",
    "PART_SUPERIEUR": "Part Bac+2/3/4", "PART_SUP_MAST": "Part Master+",
    "PROP_ENF": "Part enfants", "PROP_AD": "Part adultes <65",
    "PROP_P65": "Part 65+ ans", "PROP_AGRI": "Part agriculteurs",
    "PROP_ART_COM": "Part artisans/com.", "PROP_CADRES": "Part cadres",
    "PROP_INT_EMP": "Part prof. interm/emp", "PROP_OUVR": "Part ouvriers",
    "PROP_RETR": "Part retraités", "PROP_IMM": "Part immigrés",
    "VAR_POP0616": "Var. Pop 06-16", "VAR_EMPL0616": "Var. Emploi 06-16",
    "TAUX_EMPLOI16": "Taux emploi 2016", "TAUX_CHOM16": "Taux chômage 2016",
    "rev_med_2021": "Revenu médian", "C_Recours_Soins": "Access. soins (1-7)",
    "cambriolages_cor": "Cambriolages (‰)", "degradations_cor": "Dégradations (‰)",
    "escroqueries_cor": "Escroqueries (‰)", "stup_cor": "Stupéfiants (‰)",
    "stup_afd_cor": "AFD Stup. (‰)", "viol_phys_hors_cor": "Viol. hors fam. (‰)",
    "viol_phys_intra_cor": "Viol. intra-fam. (‰)", "viol_sex_cor": "Viol. sexuelles (‰)",
    "vol_accessoires_cor": "Vols acc. (‰)", "vol_veh_cor": "Vols ds véh. (‰)",
    "vol_de_veh_cor": "Vols de véh. (‰)", "vol_sans_viol_cor": "Vols sans viol. (‰)",
    "Taxe_foncier_bati": "Tx Foncière Bâti", "Taxe_habitation": "Tx Habitation",
    "CATAEU2010": "Cat. urbaine 2010", "GCD": "Densité (1-4)",
    "P_NP5CLA": "Centralité", "HC_ARC4": "Dist. centre majeur",
    "HC_ARC3P": "Dist. centre imp.", "HC_ARC2P": "Dist. centre inter.",
    "HC_ARC1P": "Dist. centre local", "NIV_EQUIP_2017": "Nb équipements",
    "Littoral": "Littoral (0/1)"
}

def get_label(var):
    return labels.get(var, var)


# =============================================================================
#  FONCTIONS GRAPHIQUES
# =============================================================================

def afficher_stats_variables_st(var_table, axe='PC1', top_n_pos=10, top_n_neg=10):
    df_axe = var_table[axe].copy()
    df_axe['Label'] = [get_label(idx) for idx in df_axe.index]
    
    # Affichage Label et Contribution 
    cols_show = ['Label', 'Contribution_%']

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- {axe} (+++) -")
        pos = df_axe[df_axe['Correlation'] > 0].sort_values('Contribution_%', ascending=False).head(top_n_pos)
        st.dataframe(pos[cols_show], hide_index=True)
    
    with col2:
        st.write(f"- {axe} (---) -")
        neg = df_axe[df_axe['Correlation'] < 0].sort_values('Contribution_%', ascending=False).head(top_n_neg)
        st.dataframe(neg[cols_show], hide_index=True)
        
    return list(pos.index) + list(neg.index)


def plot_cercle_correlation_st(explained_variance_ratio, var_table, features_list, axe_x=1, axe_y=2):
    cx, cy = f'PC{axe_x}', f'PC{axe_y}'
    x_vals = var_table.loc[features_list, (cx, 'Correlation')]
    y_vals = var_table.loc[features_list, (cy, 'Correlation')]
    names = [get_label(idx) for idx in features_list]
    
    hover_texts = [f"{n}<br>{cx}: {x:.2f}<br>{cy}: {y:.2f}" for n, x, y in zip(names, x_vals, y_vals)]
    
    var_x = explained_variance_ratio[axe_x - 1] * 100
    var_y = explained_variance_ratio[axe_y - 1] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, 
        mode='markers+text', 
        text=names,
        textposition="top center",
        hovertext=hover_texts, 
        hoverinfo="text",
        marker=dict(size=8, color='blue', opacity=0.5), 
        showlegend=False
    ))
    
    fleches = [dict(x=x, y=y, xref="x", yref="y", ax=0, ay=0, axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowcolor="#1f77b4") 
               for x, y in zip(x_vals, y_vals)]
    
    fig.update_layout(
        title=f"Cercle de corrélation {cx} & {cy}", 
        width=900, height=900,
        annotations=fleches,
        shapes=[dict(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, 
                     line_color="red", line_dash="dash")],
        xaxis=dict(
            title=f"{cx} ({var_x:.2f}% de variance expliquée)",
            range=[-1.1, 1.1],
            showgrid=True,
            zeroline=True,
            showline=True,
            constrain='domain'
        ),
        yaxis=dict(
            title=f"{cy} ({var_y:.2f}% de variance expliquée)",
            range=[-1.1, 1.1],
            showgrid=True,
            zeroline=True,
            showline=True,
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    st.plotly_chart(fig)


def afficher_stats_individus_st(ind_table, df_cor, axe='PC1', top_n=10):
    """Affiche les statistiques des individus pour un axe donné"""
    df_axe = ind_table[axe].copy()
    
    # Affichage : Commune, IRIS, Contribution
    cols_show = ['Commune', 'IRIS', 'Contrib_%']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"- {axe} (+++) -")
        pos = df_axe[df_axe['Coord'] > 0].sort_values('Contribution_%', ascending=False).head(top_n)
        df_pos = pd.DataFrame({
            'Commune': df_cor.loc[pos.index, 'LIBCOM'], 
            'IRIS': df_cor.loc[pos.index, 'LIB_IRIS'],
            'Contrib_%': pos['Contribution_%']
        })
        st.dataframe(df_pos[cols_show], hide_index=True)
    
    with col2:
        st.write(f"- {axe} (---) -")
        neg = df_axe[df_axe['Coord'] < 0].sort_values('Contribution_%', ascending=False).head(top_n)
        df_neg = pd.DataFrame({
            'Commune': df_cor.loc[neg.index, 'LIBCOM'], 
            'IRIS': df_cor.loc[neg.index, 'LIB_IRIS'],
            'Contrib_%': neg['Contribution_%']
        })
        st.dataframe(df_neg[cols_show], hide_index=True)
        
    return list(pos.index), list(neg.index)


def plot_nuage_individus_intelligent_st(pca_df, df_cor, ind_table, 
                         axe_x=1, axe_y=2, 
                         color_var='prix_m2_mais', 
                         price_min=None, price_max=None,
                         outliers_list=None):
    
    cx, cy = f'PC{axe_x}', f'PC{axe_y}'
    
    df_plot = pca_df[[cx, cy]].copy()
    
    infos = ['LIBCOM', 'LIB_IRIS', 'prix_m2_mais', 'prix_m2_appa']
    data_infos = df_cor[infos].copy()
    
    for col in ['prix_m2_mais', 'prix_m2_appa']:
        if data_infos[col].dtype == object:
            data_infos[col] = data_infos[col].astype(str).str.replace(r'\\s+', '', regex=True).str.replace(',', '.')
        data_infos[col] = pd.to_numeric(data_infos[col], errors='coerce')

    df_plot = df_plot.join(data_infos, how='left')
    
    # Exclusion des outliers
    if outliers_list:
        df_plot = df_plot.drop(index=[o for o in outliers_list if o in df_plot.index], errors='ignore')
    
    df_plot = df_plot.dropna(subset=['prix_m2_mais', 'prix_m2_appa'])

    label_couleur = get_label(color_var)

    df_plot['Hover'] = (
        "<b>" + df_plot['LIBCOM'].fillna('') + "</b> (" + df_plot.index.astype(str) + ")<br>" +
        "<i>" + df_plot['LIB_IRIS'].fillna('') + "</i><br>" +
        f"<b>{get_label('prix_m2_mais')} : " + df_plot['prix_m2_mais'].apply(lambda x: f"{x:.0f} €" if pd.notnull(x) else "N/A") + "</b><br>" +
        f"{get_label('prix_m2_appa')} : " + df_plot['prix_m2_appa'].apply(lambda x: f"{x:.0f} €" if pd.notnull(x) else "N/A")
    )

    titre_graphe = f"Projection IRIS/COMMUNES {cx} & {cy} - {label_couleur}"

    if price_min is not None and price_max is not None:
        titre_legende = label_couleur
        
        def categoriser(prix):
            if pd.isna(prix): return None
            if prix < price_min: return f"Moins de {price_min} €"
            if prix > price_max: return f"Plus de {price_max} €"
            return None 

        df_plot[titre_legende] = df_plot[color_var].apply(categoriser)
        df_plot = df_plot.dropna(subset=[titre_legende])
        
        color_map = {
            f"Moins de {price_min} €": "#0d0887",
            f"Plus de {price_max} €": "#f0f921"
        }
        
        fig = px.scatter(
            df_plot, x=cx, y=cy, 
            color=titre_legende, 
            color_discrete_map=color_map,
            category_orders={titre_legende: [f"Moins de {price_min} €", f"Plus de {price_max} €"]},
            title=titre_graphe,
            labels={cx: f"{cx}", cy: f"{cy}"},
            height=700
        )
    else:
        vmin, vmax = df_plot[color_var].min(), df_plot[color_var].max()
        df_plot['Color'] = df_plot[color_var]
        fig = px.scatter(
            df_plot, x=cx, y=cy, color='Color', range_color=[vmin, vmax], 
            color_continuous_scale='Plasma',
            title=titre_graphe,
            labels={cx: f"{cx}", cy: f"{cy}", 'Color': label_couleur},
            height=700
        )

    fig.update_traces(hovertemplate="%{customdata}", customdata=df_plot['Hover'])
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    x_max = max(abs(df_plot[cx].min()), abs(df_plot[cx].max())) * 1.1 if not df_plot.empty else 1
    y_max = max(abs(df_plot[cy].min()), abs(df_plot[cy].max())) * 1.1 if not df_plot.empty else 1
    fig.update_xaxes(range=[-x_max, x_max], zeroline=True)
    fig.update_yaxes(range=[-y_max, y_max], zeroline=True)
    
    st.plotly_chart(fig)

def get_top_features(var_table, axe, top_n=10):
            df_axe = var_table[axe].copy()
            pos = df_axe[df_axe['Correlation'] > 0].sort_values('Contribution_%', ascending=False).head(top_n).index.tolist()
            neg = df_axe[df_axe['Correlation'] < 0].sort_values('Contribution_%', ascending=False).head(top_n).index.tolist()
            return pos + neg

if __name__ == '__main__':
# =============================================================================
#  STREAMLIT 
# =============================================================================

    chemin_fichier = Path("/Users/christinejaloux/Documents/StreamlitACP")
    filename="df_enrich4_immo.csv"

    df_cor, X_num, features = acp_load_data(chemin_fichier,filename)
    n_components,ind_table,var_table,explained_variance_ratio,pca_df = acp_compute_components(X_num=X_num,df_cor=df_cor,features=features)

    tab1, tab2 = st.tabs(["Enrichissement des données", "Analyse ACP"])

    with tab1:
    
        st.header("Enrichissement des données")
        
        #
        st.markdown("#### Objectif : Enrichir le jeu de données avec des variables liées à la localisation des biens pour améliorer la performance prédictive des modèles.")
        
        st.markdown("""
        L'**IRIS** (Ilots Regroupés pour l'Information Statistique) présent dans notre base permet de la compléter via de nombreuses sources en Open Data 
        (Recensements INSEE, DGFIP, Ministère de l'équipement, Observatoires des territoires, SSMSI...).
        
        L'exploration des différentes bases disponibles nous a permis de construire **71 variables** caractéristiques de chaque IRIS, réparties dans les thèmes suivants :
        """)
        


        # --- THEME 1 ---
        with st.expander("**CARACTÉRISTIQUES DU PARC DE LOGEMENT**", expanded=True):
            st.markdown("""
            - **Structure du parc** : Part de maisons vs appartements
            - **Taille des logements** : Répartition par nombre de pièces (1-2 pièces, 3-4 pièces, 5+ pièces)
            - **Surface** : Répartition par tranches de surface (<30m², 30-60m², >100m²)
            - **Ancienneté** : Période de construction (Avant 1945, 1946-1990, Après 2006...)
            - **Statut d'occupation** : Propriétaires, Locataires, HLM, Résidences secondaires, Logements vacants
            - **Automobile** : Ménages avec voiture, avec place de stationnement
            - **Suroccupation** : Part de logements "surpeuplés"
            """)

        # --- THEME 2 ---
        with st.expander("**VARIABLES SOCIO-DÉMOGRAPHIQUES**"):
            st.markdown("""
            - **Niveau de diplôme** : Sans diplôme, Secondaire, Supérieur (Bac+2 à Master+)
            - **CSP** : Cadres, Ouvriers, Retraités, Agriculteurs, Artisans...
            - **Revenus** : Revenu médian
            - **Structure familiale** : Ménages avec enfants, Retraités, Étudiants
            - **Dynamique démographique et économique** : Variation de la population, Mobilité résidentielle  
            - **Chômage et Emploi**
            """)

        # --- THEME 3 ---
        with st.expander("**CRIMINALITÉ (SSMSI)**"):
            st.markdown("""
            - **Atteintes aux biens** : Cambriolages, Vols de véhicules, Dégradations
            - **Atteintes aux personnes** : Violences physiques (intra/hors famille)
            - **Trafic** :  Stupéfiants
            """)

        # --- THEME 4 ---
        with st.expander("**DISTANCE AUX ÉQUIPEMENTS ET SERVICES**"):
            st.markdown("""
            - **Accessibilité** : Distance aux centres de vie (Majeurs, Intermédiaires, Locaux)
            - **Services** : Niveau d'équipement (écoles, commerces, transports...)
            - **Santé** : Accessibilité aux soins
            - **Densité** : Classification INSEE, Littoral
            """)

        # --- THEME 5 ---
        with st.expander("**IMPÔTS LOCAUX**"):
            st.markdown("""
            - Taxe foncière sur le bâti
            - Taxe d'habitation
            """)
            
        st.info(f"**Soit un total de {len(features)} variables décrivant {len(df_cor)} zones géographiques (IRIS ou communes).**")
        
        st.divider()
        
        st.header("Réduction de dimension - ACP")

        st.markdown("""
        #### Objectif : Construire de nouvelles variables synthétiques (ou composantes principales) en préservant le maximum d'information. 
        L'ACP construit des composantes principales, combinaisons linéaires de chacune 71 variables initiales. Chaque individu (IRIS/Commune) est projeté sur sur l'axe de la composante principale, les coordonnées ainsi obtenues sont utilisées dans la modélisation.

        #### Avantages:
        - **Suppression de la multi-colinéarité :** les composantes principales sont orthogonales entre elles (indépendantes- R=0 ).
        - **Réduction de dimension :** en intégrant un nombre limité de variables (composantes principales) dans nos modèles, on réduit le risque d'overfitting en conservant le maximum de variance.
        - **Interprétabilité :** Il est possible d'interpréter les composantes principales à l'aide des variables les plus contributives, et ainsi d'améliorer l'explicabilité de nos modèles.
        """)
        
        
        
        #st.write(f"Données actuelles : {len(features)} variables actives, {len(df_cor)} individus.")


    with tab2:
        st.header("ACP sur les variables complémentaires")

        # --- SÉLECTION DES AXES ---
        st.subheader("Axes à interpréter:")

        col1, col2 = st.columns(2)
        with col1:
            axe_x_choix = st.selectbox(
                "Axe X", 
                [f"PC{i}" for i in range(1, min(15, n_components+1))],
                index=0,
                label_visibility="collapsed"
            )
        with col2:
            axe_y_choix = st.selectbox(
                "Axe Y", 
                [f"PC{i}" for i in range(1, min(15, n_components+1))],
                index=1,
                label_visibility="collapsed"
            )

        axe_x_plot = int(axe_x_choix.replace('PC', ''))
        axe_y_plot = int(axe_y_choix.replace('PC', ''))
        
        # Calcul des variances pour l'affichage
        var_x_disp = explained_variance_ratio[axe_x_plot - 1] * 100
        var_y_disp = explained_variance_ratio[axe_y_plot - 1] * 100


        # --- INTERPRÉTATION ---
        st.info(
            f"##### {axe_x_choix} ({var_x_disp:.2f}%)\n\n"
            f"##### {INTERPRETATIONS_PC.get(axe_x_choix, 'Non définie')}"
        )
        
        st.info(
            f"##### {axe_y_choix} ({var_y_disp:.2f}%)\n\n"
            f"##### {INTERPRETATIONS_PC.get(axe_y_choix, 'Non définie')}"
        )


        # --- CERCLE DE CORRÉLATION  ---
        
        

        features_to_plot = list(set(get_top_features(var_table, axe_x_choix) + get_top_features(var_table, axe_y_choix)))
        
        plot_cercle_correlation_st(var_table, features_to_plot, axe_x=axe_x_plot, axe_y=axe_y_plot)

        st.divider()


        # --- NUAGE DES INDIVIDUS ---

        type_prix = st.radio("PRIX MOYENS PAR IRIS/COMMUNE",
                            ["Prix m² maisons", "Prix m² appartements"],
                            horizontal=True)

        var_couleur = "prix_m2_mais" if type_prix == "Prix m² maisons" else "prix_m2_appa"
        borne_min_prix = None
        borne_max_prix = None

        plot_nuage_individus_intelligent_st(
            pca_df, df_cor, ind_table, 
            axe_x=axe_x_plot, axe_y=axe_y_plot,
            color_var=var_couleur, 
            price_min=borne_min_prix, 
            price_max=borne_max_prix,
            outliers_list=OUTLIERS_A_EXCLURE
        )
        
        st.divider()

        # --- STATISTIQUES  ---
        
        # Sélection de l'axe
        axe_etude = st.selectbox(
            "INDICATEURS COMPLEMENTAIRES:", 
            [f"PC{i}" for i in range(1, min(15, n_components+1))],
            index=0
        )

        # Checkbox pour les variables
        afficher_stats_var = st.checkbox("Afficher les contributions des variables")
        
        if afficher_stats_var:
            st.markdown("###### Contribution des variables aux axes")
            top_n_var = st.selectbox("Nombre de variables:", [5, 10, 20], index=1)
            
            afficher_stats_variables_st(var_table, axe=axe_etude, top_n_pos=top_n_var, top_n_neg=top_n_var)

        # Checkbox pour les (IRIS/Communes)
        afficher_stats_ind = st.checkbox("Afficher les contributions des IRIS/Communes")
        
        if afficher_stats_ind:
            st.markdown("###### Contributions des IRIS/communes aux axes")
            top_n_contrib = st.selectbox("Nombre d'individus:", [10, 20, 50], index=0)
            
            afficher_stats_individus_st(ind_table, df_cor, axe=axe_etude, top_n=top_n_contrib)



        import base64

            # --- Tableau PDF interprétation des axes---
        st.divider()
        
        afficher_pdf = st.checkbox("Afficher le tableau synthétique des axes de l'ACP")

        if afficher_pdf:
            import base64
            nom_fichier_pdf = "resume_acp.pdf"

            try:
                with open(nom_fichier_pdf, "rb") as pdf_file:
                    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                
                # Affichage HTML toute page
                pdf_display = f'''
                    <iframe 
                        src="data:application/pdf;base64,{base64_pdf}#view=FitH" 
                        width="100%" 
                        height="1000" 
                        type="application/pdf"
                        style="min-width:100%; width:100%; border:none;">
                    </iframe>
                '''
                st.markdown(pdf_display, unsafe_allow_html=True)
                
            except FileNotFoundError:
                st.error(f"Fichier introuvable.")



