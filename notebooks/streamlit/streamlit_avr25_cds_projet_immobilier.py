
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from streamlit_data_vis import DataViz
from streamlit_acp_immo import INTERPRETATIONS_PC, OUTLIERS_A_EXCLURE, acp_compute_components,  acp_preprocess_data, afficher_stats_individus_st, afficher_stats_variables_st, get_top_features, plot_cercle_correlation_st, plot_nuage_individus_intelligent_st
from streamlit_maison_app import  page_prediction_prix
from streamlit_modelisation_app import ACP_OPTION, AVEC_REGION, DECISION_TREE, DECISION_TREE_NAME, LINEAR, MODEL_NAMES, NON_LINEAR, REGION_OPTION, AVEC_ACP, XGB, XGB_NAME, flat_display_modelisation,flat_plot_decision_tree, flat_plot_xgb
from streamlit_prevision_app import  flat_plot_prevision_temps

from config import *


#  *****************************************************************************
#  calculate_metrics
#  *****************************************************************************

def calculate_metrics(actual, predicted, model_name):
    mae = mean_absolute_error(actual,predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mre = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)

    
    st.write(f"\n{model_name} performance  Metrics:")
    st.write(f"  MAE :  {mae:.2f}")
    st.write(f"  RMSE: {rmse:.2f}")
    st.write(f"  MRE : {mre:.2f}%")
    st.write(f"  R2  : {r2:.4f}")

    return mae, rmse, mre, r2

#  *****************************************************************************
#  flat_plot_decision_tree
#  *****************************************************************************

def get_type_de_bien_selection_box_index(value) :

    if value == HOUSE_NAME :
        return 0
    return 1

#  *****************************************************************************
#  main
#  *****************************************************************************

st.session_state["type_de_bien_index"] = 1

st.title(PROJECT_TITLE)
st.sidebar.title("Sommaire")
pages = PAGES
page=st.sidebar.radio("", pages)

#  *****************************************************************************
#  Page : Présentation
#  *****************************************************************************
if page == pages[0] : 
    st.subheader("Présentation")
    tab1, tab2,tab3 = st.tabs(["Introduction", "Fiche Projet","Méthodologie Projet"])
    with tab1 :
        nom_fichier_pdf = "compagnon-immobilier-1.png"
        my_path = data_dir_intro / nom_fichier_pdf
        st.image (my_path.as_posix())
    with tab2 :
        nom_fichier_pdf = "compagnon-immobilier-2.png"
        my_path = data_dir_intro / nom_fichier_pdf
        st.image (my_path.as_posix())
    with tab3 :
        nom_fichier_pdf = "compagnon-immobilier-3.png"
        my_path = data_dir_intro / nom_fichier_pdf
        st.image (my_path.as_posix())
#  *****************************************************************************
#  Page : Visualisation des data
#  *****************************************************************************
if page == pages[1] :
    DataViz(data_dir_visu)

#  *****************************************************************************
#  Page : Modelisation
#  *****************************************************************************
if page == pages[3] :
    title = "Modélisation du prix de vente au m2 des appartements et des maisons"
    st.write(title)

    index = st.session_state["type_de_bien_index"]

    house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=index)
    st.session_state["type_de_bien_index"] = get_type_de_bien_selection_box_index(house_flat)
    if house_flat == HOUSE_NAME :
        st.write(HOUSE_NAME)
    elif house_flat == FLAT_NAME :
    #  *****************************************************************************
    #  flat  Modelisation
    #  *****************************************************************************
        flat_display_modelisation(data_dir_model)
        

#  *****************************************************************************
#  Page : Time prediction
#  *****************************************************************************
if page == pages[4] : 
    # load data

    flat_plot_prevision_temps(data_dir_temps)


#  *****************************************************************************
#  Page : prediction du prix
#  *****************************************************************************

if page == pages[5] :

    page_prediction_prix(data_dir_prix=data_dir_prix)


#  *****************************************************************************
#  Page : ACP
#  *****************************************************************************
if page == pages[2] : 

    st.write("### Analyse en Composantes Principales")

    if "acp_load" not in st.session_state :
        st.session_state["acp_load"] = True
        filename = "df_enrich4_immo"
        st.session_state["df_cor"] = load_parquet_file(data_dir_acp,filename)

    df_cor = st.session_state["df_cor"]

    X_num, features = acp_preprocess_data (df_cor)
    
    n_components,ind_table,var_table,explained_variance_ratio,pca_df = acp_compute_components(X_num=X_num,df_cor=df_cor,features=features)

    tab1, tab2, tab3 = st.tabs(["Enrichissement des données", "Analyse ACP","Statistiques"])

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
        
        plot_cercle_correlation_st(explained_variance_ratio,var_table, features_to_plot, axe_x=axe_x_plot, axe_y=axe_y_plot)

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
    with tab3:

        st.header("Statistiques sur les variables complémentaires")

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

            # --- Tableau PDF interprétation des axes---
        st.divider()
        
        afficher_pdf = st.checkbox("Afficher le tableau synthétique des axes de l'ACP")

        if afficher_pdf:
 
            nom_fichier_pdf = "resume_acp-1.png"
            my_path = data_dir_acp / nom_fichier_pdf
            st.image (my_path.as_posix())
            nom_fichier_pdf = "resume_acp-2.png"
            my_path = data_dir_acp / nom_fichier_pdf
            st.image (my_path.as_posix())

#  *****************************************************************************
#  Page : Conclusion
#  *****************************************************************************
if page == pages[len(pages)-1] : 
    st.subheader("Conclusion")
    nom_fichier_pdf = "compagnon-immobilier-4.png"
    my_path = data_dir_intro / nom_fichier_pdf
    st.image (my_path.as_posix())