
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

from streamlit_data_vis import DataViz
from streamlit_acp_immo import INTERPRETATIONS_PC, OUTLIERS_A_EXCLURE, acp_compute_components,  acp_preprocess_data, afficher_stats_individus_st, afficher_stats_variables_st, get_top_features, plot_cercle_correlation_st, plot_nuage_individus_intelligent_st
from streamlit_maison_app import generate_shap_waterfall_plot,house_input_prep, house_price_pred,plot_simple_thermometer
from streamlit_modelisation_app import ACP_OPTION, DECISION_TREE, DECISION_TREE_REG, LINEAR, MODEL_NAMES, NON_LINEAR, SANS_ACP, XGB, XGB_REG, flat_plot_decision_tree, flat_plot_xgb
from streamlit_prevision_app import flat_display_exponential_predictions, flat_display_lstm_predictions, flat_display_monthly_data, flat_display_monthly_inflation_data, flat_display_prophet_inflation_predictions, flat_display_prophet_predictions, flat_merge_data_inflation, flat_plot_predictions

from config import *

#  *****************************************************************************
#  load_appartement_file
#  *****************************************************************************

def load_parquet_file (start_path, filename) :
    if filename.endswith(parquet_extension) :
        final_path = start_path / filename
    else :
        final_path = start_path / (filename + parquet_extension)
    return pd.read_parquet(final_path.as_posix())

#  *****************************************************************************
#  load_appartement_file
#  *****************************************************************************

def save_to_parquet_file (df, start_path, filename,suffix = "") :
    start_path = Path(start_path)
    if filename.endswith(parquet_extension) :
        final_path = start_path / (filename + suffix)
    else :
        final_path = start_path / (filename + suffix + parquet_extension)
    df.to_parquet(path=final_path.as_posix(),index=True,compression="gzip")

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

st.set_page_config(
    page_title="Application Streamlit Wide Mode",
    layout="wide",
)

st.session_state["type_de_bien_index"] = 1

st.title(PROJECT_TITLE)
st.sidebar.title("Sommaire")
pages = PAGES
page=st.sidebar.radio("", pages)

#  *****************************************************************************
#  Page : Présentation
#  *****************************************************************************
if page == pages[0] : 
    st.write("### Présentation")
    import base64

    nom_fichier_pdf = "Compagnon Immobilier_Soutenance-intro.pdf"
    my_path = data_dir_intro / nom_fichier_pdf
    with open(my_path.as_posix(), "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    
        # Affichage HTML toute page
        pdf_display = f'''
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}#view=FitH" 
                width="100%" 
                height="1000"
                rotate="-90deg"
                type="application/pdf"
                style="min-width:100%; width:100%; border:none;">
            </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)

#  *****************************************************************************
#  Page : Visualisation des data
#  *****************************************************************************
if page == pages[1] :
    DataViz()

#  *****************************************************************************
#  Page : Modelisation
#  *****************************************************************************
if page == pages[3] :
    title = "Modélisation du prix de vente au m2 des appartements et des maisons"
    st.write(title)

    index = st.session_state["type_de_bien_index"]

    st.subheader(r"Options pour la modelisation :")
    col1, col2, col3 = st.columns(3)
    with col1:
        house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=index)
    with col2:
        st.session_state["type_de_bien_index"] = get_type_de_bien_selection_box_index(house_flat)
        model_type = st.selectbox('Type de régression', MODEL_NAMES,index=2)
    with col3:
         with_acp = st.selectbox('ACP option',ACP_OPTION,index=0)
    st.write("")

    if house_flat == HOUSE_NAME :
        st.write(HOUSE_NAME)
    elif house_flat == FLAT_NAME :
    #  *****************************************************************************
    #  flat  Modelisation
    #  *****************************************************************************
            if with_acp == SANS_ACP :
                acp_suffix =""
                acp_suffix_reg ="-REG"
            else :
                acp_suffix = "-ACP"
                acp_suffix_reg ="-ACP-REG"

            if model_type == LINEAR :
                df = load_parquet_file(data_dir_model,"linear-regressors" + acp_suffix)
                st.write(df)
            if model_type == NON_LINEAR :
                df = load_parquet_file(data_dir_model,"non-linear-regressors" + acp_suffix)
                st.write(df)
            if model_type == XGB :
                flat_plot_xgb(data_dir_model,model_type,acp_suffix)
            if model_type == XGB_REG :
                flat_plot_xgb(data_dir_model,XGB,acp_suffix_reg)
            elif model_type == DECISION_TREE :
                flat_plot_decision_tree(data_dir_model,model_type,acp_suffix)
            elif model_type == DECISION_TREE_REG :
                flat_plot_decision_tree(data_dir_model,DECISION_TREE,acp_suffix_reg)



#  *****************************************************************************
#  Page : Time prediction
#  *****************************************************************************
if page == pages[4] : 
    # load data
    df = load_parquet_file(data_dir_temps,monthly_data_file)
    inflation = load_parquet_file(data_dir_temps,monthly_inflation_data_file)
 
    title = "Prediction en temps du prix au m2 des appartements sur la période " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
    st.write(title)

    tab1, tab2 = st.tabs(["Visualisation", "Prediction"])
    with tab1 :
        title = "Visualization des data  sur la période " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
        st.write(title)

        choices = ['prix de vente au m2', 'inflation et taux']
        option = st.selectbox('Choix de la visualisation', choices,index=0)

        if option == choices[0] :
            flat_display_monthly_data(df)
        else :
            flat_display_monthly_inflation_data (inflation,df)
    with tab2 :
        title = "Prediction du prix au m2 des appartements"
        st.write(title)
        length = 18

        df_merge = flat_merge_data_inflation(df,inflation)
        test_data = df_merge[-length:]

        choices = ['Prophet', 'Prophet avec inflation et taux','exponential smooting predictions','LSTM prediction','Summary']
        option = st.selectbox('Choix de la prédiction', choices,index=1)
        if option == choices[0] :
            forecast = flat_display_prophet_predictions(df, length)
            st.session_state['forecast'] = forecast
        elif option == choices[1] :
            forecast_inflation = flat_display_prophet_inflation_predictions(df_merge, length)
            st.session_state['forecast_inflation'] = forecast_inflation
        elif option == choices[2] :
            forecast_exponential = flat_display_exponential_predictions(df_merge, length)
            st.session_state['forecast_exponential'] = forecast_exponential
        elif option == choices[3] :
            forecast_lstm = flat_display_lstm_predictions(df_merge, length)
            st.session_state['forecast_lstm'] = forecast_lstm
        elif option == choices[4] :
            forecasts = {"Exponential smoothing": st.session_state['forecast_exponential'], 
                            "LSTM prediction" : st.session_state['forecast_lstm'], 
                            "Prophet": st.session_state['forecast'], 
                            "Prophet with inflation" : st.session_state['forecast_inflation']}
            flat_plot_predictions(forecasts,test_data,length)

#  *****************************************************************************
#  Page : prediction du prix
#  *****************************************************************************

if page == pages[5] :

    title = "Prediction du prix/m² de maison ou d'appartemment"
    st.write(title)
    
    filename = "Reference_IRIS_geo2025"
    info_geo = load_parquet_file(data_dir_prix,filename)

    house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
    if house_flat == HOUSE_NAME :
        input_house={}
        box_names=['logement_neuf',  'surface',  'surface_terrain', 'annee_construction' ,'places_parking', 'nb_pieces','nb_toilettes', 'bain',  
                    'DEP', 'REG','UU2010','CODE_IRIS','nb_log_n7',  'loyer_m2_median_n7', 'taux_rendement_n7']
        
        house_mod_box_names=['logement_neuf',  'surface',  'surface_terrain', 'annee_construction' , 'nb_pieces','nb_toilettes',   
                    'nb_log_n7',  'loyer_m2_median_n7', 'taux_rendement_n7']
        # Créez 3 colonnes
        col1, col2, col3 = st.columns(3)

        # Remplir la première colonne avec des inputs
        x=(len(house_mod_box_names))/3
        with col1:
            input_house['DEP'] = st.text_input(f'préciser le DEP',value =78)
            for i, name in enumerate(house_mod_box_names):
                if i // x == 0:  # pour s'assurer que chaque colonne a un certain nombre d'inputs
                    input_house[name] = st.text_input(f'préciser le {name}')
            input_house['places_parking'] = st.text_input('préciser le nb_places_parking')
        # Remplir la deuxième colonne avec des inputs
        with col2:
            input_house['LIBCOM'] = st.selectbox('préciser la commune',info_geo[info_geo['DEP']==input_house['DEP']]['LIBCOM'].unique())
            for i, name in enumerate(house_mod_box_names):
                if i // x == 1:
                    input_house[name] = st.text_input(f'préciser le {name}')
            input_house['bain'] = st.text_input('préciser le nb_salle de bain')
        # Remplir la troisième colonne avec des inputs
        with col3:
            input_house['LIB_IRIS'] = st.selectbox('préciser le quartier',info_geo[info_geo['LIBCOM']==input_house['LIBCOM']]['LIB_IRIS'].unique())
            for i, name in enumerate(house_mod_box_names):

                if i // x == 2:
                    input_house[name] = st.text_input(f'préciser le {name}')
        
        
        # Sélection DPE et GES
        dep_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        col4, col5,col6 = st.columns(3)
        with col4:
            expo_choices=['nord', 'sud', 'est', 'ouest']    
            input_house['expo'] = st.selectbox('Exposition', expo_choices,index=0)  
        with col5:
            input_house['dpeL'] = st.selectbox('dpeL', dep_choices, index=0)
        with col6:
            input_house['ges_class'] = st.selectbox('ges_class', dep_choices, index=0) 
     
        col7, col8 = st.columns(2)
        with col7:
            chauffage_energie_choices=['elec','gaz','fioul','bois']
            input_house['chauffage_energie'] = st.selectbox('chauffage_energie', chauffage_energie_choices,index=0)
        with col8:
            chauffage_systeme_choices=['radiateur','sol' ,'pompe à chaleur','climatisation révérsible','convecteur','poêle à bois','cheminée','chaudière']    
            input_house['chauffage_systeme'] = st.selectbox('chauffage_systeme', chauffage_systeme_choices,index=0)
        

        
        house_filter_geo = ((info_geo['DEP']==input_house['DEP']) &
                        (info_geo['LIBCOM']==input_house['LIBCOM']) & 
                        (info_geo['LIB_IRIS']==input_house['LIB_IRIS']))
        filtered_data = info_geo[house_filter_geo]
        if not filtered_data.empty:
            # Récupérer les valeurs de la première ligne du DataFrame filtré
            input_house['REG'] = int(filtered_data['REG'].values[0])
            input_house['DEP'] = int(filtered_data['DEP'].values[0])
            input_house['UU2010'] = int(filtered_data['UU2020'].values[0])
            input_house['CODE_IRIS'] = int(filtered_data['CODE_IRIS'].values[0])
        
        
        st.text("")
        st.text("")
        st.text("")
        with st.expander("**Afficher le résumé des inputs**",expanded=False):
            st.write("",input_house)
        st.text("")
        st.text("")
        st.text("")

        # Supprimer plusieurs clés
        keys_to_remove = ["LIB_IRIS", "LIBCOM"]
        LIBCOM=input_house['LIBCOM']
        for key in keys_to_remove:
            input_house.pop(key, None)  # Utiliser `None` pour éviter une erreur si la clé n'existe pas
        
        if "house_model" not in st.session_state :
            filename = "house_model.pkl"
            file_path = data_dir_prix / filename
            final_model = pickle.load(open(file_path.as_posix(), 'rb'))
            st.session_state["house_model"]=final_model
        if "pca" not in st.session_state :
            filename = "df_ACP2_IRIS_immo"
            pca=load_parquet_file(data_dir_prix,filename)
            st.session_state["pca"]=pca


        final_model=st.session_state["house_model"]
        pca = st.session_state["pca"]

        if st.button("Calculer"):
            with st.expander("**Afficher le résumé des inputs**",expanded=False):
            
                df_house_encoded=house_input_prep(input_house,box_names,pca)
                # Faire une prédiction
                
                df_encoded_reindexed , prediction = house_price_pred(df_house_encoded,final_model)
                st.session_state.prediction = prediction
            
            st.subheader(f"**** Le prix/m² estimé est de : { st.session_state.prediction[0]:.0f} € ****")

            # Generate SHAP waterfall plot for the prediction
            shap_plot = generate_shap_waterfall_plot(final_model, df_encoded_reindexed)
            
            # Display the SHAP Waterfall Plot
            st.pyplot(shap_plot)
            
            # thermometre de prix
            
            st.text("Comparaison avec la commune")
            stat_path=os.path.join(data_dir_prix,f'stat_COM_{house_flat}.parquet')
            stat=pd.read_parquet(stat_path)
            stat=stat[stat['LIBCOM']==LIBCOM]
            plot_simple_thermometer(st.session_state.prediction[0], stat['min'].values[0], stat['max'].values[0], stat['mean'].values[0])

            st.text("Comparaison dans l'IRIS")
            stat_path=os.path.join(data_dir_prix,f'stat_IRIS_{house_flat}.parquet')
            stat=pd.read_parquet(stat_path)
            stat=stat[stat['CODE_IRIS']==input_house['CODE_IRIS']]
            plot_simple_thermometer(st.session_state.prediction[0], stat['min'].values[0], stat['max'].values[0], stat['mean'].values[0])
        else:
            st.write("Cliquez sur le bouton pour calculer la prediction du  prix / m² avec Explication SHAP")
        

    elif house_flat == FLAT_NAME :
         st.write("")


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
                my_path = data_dir_acp / nom_fichier_pdf
                with open(my_path.as_posix(), "rb") as pdf_file:
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

#  *****************************************************************************
#  Page : Conclusion
#  *****************************************************************************
if page == pages[len(pages)-1] : 
    st.write("### Conclusion")
    nom_fichier_pdf = "Compagnon Immobilier_Soutenance-end.pdf"
    my_path = data_dir_intro / nom_fichier_pdf
    with open(my_path.as_posix(), "rb") as pdf_file:
        import base64
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    
        # Affichage HTML toute page
        pdf_display = f'''
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}#view=FitH" 
                width="100%" 
                height="1000" 
                rotate="-90deg"
                type="application/pdf"
                style="min-width:100%; width:100%; border:none;">
            </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
