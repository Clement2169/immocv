import pandas as pd
import os
import numpy as np
from config import FLAT_NAME,HOUSE_FLAT_CHOICE, HOUSE_NAME, load_model_file, load_parquet_file

FLAT_MODEL_NAME = "flat_model"
HOUSE_MODEL_NAME= "house_model"

def house_flat_encoding(df, model_name):
        

        if model_name == FLAT_MODEL_NAME :
            energy_prefix  = "chauf_energy" 
            ges_name = "ges_class_num"
            dpe_name = 'dpeL_num'

        else :
            energy_prefix  = "chauf_energie" 
            ges_name = "ges_class"
            dpe_name = 'dpeL'


        #  *****************************************************************************
        #  Traitement_NA
        #  *****************************************************************************

        if 'places_parking' in df.columns:
            df['places_parking']=df['places_parking'].fillna(0) #On support que les NaN n'ont pas de place parking 
            df.loc[df['surface_terrain'].isna(),'surface_terrain']=0


        #  *****************************************************************************
        #  ges_class & dpeL
        #  *****************************************************************************

        mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7 , 'F/G':7 , 'Unknown':7}
        if 'ges_class' in df.columns:
            df[ges_name]=df['ges_class'].map(mapping)
        if 'dpeL' in df.columns:
            df[dpe_name]=df['dpeL'].map(mapping)

        #  *****************************************************************************
        #  chauffage_energie
        #  *****************************************************************************

        #Regrouper les systèmes combinés
        if 'chauffage_energie' in df.columns:
            df["chauff_energie_encoded"]=df['chauffage_energie'].apply(lambda x : 'combined' if type(x)==list else x)
            #pd.dummies
            df=df.merge(pd.get_dummies(df["chauff_energie_encoded"],prefix=energy_prefix,dtype='int'),how='left', left_index=True, right_index=True)
            df.drop(columns=['chauffage_energie','chauff_energie_encoded'],inplace=True)


        #  *****************************************************************************
        #  chauffage_systeme
        #  *****************************************************************************
        #Regrouper les systèmes combinés
        if 'chauffage_systeme' in df.columns:
            df["chauff_sys_encoded"]=df['chauffage_systeme'].apply(lambda x : 'combined' if type(x)==list else x)
            #pd.dummies
            df=df.merge(pd.get_dummies(df["chauff_sys_encoded"],prefix='chauf_sys',dtype='int'),how='left', left_index=True, right_index=True)
            df.drop(columns=['chauffage_systeme','chauff_sys_encoded'],inplace=True)


        if 'chauffage_mode' in df.columns:
            df["chauf_mode_encoded"]=df['chauffage_mode'].apply(lambda x : 'combined' if type(x)==list else x)
            #pd.dummies
            df=df.merge(pd.get_dummies(df["chauf_mode_encoded"],prefix='chauf_mode',dtype='int'),how='left', left_index=True, right_index=True)
            df.drop(columns=['chauf_mode_encoded','chauffage_mode'],inplace=True)

        return df

def house_flat_exposition_streamlit (df):
        #  *****************************************************************************
        #  Exposition
        #  *****************************************************************************
        for direction in ['nord', 'sud', 'est', 'ouest']:
                df[f'has_{direction}'] = [1 if direction == df['expo'][0] else 0  ]
        df.drop(columns='expo',inplace=True)
        return df

def house_flat_add_ACP(df,pca):
        pc1=pca.iloc[:, [0]+ list(range(7, pca.shape[1]))]
        df=df.merge(pc1,how='inner',on='CODE_IRIS')
        df.drop(columns='CODE_IRIS',inplace=True)
        return df

import streamlit as st
def house_input_prep(input_house,box_names,pca,model_name):
    df_house_pred=pd.DataFrame([input_house])
    df_house_pred[box_names] = df_house_pred[box_names].apply(pd.to_numeric, errors='coerce')
    
    st.write("DF avant encodage : ")
    st.dataframe(df_house_pred)
    
    df_house_encoded=house_flat_encoding(df_house_pred,model_name)
    df_house_encoded=house_flat_exposition_streamlit(df_house_encoded)
    
    st.write("DF apres encodage : ")
    st.dataframe(df_house_encoded)

    df_house_encoded=house_flat_add_ACP(df_house_encoded,pca)
    st.write("DF apres ACP : ")
    st.dataframe(df_house_encoded)

    return df_house_encoded




import pickle
def house_flat_price_pred(df_house_encoded,final_model,columns_to_exclude):    
    # Obtenez la liste des colonnes avec lesquelles le modèle a été entraîné
    model_columns = final_model.feature_names_in_  # ou une liste de vos colonnes
    # Réindexer pour garantir que toutes les colonnes soient présentes
    df_encoded_reindexed = df_house_encoded.reindex(columns=model_columns)
    
    # Identifiez les colonnes à remplir (toutes sauf les exclues)
    columns_to_fill = [col for col in model_columns if col not in columns_to_exclude]
    # Appliquer fillna(0) uniquement sur certaines colonnes
    df_encoded_reindexed[columns_to_fill] = df_encoded_reindexed[columns_to_fill].fillna(0)

    st.write("DF df_encoded_reindexed : ")
    st.dataframe(df_encoded_reindexed)

    # Faire une prédiction
    prediction = final_model.predict(df_encoded_reindexed)
    return df_encoded_reindexed,prediction


# ******************************************
# SHAP explainer 
#*******************************************
import shap
import matplotlib.pyplot as plt

def generate_shap_waterfall_plot(model, df_row):
    """
    Generate SHAP waterfall plot for a single row prediction
    Args:
        model: Trained machine learning model
        df_row: A single-row DataFrame with the input features

    Returns:
        SHAP waterfall plot figure
    """

    # Create the SHAP Explainer
    explainer = shap.TreeExplainer(model)  # Use TreeExplainer if it's a tree-based model (e.g., XGB, Random Forest)
                                        # For other models, you can use KernelExplainer or other SHAP explainers
        # Ensure the input type matches what SHAP requires
    if df_row is None:
        raise ValueError("Input data cannot be None.")
    
    if not isinstance(df_row, pd.DataFrame) and not isinstance(df_row, np.ndarray):
        raise TypeError(f"Expected pandas DataFrame or numpy array, but got {type(df_row)}")
    
    if isinstance(df_row, pd.DataFrame) and df_row.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    # Compute SHAP values for the single prediction
    shap_values = explainer.shap_values(df_row)

    # Extract base value and feature contributions for the prediction
    base_value = explainer.expected_value
    sample_shap_values = shap_values[0]  # Assuming you use XGB/RF; use shap_values directly for non-tree models

    # Generate waterfall plot
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(values=sample_shap_values, base_values=base_value, data=df_row.iloc[0]))
    return fig




# ******************************************
# Pos pred
#*******************************************
def plot_simple_thermometer(prediction, min_price, max_price, mean_price):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Calculer la position normalisée (pour déterminer la couleur)
    position_norm = (prediction - min_price) / (max_price - min_price)
    position_norm = max(0, min(1, position_norm))  # Limiter entre 0 et 1
    
    # Créer un gradient de vert à rouge
    # Vert (0) -> Jaune (0.5) -> Rouge (1)
    if position_norm <= 0.5:
        # De vert à jaune
        r = position_norm * 2
        g = 1
        b = 0
    else:
        # De jaune à rouge
        r = 1
        g = 2 * (1 - position_norm)
        b = 0
    
    bar_color = (r, g, b)


    # Thermomètre de base (avec valeurs réelles)
    ax.barh(0, max_price - min_price, left=min_price, height=0.3, 
            color='lightgray', edgecolor='black', linewidth=2)
    
    # Remplissage jusqu'à la prédiction avec la couleur calculée
    ax.barh(0, prediction - min_price, left=min_price, height=0.25, 
            color=bar_color, alpha=0.8, edgecolor='darkgray', linewidth=1)
    
    # Ligne de moyenne
    ax.axvline(mean_price, color='blue', linestyle='--', linewidth=2, 
               label=f'Moyenne: {mean_price:,.0f}€')
    
    # Marqueur prédiction
    ax.plot(prediction, 0, 'r*', markersize=25, markeredgecolor='darkred')
    
    # Labels des prix
    ax.text(min_price, -0.25, f'{min_price:,.0f}€', ha='center', fontsize=14)
    ax.text(max_price, -0.25, f'{max_price:,.0f}€', ha='center', fontsize=14)
    ax.text(prediction, 0.25, f'{prediction:,.0f}€', ha='center', 
            fontweight='bold', fontsize=15,
            bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='red', linewidth=2))
    
    # Configurer les limites
    padding = (max_price - min_price) * 0.05
    ax.set_xlim(min_price - padding, max_price + padding)
    ax.set_ylim(-0.5, 0.5)
    
    # Enlever les axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Légende
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    st.pyplot(fig)


def page_prediction_prix_house(data_dir_prix):
    
    model_name = HOUSE_MODEL_NAME
    if model_name not in st.session_state :
        filename = HOUSE_MODEL_NAME + ".pkl"
        file_path = data_dir_prix / filename
        final_model = pickle.load(open(file_path.as_posix(), 'rb'))
        st.session_state[model_name]=final_model

    final_model=st.session_state[model_name]
    pca = st.session_state["pca"]
    info_geo = st.session_state["reference_iris"]

    input_house={}
    box_names=['logement_neuf',  'surface',  'surface_terrain', 'annee_construction' ,'places_parking', 'nb_pieces','nb_toilettes', 'bain',  
                'DEP', 'REG','UU2010','CODE_IRIS','nb_log_n7',  'loyer_m2_median_n7', 'taux_rendement_n7']
    
    box_ids=['logement_neuf',  'surface',  'surface_terrain', 'annee_construction' , 'nb_pieces','nb_toilettes', 'bain','places_parking',  
                'nb_log_n7',  'loyer_m2_median_n7', 'taux_rendement_n7']
    
    box_ids_names=['logement neuf',  'surface',  'surface terrain', 'annee construction' , 'nb pieces','nb toilettes','nb salle de bain','nb places parking',   
                'nb log n7',  'loyer m2 median n7', 'taux rendement n7']
    
    box_ids_default=None
    columns_to_exclude = ['nb_log_n7',  'loyer_m2_median_n7', 'taux_rendement_n7']
    energies=['radiateur','sol' ,'pompe à chaleur','climatisation révérsible','convecteur','poêle à bois','cheminée','chaudière']    
    chauf_mode=None


    page_prediction_prix_commun (data_dir_prix, model_name, box_ids, box_ids_names,box_ids_default, box_names,energies,chauf_mode, columns_to_exclude)


#  *****************************************************************************
#  page_prediction_prix_flat
#  *****************************************************************************

def page_prediction_prix_flat(data_dir_prix):

    model_name = FLAT_MODEL_NAME
    if model_name not in st.session_state :
        filename = "appartement_model.gz"
        final_model = load_model_file(data_dir_prix,filename)
        st.session_state[model_name]=final_model
    
    box_ids=['logement_neuf',  'surface', 'annee_construction' , 'nb_pieces','nb_etages',   
               'nb_places_parking', 'nb_logements_copro','charges_copro_m2',  'ascenseur', 'porte_digicode','cave']
    
    box_ids_names=['logement neuf',  'surface', 'annee construction' , 'nb pieces','nb etages',   
               'nb places parking', 'nb logements copro','charges copro m2',  'ascenseur', 'porte digicode','cave']
    
    box_ids_default=[0,  60, 2010 , 4,6, 1,25,2000, 1,1,0]
    
    box_names= box_ids.copy()
    box_names.extend(['DEP', 'REG','UU2010','CODE_IRIS'])
    columns_to_exclude = [['nb_log_n7',  'loyer_m2_median_n7', 'taux_rendement_n7']]
    energies = ["radiateur","pompe à chaleur","convecteur","climatisation","chaudière","sol","poêle-bois"]
    chauf_mode=['individuel', 'collectif','central']


    page_prediction_prix_commun(data_dir_prix,model_name,box_ids, box_ids_names, box_ids_default,box_names,energies,chauf_mode, columns_to_exclude)


#  *****************************************************************************
#  page_prediction_prix_commun
#  *****************************************************************************

def page_prediction_prix_commun (data_dir_prix, model_name, box_ids, box_ids_names,box_ids_default, box_names,energies,chauf_mode, columns_to_exclude):
    
    
    info_geo  = st.session_state["reference_iris"]
    pca = st.session_state["pca"]
    final_model = st.session_state[model_name]
    
    input_house={}
    
    # Créez 3 colonnes
    col1, col2, col3 = st.columns(3)

    # Remplir la première colonne avec des inputs
    x=(len(box_ids))/3
    with col1:
        input_house['DEP'] = st.text_input(f'DEP',value =78)
        for i, name in enumerate(box_ids):
            if i // x == 0:  # pour s'assurer que chaque colonne a un certain nombre d'inputs
                if box_ids_default is not None :
                    input_house[name] = st.text_input(f'{box_ids_names[i]}',value=box_ids_default[i])
                else :
                    input_house[name] = st.text_input(f'{box_ids_names[i]}')
    # Remplir la deuxième colonne avec des inputs
    with col2:
        input_house['LIBCOM'] = st.selectbox('Commune',info_geo[info_geo['DEP']==input_house['DEP']]['LIBCOM'].unique(),index=1)
        for i, name in enumerate(box_ids):
            if i // x == 1:
                if box_ids_default is not None :
                    input_house[name] = st.text_input(f'{box_ids_names[i]}',value=box_ids_default[i])
                else :
                    input_house[name] = st.text_input(f'{box_ids_names[i]}')
    # Remplir la troisième colonne avec des inputs
    with col3:
        input_house['LIB_IRIS'] = st.selectbox('Quartier',info_geo[info_geo['LIBCOM']==input_house['LIBCOM']]['LIB_IRIS'].unique(),index=6)
        for i, name in enumerate(box_ids):
            if i // x == 2:
                if box_ids_default is not None :
                    input_house[name] = st.text_input(f'{box_ids_names[i]}',value=box_ids_default[i])
                else :
                    input_house[name] = st.text_input(f'{box_ids_names[i]}')
    
    
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

    col7, col8,col9 = st.columns(3)
    with col7:
        chauffage_energie_choices=['elec','gaz','fioul','bois']
        input_house['chauffage_energie'] = st.selectbox('chauffage_energie', chauffage_energie_choices,index=0)
    with col8:
        input_house['chauffage_systeme'] = st.selectbox('chauffage_systeme', energies,index=0)
    if chauf_mode is not None :
        with col9:
            chauffage_mode_choices=chauf_mode
            input_house['chauffage_mode'] = st.selectbox('chauffage_mode', chauffage_mode_choices,index=0)
        

        
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
    with st.expander("Afficher le résumé des informations du bien",expanded=False):
        st.write("",input_house)
    st.text("")
    st.text("")
    st.text("")

    # Supprimer plusieurs clés
    keys_to_remove = ["LIB_IRIS", "LIBCOM"]
    LIBCOM=input_house['LIBCOM']
    LIB_IRIS=input_house['LIB_IRIS']
    for key in keys_to_remove:
        input_house.pop(key, None)  # Utiliser `None` pour éviter une erreur si la clé n'existe pas

    if st.button("Lancer la prédiction 🎯 "):
        with st.expander("Afficher les étapes intermédiares de calcul",expanded=False):
        
            df_house_encoded=house_input_prep(input_house,box_names,pca,model_name)
            # Faire une prédiction
            df_encoded_reindexed , prediction = house_flat_price_pred(df_house_encoded,final_model,columns_to_exclude)
            st.session_state.prediction = prediction
        
        st.subheader(f" Le prix/m² estimé est de : ⭐ { st.session_state.prediction[0]:.0f} € ")

        # Generate SHAP waterfall plot for the prediction
        shap_plot = generate_shap_waterfall_plot(final_model, df_encoded_reindexed)
        
        # Display the SHAP Waterfall Plot
        st.pyplot(shap_plot)
        
        # thermometre de prix
        
        st.write(f'##### Comparaison avec la commune "{LIBCOM}"')
        stat_path=os.path.join(data_dir_prix,f'stat_COM_{st.session_state["house_flat"]}.parquet')
        stat=pd.read_parquet(stat_path)
        stat=stat[stat['LIBCOM']==LIBCOM]
        plot_simple_thermometer(st.session_state.prediction[0], stat['min'].values[0], stat['max'].values[0], stat['mean'].values[0])

        st.write(f"##### Comparaison dans l'IRIS \"{LIB_IRIS}\"")
        stat_path=os.path.join(data_dir_prix,f'stat_IRIS_{st.session_state["house_flat"]}.parquet')
        stat=pd.read_parquet(stat_path)
        stat=stat[stat['CODE_IRIS']==input_house['CODE_IRIS']]
        plot_simple_thermometer(st.session_state.prediction[0], stat['min'].values[0], stat['max'].values[0], stat['mean'].values[0])
    else:
        st.write("Cliquez sur le bouton pour calculer la prediction du  prix / m² avec Explication SHAP")


#  *****************************************************************************
#  page_prediction_prix_flat
#  *****************************************************************************

def page_prediction_prix(data_dir_prix):

    if "reference_iris" not in st.session_state :
        filename = "Reference_IRIS_geo2025"
        info_geo = load_parquet_file(data_dir_prix,filename)
        st.session_state["reference_iris"]=info_geo

    if "pca" not in st.session_state :
        filename = "df_ACP2_IRIS_immo"
        pca=load_parquet_file(data_dir_prix,filename)
        st.session_state["pca"]=pca

    house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
    st.session_state["house_flat"]=house_flat
    if house_flat == HOUSE_NAME :
        page_prediction_prix_house(data_dir_prix)
    else :
        page_prediction_prix_flat(data_dir_prix)



