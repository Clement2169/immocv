import pandas as pd
from pathlib import Path

from config import *

def house_encoding(df):

        #  *****************************************************************************
        #  Traitement_NA
        #  *****************************************************************************

        df['places_parking']=df['places_parking'].fillna(0) #On support que les NaN n'ont pas de place parking 
        df.loc[df['surface_terrain'].isna(),'surface_terrain']=0


        #  *****************************************************************************
        #  ges_class & dpeL
        #  *****************************************************************************

        mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7 , 'F/G':7 , 'Unknown':7}
        df['ges_class']=df['ges_class'].map(mapping)
        df['dpeL']=df['dpeL'].map(mapping)

        #  *****************************************************************************
        #  chauffage_energie
        #  *****************************************************************************

        #Regrouper les systèmes combinés
        df["chauff_energie_encoded"]=df['chauffage_energie'].apply(lambda x : 'combined' if type(x)==list else x)
        #pd.dummies
        df=df.merge(pd.get_dummies(df["chauff_energie_encoded"],prefix='chauf_energie',dtype='int'),how='left', left_index=True, right_index=True)
        df.drop(columns=['chauffage_energie','chauff_energie_encoded'],inplace=True)


        #  *****************************************************************************
        #  chauffage_systeme
        #  *****************************************************************************
        #Regrouper les systèmes combinés
        df["chauff_sys_encoded"]=df['chauffage_systeme'].apply(lambda x : 'combined' if type(x)==list else x)
        #pd.dummies
        df=df.merge(pd.get_dummies(df["chauff_sys_encoded"],prefix='chauf_sys',dtype='int'),how='left', left_index=True, right_index=True)
        df.drop(columns=['chauffage_systeme','chauff_sys_encoded'],inplace=True)


        return df

def house_exposition_streamlit (df):
        #  *****************************************************************************
        #  Exposition
        #  *****************************************************************************
        for direction in ['nord', 'sud', 'est', 'ouest']:
                df[f'has_{direction}'] = [1 if direction == df['expo'][0] else 0  ]
        df.drop(columns='expo',inplace=True)
        return df

def house_add_ACP(df,pca):
        pc1=pca.iloc[:, [0]+ list(range(7, pca.shape[1]))]
        df=df.merge(pc1,how='inner',on='CODE_IRIS')
        df.drop(columns='CODE_IRIS',inplace=True)
        return df

import streamlit as st
def house_input_prep(input_house,box_names,pca):
    df_house_pred=pd.DataFrame([input_house])
    df_house_pred[box_names] = df_house_pred[box_names].apply(pd.to_numeric, errors='coerce')
    
    st.write("DF avant encodage : ")
    st.dataframe(df_house_pred)
    
    df_house_encoded=house_encoding(df_house_pred)
    df_house_encoded=house_exposition_streamlit(df_house_encoded)
    
    st.write("DF apres encodage : ")
    st.dataframe(df_house_encoded)

    df_house_encoded=house_add_ACP(df_house_encoded,pca)
    st.write("DF apres ACP : ")
    st.dataframe(df_house_encoded)
    return df_house_encoded




import pickle
def house_price_pred(df_house_encoded,final_model):    
    # Obtenez la liste des colonnes avec lesquelles le modèle a été entraîné
    model_columns = final_model.feature_names_in_  # ou une liste de vos colonnes
    # Réindexer pour garantir que toutes les colonnes soient présentes
    df_encoded_reindexed = df_house_encoded.reindex(columns=model_columns)
    
    # Liste des colonnes à exclure du remplissage
    columns_to_exclude = ['nb_log_n7',  'loyer_m2_median_n7', 'taux_rendement_n7']
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