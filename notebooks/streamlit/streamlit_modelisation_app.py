
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree

parquet_extension = ".parquet"
monthly_data_file= "monthly_data" + parquet_extension
monthly_inflation_data_file = "monthly_inflation_data" + parquet_extension

PROJECT_TITLE = "Projet immobilier - Modélisation des prix des maisons et appartements - France Métropolitaine"
PAGES = ["Présentation","Visualisation","Analyse ACP","Modélisation","Prédiction en temps","Prédiction du prix","Conclusion"]
FLAT_NAME = "appartement"
HOUSE_NAME = "maison"
HOUSE_FLAT_CHOICE = [HOUSE_NAME,FLAT_NAME]
LINEAR = "Linear Regressors"
NON_LINEAR = "Non Linear Regressors"
XGB = "XGBRegressor"
DECISION_TREE = "DecisionTreeRegressor"
MODEL_NAMES = [LINEAR,NON_LINEAR, XGB,DECISION_TREE]
AVEC_ACP = "Avec ACP"
SANS_ACP = "Sans ACP"
ACP_OPTION = [AVEC_ACP,SANS_ACP]
# define variables
immo_vis_dir = "../../../data/immo_vis/"
parquet_extension = ".parquet"
compression_extension = ".gz"
shap_app_image_extension = ".png"

metropole_appartement_file = "ventes-metropole-appartement" + parquet_extension
metropole_appartement_file_cleaned = metropole_appartement_file + "_V1_clean_" + parquet_extension

acp_appartement_file = "df_ACP_IRIS_immo_Processed"  + parquet_extension

app_model_data_file = "ventes-metropole-model-appartement" + parquet_extension
app_model_file = "ventes-metropole-model-appartement" 

app_shap_explainer_file = "shap-explainer"


DropRegDep = True
DropLoyer = True
AddACP = True

#  *****************************************************************************
#  load_appartement_file
#  *****************************************************************************

import joblib
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

import time

#  *****************************************************************************
#  load_appartement_file
#  *****************************************************************************

def load_appartement_file (start_path, filename) :
    if filename.endswith(parquet_extension) :
        final_path = start_path / filename
    else :
        final_path = start_path / (filename + parquet_extension)
    return pd.read_parquet(final_path.as_posix())



#  *****************************************************************************
#  load_model_file
#  *****************************************************************************
def load_model_file (start_path, filename,modelname,comp_ext="") :
    final_path = start_path / (filename + '-' + modelname + comp_ext)
    return joblib.load(final_path)

#  *****************************************************************************
#  apply_preprocessing
#  *****************************************************************************
def apply_preprocessing  (df,target_column,drop_region=True,drop_loyer=True) :
    y_tmp =df[target_column]
    columns = [target_column]
    if drop_region :
        columns.append("DEP")
        columns.append("REG")
    if drop_loyer :
        columns.extend(['loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7'])
        
    X_tmp =df.drop(columns=columns)
    return X_tmp,y_tmp

#  *****************************************************************************
#  create_train_test_data
#  *****************************************************************************
def create_train_test_data (X ,y) :
    from sklearn.model_selection import train_test_split
    #  build the train and test data
    print (X.shape)
    return train_test_split(X,y,test_size=0.2,random_state=42)

#  *****************************************************************************
#  create_train_test_data
#  *****************************************************************************
def create_train_test_data_subset (X ,y,subset_size = 1.0) :
    from sklearn.model_selection import train_test_split
    #  build the train and test data
    print (X.shape)
    return train_test_split(X,y,test_size=0.2,train_size=subset_size, random_state=42)

#  *****************************************************************************
#  create_train_test_data
#  *****************************************************************************
def train_models (models,X_train, y_train,X_test,y_test) :
    results = []
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        
        # Mesurer le temps d'entraînement
        start_time = time.time()
        fit_time = time.time() - start_time
        
        # Mesurer le temps de prédiction
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Calculer les métriques
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'R²': r2,
            'Fit_Time': f"{fit_time:.4f}s",
            'Predict_Time': f"{predict_time:.4f}s",
            'Total_Time': f"{fit_time + predict_time:.4f}s"
        })

    # Afficher les résultats
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')


    return results_df

#  *****************************************************************************
#  flat_plot_shap
#  *****************************************************************************

def flat_plot_xgb (current_dir, model_type,acp_suffix) :

    summary, bar,waterfall,dependency = st.tabs(["Shap summary plot", "Shap bar plot","Shap waterfall plot", "Shap dependency plot"])
    with summary :
        st.write("1. SUMMARY PLOT (Feature Importance)")
        st.write("------------------------------------------------------------")
        st.write("Shows: ")
        st.write("• Which features are most important globally")
        st.write("• Distribution of SHAP values for each feature")
        st.write("• Color: feature value (red=high, blue=low\n")
        st.write("Shap summary plot")
        # Afficher les résultats
        filename = current_dir / ("shap_summary_plot"  + acp_suffix + "-" + model_type + shap_app_image_extension)
        st.image (filename.as_posix())
    with bar :
        st.write("2. BAR  PLOT (Mean Feature Importance)")
        st.write("------------------------------------------------------------")
        st.write("Shows average impact magnitude of each feature")
        # Afficher les résultats
        filename = current_dir / ("shap_bar_plot"  + acp_suffix + "-" + model_type + shap_app_image_extension)
        st.image (filename.as_posix())
    with waterfall:
        st.write("3. WATERFALL PLOT (Single Prediction Breakdown)")
        st.write("------------------------------------------------------------")
        st.write("Shows how features contribute to ONE specific prediction: ")
        st.write("Reading from bottom to top: ")
        st.write("• Starts at base value (average prediction)")
        st.write("• Each bar shows a feature's contribution")
        st.write("• Ends at the final prediction")

        # Afficher les résultats
        filename = current_dir / ("shap_waterfall_plot"  + acp_suffix + "-" + model_type + shap_app_image_extension)
        st.image (filename.as_posix())
    with dependency:
        st.write("4. DEPENDENCE PLOTS (Individual Feature Effects)")
        st.write("------------------------------------------------------------")
        st.write("Shows how a feature's value affects predictions")
        st.write("Color shows interaction with another feature")
        st.write(title)
        # Afficher les résultats
        filename = current_dir / ("shap_dependency_plot"   + acp_suffix + "-" + model_type + shap_app_image_extension)
        st.image (filename.as_posix())

#  *****************************************************************************
#  flat_plot_decision_tree
#  *****************************************************************************

def flat_plot_decision_tree (current_dir,model_type,acp_suffix) :

    feature_importance, decision_tree = st.tabs(["Feature importances", "Decision Tree"])
    with feature_importance :
        filename = current_dir / ("decision_tree_importance" + acp_suffix + "-" + model_type + shap_app_image_extension)
        st.image (filename.as_posix())
    with decision_tree :
        filename = current_dir / ("decision_tree_tree" + acp_suffix + "-" + model_type + shap_app_image_extension)
        st.image (filename.as_posix())

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

if __name__ == '__main__':
    
    current_dir = Path(__file__).parent

    current_dir = Path(immo_vis_dir)

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

    #  *****************************************************************************
    #  Page : Visualisation des data
    #  *****************************************************************************
    if page == pages[1] :
        title = "Visualization et traitement sur les données"
        st.write(title)

        house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
        if house_flat == HOUSE_NAME :
            st.write(HOUSE_NAME)
        elif house_flat == FLAT_NAME :
            st.write(FLAT_NAME)

    #  *****************************************************************************
    #  Page : ACP
    #  *****************************************************************************
    if page == pages[2] : 
        st.write("### Analyse en Composantes Principales")

    #  *****************************************************************************
    #  Page : Modelisation
    #  *****************************************************************************
    if page == pages[3] :
        title = "Modélisation du prix de vente au m2 des appartements et des maisons"
        st.write(title)

   
        index = st.session_state["type_de_bien_index"]
        house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=index)
        st.session_state["type_de_bien_index"] = get_type_de_bien_selection_box_index(house_flat)
        model_type = st.selectbox('Type de régression', MODEL_NAMES,index=0)
        with_acp = st.selectbox('ACP option',ACP_OPTION,index=0)


        if house_flat == HOUSE_NAME :
            st.write(HOUSE_NAME)
        elif house_flat == FLAT_NAME :
    #  *****************************************************************************
    #  flat  Modelisation
    #  *****************************************************************************
            if with_acp == SANS_ACP :
                acp_suffix =""
            else :
                acp_suffix = "-ACP"
            if model_type == LINEAR :
                df = load_appartement_file(current_dir,"linear-regressors" + acp_suffix)
                st.write(df)
            if model_type == NON_LINEAR :
                df = load_appartement_file(current_dir,"non-linear-regressors" + acp_suffix)
                st.write(df)
            if model_type == XGB :
                flat_plot_xgb(current_dir,model_type,acp_suffix)
            elif model_type == DECISION_TREE :
                flat_plot_decision_tree(current_dir,model_type,acp_suffix)
 

    #  *****************************************************************************
    #  Page : Time prediction
    #  *****************************************************************************
    if page == pages[4] : 
        title = "Prediction en temps du prix au m2 des appartements sur la période " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
        st.write(title)

        tab1, tab2 = st.tabs(["Visualisation", "Prediction"])
        with tab1 :
            title = "Visualization des data  sur la période " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
            st.write(title)

            choices = ['prix de vente au m2', 'variables économiques']
            option = st.selectbox('Choix de la visualisation', choices,index=0)

        with tab2 :
            title = "Prediction du prix au m2 des appartements"
            st.write(title)

    #  *****************************************************************************
    #  Page : prediction du prix
    #  *****************************************************************************
    if page == pages[5] : 
        title = "Prediction du prix ou d'un appartemment"
        st.write(title)

    #  *****************************************************************************
    #  Page : Conclusion
    #  *****************************************************************************
    if page == pages[len(pages)-1] : 
        st.write("### Conclusion")
