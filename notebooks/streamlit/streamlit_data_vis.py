import streamlit as st 
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import seaborn as sns 
import numpy as np
#  stats on Nans, display Nans for all columns
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick


# def NAN_chart(df):
#     plt.rcParams['xtick.labelsize'] = 15
#     nb_rows= df.shape[0]
#     nb_cols= df.shape[1]


#     missing_values = df.isna().sum()
#     missing_values = missing_values.sort_values(ascending=False)
#     full_columns = df.columns[ df.isna().any() == False ]
#     print (f" nb_rows {nb_rows} nb_cols {nb_cols}")
#     # print(f"columns without Nan values {full_columns.size} / {nb_cols}" )
#     # print (f"columns without Nans {full_columns}")
#     # print (missing_values)
#     fig=plt.figure(figsize=(20,10))
#     # ax = plt.subplot(121)
#     # plt.plot (missing_values.index,missing_values.values)
#     # plt.axhline(y=int(nb_rows/2), color='r', linestyle='--', label='50%')
#     # plt.axhline(y=int(nb_rows*0.9), color='b', linestyle='--', label='90%')
#     # plt.legend()

#     # plt.xticks(rotation=80)
#     # plt.title(f"Missing values  nb-rows = {nb_rows}")
#     ax2 = plt.subplot(111)
#     missing_values_percent = (missing_values/nb_rows)*100.0
#     plt.plot (missing_values_percent.index,missing_values_percent.values)
#     plt.axhline(y=50.0, color='r', linestyle='--', label='50%')
#     plt.axhline(y=90.0, color='b', linestyle='--', label='90%')
#     plt.xticks(rotation=80)
#     ax2.yaxis.set_major_formatter(mtick.PercentFormatter(100)) 
#     plt.title(f"Missing values  percentage")
#     plt.legend()
#     plt.show()
#     return fig

def corr_plots(df,corr_type,labels,corr_threshold=0.1):
    df_num_corr=df[df['typedebien'].isin(labels)].select_dtypes(include='number').corr(corr_type)
    fig, ax = plt.subplots(figsize=(5, 7)) 
    param = {
        'vmax': 1, 
        'vmin': -1, 
        'cmap': 'coolwarm', 
        'annot': True,
        #'square': True,           # Cellules carrées
        'fmt': '.2f',            # Format des nombres (3 décimales)
        'cbar_kws': {'label': f'Corrélation {corr_type}'},  # Label de la colorbar
           
    }
    sns.heatmap(data=df_num_corr[np.abs(df_num_corr)['prix_m2_vente']>corr_threshold][['prix_m2_vente']],ax=ax,**param)
    st.pyplot(fig, use_container_width=True)


if __name__=='__main__':

    @st.cache_data
    def import_raw_data():
        return pd.read_parquet(r'raw_data\From_Clement\ventes-metropole.parquet',engine='fastparquet')

    df=import_raw_data()
    house_labels = ['m', 'mn', 'Maison/Villa neuve' ]
    flat_labels = ['a', 'an']
   
    PROJECT_TITLE = "Projet immobilier - Modélisation des prix des maisons et appartements - France Métropolitaine"
    PAGES = ["Présentation","Visualisation","Analyse ACP","Modélisation","Prédiction en temps","Prédiction du prix","Conclusion"]
    FLAT_NAME = "appartement"
    HOUSE_NAME = "maison"
    HOUSE_FLAT_CHOICE = [HOUSE_NAME,FLAT_NAME]
    REGRESSION_MODEL_LGB = "LgbRegressor"
    REGRESSION_MODEL_DECISION_TREE = "DecisionTreeRegressor"
    REGRESSION_MODEL_XGB = "XGBRegressor"

    REGRESSION_MODEL_CHOICE = [REGRESSION_MODEL_LGB,REGRESSION_MODEL_DECISION_TREE,REGRESSION_MODEL_XGB]

    current_dir = Path(__file__).parent

    data_dir = current_dir / "data-st"
    
    #  *****************************************************************************
    #  main
    #  *****************************************************************************


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
        
        st.write('Description du dataset initial : ')  
        filepath = os.path.join(data_dir, 'prop_maison_app_plot.png')
        st.image(filepath, )
        st.table(pd.read_pickle(os.path.join(data_dir,'DataSummaryTable.pkl')))
        
        st.subheader(r"Figure montrant le % des NAN dans le dataset initial :")
        filepath = os.path.join(data_dir, 'percentage_NA_alldata_plot.png')
        st.image(filepath, )
        
        st.subheader(r"Corrélation des variables avec le prix/m² :")
        col1, col2, col3 = st.columns(3)
        with col1:
            house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
        with col2:
            corr_type = st.selectbox('Type de corrélation à considérer', ['pearson','spearman'],index=0)
        with col3:
            corr_threshold = float(st.text_input("Quel seuil à considérer pour l'affichage ? ",value=0.1))

        if st.button("Préparer le graph"):
            if house_flat == HOUSE_NAME :
                corr_plots(df,corr_type,house_labels,corr_threshold)
            elif house_flat == FLAT_NAME :
                corr_plots(df,corr_type,flat_labels,corr_threshold)