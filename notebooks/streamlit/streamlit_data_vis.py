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

from config import *

def corr_plots(df_num_corr,corr_type,labels,corr_threshold=0.1):
    fig, ax = plt.subplots() 
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




def DataViz() :
  
    house_labels = ['m', 'mn', 'Maison/Villa neuve' ]
    flat_labels = ['a', 'an']
    title = "Visualization et traitement sur les données"
    st.write(title)
    
    intial_tab, nan_tab,cor_tab = st.tabs(['Dataset initial', r"NAN dans le dataset","Corrélation des variables avec le prix/m²"])
    with intial_tab :
        st.write('Description du dataset initial : ')  
        filepath = os.path.join(data_dir_visu, 'prop_maison_app_plot.png')
        st.image(filepath, )
        st.table(pd.read_pickle(os.path.join(data_dir_visu,'DataSummaryTable.pkl')))

    with nan_tab :  
        st.subheader(r"Figure montrant le % des NAN dans le dataset initial :")
        filepath = os.path.join(data_dir_visu, 'percentage_NA_alldata_plot.png')
        st.image(filepath, )
    
    with cor_tab :  
        st.subheader(r"Corrélation des variables avec le prix/m² :")
        col1, col2, col3 = st.columns(3)
        with col1:
            house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
        with col2:
            corr_type = st.selectbox('Type de corrélation', ['pearson','spearman'],index=0)
        with col3:
            corr_threshold = float(st.text_input("Seuil de corrélation (%)? ",value=10))
    
        corr_file_path=os.path.join(data_dir_visu, f'corr_df_{house_flat}_{corr_type}.pkl')
        df_num_corr=pd.read_pickle(corr_file_path)
        
        if st.button("Afficher"):
            if house_flat == HOUSE_NAME :
                corr_plots(df_num_corr,corr_type,house_labels,corr_threshold=corr_threshold/100)
            elif house_flat == FLAT_NAME :
                corr_plots(df_num_corr,corr_type,flat_labels,corr_threshold=corr_threshold/100)






if __name__=='__main__':
    
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
        DataViz()