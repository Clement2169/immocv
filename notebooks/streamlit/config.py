from pathlib import Path
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


current_dir = Path(__file__).parent
data_dir = current_dir / "data"

data_dir_model = data_dir/ "model"
data_dir_visu = data_dir/ "visu"
data_dir_temps = data_dir/ "pred-temps"
data_dir_prix = data_dir/ "pred-prix"
data_dir_acp = data_dir/ "acp"
data_dir_intro = data_dir/ "intro"