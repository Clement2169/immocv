from pathlib import Path
import joblib
import pandas as pd

parquet_extension = ".parquet"
monthly_data_file= "monthly_data" + parquet_extension
monthly_inflation_data_file = "monthly_inflation_data" + parquet_extension

PROJECT_TITLE = "Projet immobilier - Modélisation des prix des maisons et appartements - France Métropolitaine"
PAGES = ["Présentation","Visualisation","Enrichissement ACP","Modélisation","Prédiction dans le temps","Prédiction du prix","Conclusion"]
FLAT_NAME = "appartement"
HOUSE_NAME = "maison"
HOUSE_FLAT_CHOICE = [HOUSE_NAME,FLAT_NAME]

current_dir = Path(__file__).parent
data_dir = current_dir / "data"

data_dir_model = data_dir/ "model"
data_dir_visu = data_dir/ "visu"
data_dir_temps = data_dir/ "pred-temps"
data_dir_prix = data_dir/ "pred-prix"
data_dir_acp = data_dir/ "acp"
data_dir_intro = data_dir/ "intro"


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
#  load_model_file (joblib)
#  *****************************************************************************
def load_model_file (input_path,filename) :
    start_path = Path(input_path)
    final_path = start_path / filename
    return joblib.load(final_path)
