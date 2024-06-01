import yaml
import pandas as pd
from sklearn import preprocessing
import sys

def data_cleaning(params):

    Features = pd.read_csv(params["data_load"]["features_data"])
    Targets = pd.read_csv(params["data_load"]["targets_data"])

    values_to_drop = []
    percent_missing = Features.isnull().sum() * 100 / len(Features)
    for v in percent_missing.keys():

        if percent_missing[v] > 20:

            values_to_drop.append(v)

    targets_reduced_data = Targets.copy()
    features_reduced_data = Features.drop(columns =values_to_drop)
    
    targets_reduced_data = Targets.copy()
    numerical_features = features_reduced_data.select_dtypes(include='number')
    categorical_features = features_reduced_data.select_dtypes(exclude='number')

    numerical_features["Age"] = numerical_features["Age"].fillna(numerical_features["Age"].median())
    numerical_features["BMI"] = numerical_features["BMI"].fillna(numerical_features["BMI"].mean())
    numerical_features["Height"] = numerical_features["Height"].fillna(numerical_features["Height"].mean())
    numerical_features["Weight"] = numerical_features["Weight"].fillna(numerical_features["Weight"].mean())
    numerical_features["Length_of_Stay"] = numerical_features["Length_of_Stay"].fillna(numerical_features["Length_of_Stay"].mean())
    numerical_features["Alvarado_Score"] = numerical_features["Alvarado_Score"].fillna(numerical_features["Alvarado_Score"].mean())
    numerical_features["Paedriatic_Appendicitis_Score"] = numerical_features["Paedriatic_Appendicitis_Score"].fillna(numerical_features["Paedriatic_Appendicitis_Score"].mean())
    numerical_features["Body_Temperature"] = numerical_features["Body_Temperature"].fillna(numerical_features["Body_Temperature"].mean())
    numerical_features["WBC_Count"] = numerical_features["WBC_Count"].fillna(numerical_features["WBC_Count"].median())
    numerical_features["Neutrophil_Percentage"] = numerical_features["Neutrophil_Percentage"].fillna(numerical_features["Neutrophil_Percentage"].median())
    numerical_features["RBC_Count"] = numerical_features["RBC_Count"].fillna(numerical_features["RBC_Count"].median())
    numerical_features["Hemoglobin"] = numerical_features["Hemoglobin"].fillna(numerical_features["Hemoglobin"].median())
    numerical_features["RDW"] = numerical_features["RDW"].fillna(numerical_features["RDW"].median())
    numerical_features["Thrombocyte_Count"] = numerical_features["Thrombocyte_Count"].fillna(numerical_features["Thrombocyte_Count"].median())
    numerical_features["CRP"] = numerical_features["CRP"].fillna(numerical_features["CRP"].median())

    categorical_features["Sex"] = categorical_features["Sex"].fillna(categorical_features["Sex"].value_counts().index[0])
    categorical_features["Appendix_on_US"] = categorical_features["Appendix_on_US"].fillna(categorical_features["Appendix_on_US"].value_counts().index[0])
    categorical_features["Migratory_Pain"] = categorical_features["Migratory_Pain"].fillna(categorical_features["Migratory_Pain"].value_counts().index[0])
    categorical_features["Lower_Right_Abd_Pain"] = categorical_features["Lower_Right_Abd_Pain"].fillna(categorical_features["Lower_Right_Abd_Pain"].value_counts().index[0])
    categorical_features["Contralateral_Rebound_Tenderness"] = categorical_features["Contralateral_Rebound_Tenderness"].fillna(categorical_features["Contralateral_Rebound_Tenderness"].value_counts().index[0])
    categorical_features["Coughing_Pain"] = categorical_features["Coughing_Pain"].fillna(categorical_features["Coughing_Pain"].value_counts().index[0])
    categorical_features["Nausea"] = categorical_features["Nausea"].fillna(categorical_features["Nausea"].value_counts().index[0])
    categorical_features["Loss_of_Appetite"] = categorical_features["Loss_of_Appetite"].fillna(categorical_features["Loss_of_Appetite"].value_counts().index[0])
    categorical_features["Neutrophilia"] = categorical_features["Neutrophilia"].fillna(categorical_features["Neutrophilia"].value_counts().index[0])
    categorical_features["Dysuria"] = categorical_features["Dysuria"].fillna(categorical_features["Dysuria"].value_counts().index[0])
    categorical_features["Stool"] = categorical_features["Stool"].fillna(categorical_features["Stool"].value_counts().index[0])
    categorical_features["Peritonitis"] = categorical_features["Peritonitis"].fillna(categorical_features["Peritonitis"].value_counts().index[0])
    categorical_features["Psoas_Sign"] = categorical_features["Psoas_Sign"].fillna(categorical_features["Psoas_Sign"].value_counts().index[0])
    categorical_features["US_Performed"] = categorical_features["US_Performed"].fillna(categorical_features["US_Performed"].value_counts().index[0])
    categorical_features["Free_Fluids"] = categorical_features["Free_Fluids"].fillna(categorical_features["Free_Fluids"].value_counts().index[0])

    targets_reduced_data["Management"] = targets_reduced_data["Management"].fillna(targets_reduced_data["Management"].value_counts().index[0])
    targets_reduced_data["Severity"] = targets_reduced_data["Severity"].fillna(targets_reduced_data["Severity"].value_counts().index[0])
    targets_reduced_data["Diagnosis"] = targets_reduced_data["Diagnosis"].fillna(targets_reduced_data["Diagnosis"].value_counts().index[0])

    numerical_features.drop(labels=[221, 203, 564, 541, 557, 586], axis="index", inplace=True)
    categorical_features.drop(labels=[221, 203, 564, 541, 557, 586], axis="index", inplace=True)
    targets_reduced_data.drop(labels=[221, 203, 564, 541, 557, 586], axis="index", inplace=True)

    numerical_features.reset_index(drop=True)
    categorical_features.reset_index(drop=True)
    targets_reduced_data.reset_index(drop=True)

    numerical_features.to_csv(params["pre_process"]["processed_num"], index=False)
    categorical_features.to_csv(params["pre_process"]["processed_cat"], index=False)
    targets_reduced_data.to_csv(params["pre_process"]["processed_targets"], index=False)

def prepare_pipeline(params):

    numerical_features = pd.read_csv(params["pre_process"]["processed_num"])
    categorical_features = pd.read_csv(params["pre_process"]["processed_cat"])
    targets_reduced_data = pd.read_csv(params["pre_process"]["processed_targets"])

    cat_to_num = pd.get_dummies(categorical_features)
    cat_to_num.reset_index(drop=True)

    X = pd.concat([numerical_features, cat_to_num], axis= 1)

    new_targets = targets_reduced_data.drop(labels= ['Management', 'Severity'], axis= 1)

    le = preprocessing.LabelEncoder()
    le.fit(new_targets.values.ravel())

    target_array = le.transform(new_targets.values.ravel())
    y = pd.DataFrame(target_array)
    y.rename(columns = {0:'Diagnosis'}, inplace = True)

    X_copy = X.copy()

    # apply normalization techniques
    Sc = preprocessing.StandardScaler().fit_transform(X_copy)
    X_scaled = pd.DataFrame(Sc)

    X_scaled.to_csv(params["pre_process"]["pipeline_features"], index=False)
    y.to_csv(params["pre_process"]["pipeline_targets"], index=False)

if __name__ == '__main__':

    data_path = sys.argv[1]

    with open(data_path, "r") as f:
        params = yaml.safe_load(f)

    data_cleaning(params)
    prepare_pipeline(params)