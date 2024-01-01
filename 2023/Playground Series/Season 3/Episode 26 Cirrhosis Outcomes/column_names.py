target = "Status"
multi_output_columns = ("Status_C", "Status_CL", "Status_D")
id_col = "id"

quanti_var = [
    "N_Days",
    "Age",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin",
]

quali_var = [
    "Drug",
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema",
    "Stage",
]

quali_var_binary = [
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema",
]

quali_var_for_ohe = [
    "Drug",
    "Stage",
]
