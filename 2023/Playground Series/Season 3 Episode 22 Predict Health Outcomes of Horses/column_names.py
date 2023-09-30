target = "outcome"
id_col = "id"

quanti_var = [
    # 'hospital_number',
    "rectal_temp",
    "pulse",
    "respiratory_rate",
    "nasogastric_reflux_ph",
    "packed_cell_volume",
    "total_protein",
    "abdomo_protein",
    "lesion_1",
    # 'lesion_2',
    # 'lesion_3',
]

quali_var = [
    "surgery",
    "age",
    "temp_of_extremities",
    "peripheral_pulse",
    "mucous_membrane",
    "capillary_refill_time",
    "pain",
    "peristalsis",
    "abdominal_distention",
    "nasogastric_tube",
    "nasogastric_reflux",
    "rectal_exam_feces",
    "abdomen",
    "abdomo_appearance",
    "surgical_lesion",
    "cp_data",
]

quali_var_binary = [
    "surgery",
    "age",
    "capillary_refill_time",
    "nasogastric_reflux",
    "surgical_lesion",
    "cp_data",
]

quali_var_for_ohe = [
    "temp_of_extremities",
    "peripheral_pulse",
    "mucous_membrane",
    "pain",
    "peristalsis",
    "abdominal_distention",
    "nasogastric_tube",
    "rectal_exam_feces",
    "abdomen",
    "abdomo_appearance",
]
