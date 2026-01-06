# Create the mapping for the final report
id_to_name = {v: k for k, v in disease_omop_map.items()}

def evaluate_research_impact(model, test_loader):
    # 1. Get all predictions
    # 2. Join with test_df metadata (ADI, Rural Penalty)
    # 3. Calculate AUC for 'High Stress' vs 'Low Stress' patients
    # 4. Print Table: Disease | Global AUC | High ADI AUC | Low ADI AUC