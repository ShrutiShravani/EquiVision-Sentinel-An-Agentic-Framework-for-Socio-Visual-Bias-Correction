import pandas as pd


def bridge_linkage_engine(clinical_df: pd.DataFrame, adi_df: pd.DataFrame):

    # Pre-filter
    high_risk_adi = adi_df[adi_df['ADI_STATERNK'] >= 8].reset_index(drop=True)

    n = len(clinical_df)

    # Pre-sample once
    general_samples = adi_df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)
    high_risk_samples = high_risk_adi.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

    # Risk mask
    age = clinical_df['Patient Age']
    is_high_risk = (age >= 65) | (age <= 24)

    # Assign ADI
    assigned_adi = general_samples.copy()
    assigned_adi.loc[is_high_risk.values] = high_risk_samples.loc[is_high_risk.values]

    # Merge (NO column loss)
    master_df = pd.concat(
        [clinical_df.reset_index(drop=True), assigned_adi],
        axis=1
    )

    # Engineered features
    adi_rank = master_df['ADI_STATERNK']

    master_df['is_top_15_risk'] = (adi_rank >= 8.5).astype(int)
    master_df['rural_penalty'] = ((master_df['Patient Age'] >= 65) & (adi_rank >= 9)).map({True:1.25, False:1.0})
    master_df['pediatric_idx'] = (master_df['Patient Age'] <= 18).map({True:1.2, False:1.0})
    master_df['psych_stress'] = (master_df['Patient Gender'] == 'F').map({True:1.05, False:1.0})
    master_df['synthetic_context'] = True

    return master_df
