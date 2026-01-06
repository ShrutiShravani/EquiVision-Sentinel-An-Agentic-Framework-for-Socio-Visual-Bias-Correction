Informatics & Discovery-

1. Demographic Integrity 
Total Records: [Enter Total Row Count]

Age Outliers: Found 16 records where Patient Age > 100.

Action: Records will be flagged for removal in Step 1 (ETL) to prevent model skew.

2. Disease Cardinality Audit
Objective: Determine if the task is single-label or multi-label classification and quantify complexity.

Findings:

Total "No Finding" (Class 0): 60,361 images.

Single Disease (Class 1): 51,759 images.

Complex Multi-Label (>1 Disease): [Sum your counts for 2 through 9].

Maximum Complexity: 9 simultaneous diseases found in a single patient record.

Engineering Implications:

Loss Function: Must use Binary Cross-Entropy (BCE) with Logits Loss rather than Categorical Cross-Entropy, as classes are not mutually exclusive.

Activation: The final layer of the ViT must use a Sigmoid activation (0 to 1 per class) instead of Softmax.

Data Imbalance: "No Finding" represents ~54% of the dataset. We will require Weighted Random Sampling or Focal Loss to ensure the model learns rare, complex multi-label cases.

3. Clinical Label Audit (Head vs. Tail)
Objective: Identify class imbalance to prevent the model from ignoring rare but critical pathologies.

Metric: Prevalence = (Count of Disease / Total Patient Records) * 100.

Head Classes (>1%): [Insert your head_classes list]

Tail Classes (<1%): [Insert your tail_classes list]

Strategy Note: The "Tail" diseases (like Hernia/Pneumonia) are high-risk for under-diagnosis. Our Step 8 (Performance Engineering) will implement Loss Weighting to give these rare classes more "importance" during training.

Class Imbalance Mitigation StrategyObjective: Prevent model collapse toward the "No Finding" majority class.Problem: "No Finding" (60k) vs "Hernia" (approx. 0.2%).

Solution A (Data Level): Implement a WeightedRandomSampler to over-sample the Tail Classes identified in Audit #3 during training batches.

Solution B (Algorithmic Level): Utilize Focal Loss with an alpha parameter to balance class importance and a gamma parameter to focus on hard-to-classify "Tail" samples.

Equity Justification: Rare diseases must be treated with high priority to ensure that patients with less common conditions receive the same diagnostic quality as those with common ones.

4. Integrated Data Splitting & Leakage ProtocolProblem: A "Power Law" distribution where 56.8% of patients have only 1 image, while a small group of "Chronic" patients contributes up to 184 images each. Simple random splitting will cause Major Data Leakage.

Strict Grouping (The Anchor): We will implement GroupKFold using Patient ID as the grouping variable.

Rule: All images belonging to a single person MUST stay within the same fold (Train, Val, or Test). This prevents the model from "memorizing" specific ribs or heart shapes.

Frequency Capping & Weighting (The Balance): To prevent a patient with 184 images from dominating the gradient updates:

Action: During the ETL (Step 1), we will "Soft-Cap" the training contributions by using Weighted Batch Sampling. We prioritize diversity of Patient ID over total image count in every training step.

Socio-Clinical Stratification (The Equity Layer):We will cross-reference the Recidivism Ratio during the split.
Metric: Recidivism Ratio = Avg Images (High ADI)/{Avg Images (Low ADI)}Goal: Ensure the Train/Test sets have a mirrored ratio so the model learns the "Chronic Disease Patterns" seen in high-deprivation areas equally.

5.View Position Confounder Analysis
Finding: Significant "Clinical Shortcut" detected in Edema (11x skew) and Consolidation (3x skew) toward AP views.

No Finding Bias: PA views are 11% more likely to be labeled "Healthy," reflecting an outpatient "well-patient" bias.

Requirement for Phase 3: Implement Late-Fusion Metadata Injection (Step 6.to include View Position as a feature, preventing the ViT from using camera angle as a proxy for disease severity.)


6. Synthesis & Linkage Constraints
Objective: Overcome the lack of direct identifiers (Zip Codes) in Clinical Data.

Discovery: NIH Data_Entry_2017 contains no geographic identifiers. ADI data contains 13.7% suppressed/categorical codes (GQ, U, PH).

Impact: Direct SQL-style joining is impossible.

Strategic implementation (Step 3): * Data Imputation: Categorical codes (GQ, U,PH) will be treated as a separate "Socio-economic Category" or imputed using the state-wide median to preserve dataset size.

Data Linkage & Ecological Inference Logic
Objective: To overcome the lack of geographic identifiers (Zip Codes) in the NIH Clinical Dataset by applying a Demographic-to-Socioeconomic Crosswalk using the Wisconsin-specific health landscape.

a. The Method: Ecological Imputation We utilize the methodology established by Kind et al. (2014), which identifies the "Top 15%" most disadvantaged neighborhoods as a primary clinical predictor for pulmonary rehospitalization. Since individual addresses are de-identified in the NIH dataset, we map patients to these cohorts using Ecological Inference—assigning group-level environmental probabilities to individuals based on their demographic signatures.

b. Key Research Pillars & Findings (Source: County Health Rankings & Roadmaps 2024)

Income Inequality & Cardiovascular Stress (Finding #1): Income inequality acts as a "social stressor," independent of individual income. We integrate this by recognizing that high-inequality communities in Wisconsin experience a loss of social connectedness, directly correlating with increased cardiovascular and respiratory mortality.

Pediatric Poverty & Chronic Asthma (Finding #2): Children in low-income households face an increased risk of chronic conditions like asthma and diabetes due to unsafe environments. We apply a Pediatric Risk Multiplier to patients under 18 to account for this susceptibility.

The Rural Treatment Gap (Finding #3): Rural areas in Wisconsin show higher unintentional injury death rates and significant difficulties in obtaining rapid emergency treatment. Since the elderly demographic (65+) is geographically concentrated in these rural "High-ADI" blocks, we apply a Rural Access Penalty to this cohort.

Gender Pay Gap & Mood Disorders (Finding #3.1): Systemic pay inequity is linked to higher rates of depression and anxiety in women. We include a Psychosocial Stress Factor for female patient profiles to account for the cumulative health impact of these economic disparities.

Structural Racism & Residential Segregation (Finding #5): Residential segregation is a fundamental cause of health disparities, linked to poor-quality housing and environmental toxin exposure. The ADI (Area Deprivation Index) serves as our proxy for this structural risk.

Disconnected Youth Risk (Finding #6): Approximately 1 in 9 teenagers are disconnected from school or work, facing higher rates of poverty and early mortality. We map the 15–24 age bracket to higher deprivation scores to reflect this vulnerability.

Social & Emotional Support Deficit (Finding #7): 1 in 4 US adults lack social support, a factor linked to heart disease and premature death. This deficit is highest among transgender, gender nonconforming, and low-income individuals, justifying the use of Gender and Income proxies in our linkage logic.

c. Goal of Synthetic Linkage To create a Synthetic Control Cohort that allows the Multimodal Vision Transformer (ViT) to distinguish between visual lung morphology and the probabilistic impact of environmental and structural deprivation.


7. Weakly Supervised Strategy (Handling the BBox Gap)(Attention-MIL)

Architecture choice: Vision Transformer (ViT).

Mechanism: We treat the X-ray as a Bag of Patches.

Learning Logic: By using Global Labels, the ViT's self-attention mechanism acts as an MIL-Aggregator, automatically localizing diseases without requiring manual bounding boxes.

Audit Logic: We will extract Attention Heatmaps from the Transformer's last layer to verify that the "Instances" the model thinks are important actually contain the pathology.

8.Visual & Hardware Infrastructure Audit
Objective: Validate file-system integrity and establish a latency baseline for training.

Integrity Audit: * Success Rate: [Your Linkage Success Number] / 112,120.

Status: [Success/Warning]. We have synchronized the Data_Entry_2017.csv to match physical disk availability.

Hardware Profile:

Resolution: 1024x1024 (Standard NIH).

Latency: [Your latency] ms (Disk-to-RAM).

Engineering Impact (Step 2: ETL): * Action: Because 1024x1024 is too large for a Vision Transformer (ViT) to process efficiently, we must implement On-the-fly Resizing to 224x224 in the DataLoader.

Optimization: With a latency of X ms, we will require num_workers > 4 in PyTorch to prevent the GPU from waiting for the disk (I/O Bottleneck).