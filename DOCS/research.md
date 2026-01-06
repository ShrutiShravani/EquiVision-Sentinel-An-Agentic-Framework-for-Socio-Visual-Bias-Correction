## 📚 Research & Literature Foundation
This project is built upon state-of-the-art research in clinical informatics and health equity.

### Key Literature
1. **Clinical Standards:** Developed in alignment with the *OHDSI/OMOP Common Data Model* to ensure international research interoperability.
2. **Socioeconomic Modeling:** Implementation of the *Area Deprivation Index (ADI)* based on the methodology of Dr. Amy Kind (UW-Madison), linking geography to clinical outcomes.
3. **Multimodal Fusion:** Inspired by "Multimodal Healthcare Modeling" (MDPI 2024), utilizing late-fusion techniques for pixel and tabular data.

### Research Methodology
- **Data Harmonization:** Applied Deterministic Record Linkage to bridge de-identified NIH datasets with spatial UW Atlas data.
- **Ethics & Bias:** Documented a "Selection Bias Mitigation" strategy in `docs/research/bias_audit.md`.


# Research Methodology & Theoretical Foundation
**Project:** WHEP-IS (Wisconsin Health Equity Predictor)

## 1. Problem Statement: The "Social Blind Spot"
Traditional Computer Vision (CV) in healthcare operates on a "Unimodal" basis, ignoring the Social Determinants of Health (SDOH). This research addresses the diagnostic gap by integrating neighborhood-level socioeconomic priors into the model's attention mechanism.hy this makes you a "Senior" Researcher: When you talk to an interviewer, you say:

"I standardized my metadata using OMOP CDM. This ensures that if my model is deployed in a real hospital system (like Epic or Cerner), the data structure is already compatible with their clinical databases."

## 2. Theoretical Pillars (Literature Review)
* **Socioeconomic Framework:** *Kind et al. (2014)*. We utilize the Area Deprivation Index (ADI) to quantify neighborhood disadvantage.
* **Vision Architecture:** *Dosovitskiy et al. (2020)*. We move beyond CNNs to use Vision Transformers (ViT), allowing for global pixel dependency.
* **Multimodal Fusion:** *Huang et al. (2020)*. We apply "Late-Fusion" with an Attention-Gating mechanism to prevent one data modality from overwhelming another.

## 3. Algorithmic Innovation: Attention-Gated Spatial Fusion
Instead of treating ADI as a simple tabular feature, our innovation uses the ADI score as a **Spatial Prior**. 

> **Hypothesis:** By weighting the transformer's self-attention layers with ADI-derived risk scores, the model will prioritize subtle radiographic features that correlate with long-term exposure to environmental stressors (e.g., pollution or chronic stress in high-ADI zones).

## 4. Evaluation Metrics for Equity
We do not just use Accuracy. To evaluate the "Unknown," we measure:
* **Fairness Discrepancy:** Comparing model performance across ADI Deciles (1 vs 10).
* **Clinical Utility:** Measuring the reduction in "False Negatives" for high-vulnerability populations.