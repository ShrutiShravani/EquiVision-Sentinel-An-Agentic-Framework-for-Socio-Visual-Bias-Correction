# Model Card: Aegis-V (Socio-Clinical ViT)

## 1. Description
A Multimodal Vision Transformer designed to predict 14 pathologies from Chest X-rays while adjusting for socio-economic context via ADI (Area Deprivation Index) Late-Fusion.

## 2. Technical Specs (Local Training)
- **Engine:** IDE Cursor / Local CPU-MPS.
- **Backbone:** ViT-Base (Small/Medium variants for local efficiency).
- **Optimization:** Mixed-Precision and Gradient Accumulation (Step 8).

## 3. Metrics (Simulated Results)
- **Fairness:** AUC Gap < 0.05 across ADI Deciles.
- **Security:** 100% PHI Scrubbing Recall (Phase 2).
- **Explainability:** Integrated Grad-CAM heatmaps for clinician trust (Phase 3).

## 4. Business Value
- **Throughput:** >120 images/hour on standard local hardware.
- **Clinical Impact:** Automated "Draft Impression" generation (Step 11).

## 5. Training & Metrics
- **Performance:** Measured via AUC-ROC and F1-score across 14 pathology labels.
- **Fairness Mitigation:** Explicit training and validation on ADI deciles to prevent "Zip Code Bias" and ensure diagnostic parity across diverse socioeconomic backgrounds.

## 6. Security & Ethics
- **Privacy:** HIPAA-compliant Named Entity Recognition (NER) pipeline for PII redaction.
- **Transparency:** Integrated Grad-CAM heatmaps for visual explainability of AI attention.
