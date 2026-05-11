# 🔬 Research & Theoretical Foundation
## Project: EquiVision Sentinel (SVA)

### 1. Problem Statement: The "Hardware-Equity Gap"
Traditional Medical AI assumes high-fidelity input. Research shows that models trained on high-resource clinical data (low ADI) suffer **Performance Collapse** when deployed in low-resource settings (high ADI) due to hardware-induced noise and artifacts. This project moves beyond "prediction" to **Active Bias Neutralization.**

---

### 2. Theoretical Pillars

#### A. Vision Architecture: Transformers over CNNs
*Based on Dosovitskiy et al. (2020).* We utilize **Vision Transformers (ViT)** as our diagnostic backbone. Unlike traditional CNNs, ViT’s **Global Self-Attention** allows the model to correlate distant radiographic features, which is essential for identifying patterns across varying levels of image degradation (noise) found in different socio-economic tiers.

#### B. Domain Adaptation via Diffusion
*Based on Rombach et al. (2022).* We leverage **Latent Diffusion Models (LDM)** to perform Image-to-Image translation, mapping high-fidelity "Urban" images into "Rural/Low-Resource" styles without losing clinical significance.

#### C. Structural Integrity (Anatomical Locking)
*Based on Zhang & Agrawala (2023) - ControlNet.* We implement spatial conditioning to ensure that while the "Socio-Visual Style" changes (noise/contrast), the **Anatomical Truth** (bones/organs) remains 100% invariant.

#### D. Agentic Evaluation (Auditor-Critic)
We apply research in **Agentic Workflows** via LangGraph to automate the quality assurance of synthetic medical data, ensuring a closed-loop system that self-corrects hallucinations.

---

### 3. Algorithmic Innovation: Adversarial Fairness Synthesis (AFS)
Our primary innovation is the **Closed-Loop Agentic Factory**. Instead of static data augmentation, we use an autonomous agent to:

*   **Identify:** Detect performance gaps in specific ADI deciles (Phase 1).
*   **Synthesize:** Generate "Anatomically-Locked" counterfactuals using ADI-driven prompts (Phase 2).
*   **Verify:** A Critic Agent (Med-CLIP) validates the medical accuracy of the synthetic data before retraining.

---

### 4. Methodology: The Socio-Visual Aligner (SVA) Loop

*   **Socio-Technical Mapping:** We translated **ADI (Area Deprivation Index)** ranks into a taxonomy of **Radiographic Artifacts** (e.g., ADI-10 = Quantum Mottle + Grid Cutoff).
*   **Validation Metrics:**
    *   **FID (Fréchet Inception Distance):** To ensure synthetic image quality.
    *   **Anatomical SSIM:** To guarantee zero structural hallucination.
    *   **Equity Gain:** Measuring the reduction in loss discrepancy between ADI deciles after retraining.