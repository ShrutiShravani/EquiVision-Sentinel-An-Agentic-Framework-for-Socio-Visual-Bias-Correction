# Aegis-V Project Metrics Framework (CPU/Edge Optimized)

## 1. Business Metrics (Clinical & Social Impact)
| Metric | Definition | At-Home Measurement Method |
| :--- | :--- | :--- |
| **Diagnostic Parity** | The AUC variance between ADI Decile 1 (Wealthy) and Decile 10 (Poor). | Subset the NIH test set by ADI; compare ROC curves using `sklearn`. |
| **Simulated Triage TAT** | Time saved by AI-drafting vs. manual reporting baseline. | Measure end-to-end latency and extrapolate 24-hr time savings. |
| **Socioeconomic Bias Gap** | Accuracy delta across ADI deciles. | Statistical parity test on model predictions across different zip-code vectors. |

## 2. Technical Metrics (System Efficiency - CPU/Local)
| Metric | Definition | At-Home Measurement Method |
| :--- | :--- | :--- |
| **P99 Inference Latency** | Time taken for the slowest 1% of inferences. | Use `time.perf_counter()` over 50 test runs in Cursor. |
| **Peak System RAM Usage** | Maximum RAM consumed during high-res ViT inference. | Track `psutil.virtual_memory()` peak during the forward pass. |
| **Quantization Fidelity** | Accuracy retained after converting FP32 to INT8. | Compare AUC of the original model vs. the quantized "Edge" version. |
| **Task Queue Reliability** | Success rate of Celery tasks under local resource constraints. | Monitor Redis for "OOM (Out of Memory)" kills or task timeouts. |

## 3. Component Metrics (Pipeline Accuracy)
| Metric | Definition | At-Home Measurement Method |
| :--- | :--- | :--- |
| **NER Scrubbing Recall** | % of PII successfully hidden from clinical metadata. | Run the Scrubber on 100 synthetic NIH header samples; target 100%. |
| **Grad-CAM IoU** | Alignment between AI attention and NIH Bounding Boxes. | Use `cv2` to calculate IoU between the heatmap mask and expert BBoxes. |
| **OMOP Mapping Fidelity** | % of raw fields correctly transformed to CDM v5.4. | Run a JSON-schema validator against the mapped OMOP output. |
| **ADI Injection Influence** | The weight/impact of the ADI vector on the final layer. | Perform "Feature Importance" (SHAP/LIME) to prove ADI affects the prediction. |

fairness bias
ablation study