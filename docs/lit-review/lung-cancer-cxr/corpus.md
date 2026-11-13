# Screened Corpus: LLMs for Lung Cancer on Chest X-rays

Built 2026-11-12. Screening was done at abstract level only. No full texts or PDFs were read. All bibliographic facts come from tool output: PubMed, arXiv, OpenAlex, Europe PMC, Consensus and WebSearch. A field is marked "n/s" (not shown) when the tool output did not include it. Where a venue is inferred only from a DOI prefix, it is labelled "(per DOI prefix)".

Scope: the imaging modality is CXR, and the target is lung nodules, masses, cancer, or screening and risk. The method is a VLM/MLLM, a text-LLM, or a VL-foundation model. A capped set of CNN/ViT baselines, datasets and benchmarks is also included.

---

## 1. Search log

"Hits" is the number of records the tool returned. "Kept" counts new papers added to the corpus from that query; duplicates are not counted again.

| # | Source | Query | Hits | Kept (new) |
|---|---|---|---|---|
| 1 | arXiv | "lung nodule" AND ("chest radiograph" OR "chest X-ray") AND ("vision-language" OR "large language model" OR "multimodal") [2022+] | 0 | 0 |
| 2 | PubMed | vision-language model lung nodule chest radiograph | 1 | 1 (Nishio) |
| 3 | PubMed | GPT-4V chest radiograph nodule | 0 | 0 |
| 4 | PubMed | multimodal large language model lung cancer chest radiograph | 2 | 1 (Cusick) |
| 5 | arXiv | nodule "chest X-ray" "vision-language" [2022+] | 25 | 3 (NODE21, NoduLoCC2026, CheXagent) |
| 6 | PubMed | large language model lung nodule follow-up recommendation radiology report | 4 | 2 (Gamble, Zhou T) |
| 7 | PubMed | pulmonary nodule detection chest radiographs multimodal LLM | 1 | 1 (Khovanova) |
| 8 | PubMed | ChatGPT chest X-ray lung cancer detection | 4 | 0 |
| 9 | arXiv | "lung cancer" "chest X-ray" "large language model" [2022+] | 20 | 2 (CXR-LLaVA preprint, Kotevska) |
| 10 | arXiv | get_abstract 2608.14657 | 1 | 1 (LUNG-KGMM) |
| 11 | PubMed | CXR-LLaVA | 5 | 0 (CXR-LLaVA Eur Radiol version merged with L5) |
| 12 | PubMed | MedGemma | 8 | 1 (MedGemma Nat Med) |
| 13 | PubMed | deep learning chest radiograph lung cancer screening NLST PLCO risk | 3 | 1 (Lu CXR-LC) |
| 14 | PubMed | deep learning malignant pulmonary nodules chest radiographs Nam Radiology | 2 | 1 (Nam 2019) |
| 15 | PubMed | artificial intelligence chest radiograph lung cancer screening population Yoo | 0 | 0 |
| 16 | PubMed | commercial AI lung nodule detection chest radiographs randomized | 0 | 0 |
| 17 | PubMed | AI-based computer-aided detection chest radiographs lung cancer screening JAMA Network Open | 0 | 0 |
| 18 | PubMed | actionable lung nodules chest radiographs artificial intelligence randomized controlled trial | 2 | 1 (Nam 2023 RCT) |
| 19 | PubMed | deep learning lung cancer chest radiograph missed retrospective detection | 4 | 2 (Lee 2022, Topff 2024) |
| 20 | PubMed | deep learning algorithm major thoracic diseases chest radiographs multicenter | 1 | 1 (Hwang 2019) |
| 21 | Semantic Scholar | Yoo JAMA Netw Open 2020 normal CXR lung cancer NLST | 0 | 0 |
| 22 | PubMed | Majkowska chest radiograph interpretation deep learning adjudicated reference standard | 1 | 1 (Majkowska) |
| 23 | PubMed | large language model chest radiograph reports lung nodule extraction | 0 | 0 |
| 24 | PubMed | GPT-4 radiology reports incidental pulmonary nodule | 1 | 0 (dup Gamble) |
| 25 | Europe PMC | GPT-4V chest radiograph nodule mass detection | 15 | 0 |
| 26 | Europe PMC | large language model chest radiograph report nodule follow-up | 15 | 0 |
| 27 | Consensus | multimodal large language models detecting lung nodules or lung cancer on chest radiographs | 2 | 0 (dup Khovanova; review listed as excluded) |
| 28 | Consensus | GPT-4V Gemini chest X-ray diagnostic accuracy nodule mass | 3 | 2 (Chen CH 2026, Essa meta-analysis) |
| 29 | arXiv | abs:nodule AND abs:"chest X-ray" AND (zero-shot OR CLIP OR foundation model) | 1 | 0 |
| 30 | arXiv | abs:nodule AND (radiograph OR "chest X-ray") AND (LLM OR multimodal large language) | 0 | 0 |
| 31 | arXiv | lung nodule chest radiograph GPT-4o vision language model evaluation [2023+] | 20 | 3 (XrayGPT, REVEAL-CXR; CLIP-CT paper excluded) |
| 32 | arXiv | lung cancer risk prediction chest radiograph deep learning | 15 | 1 (Dissez) |
| 33 | arXiv | chest X-ray VQA benchmark nodule mass grounding [2023+] | 20 | 2 (ReXVQA, VinDr-CXR-VQA) |
| 34 | arXiv | ReXVQA chest radiology VQA | 8 | 0 (dup) |
| 35 | arXiv | ti:MAIRA-2 OR ti:RadVLM OR ti:MedRAX OR ti:ELIXR | 5 | 3 (MAIRA-2, RadVLM, MedRAX) |
| 36 | arXiv | ti:LLaVA-Rad OR ... OR ti:BiomedCLIP OR ti:CheXzero | 6 | 1 (BiomedCLIP) |
| 37 | arXiv | "LLaVA-Rad" radiology report generation small multimodal model | 5 | 0 |
| 38 | PubMed | expert-level detection of pathologies ... self-supervised (CheXzero) | 1 | 0 (excluded: generic) |
| 39 | arXiv | lung nodule report generation chest X-ray LLM malignancy [2023-06+] | 20 | 0 |
| 40 | Semantic Scholar | vision language model lung nodule detection chest radiograph zero-shot [2023-] | 0 | 0 |
| 41 | OpenAlex | GPT-4 vision chest radiograph lung nodule | 15 | 1 (Tang 2023) |
| 42 | OpenAlex | large language model chest X-ray lung cancer | 15 | 1 (Megat Ramli SR) |
| 43 | PubMed | clinically accessible radiology foundation model LLaVA-Rad | 1 | 1 (LLaVA-Rad) |
| 44 | PubMed | VinDr-CXR open dataset chest X-rays radiologist annotations | 1 | 1 (VinDr-CXR) |
| 45 | PubMed | JSRT digital image database chest radiographs with and without lung nodule | 2 | 0 (original JSRT paper not retrieved) |
| 46 | arXiv | ti:ChestX-ray8 OR ti:CheXpert ... OR ti:MIMIC-CXR OR ti:ChestX-Det | 5 | 3 (ChestX-ray8, CheXpert, MIMIC-CXR-JPG) |
| 47 | Europe PMC | LLM radiology reports pulmonary nodule identification follow-up chest radiograph CT | 15 | 0 |
| 48 | PubMed | large language model Lung-RADS classification radiology reports | 2 | 1 (Gulati) |
| 49 | arXiv | ChestX-Det chest X-ray detection dataset instance-level annotations | 5 | 1 (ChestX-Det10) |
| 50 | PubMed | chest radiograph reports LLM incidental findings follow-up recommendations | 0 | 0 |
| 51 | Semantic Scholar | LLM extraction lung nodule chest X-ray reports | 0 | 0 |
| 52 | Consensus | LLM chest radiograph report lung nodule mass follow-up recommendation | 3 | 1 (Wen 2025) |
| 53 | arXiv | lung cancer chest radiograph multimodal foundation model risk prediction reports [2024+] | 15 | 0 (dup LUNG-KGMM) |
| 54 | arXiv | pulmonary nodule chest radiograph agent MLLM tool [2024-06+] | 15 | 0 |
| 55 | WebSearch | Yoo JAMA Network Open AI chest radiograph lung cancer screening NLST nodules | 10 | 0 (verified via PubMed below) |
| 56 | WebSearch | GPT-4V chest radiograph evaluation Radiology 2024 lung nodule mass | 10 | 0 (Zhou Y GPT-4V; excluded) |
| 57 | arXiv | get_abstract 2403.15528 | 1 | 0 |
| 58 | PubMed | Validation of a DL Algorithm for Detection of Malignant Pulmonary Nodules in CXR | 3 | 2 (Yoo 2020, Shimazaki 2022) |
| 59 | PubMed | Treasure Trove Hidden in Plain Sight GPT-4 chest radiograph | 1 | 0 |
| 60 | PubMed | AI-based improvement lung cancer detection CXR multi-reader NLST | 1 | 1 (Yoo 2021) |
| 61 | arXiv | "lung lesion" OR "lung nodule" chest X-ray CLIP zero-shot [2022-06+] | 15 | 1 (Ohlmann-Knafo) |
| 62 | PubMed | Gemini GPT-4o chest radiograph lung mass detection accuracy | 0 | 0 |
| 63 | PubMed | vision LLM chest X-ray tumor nodule diagnostic performance radiologists | 0 | 0 |

Totals: this table lists 63 calls, reconstructed after a context compaction, so it is partial. It covers every query that contributed a kept paper. A count of the raw session transcript shows about 85-90 search and lookup calls in total, by name: PubMed about 37, arXiv search about 23, arXiv get_abstract about 5, Consensus about 6, Europe PMC about 6, Semantic Scholar about 6, OpenAlex about 5, WebSearch about 5. Each count may be 1 too high because tool-schema loads also match. The unlisted calls were mostly known-item lookups (by title or author) and zero-hit variants. Records seen: about 470 for the queries listed above, and roughly 600 or more across all calls, including duplicates. 44 papers were kept.

---

## 2. Kept corpus (44 papers)

Abbreviations: PR = peer-reviewed (journal or proceedings indicated by the tool). PP = preprint, meaning arXiv-only in the tool output. Some arXiv items are known to have published versions (for example CheXpert and ChestX-ray8), but those versions were not verified, so the items stay marked PP.

### 2a. VLM / MLLM (image-reading LLMs)

| ID | Authors | Year | Title | Venue | PR? | DOI / arXiv | Category | Task | Data | Key quantitative result (verbatim from abstract) | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | Khovanova D et al. | 2026 (PubMed lists 2025) | A comparative accuracy study of multimodal LLMs, VLM and agent-based framework for pulmonary nodule detection on chest radiographs | Frontiers in Digital Health | PR | 10.3389/fdgth.2025.1674835 | VLM (+ VL-foundation + agent) | detection | 100 CXR, 50 with nodules, single centre, PNG | "MedRAX framework and BiomedCLIP ... accuracy of 0.711 (95% CI 0.613-0.808)"; "Claude 3.7 Sonnet ... accuracy 0.651 (0.548-0.753)"; Llama 3.2 Vision 90B, Claude 3.5 Sonnet and Gemini 2.0 Pro "0.602 (0.497-0.708)" | Core paper. Head-to-head comparison of nine models (general MLLMs, CXR-LLaVA, XrayGPT, BiomedCLIP, MedRAX agent) on CXR nodules |
| L2 | Nishio M et al. | 2026 | Zero-Shot Performance of Six Vision-Language Models for Lung Nodule Detection on Chest Radiographs | AJR | PR | 10.2214/AJR.26.35421 | VLM | detection | JSRT, 247 CXR (154 nodule, 93 without) | "accuracies of 36.4–44.5%, sensitivities of 0.6–27.3%, and specificities of 71.0–100.0%"; RadVLM "accuracy (44.5%...)... sensitivity was low (11.0%)"; LLaVA-Rad "highest sensitivity (27.3%)"; MedGemma-4b-it "sensitivity of only 5.2%" | Core paper. Shows zero-shot VLMs fail on subtle nodules. Uses a public benchmark (JSRT) |
| L3 | Chen CH et al. | 2026 | Limitations in Chest X-Ray Interpretation by Vision-Capable LLMs, Gemini 1.0, Gemini 1.5 Pro, GPT-4 Turbo, and GPT-4o | Diagnostics | PR | 10.3390/diagnostics16030376 | VLM | detection (multi-diagnosis) | 247 CXR, 13 diagnostic categories | Higher sensitivity for "large, bilateral, multiple lesions ... multiple malignancies"; "Gemini 1.5 Pro achieved the highest overall detection rate"; limitations "in detecting small lesions" | Includes a malignancy category. Supports the finding that small lesions are missed |
| L4 | Cusick A et al. | 2026 | The Effect of Structured Context on CXR Interpretation by a Multimodal LLM: A Pilot Comparative Study | Cureus | PR | 10.7759/cureus.108715 | VLM | detection (diagnosis) | 50 CXR, 10 diagnoses | Overall accuracy "28% at baseline to 62%"; "Performance remained limited for lung cancer (20%)" | Small pilot. Lung-cancer-specific accuracy for ChatGPT |
| L5 | Lee S et al. | 2025 | CXR-LLaVA: a multimodal large language model for interpreting chest X-ray images | European Radiology | PR | 10.1007/s00330-024-11339-6; arXiv 2310.18341 | VLM (foundational) | report-gen | 592,580 public CXRs; MIMIC + external test | "average F1 score of 0.81 for six major pathological findings in the MIMIC internal test set and 0.56 ... external"; "72.7% success rate in autonomous reporting ... below the 84.0%" | Open CXR MLLM. Evaluated on nodules in L1 |
| L6 | Chen Z et al. | 2024 | A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation (CheXagent) | arXiv | PP | arXiv 2401.12208 | VLM (foundational) | other (multi-task, 8 task types) | CheXinstruct; CheXbench | "36% time saving for residents using CheXagent-drafted reports" | Background model. No nodule-specific result in the abstract |
| L7 | Sellergren A et al. | 2026 | An open vision-language model for diverse medical applications (MedGemma) | Nature Medicine | PR | 10.1038/s41591-026-04626-w | VLM (foundational) + MedSigLIP encoder | other (multi-task) | multi-domain | "15.5-18.1% in chest X-ray finding classification" improvement over base | Background model. Evaluated on nodules in L2 (sensitivity 5.2%) |
| L8 | Bannur S et al. | 2024 | MAIRA-2: Grounded Radiology Report Generation | arXiv | PP | arXiv 2406.04449 | VLM (foundational) | report-gen (grounded) | MIMIC-CXR et al. | "achieves state of the art on existing report generation benchmarks" (no numbers in abstract) | Background model for grounded findings, relevant to nodule localization |
| L9 | Deperrois N et al. | 2025 | RadVLM: A Multitask Conversational Vision-Language Model for Radiology | arXiv | PP | arXiv 2502.03333 | VLM (foundational) | VQA-grounding | >1M image-instruction pairs | "state-of-the-art performance in conversational capabilities and visual grounding" (no numbers) | Background model. Highest accuracy in L2 but 11% sensitivity |
| L10 | Zambrano Chaves JM et al. | 2025 | A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings (LLaVA-Rad) | Nature Communications (per DOI prefix) | PR | 10.1038/s41467-025-58344-x | VLM (foundational) | report-gen | 697K image-text pairs | "LLaVA-Rad (7B) ... outperforming much larger models like GPT-4V and Med-PaLM M (84B)" | Background model. Highest sensitivity in L2 |
| L11 | Fallahpour A et al. | 2025 | MedRAX: Medical Reasoning Agent for Chest X-ray | arXiv | PP | arXiv 2502.02673 | VLM agent | VQA (agentic) | ChestAgentBench, 2,500 queries | "state-of-the-art performance compared to both open-source and proprietary models" (no numbers) | Agent framework. Tied best on nodules in L1 (0.711) |
| L12 | Thawakar O et al. | 2023 | XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models | arXiv | PP | arXiv 2306.07971 | VLM | report-gen (summary) / VQA | ~217k report summaries (MIMIC/OpenI) | No quantitative result in abstract | Evaluated on nodules in L1 |

### 2b. VL-foundation / multimodal

| ID | Authors | Year | Title | Venue | PR? | DOI / arXiv | Category | Task | Data | Key quantitative result | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L13 | Yang C et al. | 2026 | LUNG-KGMM: Knowledge-Guided Multimodal Learning for Lung Cancer Incidence Prediction | arXiv | PP | arXiv 2608.14657 | VL-foundation / multimodal (CXR representation + radiology reports + EHR + guideline KG) | screening-risk | MIMIC cohort; Xiamen external cohort | No numbers in abstract ("superior performance over state-of-the-art methods") | Only multimodal CXR+report lung cancer risk model found. Abstract does not say whether an LLM is used |
| L14 | Zhang S et al. | 2023 | BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs | arXiv | PP | arXiv 2303.00915 | VL-foundation (CLIP) | other (classification/retrieval/VQA) | PMC-15M | No CXR-nodule numbers. Abstract says it "outperforms ... BioViL ... RSNA pneumonia detection" | Tied best on nodules in L1 (accuracy 0.711) |

### 2c. Text-only LLMs on nodule reports (adjacent: no CXR-report study was found)

| ID | Authors | Year | Title | Venue | PR? | DOI / arXiv | Category | Task | Data | Key quantitative result | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L15 | Gamble JL et al. | 2024 | Limitations of GPT-3.5 and GPT-4 in Applying Fleischner Society Guidelines to Incidental Lung Nodules | journal n/s (SAGE DOI) | PR | 10.1177/08465371231218250 | text-LLM | other (follow-up recommendation) | 120 fictitious nodule descriptions | "GPT-3.5 accuracy ... 0.058"; "GPT-4 ... 0.15"; with guidelines in prompt "0.42 ... and to 0.66"; fine-tuned GPT-3.5 "0.46" | Key negative result on guideline reasoning. Reports are synthetic, not CXR |
| L16 | Gulati V et al. | 2026 | Performance of ChatGPT (GPT-4o) in automated Lung-RADS classification: a category-stratified evaluation of prompting strategies | Clinical Radiology (per DOI prefix) | PR | 10.1016/j.crad.2026.107466 | text-LLM | other (Lung-RADS classification) | 327 LDCT reports | Structured prompt "accuracy was 68.8% with a weighted Cohen's kappa of 0.758"; unstructured "58.1% ... 0.695" | CT reports. Adjacent evidence on prompting |
| L17 | Zhou T et al. | 2024 | Automating Lung-RADS Categorization And Follow-Up Recommendations Using In-Context Learning With LLMs | venue n/s (PubMed 41726431) | PR (PubMed-indexed; likely proceedings) | n/s | text-LLM | other (Lung-RADS / follow-up) | LDCT reports | No numbers in abstract | Adjacent (CT). Method: in-context learning |
| L18 | Wen J et al. | 2025 | Evaluation of large language models in generating pulmonary nodule follow-up recommendations | European Journal of Radiology Open | PR | n/s (Consensus free tier) | text-LLM | other (follow-up recommendation) | 1,009 CT follow-up reports | "ERNIE-4.0-Turbo-8K and GPT-4o-mini ... (94.6 % vs 92.8 %, P = 0.07)"; harmfulness "(2.9 % vs 3.5 %...)" | Adjacent (CT). Large real-report evaluation |

### 2d. CNN/ViT baselines (comparators)

| ID | Authors | Year | Title | Venue | PR? | DOI / arXiv | Category | Task | Data | Key quantitative result | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L19 | Nam JG et al. | 2019 | Development and Validation of DL-based Automatic Detection Algorithm for Malignant Pulmonary Nodules on Chest Radiographs | Radiology | PR | 10.1148/radiol.2018180237 | CNN baseline | detection | 43,292 CXR; 1 internal + 4 external sets | "0.92-0.99 (AUROC) and 0.831-0.924 (JAFROC FOM)"; higher than "17 of 18 ... physicians" | Landmark malignant-nodule detector (Lunit precursor) |
| L20 | Nam JG et al. | 2023 | AI Improves Nodule Detection on Chest Radiographs in a Health Screening Population: A Randomized Controlled Trial | Radiology | PR | 10.1148/radiol.221894 | CNN baseline (commercial CAD) | detection (prospective RCT) | 10,476 participants | "0.59% [31 of 5238] vs 0.25% [13 of 5238] ... odds ratio, 2.4" | Only RCT-level evidence. Sets the clinical bar |
| L21 | Yoo H et al. | 2020 | Validation of a Deep Learning Algorithm for the Detection of Malignant Pulmonary Nodules in Chest Radiographs | JAMA Network Open | PR | 10.1001/jamanetworkopen.2020.17135 | CNN baseline (Lunit INSIGHT CXR per web search) | detection / screening | NLST, 5,485 participants | Nodule "sensitivity and specificity ... 86.2% ... and 85.0%"; all cancers "sensitivity was 75.0% ... positive predictive value was 3.8%" | Landmark NLST validation |
| L22 | Yoo H et al. | 2021 | AI-based improvement in lung cancer detection on chest radiographs: results of a multi-reader study in NLST dataset | European Radiology | PR | 10.1007/s00330-021-08074-7 | CNN baseline | detection (reader study) | NLST, 519 images, 8 readers | Residents' sensitivity "0.61 ... vs. 0.72"; radiologists' FPPI "0.24 ... vs. 0.17" | Human-AI second-reader effect |
| L23 | Lu MT et al. | 2020 | Deep Learning Using Chest Radiographs to Identify High-Risk Smokers for Lung Cancer Screening CT (CXR-LC) | Annals of Internal Medicine (per DOI prefix) | PR | 10.7326/M20-1868 | CNN baseline | screening-risk | PLCO (n=41,856 dev), NLST validation | "PLCO AUC, 0.755 vs. 0.634"; NLST "0.659 vs. 0.650" (vs PLCOm2012) | Landmark CXR risk model |
| L24 | Lee JH et al. | 2022 | Deep Learning to Optimize Candidate Selection for Lung Cancer CT Screening: Advancing the 2021 USPSTF Recommendations | Radiology | PR | 10.1148/radiol.212877 | CNN baseline | screening-risk | 19,488 Korean health-check smokers | "AUCs for incident lung cancers were 0.68 ... and 0.75" | External validation of CXR-LC |
| L25 | Hwang EJ et al. | 2019 | Development and Validation of a DL-Based Automated Detection Algorithm for Major Thoracic Diseases on Chest Radiographs | JAMA Network Open | PR | 10.1001/jamanetworkopen.2019.1095 | CNN baseline | detection (incl. malignant neoplasm) | 89,834 dev CXR; 5-institution external | "AUC of 0.979 ... image-wise ... 0.972 ... lesion-wise localization" | Multi-disease detector that includes malignancy |
| L26 | Majkowska A et al. | 2020 | Chest Radiograph Interpretation with DL Models: Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation | Radiology | PR | 10.1148/radiol.2019191293 | CNN baseline (Google) | detection (nodule or mass) | 759,611 DS1 images + ChestX-ray14 | Nodule or mass population-adjusted AUC "0.72"; ChestX-ray14 "0.91" | Google CXR nodule/mass result. Released adjudicated ChestX-ray14 labels |
| L27 | Dissez G et al. | 2022 | Enhancing Early Lung Cancer Detection on Chest Radiographs with AI-assistance: A Multi-Reader Study | arXiv | PP | arXiv 2208.14742 | CNN baseline (commercial, Behold.ai) | detection (reader study) | 400 NHS CXR, 11 clinicians | "overall increase of 17.4% of lung cancers being identified" | UK lung-cancer CXR reader study |
| L28 | Topff L et al. | 2024 | AI-assisted double reading of chest radiographs to detect clinically relevant missed findings: a two-centre evaluation | European Radiology | PR | 10.1007/s00330-024-10676-w | CNN + NLP (report-image discrepancy) | detection (missed findings) | 25,104 CXR | Missed findings "lung nodules (71.4%, 25 of 35)"; "0.1% of all cases" | Image-report comparison, a precursor to LLM report QA |
| L29 | Shimazaki A et al. | 2022 | Deep learning-based algorithm for lung cancer detection on chest radiographs using the segmentation method | Scientific Reports (per DOI prefix) | PR | 10.1038/s41598-021-04667-w | CNN baseline | detection | 629 train / 151 test CXR (pathology-confirmed cancers) | "sensitivity of 0.73 with 0.13 mFPI"; blind spots "0.50-0.64 compared with ... 0.87" | Characterizes blind-spot failures |
| L30 | Kotevska O et al. | 2026 | Extending the Horizon of Early Diagnosis: Lung Cancer Prediction with Vision Transformers | arXiv | PP | arXiv 2608.21571 | ViT baseline | screening-risk (1-2 yr before diagnosis) | 259,361 VA CXRs | "pretrained models exceeding the scratch baseline by 6-10 percentage points in AUC" | Recent ViT pre-diagnostic prediction |
| L31 | Ohlmann-Knafo S et al. | 2022 | AI-based software for lung nodule detection in chest X-rays -- Time for a second reader approach? | arXiv | PP | arXiv 2206.10912 | CNN baseline (commercial) | detection | JSRT (n=247), 8 radiologists | Radiologists "average AUC value was 0.77 ± 0.07"; AI modes "increase in sensitivity (by 14% and 12%)" | Same JSRT set as L2, so VLM, CNN and radiologist results can be compared directly |
| L32 | Tang JSN et al. | 2023 | Impact of Different AI User Interfaces on Lung Nodule and Mass Detection on Chest Radiographs | Radiology: Artificial Intelligence | PR | 10.1148/ryai.220079 | CNN baseline (human-AI UI) | detection (reader study) | 140 CXR, 10 radiologists | "text-only output compared with no AI (0.87 vs 0.82; P < .001)" | Text-only AI output helped most, which is relevant to LLM text interfaces |
| L33 | Essa MEA | 2025 | Diagnostic accuracy of AI in chest radiography for pneumonia and lung cancer: A meta-analysis | European Journal of Radiology Open | PR | n/s | review / meta-analysis (CNN evidence) | detection | 15 studies, ≈12,000 CXRs | "For lung nodules, pooled AI sensitivity was ≈ 72 % and specificity ≈ 95 %"; second reader "+9-10 %age points" | Pooled baseline numbers |
| L34 | Megat Ramli PN et al. | 2025 | A Systematic Review: The Role of AI in Lung Cancer Screening in Detecting Lung Nodules on Chest X-Rays | Diagnostics | PR | 10.3390/diagnostics15030246 | review / SR (CNN evidence) | detection | 34 studies (2020–2024) | "sensitivity rates of 56.4–95.7% and specificities of 71.9–97.5% ... AUROC ... 0.89 and 0.99, compared to radiologists' mean ... 0.81" | Range of baseline performance. Notes GGO and hilar failures |

### 2e. Datasets / benchmarks

| ID | Authors | Year | Title | Venue | PR? | DOI / arXiv | Category | Task | Data | Key quantitative result | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L35 | Sogancioglu E et al. | 2024 | Nodule detection and generation on chest X-rays: NODE21 Challenge | arXiv | PP | arXiv 2401.02192 | dataset/challenge | detection | NODE21 public set | No numbers in abstract | Main public CXR nodule benchmark |
| L36 | Mustafic A et al. | 2026 | NoduLoCC2026: Lung Nodule Localization and Classification Contest from Chest X-Ray Images | arXiv | PP | arXiv 2606.21290 | dataset/challenge | detection + localization | Challenge set + external test; 5 teams | "balanced accuracy score of 0.72 and AUC-ROC of 0.79"; localization "correct number of nodules on 53% ... median distance of 12.83mm" | Newest nodule localization benchmark. Shows localization is the hard part |
| L37 | Nguyen HQ et al. | 2022 | VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations | Scientific Data (per DOI prefix) | PR | 10.1038/s41597-022-01498-w | dataset | detection (boxes) | 18,000 images, 22 local labels, 17 radiologists | "22 local labels of rectangles"; the abstract does not list label names (nodule/mass labels not confirmed from the abstract) | Box-level labels for grounding |
| L38 | Wang X et al. | 2017 | ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks ... | arXiv (CVPR version not verified) | PP | arXiv 1705.02315 | dataset | detection (weak localization) | 108,948 images, 8 labels | Abstract does not list the labels (Nodule/Mass not confirmed from the abstract) | Origin of ChestX-ray14 Nodule/Mass labels |
| L39 | Irvin J et al. | 2019 | CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison | arXiv (AAAI version not verified) | PP | arXiv 1901.07031 | dataset | other (classification) | 224,316 CXR, 14 observations | The abstract does not name the "Lung Lesion" label | Lung Lesion label used in zero-shot evaluations |
| L40 | Johnson AEW et al. | 2019 | MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs | arXiv | PP | arXiv 1901.07042 | dataset | other (classification / report source) | 377,110 images, 14 labels | n/a | Training source for nearly all CXR VLMs. LUNG-KGMM cohort |
| L41 | Liu J et al. | 2020 | ChestX-Det10: Chest X-ray Dataset on Detection of Thoracic Abnormalities | arXiv | PP | arXiv 2006.10550 | dataset | detection (boxes) | ~3,500 images, 10 categories | Categories not listed in abstract | Box-level NIH subset |
| L42 | Pal A et al. | 2025 | ReXVQA: A Large-scale VQA Benchmark for Generalist Chest X-ray Understanding | arXiv | PP | arXiv 2506.04353 | dataset/benchmark | VQA | ~696,000 Qs, 160,000 studies | "MedGemma achieves 83.24% overall accuracy"; reader study "83.84% ... best radiology resident: 77.27%" | Large CXR VQA benchmark. Nodule question subset not confirmed from the abstract |
| L43 | Nguyen DH et al. | 2025 | VinDr-CXR-VQA: A VQA Dataset for Explainable Chest X-Ray Analysis with Multi-Task Learning | arXiv | PP | arXiv 2511.00504 | dataset/benchmark | VQA-grounding | 17,597 QA pairs, 4,394 images with boxes | "MedGemma-4B-it ... (F1 = 0.624, +11.8% over baseline)" | Grounded VQA built on VinDr boxes, a candidate for nodule grounding |
| L44 | Wei Y et al. | 2026 | RSNA LLM Benchmark Dataset for Chest Radiographs of Cardiothoracic Disease (REVEAL-CXR) | arXiv | PP | arXiv 2601.15129 | dataset/benchmark | other (MLLM benchmark, 12 labels) | 200 studies, 17 radiologists | "A benchmark of 200 chest radiographic studies with 12 benchmark labels" | RSNA MLLM benchmark. Label list (nodule/mass) not confirmed from the abstract |

---

## 3. Title-only / unverified

These items were seen, but no abstract was available in tool output, or they were not retrieved at all. They have not been screened.

- Bustos A et al. 2020. "PadChest: A large chest x-ray image dataset with multi-label annotated reports". Medical Image Analysis, DOI 10.1016/j.media.2020.101797 (OpenAlex; abstract empty). Expected to contain nodule/mass labels; not verified.
- "An Artificial Intelligence-Based Chest X-ray Model on Human Nodule Detection Accuracy From a Multicenter Study". JAMA Network Open (from a WebSearch snippet only; authors, year and DOI not verified).
- Shiraishi J et al. (JSRT database original paper, about 2000). Not retrieved by any tool. It is used as the dataset in L2 and L31.

## 4. Excluded but notable (max 15)

| # | Paper | Reason |
|---|---|---|
| 1 | Hwang EJ, Goo JM, Park CM 2025, "AI Applications for Thoracic Imaging: Considerations for Best Practice", Radiology, 10.1148/radiol.240650 | Review. Useful framing of LLMs in thoracic imaging |
| 2 | Zhong J et al. 2025, "A Narrative Review on Large AI Models in Lung Cancer Screening, Diagnosis, and Treatment Planning", arXiv (id n/s) | Review (mostly CT) |
| 3 | Luo Y et al. 2025, "Opportunities and challenges in lung cancer care in the era of LLMs and VLMs", Transl Lung Cancer Res, 10.21037/tlcr-24-801 | Review |
| 4 | Zhou Y et al. 2024, "Evaluating GPT-4 with Vision on Detection of Radiological Findings on Chest Radiographs", arXiv 2403.15528 (Radiology 2024 per WebSearch) | Generic GPT-4V CXR evaluation. No nodule-specific result in the abstract |
| 5 | Dehdab R et al. 2024, "Evaluating ChatGPT-4V in chest CT diagnostics", Jpn J Radiol?, 10.1007/s11604-024-01606-3 | CT only (NSCLC sensitivity 27.27%) |
| 6 | Zhuang L et al. 2025, "VLM-Based Semantic-Guided Imaging Biomarker for Lung Nodule Malignancy Prediction", arXiv 2504.21344 | CT (CLIP on NLST LDCT) |
| 7 | Dutta P et al. 2026, "FZ-VLM ... Pulmonary Nodule Characterization", arXiv 2608.15004 | CT |
| 8 | Kim TH et al. 2026, "Performance evaluation of domain-specific and general-purpose AI models for CXR interpretation" (M4CXR vs GPT-4o), BMC Med Imaging, 10.1186/s12880-026-02557-z | Generic CXR. No nodule-specific result |
| 9 | Lee KH et al. 2023, "Validation of a DL CXR Interpretation Model ... Comparative Analysis with ChatGPT" (KARA-CXR), Diagnostics, 10.3390/diagnostics14010090 | Generic CXR |
| 10 | Xu S et al. 2023, ELIXR, arXiv 2308.01317 | Generic VL foundation. Abstract reports no nodule result (zero-shot "mean AUC of 0.850 across 13 findings") |
| 11 | Tiu E et al. 2022, CheXzero, Nat Biomed Eng, 10.1038/s41551-022-00936-9 | Generic. No nodule result in abstract |
| 12 | Hu D et al. 2025, "LLMs in Summarizing Radiology Report Impressions for Lung Cancer in Chinese", JMIR, 10.2196/65547 | CT/PET-CT/US reports, not CXR |
| 13 | Rafferty A et al. 2024/2025, "Leveraging Expert Input for Robust and Explainable AI-Assisted Lung Cancer Detection in Chest X-rays" (ClinicXAI), arXiv 2403.19444 / IEEE ICHI 2025 | CNN XAI on CXR lung cancer. Not a landmark baseline; cap reached |
| 14 | Blebea C et al. 2025, "A Breath of Fresh AI: Using LLMs to Detect Lung Nodule Characteristics and Follow-up Recommendations", AJRCCM (conference abstract) | CT reports, conference abstract (92% accuracy, 3% hallucination) |
| 15 | Moore CL et al. 2025, "Using NLP to identify ED patients with incidental lung nodules requiring follow-up", 10.1111/acem.15080 | CT reports. RoBERTa (encoder) NLP, not a generative LLM |

## 6. Undermind deep-search additions (round 2, 2026-11-12)

Source: an Undermind deep search in the workspace "LLMs for Lung Cancer on Chest X-rays" (101 ranked papers). It covers abstracts only. I read the full PDFs of 4 key papers with Undermind `read_pdfs`: Kho26, Mut26, Gos25 and Cho26. I compared all 101 papers with L1–L44 and kept 18 new ones as L45–L62. The other papers either duplicate L1–L44 or are lower-ranked CNN CAD papers. Those are listed in the deep search (r < 0.55) but not added here.

**Correction to L1 (from the full text):**
- Only 83 of the 100 CXRs were analysed (38 nodule, 45 normal), because 17 images could not be processed by several models.
- The source is MosMedData type VII from the Moscow Experiment. Nodules are solid, 6–30 mm, confirmed by CT within 14 days.
- Best AUC: BiomedCLIP 0.743, then MedRAX 0.704. Best of the general MLLMs: Claude 3.7 Sonnet, AUC 0.625 (sensitivity 0.316, specificity 0.933).
- CXR-LLaVA reached sensitivity 0.079.
- There is no head-to-head comparison with CNN CAD on the same cohort.

| ID | Cite key (Undermind) | Year | Title | Venue | PR? | DOI / arXiv | Category | Task | Key result | Relevance |
|---|---|---|---|---|---|---|---|---|---|---|
| L45 | Arz24 | 2024 | Independent evaluation of the accuracy of 5 AI software for detecting lung nodules on chest X-rays | Quant Imaging Med Surg | PR | 10.21037/qims-24-160 | CNN baseline (commercial: Celsus, Lunit INSIGHT CXR, qXR…) | detection + segmentation | Best AUC 0.956 (probability stage). With radiologist-checked segmentation and classification, best AUC 0.812 | **Same dataset source as L1** (MosMedData VII, 100 CXR, 50/50) and same research group. This is the closest CNN-vs-LLM comparison point; L1 does not confirm the exact sample is the same |
| L46 | Nis26b | 2026 | Performance of VLMs for Zero-Shot Lung Nodule Detection on CXR | medRxiv | PP | 10.64898/2026.05.31.26354565 | VLM | detection | Same numbers as L2. Adds: detection highest for obvious nodules, limited for subtle ones | Preprint of L2. Merge with L2 |
| L47 | Mut26 | 2026 | Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest Radiography | arXiv | PP | arXiv 2606.11606 | VL-foundation (mechanistic probe) | other (representation analysis) | ChestX-Det10 nodule, bbox-level AUC with CLS → patch-local pooling: BiomedCLIP 0.688 → 0.899; MedSigLIP 0.805 → 0.955. Image-level CLS for MedSigLIP: small nodules 0.627 vs large 0.844 | **Candidate mechanism** for why VLMs miss small nodules: global pooling discards small-lesion signal. Requires an oracle ROI (full text read) |
| L48 | Gos25 | 2025 | Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in CXR | PMLR (ML4H) | PR | arXiv 2509.18015 | VLM | VQA-grounding (localization) | Lung lesion hit rate: GPT-4 35.7%, GPT-5 35.7%, MedGemma 14.3%, CNN 29.0%, radiologist 85.0%. Mean over 9 findings: GPT-5 49.7% | Only paper found with a lung-lesion localization number for MLLMs (full text read) |
| L49 | Cho26 | 2026 | CheXpercept: Expert-Level Lesion Perception in CXR | arXiv | PP | arXiv 2606.21020 | benchmark | VQA-grounding | Coarse detection about 92%. Contour revision ≤ 2.9%; attributes ≤ 25.7% (OP). Medical VLMs show no advantage over general ones | **Does not include nodule or mass** (full text read). Shows a design for a multi-level perception benchmark |
| L50 | Fis25 | 2025 | VL Foundation Models Do Not Transfer to Medical Imaging Classification (negative result) | medRxiv | PP | 10.64898/2025.12.06.25341759 | VL-foundation | detection (ChestX-ray14 multi-label) | Best VLM F1 = 0.203 vs CNN 0.811. Fine-tuning collapsed sensitivity to < 15% | Negative result. Does not report a nodule-specific number |
| L51 | Abo26 | 2026 | Automated reporting of chest radiographs: pilot study with two VLMs (in German) | RöFo | PR | 10.1055/s-0046-1817587 | VLM | report-gen | 100 patients, 25 with a pulmonary mass. Claude 3.5 Sonnet main-finding accuracy 62% vs GPT-4 51%. Masses were "less often correctly identified" | Report generation with a mass subgroup |
| L52 | Lac25 | 2025 | Accuracy of ChatGPT-4o in Interpreting Chest and Abdominal X-Rays | J Pers Med | PR | 10.3390/jpm15050194 | VLM | detection | CXR 66.15% correct. Tumor was among the highest detection rates | **Counterpoint**: tumors are easier when large or obvious |
| L53 | Sta25 | 2025 | Sensitivity/specificity of generative AI chatbots in CXR interpretation | ERS Congress abstract (Eur Respir J suppl.) | conference abstract | 10.1183/13993003.congress-2025.pa4975 | VLM | detection | 180 CXR. Nodule sensitivity (Mistral) 41.3%. Accuracy ≤ 38% | Low-grade evidence |
| L54 | Che26 | 2026 | Limitations in CXR interpretation by vision-capable LLMs | Diagnostics | PR | (= L3) | VLM | detection | — | Duplicate of L3 |
| L55 | Muk23 | 2023 | Feasibility of the privacy-preserving LLM Vicuna for labeling radiology reports | Radiology | PR | 10.1148/radiol.231147 | text-LLM | other (CXR report labeling) | 13 findings. Median κ 0.57–0.64 vs CheXpert/CheXbert. Median AUC 0.84 | **Fills the "no text-LLM on CXR reports" gap**. Covers CheXpert findings (incl. lung lesion), but no nodule-specific number in the abstract |
| L56 | Sav24 | 2024 | General-purpose LLMs vs a domain-specific NLP tool for label extraction from CXR reports | AJR | PR | 10.2214/AJR.23.30573 | text-LLM | other (CXR report labeling) | GPT-4 beat the domain-specific NLP model (no numbers in the abstract) | Text-LLM on CXR reports |
| L57 | Par24b | 2024 | Extraction of clinical data on major pulmonary diseases from unstructured radiologic reports using an LLM | PLOS ONE | PR | 10.1371/journal.pone.0314136 | text-LLM | other (extraction incl. lung cancer) | 1,800 CXR + CT reports, 7 diseases incl. lung cancer. GPT-4 sensitivity 0.71–1.00, specificity 0.89–1.00 | Text-LLM, includes lung cancer and CXR reports |
| L58 | Yas24 | 2024 | Fine-tuned LLM for extracting patients on pretreatment for lung cancer from PACS radiological reports | J Imaging Inform Med | PR | 10.1007/s10278-024-01186-8 | text-LLM (fine-tuned) | other (cohort extraction) | Accuracy 0.983 vs radiologists 0.969. 46 s vs 1,538–2,539 s | Lung cancer text-LLM. The abstract does not state the modality |
| L59 | Led24 | 2024 | LiteGPT: LVLM for joint CXR localization and classification | arXiv | PP | arXiv 2407.12064 | VLM | VQA-grounding | SOTA classification on VinDr-CXR (no numbers in the abstract) | VLM grounding on VinDr, which has nodule/mass boxes |
| L60 | Mus26b | 2026 | NoduLoCC: a dataset for lung nodule localization and classification from CXR | ICCVIT proceedings | PR | (Undermind; DOI n/s) | dataset | detection + localization | — | Peer-reviewed version of the L36 dataset |
| L61 | Kim26 | 2026 | Lung Cancer Risk Prediction Using CXR and LLM-Embedded Clinical Features | J Thorac Oncol (conference abstract P1.079) | conference abstract | 10.1016/j.jtho.2026.104482 | multimodal (CXR + LLM embedding) | screening-risk | Abstract unavailable | **Title only.** The only CXR + LLM risk model besides L13. Monitor for a full paper |
| L62 | Chu26 | 2026 | Caution Needed When Applying VLMs for CXR Nodule Screening | AJR (commentary) | PR (commentary) | 10.2214/AJR.26.35708 | commentary | — | Abstract unavailable | Editorial on L2. Title only |

Other CNN CAD papers in the deep search that may be worth citing as baselines (not screened in detail): Sim19 (Radiology 2019, 10.1148/radiol.2019182465), Nam20 (Radiol Cardiothorac Imaging, missed lung cancer), Lee20 (Radiology, health screening), Jan20 (Radiology, overlooked lung cancers), Lee23b (Radiology, human–AI interaction), Rag22 (JAMA Netw Open, CXR + EMR risk model), Ued21 (BMC Cancer), Luo22 (Radiol AI, annotation granularity vs shortcut), Rob24 (Acad Radiol, multicentre reader study), Kim24 (Radiology Advances, unsuspected lung cancer).

Updated totals: **62 entries** (L1–L62). Two are duplicates or preprints of existing entries (L46 = L2, L54 = L3), which leaves **60 unique papers**. Two more are title-only (L61, L62).

## 5. arXiv IDs in kept corpus

2310.18341, 2401.12208, 2406.04449, 2502.03333, 2502.02673, 2306.07971, 2608.14657, 2303.00915, 2208.14742, 2608.21571, 2206.10912, 2401.02192, 2606.21290, 1705.02315, 1901.07031, 1901.07042, 2006.10550, 2506.04353, 2511.00504, 2601.15129, 2606.11606, 2509.18015, 2606.21020, 2407.12064

## 7. Scopus prestige-journal additions (2026-11-12)

Source: Scopus (VŠB institutional access), queries Q1–Q3 in `04-scopus-prestige-search.md`; DOIs verified via Crossref/Undermind. "Read" = depth of reading: FT = full text available in `papers/`, AB = abstract only (RSNA, see `papers/_RSNA-abstract-notes.md`), T = title/venue only so far.

**Updates to earlier entries:** L9 RadVLM → *Scientific Reports* 2026 (10.1038/s41598-026-66181-1). L35 NODE21 → *IEEE TMI* 2024 (10.1109/tmi.2024.3382042).

| ID | First author | Year | Title (short) | Venue | DOI | Cluster | Read | Key result |
|---|---|---|---|---|---|---|---|---|
| L63 | Hong EK | 2025 | Domain-specific multimodal generative AI for CXR report generation | Radiology | 10.1148/radiol.241476 | T2 | AB | 8.8M pairs; acceptance 70.5% vs radiologist 73.3% vs GPT-4V 29.6% |
| L64 | Volkov E | 2025 | VLM fine-tuning via LoRA for structured reports on lung X-ray | SCM 2025 (IEEE) | 10.1109/scm66446.2025.11060140 | T2 | FT (read) | SmolVLM 2.2B + QLoRA (0.47% params) on MultiCaRe captions; only NLG metrics (ROUGE-L 0.13→0.58); no detection/localization or clinical accuracy |
| L65 | Zhang | 2026 | CXR screening using LLMs + deep CNNs on MIMIC-CXR | Front Digit Health | 10.3389/fdgth.2026.1883337 | T2 | FT | — |
| L66 | Hoq ME | 2025 | Native-resolution 2D embeddings, RAD-DINO, lung cancer classification | J Imaging Inform Med | 10.1007/s10278-025-01748-4 | T3 | FT | — |
| L67 | Hayeso | 2026 | Lightweight hybrid foundation model, lung cancer prognosis, low-dose CXR | Transl Lung Cancer Res | 10.21037/tlcr-2025-aw-1299 | T8 | FT | — |
| L68 | Bannur S | 2023 | BioViL-T: temporal structure for biomedical VLP | CVPR | 10.1109/cvpr52729.2023.01442 | T3 | FT | — |
| L69 | Gao J | 2026 | S2D-Align: anatomically grounded report generation | AAAI | 10.1609/aaai.v40i36.40335 | T4 | FT | — |
| L70 | Zhang | 2026 | Generalized medical phrase grounding | IEEE TMI | 10.1109/tmi.2026.3707568 | T4 | FT | — |
| L71 | Xu | 2025 | MedGround-R1: grounding via spatial-semantic rewarded GRPO | MICCAI | 10.1007/978-3-032-04971-1_37 | T4 | FT | — |
| L72 | Cid YD | 2024 | X-Raydar: open-source DNNs for comprehensive CXR reading | Lancet Digit Health | 10.1016/s2589-7500(23)00218-2 | T6 | FT | — |
| L73 | Pesce E | 2019 | Pulmonary lesion detection with visual attention networks | Med Image Anal | 10.1016/j.media.2018.12.007 | T6 | FT | — |
| L74 | Jang S | 2020 | DLAD reduces overlooked lung cancers on CXR | Radiology | 10.1148/radiol.2020200165 | T7 | AB | Reader AUC 0.67→0.76; sens 40%→53% on missed cancers |
| L75 | Lee JH | 2020 | DL vs radiologists, lung cancer detection, screening population | Radiology | 10.1148/radiol.2020201240 | T6 | AB | AUC 0.97–0.99; sens 83–90%; FPR ~3% |
| L76 | Sung J | 2021 | DL detection system for multiple findings (randomized crossover) | Radiology | 10.1148/radiol.2021202818 | T7 | AB | JAFROC 0.90→0.95; DLD alone 0.96 |
| L77 | Hwang EJ | 2021 | DL CAD for pulmonary metastasis on CXR (real-world) | Radiology | 10.1148/radiol.2021210578 | T6 | AB | Yield 0.32%→0.86%, false referral non-inferior |
| L78 | Shen | 2023 | Disentangled image synthesis for nodule augmentation/detection | Med Image Anal | 10.1016/j.media.2022.102708 | T6 | FT | — |
| L79 | Kim | 2023 | Multicentre external validation of commercial CXR AI, low prevalence | Eur Radiol | 10.1007/s00330-022-09315-z | T6 | FT | — |
| L80 | van Leeuwen KG | 2024 | Commercial AI for radiograph nodule detection and bone age | Radiology | 10.1148/radiol.230981 | T6 | FT | 7 nodule algorithms; 4 beat readers (AUC 0.86–0.93 vs 0.81) |
| L81 | Nam JG | 2020 | Undetected lung cancer on PA CXR: role of DL algorithm | Radiol Cardiothorac Imaging | 10.1148/ryct.2020190222 | T6 | FT | AUROC 0.899, AUFROC 0.744 vs radiologists 0.63–0.66 |
| L82 | Sabottke CF | 2020 | Effect of image resolution on DL in radiography | Radiology AI | 10.1148/ryai.2019190015 | M | FT | Nodule AUC at 64 px = 80.7% of AUC at 320 px; best 256–448 px |
| L83 | Kim | 2020 | Optimal matrix size of CXR for nodule/mass CAD | Eur Radiol | 10.1007/s00330-020-06892-9 | M | FT | — |
| L84 | Mosquera C | 2024 | Class imbalance: evaluation practices (discrimination, calibration) | Eur Radiol | 10.1007/s00330-024-10834-0 | M | FT | — |
| L85 | Sim Y | 2020 | DCNN software improves radiologist detection of malignant nodules | Radiology | 10.1148/radiol.2019182465 | T7 | AB | Radiologist sens 65.1%→70.3% (600 cancers, 4 centres) |
| L86 | Homayounieh F | 2021 | AI CXR model on human nodule detection accuracy | JAMA Netw Open | 10.1001/jamanetworkopen.2021.41096 | T7 | FT | — |
| L87 | Ahn JS | 2022 | AI-aided CXR interpretation and reader performance | JAMA Netw Open | 10.1001/jamanetworkopen.2022.29289 | T7 | FT | — |
| L88 | Bernstein MH | 2023 | Incorrect AI results impact radiologists | Eur Radiol | 10.1007/s00330-023-09747-1 | T7 | FT | — |
| L89 | Bennani S | 2023 | AI improves radiologist detection of CXR abnormalities | Radiology | 10.1148/radiol.230860 | T7 | AB | Nodule sens +12% absolute; reading time −31% |
| L90 | Lee JH | 2023 | Human–AI interaction, malignant nodule detection | Radiology | 10.1148/radiol.222976 | T7 | AB | High-acc AI: reader AUC 0.77→0.82; susceptibility 67% |
| L91 | Raghu VK | 2022 | DL lung cancer risk from CXR + EMR | JAMA Netw Open | 10.1001/jamanetworkopen.2022.48793 | T8 | FT | — |
| L92 | Aggarwal R | 2021 | DL diagnostic accuracy in medical imaging: SR/MA | npj Digit Med | 10.1038/s41746-021-00438-z | T10 | FT | — |

"M" = new methods cluster (resolution, class imbalance, PEFT/audit methods). Updated totals: **92 entries (L1–L92), 90 unique papers** (L46 = L2, L54 = L3). Full text available locally: 4 read earlier (L1, L47, L48, L49) + 22 PDFs of L63–L92 (L64 read in full).
