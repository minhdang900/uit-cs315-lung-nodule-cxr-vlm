# Tìm kiếm bổ sung trên Scopus (tạp chí và hội nghị uy tín)

**Ngày:** 2026-11-12
**Công cụ:** Scopus qua tài khoản trường (trình duyệt), DOI kiểm chứng bằng Crossref API.
**Mục tiêu:** bổ sung bài từ tạp chí Q1/Q2 và hội nghị A* cho corpus L1–L62, nhất là các cụm còn mỏng (T3, T4, T6, T7).
**Mức đọc:** chỉ đọc tiêu đề, nguồn, năm và số trích dẫn trên Scopus. Chưa đọc tóm tắt hay toàn văn.

---

## 1. Nhật ký truy vấn

Mọi truy vấn dùng trường `TITLE-ABS-KEY` và sắp theo số trích dẫn giảm dần.

| # | Truy vấn (rút gọn) | Bộ lọc | Kết quả | Giữ |
|---|---|---|---|---|
| Q1 | (LLM / VLM / MLLM / GPT-4 / foundation model) AND (nodule / lung cancer / lung mass / lung lesion) AND (chest radiograph / chest x ray / CXR) | PUBYEAR > 2022 | 36 | 6 mới, 2 trùng (L1, L3) |
| Q2 | (nodule / lung cancer) AND (chest radiograph / chest x ray) AND (deep learning / AI / CAD / CNN) | PUBYEAR > 2018; 13 tạp chí: Radiology, Radiology AI, Lancet Digital Health, Nature Medicine, Nature Communications, npj Digital Medicine, MedIA, IEEE TMI, European Radiology, JAMA Netw Open, Lancet Oncology, JTO, Chest | 116 (đọc 50 bài trích dẫn nhiều nhất) | 20 mới, 1 nâng cấp (L35) |
| Q3 | (VLM / MLLM / LVLM) AND (CXR) AND (grounding / localization / bounding box / lesion detection) AND (fine-tuning / LoRA / PEFT / instruction tuning) | PUBYEAR > 2022 | 23 | 4 mới, 1 nâng cấp (L9) |

Lần đầu chạy Q1 trả về 0 kết quả, vì Scopus không nhận ký tự `*` sau dấu gạch nối (ví dụ `"chest x-ray*"`). Mình đã viết lại bằng các dạng số ít/số nhiều liệt kê rõ.

**Loại:** bài về CT, X-quang nhi khoa, bệnh bụi phổi silic, tăng áp động mạch phổi, tràn khí màng phổi; NLP báo cáo không liên quan đến phổi; mục lục kỷ yếu hội nghị; bài trùng corpus.

---

## 2. Cập nhật bài đã có

| Mã | Thay đổi |
|---|---|
| L9 | RadVLM (Deperrois) nay đã đăng ở *Scientific Reports* 2026, DOI 10.1038/s41598-026-66181-1. Trước đó corpus ghi là arXiv. |
| L35 | NODE21 (Sogancioglu) nay đã đăng ở *IEEE TMI* 2024, DOI 10.1109/tmi.2024.3382042. Trước đó corpus ghi là arXiv. |

---

## 3. Bài mới (L63–L92, 30 bài)

Cột "Hạng" ghi hạng nguồn theo hiểu biết chung (SJR/CORE), chưa tra lại từng bài.

### T1 / T2: VLM và LLM với ung thư phổi trên X-quang ngực

| Mã | Bài | Nguồn | Năm | Trích dẫn | DOI | Hạng | Cụm |
|---|---|---|---|---|---|---|---|
| L63 | Hong EK et al. Domain-specific multimodal generative AI model for CXR report generation | Radiology | 2025 | 36 | 10.1148/radiol.241476 | Q1 | T2 |
| L64 | Volkov et al. VLM fine-tuning via LoRA for structured medical reports for lung X-ray scans | SCM 2025 (kỷ yếu IEEE) | 2025 | 3 | 10.1109/scm66446.2025.11060140 | Hội nghị thấp | T2 (gần nhất với phương pháp) |
| L65 | Zhang et al. Automated CXR disease screening using LLMs and deep CNNs on MIMIC-CXR | Frontiers in Digital Health | 2026 | 0 | 10.3389/fdgth.2026.1883337 | Q2 | T2 |
| L66 | Hoq ME et al. Native-resolution 2D embeddings for lung cancer classification with RAD-DINO | J Imaging Inform Med | 2025 | 4 | 10.1007/s10278-025-01748-4 | Q1/Q2 | T3 |
| L67 | Hayeso et al. Lightweight hybrid foundation model for lung cancer prognosis on low-dose CXR | Transl Lung Cancer Res | 2026 | 0 | 10.21037/tlcr-2025-aw-1299 | Q1/Q2 | T8 |

**Ghi chú:**
- L64 là bài duy nhất tìm thấy có tinh chỉnh LoRA một VLM trên X-quang phổi, nhưng mục tiêu là sinh báo cáo có cấu trúc, không phải phát hiện hay định vị nốt, và nguồn ở hạng thấp. Cần đọc toàn văn để xác nhận nó không làm mất khoảng trống của đề tài.
- L66 chạy embedding ở độ phân giải gốc, liên quan trực tiếp đến rủi ro "nốt nhỏ mất khi resize" trong spec.

### T3 / T4: Biểu diễn thị giác–ngôn ngữ và định vị

| Mã | Bài | Nguồn | Năm | Trích dẫn | DOI | Hạng | Cụm |
|---|---|---|---|---|---|---|---|
| L68 | Bannur S et al. Learning to exploit temporal structure for biomedical VLP (BioViL-T) | CVPR | 2023 | 289 | 10.1109/cvpr52729.2023.01442 | A* | T3 |
| L69 | Gao et al. S2D-Align: anatomically grounded radiology report generation | AAAI | 2026 | 1 | 10.1609/aaai.v40i36.40335 | A* | T4 |
| L70 | Zhang et al. Generalized medical phrase grounding | IEEE TMI | 2026 | 0 | 10.1109/tmi.2026.3707568 | Q1 | T4 |
| L71 | Xu et al. MedGround-R1: medical image grounding via spatial-semantic rewarded GRPO | MICCAI (LNCS) | 2025 | 0 | 10.1007/978-3-032-04971-1_37 | A (MICCAI) | T4 |

### T6: CAD học sâu phát hiện nốt và ung thư phổi

| Mã | Bài | Nguồn | Năm | Trích dẫn | DOI | Hạng |
|---|---|---|---|---|---|---|
| L72 | Cid YD et al. Open-source DNNs for comprehensive CXR reading (X-Raydar) | Lancet Digital Health | 2024 | 43 | 10.1016/s2589-7500(23)00218-2 | Q1 |
| L73 | Pesce E et al. Detecting CXRs containing pulmonary lesions with visual attention networks | Medical Image Analysis | 2019 | 95 | 10.1016/j.media.2018.12.007 | Q1 |
| L74 | Jang S et al. DL detection algorithm for reducing overlooked lung cancers on CXR | Radiology | 2020 | 71 | 10.1148/radiol.2020200165 | Q1 |
| L75 | Lee JH et al. DL vs radiologists for lung cancer detection on CXR in a health screening population | Radiology | 2020 | 76 | 10.1148/radiol.2020201240 | Q1 |
| L76 | Sung J et al. Added value of a DL detection system for multiple major findings on CXR | Radiology | 2021 | 68 | 10.1148/radiol.2021202818 | Q1 |
| L77 | Hwang EJ et al. DL for detection of pulmonary metastasis on CXR | Radiology | 2021 | 33 | 10.1148/radiol.2021210578 | Q1 |
| L78 | Shen et al. Image synthesis with disentangled attributes for CXR nodule augmentation and detection | Medical Image Analysis | 2023 | 37 | 10.1016/j.media.2022.102708 | Q1 |
| L79 | Kim et al. Multicentre external validation of a commercial AI software for CXR | European Radiology | 2023 | 32 | 10.1007/s00330-022-09315-z | Q1 |
| L80 | van Leeuwen KG et al. Commercial AI software for radiograph lung nodule detection and bone age prediction | Radiology | 2024 | 40 | 10.1148/radiol.230981 | Q1 |
| L81 | Nam JG et al. Undetected lung cancer at PA chest radiography: potential role of a DL algorithm | Radiology: Cardiothoracic Imaging | 2020 | 32 | 10.1148/ryct.2020190222 | Q1/Q2 |

### Phương pháp: độ phân giải và mất cân bằng lớp

Nhóm này hỗ trợ trực tiếp các quyết định thiết kế trong spec.

| Mã | Bài | Nguồn | Năm | Trích dẫn | DOI | Liên quan spec |
|---|---|---|---|---|---|---|
| L82 | Sabottke CF, Spieler BM. The effect of image resolution on DL in radiography | Radiology: AI | 2020 | 289 | 10.1148/ryai.2019190015 | Rủi ro resize làm mất nốt nhỏ; chọn độ phân giải đầu vào |
| L83 | Kim et al. Optimal matrix size of CXR for CAD of lung nodule or mass with DL | European Radiology | 2020 | 22 | 10.1007/s00330-020-06892-9 | Như trên, riêng cho nốt/khối |
| L84 | Mosquera C et al. Class imbalance on medical image classification: evaluation practices for discrimination and calibration | European Radiology | 2024 | 87 | 10.1007/s00330-024-10834-0 | C3 (cân bằng lớp); chọn chỉ số khi ca dương hiếm |

### T7: Tương tác người–AI và tác động lâm sàng

| Mã | Bài | Nguồn | Năm | Trích dẫn | DOI | Hạng |
|---|---|---|---|---|---|---|
| L85 | Sim Y et al. DCNN software improves radiologist detection of malignant lung nodules on CXR | Radiology | 2020 | 233 | 10.1148/radiol.2019182465 | Q1 |
| L86 | Homayounieh F et al. AI-based CXR model on human nodule detection accuracy (multicenter) | JAMA Netw Open | 2021 | 103 | 10.1001/jamanetworkopen.2021.41096 | Q1 |
| L87 | Ahn JS et al. AI-aided CXR interpretation and reader performance | JAMA Netw Open | 2022 | 126 | 10.1001/jamanetworkopen.2022.29289 | Q1 |
| L88 | Bernstein MH et al. Can incorrect AI results impact radiologists? (CXR lung nodules) | European Radiology | 2023 | — | 10.1007/s00330-023-09747-1 | Q1 |
| L89 | Bennani S et al. Using AI to improve radiologist performance in detecting abnormalities on CXR | Radiology | 2023 | 94 | 10.1148/radiol.230860 | Q1 |
| L90 | Lee et al. Effect of human-AI interaction on detection of malignant lung nodules on CXR | Radiology | 2023 | 44 | 10.1148/radiol.222976 | Q1 |

### T8 / T10: Nguy cơ và tổng quan

| Mã | Bài | Nguồn | Năm | Trích dẫn | DOI | Cụm |
|---|---|---|---|---|---|---|
| L91 | Raghu VK et al. DL model predicting lung cancer risk from CXR and EMR data | JAMA Netw Open | 2022 | 31 | 10.1001/jamanetworkopen.2022.48793 | T8 |
| L92 | Aggarwal R et al. Diagnostic accuracy of DL in medical imaging: systematic review and meta-analysis | npj Digital Medicine | 2021 | 896 | 10.1038/s41746-021-00438-z | T10 |

---

## 4. Nhận xét

1. **Khoảng trống vẫn còn.** Q1 và Q3 không tìm thấy bài nào ở tạp chí Q1/Q2 hay hội nghị A* tinh chỉnh LoRA một VLM để phát hiện hoặc định vị nốt phổi trên X-quang ngực. Bài gần nhất là L64, nhưng làm sinh báo cáo và ở hội nghị hạng thấp.
2. **Bằng chứng cho mốc CAD mạnh hơn.** T6 và T7 thêm 16 bài Radiology, Lancet Digital Health, JAMA Netw Open và MedIA, gồm kiểm định ngoài đa trung tâm (L79, L80) và nghiên cứu đa người đọc (L85–L90). Mốc "CAD mạnh, VLM yếu" vì vậy có cơ sở vững hơn.
3. **Rủi ro độ phân giải có tài liệu.** L82, L83 và L66 cho thấy độ phân giải đầu vào ảnh hưởng mạnh đến phát hiện nốt. Spec nên ghi rõ độ phân giải ảnh của MedGemma và Qwen2.5-VL, và cân nhắc thêm một thí nghiệm phụ chia ô hoặc cắt vùng phổi.
4. **Định vị bằng VLM chủ yếu ở hội nghị.** Các bài grounding (L69–L71) đều ở AAAI, MICCAI và IEEE TMI, và chủ yếu làm phrase grounding trên MIMIC/MS-CXR, không riêng cho nốt.

## 5. Việc tiếp theo

- **Trạng thái tải PDF (2026-11-12): đã xong.** Có 22/30 bài toàn văn trong `papers/T*/`.
  - Quyền truy cập tổ chức là của VŠB – Technical University of Ostrava, dùng được cho Springer, Elsevier/Lancet và IEEE.
  - Ba bài RSNA L80, L81, L82 lấy được PDF qua Undermind.
  - 8 bài RSNA còn lại (L63, L74–L77, L85, L89, L90) chỉ có abstract, ghi trong `papers/_RSNA-abstract-notes.md`. RSNA không hỗ trợ đăng nhập qua eduID.cz hay UIT/VNU.
  - DOI của L80 đã xác nhận qua Undermind: 10.1148/radiol.230981.
- Đọc toàn văn L64 (kiểm tra khoảng trống), L80 và L82.
- Gộp L63–L92 vào `corpus.md` và `03-thematic-clusters.md` sau khi đọc tóm tắt.
