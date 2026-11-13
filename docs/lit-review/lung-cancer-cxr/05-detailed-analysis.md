# Phân tích chi tiết từng bài: VLM/LLM cho nốt và ung thư phổi trên X-quang ngực

*Phiên bản 1, ngày 2026-11-12. Phạm vi: 90 bài duy nhất trong `corpus.md` (L1–L92). L46 trùng L2 và L54 trùng L3 nên mỗi cặp chỉ phân tích một lần. Thứ tự cụm theo `03-thematic-clusters.md`.*

## Quy ước mức đọc (trung thực)

| Thẻ | Nghĩa |
|---|---|
| **[FT]** | Đọc toàn văn PDF cục bộ **trong lượt này**, bằng `pdftotext`. Tập trung vào abstract, phương pháp, kết quả, hạn chế và các bảng chính. Không đọc từng dòng phụ lục. |
| **[AB]** | Chỉ dựa trên abstract, gồm trích dẫn nguyên văn trong `corpus.md`, `papers/_RSNA-abstract-notes.md` và `03-thematic-clusters.md`. |
| **[T]** | Chỉ có tiêu đề hoặc metadata. |

**Lưu ý về nguồn**
- **L1, L47, L48, L49** đã được đọc toàn văn ở một phiên trước, qua Undermind `read_pdfs`. Ghi chú của phiên đó nằm trong `corpus.md` §6. Lượt này không đọc lại được vì không có PDF cục bộ, và Undermind MCP đòi đăng nhập OAuth mà worker không làm được. Vì vậy các bài này gắn **[AB]**, kèm câu "theo ghi chú toàn văn phiên trước".
- **L39 (CheXpert):** file `papers/T9-datasets/[16]_Irvin2019_CheXpert...pdf` thực ra chứa một bài khác: Saralajew et al., *Robustness of Generalized LVQ Models against Adversarial Attacks*, arXiv 1902.00577. Vì vậy L39 gắn [AB]. **Cần thay file PDF.**
- **L40:** PDF cục bộ là bài MIMIC-CXR ở *Scientific Data* (bản DICOM kèm báo cáo), không phải bài MIMIC-CXR-JPG trên arXiv được liệt kê trong corpus. Hai bài cùng nhóm tác giả và cùng bộ dữ liệu. L40 gắn [AB], và có ghi chú những gì đọc được từ bài Sci Data.
- Các PDF khác trong `papers/` **không thuộc L1–L92** nên không phân tích ở đây: [1] CheXNet, [2] CheXzero (đã loại trong corpus §4), [4] LLaVA-Med, [5] Med-PaLM M, [6] RadFM, [10] Qwen2.5-VL, [22] BioViL, [23] GEMeX, [29] Brin 2025, [94] M4CXR, [104] Look-and-Mark, và nhóm phương pháp [11] LoRA, [12] PeFoMed, [43] Geirhos, [44] Zech, [55] ROAR, [56] Fong, [57] Adebayo, [108] Jimenez-Sanchez. Đây là nguồn tham khảo phương pháp, nên cân nhắc thêm vào corpus.
- Quy tắc số liệu: con số nào không có trong nguồn đã đọc thì ghi "không nêu trong nguồn đã đọc".

---

## T1. Đánh giá zero-shot MLLM/VLM cho nốt và ung thư phổi trên CXR

> *Cập nhật 2026-11-12: đã đọc toàn văn cụm T1. Nguồn: Undermind `read_pdfs` cho L1, L3, L4, L52; bản preprint medRxiv toàn văn cho L2/L46, kèm Bảng 2 và Bảng 3; trang Thieme cho L51. L51 và L53 vốn chỉ là tóm tắt hội nghị. L62 không truy cập được: AJR bị chặn, Undermind không có PDF.*

### L1 — Khovanova et al. 2026, *Frontiers in Digital Health* [FT]

**Bối cảnh & luận điểm chính.** So sánh 9 mô hình phát hiện nốt phổi trên CXR:
- 5 MLLM đa năng: Llama 3.2 Vision 90B, Claude 3.5 Sonnet, Claude 3.7 Sonnet, Gemini 2.0 Pro, Perplexity.
- 2 VLM chuyên CXR: CXR-LLaVA, XrayGPT.
- BiomedCLIP.
- Agent MedRAX, dùng LLM Qwen2.5 14B.

**Phương pháp**
- **Dữ liệu:** MosMedData loại VII (Moscow, 2020–2022).
  - Dự kiến 100 ảnh (50 có nốt, 50 bình thường). Thực tế đánh giá 83 ảnh (38 có nốt, 45 bình thường), vì 17 ảnh bị lỗi xử lý ở một số mô hình.
  - Nốt đặc 6–30 mm, xác nhận bằng CT trong 14 ngày. Không có phân bố kích thước.
- **Đầu vào:** ảnh PNG tư thế AP; độ phân giải không nêu.
- **Prompt cho MLLM:** đóng vai bác sĩ, trả lời 1 hoặc 0 cho "có nốt?". Chạy trên LMArena, temperature 0,7, mỗi ảnh chạy một lần.
- **Mô hình chuyên ngành:** mỗi mô hình một quy trình riêng.
  - CXR-LLaVA: chỉ tính dương khi vừa trả lời "có nốt" vừa mô tả đám mờ.
  - MedRAX: dùng xác suất lớp "Nodule".
  - BiomedCLIP: dùng xác suất lớp "lung cancer".
- **Ngưỡng:** chọn theo chỉ số Youden.

**Kết quả chính (Bảng 1)**

| Mô hình | AUC | Sensitivity | Specificity |
|---|---|---|---|
| BiomedCLIP | 0,743 | 0,737 | 0,689 |
| MedRAX | 0,704 | 0,553 | 0,844 |
| Claude 3.7 | 0,625 | 0,316 | 0,933 |
| Claude 3.5 | 0,592 | 0,474 | — |
| Gemini 2.0 Pro | 0,580 | 0,316 | — |
| Llama 3.2 90B | 0,582 | 0,342 | — |
| Perplexity | 0,569 | 0,316 | — |
| XrayGPT | 0,506 | 0,500 | 0,511 |
| CXR-LLaVA | 0,517 | 0,079 | 0,956 |

- Accuracy cao nhất là 0,711, đạt bởi cả BiomedCLIP và MedRAX.
- Bài **không đánh giá định vị** và **không phân tích theo kích thước nốt**.
- **Không thử MedGemma.** Qwen chỉ có mặt gián tiếp, làm LLM bên trong MedRAX.

**Hạn chế**
- Tác giả nêu: mỗi ảnh chỉ hỏi một lần; prompt khác nhau giữa các nhóm mô hình; dùng PNG thay DICOM; mẫu nhỏ; một trung tâm; kết quả mang tính thăm dò.
- Phê bình riêng:
  - Loại 17 ảnh có thể gây thiên lệch chọn mẫu.
  - Chọn ngưỡng Youden trên chính tập test làm hiệu năng của BiomedCLIP và MedRAX bị thổi phồng.
  - MLLM chỉ trả lời nhị phân, nên AUC của chúng không thực sự là AUC đầy đủ.

**Liên quan đề tài**
- Bằng chứng lõi cho §1 và E2.
- Mô hình có điểm liên tục (BiomedCLIP, MedRAX) tốt hơn mô hình sinh văn bản. Điều này ủng hộ việc đề tài trích xác suất hoặc logit từ VLM thay vì chỉ đọc văn bản.

### L2 (+L46) — Nishio et al. 2026, *AJR*; preprint medRxiv là L46 [FT, đọc qua preprint]

**Bối cảnh & luận điểm chính**
- So sánh zero-shot 6 VLM trên một benchmark ngoài chuẩn hóa là JSRT.
- Tác giả lập luận rằng quy đầu ra văn bản về nhị phân có/không có nốt sẽ tránh được nhược điểm của các chỉ số đánh giá văn bản.

**Phương pháp**
- **Dữ liệu:** 247 CXR từ 14 cơ sở; 154 có nốt (100 ác tính, 54 lành tính), 93 không có nốt.
- **Tiền xử lý:** ảnh gốc 2048×2048, 12-bit → tăng tương phản theo phân vị 2–98 → chuyển 8-bit JPEG, không cắt ảnh → resize theo yêu cầu đầu vào của từng mô hình.
- **Mô hình:** RadVLM, gpt-4o-mini, Qwen3-VL-8B-Instruct, **MedGemma-4b-it**, LLaVA-Rad, CheXpert Plus.
- **Prompt:** "Draft a concise report for this image". Nếu nhà phát triển có prompt khuyến nghị thì dùng prompt đó.
- **Chấm điểm:** hai bác sĩ, không biết mô hình nào, đọc báo cáo và quy về nhị phân. Cách diễn đạt mơ hồ (ví dụ "possible nodule") được tính là dương.
- **Thống kê:** McNemar với hiệu chỉnh Holm; phân tích theo 5 mức độ khó thấy của nốt (subtlety grade).

**Kết quả chính (Bảng 2)**

| Mô hình | Sensitivity | Specificity | Accuracy | F1 |
|---|---|---|---|---|
| RadVLM | 11,0% | 100% | 44,5% | 19,9% |
| gpt-4o-mini | 6,5% | 93,5% | 39,3% | — |
| Qwen3-VL-8B | 7,8% | 88,2% | 38,1% | — |
| **MedGemma-4b-it** | **5,2% (8/154)** | **100%** | **40,9%** | **9,9%** |
| LLaVA-Rad | 27,3% | 71,0% | 43,7% | 37,7% |
| CheXpert Plus | **0,6% (1/154)** | 95,7% | 36,4% | — |

- Như vậy, MedGemma đứng **áp chót** về sensitivity, không phải thấp nhất. Mô hình thấp nhất là CheXpert Plus.

**Kết quả theo độ khó thấy (Bảng 3)**

| Mô hình | Grade 5 (rõ) | Grade 4 | Grade 3 | Grade 2 | Grade 1 (rất mờ) |
|---|---|---|---|---|---|
| MedGemma | 3/12 (25,0%) | 3/38 (7,9%) | 1/50 (2,0%) | 1/29 (3,5%) | 0/25 |
| LLaVA-Rad | 83,3% | 44,7% | 20,0% | 13,8% | 4,0% |
| RadVLM | 58,3% | 23,7% | 2,0% | 0% | 0% |

- Ở grade 1, mọi mô hình đều phát hiện ≤ 12%.

**Hạn chế**
- Tác giả nêu:
  - Không có điểm liên tục, nên không có ROC và không tối ưu được ngưỡng.
  - Không có nghiên cứu với người đọc.
  - Không phân tích theo kích thước, vị trí hay tính chất lành/ác.
  - Chỉ dùng một tập ngoài.
  - Không so với CAD.
  - Prompt và quy trình suy luận khác nhau giữa các mô hình.
- Phê bình riêng: ảnh JSRT kiểu cũ và tỷ lệ nốt 62% tạo lệch miền so với dữ liệu huấn luyện, như tác giả cũng thừa nhận.

**Liên quan đề tài**
- Đây là mốc zero-shot trực tiếp của MedGemma-4b-it. Mô hình "bảo thủ": specificity 100%, sensitivity 5,2%. Sự bảo thủ này chính là lý do cần LoRA và cân bằng lớp.
- Bài dùng **Qwen3-VL-8B** chứ không phải Qwen2.5-VL-3B, nên cần ghi rõ khi so sánh.
- Bảng theo độ khó là khung trực tiếp cho phân tích phân tầng trên JSRT của đề tài.
- Spec cần sửa câu "MedGemma thấp nhất".

### L3 (=L54) — Chen CH et al. 2026, *Diagnostics* [FT]

DOI đúng: 10.3390/diagnostics16030376.

**Bối cảnh & luận điểm chính.** Bốn MLLM (Gemini 1.0, Gemini 1.5 Pro, GPT-4 Turbo, GPT-4o; phiên bản 06/2024) mô tả CXR ở dạng tự do.

**Phương pháp**
- **Dữ liệu:** 247 CXR thẳng từ NIH ChestX-ray. Hai bác sĩ hô hấp chọn ca **rõ và điển hình**, gồm 13 nhóm chẩn đoán.
- **Nhóm ác tính:** 67 ca, chia thành tổn thương đơn ổ (22), đa ổ (30) và phân bố trung tâm (15). **Không có nhóm "nốt" riêng.**
- **Prompt:** "Please describe the image"; tham số mặc định; mỗi ảnh chạy một lần.
- **Chấm:** đúng hoàn toàn, đúng một phần hoặc sai.

**Kết quả chính**
- **Phát hiện chung** (đúng hoàn toàn + đúng một phần): Gemini 1.5 Pro 51,0%, Gemini 1.0 47,4%, GPT-4o 40,5%, GPT-4 Turbo 25,9%.
- **Ác tính đơn ổ** (gần với nốt đơn độc nhất): chỉ phát hiện 4,5–18,2%; gộp các mô hình được 10,2%.
- **Ác tính đa ổ:** gộp được 55,0%.
- **Ác tính trung tâm:** 0–13,3%.
- Tràn dịch màng phổi được phát hiện tốt hơn khi lượng dịch nhiều hơn: 5% → 47,5%.

**Hạn chế**
- Tác giả nêu:
  - Chỉ chọn ca rõ, nên hiệu năng bị thổi phồng.
  - Không có bệnh sử.
  - Không tiền xử lý ảnh.
  - Kết quả gắn với phiên bản mô hình đã thử.
  - Cỡ nhóm không đều.
  - Một nguồn dữ liệu.
- Phê bình riêng: "đúng một phần" bao cả sai bên và sai chẩn đoán, nên tỷ lệ "phát hiện" bị nới rộng.

**Liên quan đề tài**
- Bằng chứng định lượng mạnh rằng tổn thương đơn ổ bị bỏ sót (khoảng 10%), còn đa ổ thì không (khoảng 55%).
- Ủng hộ phân tầng theo số ổ và kích thước.
- Gợi ý rằng cắt vùng hoặc chia ô ảnh có thể giúp.

### L4 — Cusick et al. 2026, *Cureus* [FT]

**Bối cảnh & luận điểm chính.** Pilot xem cách cho ChatGPT học trước một module dạy học (100 ảnh từ Radiopaedia) kèm khung diễn giải có cấu trúc có cải thiện việc đọc CXR không. Bài không nêu phiên bản mô hình.

**Phương pháp**
- 50 CXR, 10 chẩn đoán.
- Không nêu số ca ung thư phổi.
- Prompt: "Diagnose this X-ray image".
- Hai điều kiện: không có ngữ cảnh, và có ngữ cảnh cấu trúc.

**Kết quả chính**
- Accuracy chung tăng từ 28% lên 62%.
- **Ung thư phổi đứng yên ở 20%** trong cả hai điều kiện.
- Không có nhóm nốt riêng.

**Hạn chế**
- Tác giả nêu:
  - Mẫu nhỏ.
  - Dùng lại cùng bộ ảnh trước và sau can thiệp.
  - Chấm điểm một phần mang tính chủ quan.
  - Không tách được tác dụng của ví dụ minh họa và của khung prompt.
  - Prompt dễ vỡ và mô hình có tính ngẫu nhiên.
- Phê bình riêng: không nêu phiên bản mô hình, nên khó tái lập.

**Liên quan đề tài.** Ngữ cảnh cải thiện các bệnh lan tỏa nhưng **không** cải thiện được ung thư phổi. Điều này ủng hộ việc phải tinh chỉnh thị giác (LoRA) thay vì chỉ thiết kế prompt.

### L51 — Abou-Zeid et al. 2026, *RöFo* (tóm tắt hội nghị RöKo) [FT, toàn bộ nội dung là tóm tắt hội nghị]

**Bối cảnh & luận điểm chính.** Dùng hai VLM viết báo cáo CXR tự động. Bài bằng tiếng Đức.

**Phương pháp**
- 100 bệnh nhân (tuổi trung vị 74), mỗi nhóm 25 ca: tràn khí màng phổi, tràn dịch màng phổi, thâm nhiễm viêm phổi, khối phổi.
- Đầu vào gồm ảnh, **bệnh sử và câu hỏi lâm sàng**, với prompt chuẩn hóa.
- Mô hình: GPT-4.0 và Claude 3.5 Sonnet.
- Hai bác sĩ chấm: phát hiện nhị phân và chất lượng theo thang Likert 1–5.
- Thống kê: McNemar và Wilcoxon.

**Kết quả chính**
- Accuracy với bệnh lý chính: Claude 62%, GPT-4.0 51% (p = 0,04).
- Tràn khí màng phổi: 36% so với 4%.
- Thâm nhiễm: GPT-4.0 80%, Claude 72%.
- Khối phổi được "ít nhận đúng hơn". Tóm tắt **không có con số riêng cho nhóm khối**.

**Hạn chế**
- Chỉ là tóm tắt hội nghị.
- Một trung tâm.
- Prompt có bệnh sử, nên có thể gợi ý đáp án.

**Liên quan đề tài.** Thêm bằng chứng VLM yếu với khối phổi, ngay cả khi có bệnh sử.

### L52 — Lacaita et al. 2025, *Journal of Personalized Medicine* [FT]

**Bối cảnh & luận điểm chính.** ChatGPT-4o đọc X-quang ngực và bụng.

**Phương pháp**
- 257 CXR từ Radiopaedia, gồm 9 nhóm.
- 10 ca "lung tumor", **cố ý chọn khối > 3 cm** và dấu hiệu rõ.
- Prompt: "What is going on?".

**Kết quả chính**
- CXR: đúng 170/257 (66,15%).
- **Khối u phổi: 9/10 (90%).**

**Hạn chế**
- Tác giả nêu:
  - Chỉ thử một mô hình.
  - Không có bệnh sử.
  - Bộ ca bệnh chọn lọc.
  - Đánh giá an toàn mang tính chủ quan.
- Phê bình riêng: con số 90% chỉ áp dụng cho khối > 3 cm. Theo định nghĩa, nốt là tổn thương ≤ 3 cm, nên kết quả này không nói gì về nốt.

**Liên quan đề tài.** Phản ví dụ có điều kiện: khối lớn thì VLM phát hiện được. Điều này củng cố việc tách Nodule khỏi Mass trong phân tích phụ, dù nhãn dương tính gộp cả hai.

### L53 — Starychenko et al. 2025, *ERS Congress* (tóm tắt hội nghị, Eur Respir J suppl.) [AB, toàn bộ nội dung là tóm tắt hội nghị]
- **Bối cảnh & luận điểm chính:** Sensitivity và specificity của các chatbot AI tạo sinh khi đọc CXR.
- **Phương pháp:** 180 CXR, nhiều chatbot, trong đó có Mistral. Thiết kế chi tiết: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Sensitivity với nốt tối đa 41,3% (Mistral). Accuracy chung tối đa 38%.
- **Hạn chế:** Chỉ là tóm tắt hội nghị, chưa bình duyệt đầy đủ, ít chi tiết phương pháp.
- **Liên quan đề tài:** Bằng chứng cấp thấp nhưng cùng hướng: sensitivity với nốt thấp.

### L62 — Chu 2026, *AJR* (commentary) [T]
- **Bối cảnh & luận điểm chính:** Bình luận về L2, cảnh báo cần thận trọng khi dùng VLM sàng lọc nốt trên CXR. Chỉ có tiêu đề.
- **Phương pháp:** Không áp dụng (bài bình luận). Không nêu trong nguồn đã đọc.
- **Kết quả chính:** Không nêu trong nguồn đã đọc.
- **Hạn chế:** Chưa đọc được nội dung.
- **Liên quan đề tài:** Cho thấy cộng đồng chẩn đoán hình ảnh đã chú ý vấn đề này. Có thể trích ở phần động lực.

---

## T2. VLM, agent và mô hình tạo sinh chuyên cho X-quang ngực

### L5 — Lee S et al. 2025, *European Radiology* [AB]
- **Bối cảnh & luận điểm chính:** CXR-LLaVA, một MLLM mã nguồn mở diễn giải CXR và tự viết báo cáo.
- **Phương pháp:** 592.580 CXR công khai. Test nội bộ trên MIMIC và test ngoài. Chỉ số: F1 cho 6 dấu hiệu chính, tỷ lệ báo cáo tự động được chấp nhận.
- **Kết quả chính:** F1 trung bình 0,81 trên tập nội bộ MIMIC, 0,56 trên tập ngoài. Tỷ lệ tự viết báo cáo thành công 72,7%, thấp hơn mức 84,0% của bác sĩ.
- **Hạn chế:** F1 giảm mạnh khi ra ngoài phân phối huấn luyện. Abstract không có số liệu riêng cho nốt. Ở L1, CXR-LLaVA chỉ đạt sensitivity 0,079 với nốt.
- **Liên quan đề tài:** VLM chuyên CXR vẫn thất bại với nốt. Huấn luyện chuyên ngành chưa đủ nếu không nhắm vào tác vụ nốt.

### L6 — Chen Z et al. 2024, *arXiv* (CheXagent) [FT]
- **Bối cảnh & luận điểm chính:** Thiếu dữ liệu và benchmark lớn công khai cản trở việc xây mô hình nền cho CXR. Nhóm tác giả xây CheXinstruct (dữ liệu instruction), CheXagent (VLM) và CheXbench (benchmark 8 loại tác vụ).
- **Phương pháp:** Encoder SigLIP huấn luyện lại trên 1.052.257 cặp ảnh–văn bản. Độ phân giải đầu vào nâng từ 384 lên 512 bằng nội suy positional encoding. Projector là MLP hai lớp (1.024 → 2.560 chiều). Đánh giá lâm sàng với 8 bác sĩ chẩn đoán hình ảnh.
- **Kết quả chính:** Bác sĩ nội trú tiết kiệm 36% thời gian khi dùng bản nháp của CheXagent. Bác sĩ chính không thấy khác biệt thời gian. Hiệu suất viết báo cáo cải thiện ở 81% ca (nội trú) và 61% ca (bác sĩ chính).
- **Hạn chế:** Phần đã đọc không có kết quả riêng cho nốt. Độ phân giải 512 vẫn thấp cho nốt nhỏ (xem L82, L83). Benchmark do chính nhóm tác giả xây.
- **Liên quan đề tài:** Ứng viên baseline phụ. Là ví dụ về việc nâng độ phân giải encoder bằng nội suy positional encoding.

### L7 — Sellergren et al. 2026, *Nature Medicine* (MedGemma) [FT]
- **Bối cảnh & luận điểm chính:** MedGemma là bộ VLM y khoa mở dựa trên Gemma 3 (4B và 27B), kèm encoder MedSigLIP. Mục tiêu là mô hình nền y khoa cần ít dữ liệu tinh chỉnh.
- **Phương pháp:** Ảnh được resize về 896×896. Đánh giá trên QA văn bản, phân loại ảnh, VQA, sinh báo cáo MIMIC-CXR (RadGraph F1) và agent. Phân loại CXR zero-shot: macro F1 trên 5 bệnh (MIMIC, CheXpert) và trên 3 nhãn (CXR14: lung opacity, pneumothorax, fracture). Thí nghiệm tinh chỉnh dùng SFT toàn tham số, không dùng LoRA.
- **Kết quả chính:** Cải thiện 15,5–18,1% về phân loại CXR so với mô hình gốc. Macro F1 trên MIMIC: 88,9 (MedGemma 4B) so với 81,2 (Gemma 3 4B). Trên CheXpert ngoài phân phối: 48,1 so với 32,6. Tinh chỉnh đạt mức SOTA chuyên biệt cho pneumothorax trên SIIM-ACR.
- **Hạn chế:** Tác giả nêu: benchmark tự động chưa phản ánh giá trị thực tế, và một số benchmark đã bão hòa. Phê bình riêng: **không có nhãn nốt hay khối** trong các tác vụ phân loại CXR. MedSigLIP chỉ được đánh giá với nhãn "Lung lesion". Không thử LoRA.
- **Liên quan đề tài:** Mô hình chính. Độ phân giải 896 trùng mức tối ưu của L83. Khoảng trống: chưa ai đánh giá hay tinh chỉnh MedGemma cho nốt.

### L8 — Bannur et al. 2024, *arXiv* (MAIRA-2) [FT]
- **Bối cảnh & luận điểm chính:** Đề xuất "grounded report generation": mỗi câu trong báo cáo kèm bounding box, để báo cáo dễ kiểm chứng. Đồng thời đề xuất RadFact, khung đánh giá dùng LLM suy luận logic ở mức từng câu.
- **Phương pháp:** Encoder Rad-DINO đông cứng (ViT-B 87M), ảnh 518×518 với patch 14, cho ra 1.369 token; không dùng token CLS. Adapter MLP 4 lớp, LLM Vicuna-7B. Đầu vào thêm ảnh bên, ảnh và báo cáo trước, cùng các mục Indication, Technique, Comparison. Dữ liệu: MIMIC-CXR, PadChest, USMix (riêng). IU-Xray dùng làm tập test ngoài. Tọa độ box được mã hóa thành token.
- **Kết quả chính:** SOTA trên các benchmark sinh báo cáo hiện có. Số liệu chi tiết nằm trong bảng, không trích đầy đủ.
- **Hạn chế:** Grounding đánh giá trên mọi dấu hiệu, không riêng nốt. Dùng dữ liệu riêng. RadFact phụ thuộc LLM chấm điểm.
- **Liên quan đề tài:** Tiền lệ thiết kế cho đầu ra dạng box qua token và cho việc tránh token CLS (khớp cơ chế ở L47). Cũng là mẫu cho chỉ số "spatial precision".

### L9 — Deperrois et al. 2026, *Scientific Reports* (trước là arXiv) (RadVLM) [AB]
- **Bối cảnh & luận điểm chính:** VLM hội thoại đa tác vụ cho X-quang, có visual grounding.
- **Phương pháp:** Hơn 1 triệu cặp ảnh–instruction. Kiến trúc chi tiết: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Tác giả báo SOTA về hội thoại và grounding nhưng không nêu con số. Ở L2, RadVLM có accuracy cao nhất (44,5%) nhưng sensitivity chỉ 11,0%.
- **Hạn chế:** Có năng lực grounding nhưng không chuyển thành phát hiện nốt. Có thể do mất cân bằng nốt trong dữ liệu instruction.
- **Liên quan đề tài:** Mô hình có grounding vẫn bỏ sót nốt. Grounding chung chung là chưa đủ.

### L10 — Zambrano Chaves et al. 2025, *Nature Communications* (LLaVA-Rad) [FT]
- **Bối cảnh & luận điểm chính:** Mô hình đa phương thức nhỏ (7B), mở, dễ triển khai lâm sàng, sinh mục Findings từ CXR. Kèm CheXprompt, một chỉ số factuality dựa trên GPT-4.
- **Phương pháp:** 697.435 cặp ảnh–báo cáo từ 7 nguồn. Với nguồn chỉ có nhãn, dùng GPT-4 tổng hợp báo cáo. Huấn luyện encoder chuyên ngành BiomedCLIP-CXR, rồi nối với LLM qua adapter nhẹ. Đánh giá bằng ROUGE-L, F1-RadGraph, F1-CheXbert và CheXprompt, có đối chiếu với bác sĩ.
- **Kết quả chính:** So với Med-PaLM M (84B): ROUGE-L tốt hơn tương đối 12,1%, F1-RadGraph tốt hơn 10,1%. Med-PaLM M nhỉnh hơn một chút về F1-5 CheXbert (dưới 1%). LLaVA-Rad vượt GPT-4V.
- **Hạn chế:** Tác giả nêu: chưa sẵn sàng triển khai thời gian thực. Báo cáo tổng hợp bằng GPT-4 thiếu định vị và chi tiết tinh tế. Phê bình riêng: không có đánh giá riêng cho nốt; ở L2 sensitivity với nốt chỉ 27,3%.
- **Liên quan đề tài:** Baseline zero-shot mạnh nhất về sensitivity trong L2. Cho thấy hạn chế của dữ liệu thiếu định vị.

### L11 — Fallahpour et al. 2025, *arXiv* (MedRAX) [AB]
- **Bối cảnh & luận điểm chính:** Agent suy luận cho CXR, gọi các công cụ chuyên dụng như phân loại, phân đoạn và grounding.
- **Phương pháp:** Benchmark ChestAgentBench, 2.500 truy vấn. Danh sách công cụ: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Tác giả báo SOTA nhưng không nêu con số. Ở L1, MedRAX có accuracy 0,711 và AUC 0,704.
- **Hạn chế:** Hiệu năng phụ thuộc công cụ CNN bên dưới, nên khó quy cho LLM. Benchmark do chính nhóm tác giả xây.
- **Liên quan đề tài:** Gợi ý một hướng thay thế (VLM gọi detector) để so với LoRA trực tiếp.

### L12 — Thawakar et al. 2023, *arXiv* (XrayGPT) [AB]
- **Bối cảnh & luận điểm chính:** VLM y khoa tóm tắt và hỏi đáp về CXR.
- **Phương pháp:** Khoảng 217.000 tóm tắt báo cáo từ MIMIC và OpenI. Kiến trúc chi tiết: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Abstract không có kết quả định lượng. Được đánh giá trong L1, với kết quả thấp.
- **Hạn chế:** Không có đánh giá định lượng trong abstract. Dữ liệu là tóm tắt văn bản nên yếu về thị giác.
- **Liên quan đề tài:** Ví dụ VLM CXR thế hệ đầu. Chỉ dùng làm bối cảnh.

### L59 — Le-Duc et al. 2024, *arXiv* (LiteGPT) [AB]
- **Bối cảnh & luận điểm chính:** LVLM nhỏ vừa định vị vừa phân loại trên CXR.
- **Phương pháp:** VinDr-CXR (có box Nodule/Mass). Chi tiết: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Tác giả báo SOTA về phân loại trên VinDr-CXR nhưng không nêu con số.
- **Hạn chế:** Không có số liệu định vị riêng cho nốt. Chỉ là preprint.
- **Liên quan đề tài:** Tiền lệ gần nhất về VLM vừa định vị vừa phân loại trên VinDr, nguồn dữ liệu ứng viên của đề tài.

### L63 — Hong EK et al. 2025, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** AI tạo sinh đa phương thức chuyên ngành viết báo cáo CXR sơ bộ.
- **Phương pháp:** Khoảng 8,8 triệu cặp ảnh–báo cáo từ 42 bệnh viện. Test trên 2.145 ảnh (PadChest, Open-i, VinDr-CXR, MIMIC-CXR-JPG), 13 dấu hiệu. So với bác sĩ và GPT-4V.
- **Kết quả chính:** Bác sĩ chấp nhận không sửa 70,5% báo cáo của mô hình, so với 73,3% của bác sĩ và 29,6% của GPT-4V. Sensitivity tràn khí màng phổi 95,3%. RADPEER trung vị 4, so với 1 của GPT-4V.
- **Hạn chế:** Abstract không có sensitivity riêng cho nốt hay khối. Cần dữ liệu cực lớn, khó tái lập.
- **Liên quan đề tài:** Mô hình chuyên ngành vượt xa VLM đa năng, ủng hộ việc thích nghi miền.

### L64 — Volkov et al. 2025, *SCM 2025 (IEEE)* [FT]
- **Bối cảnh & luận điểm chính:** Tinh chỉnh LoRA một VLM nhỏ để sinh báo cáo X-quang phổi có cấu trúc gồm 4 trường: Image Labels, Gender, Age, Image Caption.
- **Phương pháp:** SmolVLM-Instruct (khoảng 2,26 tỷ tham số; SigLIP 384×384, 81 token ảnh). QLoRA 4-bit NF4 trên q/k/v/o và gate/up/down, 10.536.960 tham số học được (0,47%). Dữ liệu: tập con CXR của MultiCaRe (chú thích hình trong case report trên PMC); số mẫu: không nêu trong nguồn đã đọc. Chạy trên một A100 40 GB. Chỉ số: ROUGE, BLEU, METEOR, BERTScore.
- **Kết quả chính:** ROUGE-L tăng từ 0,128 lên 0,580. BLEU từ 0,019 lên 0,379. METEOR từ 0,142 lên 0,450. BERTScore F1 từ 0,820 lên 0,938.
- **Hạn chế:** Không có hạn chế tác giả tự nêu. Phần mở đầu hứa có đánh giá của bác sĩ nhưng không thấy kết quả. Phê bình riêng: chỉ số chủ yếu đo việc tuân thủ khuôn mẫu. Ví dụ minh họa chính của bài sai lâm sàng: kỳ vọng "Chilaiditis sign", mô hình sinh "right upper lobe consolidation". Không phát hiện hay định vị nốt. Chú thích PMC không phải báo cáo thật.
- **Liên quan đề tài:** Tiền lệ LoRA cho VLM nhỏ trên CXR, nhưng khác tác vụ. Phải phân biệt rõ ở §3. Cũng là cảnh báo: chỉ số NLG không đo được độ đúng lâm sàng.

### L65 — Zhang Q et al. 2026, *Frontiers in Digital Health* [FT]
- **Bối cảnh & luận điểm chính:** Dùng GPT-4o gán nhãn nhị phân "diseased/no disease" từ báo cáo CXR, rồi dùng nhãn đó huấn luyện CNN sàng lọc.
- **Phương pháp:** Tập con MIMIC-CXR v2.1.0 gồm 623 bệnh nhân và 2.105 nghiên cứu (2.097 sau lọc). Chia theo bệnh nhân 70/10/20, chạy 5 seed. Huấn luyện ResNet-18, DenseNet-121, EfficientNet-B1 và ConvNeXt-Tiny. Chất lượng nhãn kiểm tra trên 210 báo cáo so với một bác sĩ.
- **Kết quả chính:** Nhãn GPT-4o đạt accuracy 92,9% (diseased: precision 97,4%, recall 90,5%). ConvNeXt-Tiny tốt nhất: AUC 0,832 (95% CI 0,801–0,863), balanced accuracy 0,739. Mọi CNN đều có AUC trên 0,79.
- **Hạn chế:** Tác giả nêu: GPT-4o không thấy ảnh nên nhãn có nhiễu; một cơ sở; nhãn nhị phân quá đơn giản; 5 seed là ít. Phê bình riêng: không có nhãn nốt; mẫu nhỏ so với cỡ MIMIC; chỉ một bác sĩ đối chiếu.
- **Liên quan đề tài:** Gợi ý dùng LLM trích nhãn nốt từ báo cáo MIMIC để có thêm dữ liệu yếu. Cần kiểm định nhãn.

---

## T3. Bộ mã hóa thị giác–ngôn ngữ và giới hạn biểu diễn

### L14 — Zhang S et al. 2023, *arXiv* (BiomedCLIP) [AB]
- **Bối cảnh & luận điểm chính:** Mô hình nền CLIP y sinh tiền huấn luyện trên 15 triệu cặp ảnh–chú thích từ bài báo khoa học (PMC-15M).
- **Phương pháp:** Tiền huấn luyện đối chiếu (contrastive) trên PMC-15M. Đánh giá phân loại, truy hồi và VQA.
- **Kết quả chính:** Vượt BioViL trên RSNA pneumonia (con số: không nêu trong nguồn đã đọc). Abstract không có số liệu nốt CXR. Ở L1 đạt AUC tốt nhất (0,743).
- **Hạn chế:** Dữ liệu từ hình trong bài báo, khác ảnh lâm sàng thô. Gộp đặc trưng qua CLS làm mất tín hiệu nốt nhỏ (L47: AUC 0,688 với CLS).
- **Liên quan đề tài:** Baseline kiểu CLIP. Mô hình phân biệt (discriminative) vượt mô hình sinh văn bản ở L1, gợi ý thêm một đầu phân loại hoặc probe.

### L47 — Muthyala et al. 2026, *arXiv* [AB]
- **Bối cảnh & luận điểm chính:** Embedding đông cứng của mô hình nền làm mất tín hiệu tổn thương nhỏ trên CXR, do cách gộp đặc trưng toàn cục (CLS).
- **Phương pháp:** ChestX-Det10 (nốt). So sánh probe trên CLS với probe gộp patch-local (dùng ROI biết trước), cho BiomedCLIP và MedSigLIP. Phân tầng theo kích thước. (Theo ghi chú toàn văn phiên trước.)
- **Kết quả chính:** AUC mức box: BiomedCLIP 0,688 → 0,899; MedSigLIP 0,805 → 0,955. CLS mức ảnh của MedSigLIP: nốt nhỏ 0,627, nốt lớn 0,844.
- **Hạn chế:** Cách gộp patch-local cần ROI biết trước (oracle), nên chưa phải bộ phát hiện thực. Chỉ là preprint. Một dataset.
- **Liên quan đề tài:** Cơ chế thất bại cốt lõi. MedSigLIP là encoder của MedGemma: thông tin nốt còn trong patch nhưng mất khi gộp. Đây là luận cứ để LoRA và chia ô (tiling) có tác dụng.

### L50 — Fisher 2025, *medRxiv* [AB]
- **Bối cảnh & luận điểm chính:** Kết quả âm tính: mô hình nền thị giác–ngôn ngữ không chuyển giao tốt sang phân loại ảnh y khoa.
- **Phương pháp:** ChestX-ray14, phân loại đa nhãn. So VLM với CNN, có thử tinh chỉnh. Cỡ mẫu test: không nêu trong nguồn đã đọc.
- **Kết quả chính:** F1 của VLM tốt nhất là 0,203, so với 0,811 của CNN. Tinh chỉnh làm sensitivity sụp xuống dưới 15%.
- **Hạn chế:** Không có số liệu riêng cho nốt. Chưa bình duyệt. Chi tiết cách tinh chỉnh không rõ trong nguồn đã đọc; sensitivity sụp có thể do mất cân bằng lớp hoặc ngưỡng.
- **Liên quan đề tài:** Rủi ro trực tiếp cho kế hoạch LoRA: cần cân bằng lớp (C3), hiệu chỉnh ngưỡng và theo dõi sensitivity khi huấn luyện.

### L66 — Hoq et al. 2025, *Journal of Imaging Informatics in Medicine* [FT]
- **Bối cảnh & luận điểm chính:** Embedding 2D ở độ phân giải gốc từ RAD-DINO đông cứng, kèm MLP nhẹ, để phân loại nguy cơ ung thư phổi. **Lưu ý: đây là CT liều thấp, không phải CXR.** RAD-DINO được tiền huấn luyện trên CXR rồi áp dụng cho lát cắt CT.
- **Phương pháp:** Nhánh CT của NLST, kết cục trong 0–24 tháng. Chia cố định theo bệnh nhân. 25 lần rút tập test mất cân bằng (tỷ lệ bệnh khoảng 6%). Gộp trung bình theo lát và hiệu chỉnh isotonic. So với CT-Foundation.
- **Kết quả chính:** PR-AUC 0,705 sau hiệu chỉnh (0,554 thô). ROC-AUC 0,817 thô (0,736 sau hiệu chỉnh). Tập cân bằng (N = 1.984): ROC-AUC 0,912. CT-Foundation: AUC 0,793. Truy hồi Precision@5 = 0,853.
- **Hạn chế:** Tác giả nêu: một cohort, NLST công khai thiếu nhãn Lung-RADS, lệch miền khi tiền huấn luyện trên CXR rồi dùng cho CT. Phê bình riêng: `03-thematic-clusters.md` coi bài này là bằng chứng về độ phân giải trên CXR. Điều đó **không đúng modality**, cần sửa.
- **Liên quan đề tài:** Hữu ích về phương pháp đánh giá ở tỷ lệ bệnh thực tế (PR-AUC, hiệu chỉnh xác suất), không phải bằng chứng độ phân giải CXR.

### L68 — Bannur et al. 2023, *CVPR* (BioViL-T) [FT]
- **Bối cảnh & luận điểm chính:** Tiền huấn luyện thị giác–ngôn ngữ thường bỏ qua ảnh và báo cáo trước. BioViL-T khai thác cấu trúc thời gian bằng encoder lai CNN–Transformer nhiều ảnh, huấn luyện cùng encoder văn bản (CXR-BERT).
- **Phương pháp:** MIMIC-CXR. Stem ResNet-50 kèm transformer không gian–thời gian. Đánh giá phân loại tiến triển, phrase grounding trên MS-CXR (mIoU, CNR) và sinh báo cáo. Công bố MS-CXR-T: 1.326 cặp ảnh cho 5 dấu hiệu, và 361 cặp câu.
- **Kết quả chính:** Phrase grounding: CNR 1,33 so với 1,07 của BioViL; mIoU 0,243 so với 0,229. SOTA về phân loại tiến triển và sinh báo cáo.
- **Hạn chế:** mIoU tuyệt đối còn thấp (khoảng 0,24). Không đánh giá riêng nốt. Phụ thuộc có ảnh trước.
- **Liên quan đề tài:** Mốc grounding của mô hình CLIP-like. Nếu có ảnh trước, so sánh theo thời gian là hướng mở rộng.

---

## T4. Định vị, grounding và benchmark nhận thức/VQA

### L48 — Gosai et al. 2025, *PMLR (ML4H)* [AB]
- **Bối cảnh & luận điểm chính:** MLLM không chỉ cần chẩn đoán đúng mà còn phải chỉ ra vị trí bệnh lý trên CXR. Bài đánh giá khả năng định vị của chúng.
- **Phương pháp:** 9 dấu hiệu, có lung lesion. So sánh GPT-4, GPT-5, MedGemma, một CNN và bác sĩ. Chỉ số: hit rate, tức vị trí dự đoán rơi vào vùng đúng. (Theo ghi chú toàn văn phiên trước.)
- **Kết quả chính:** Hit rate với lung lesion: GPT-4 35,7%, GPT-5 35,7%, MedGemma 14,3%, CNN 29,0%, bác sĩ 85,0%. Trung bình 9 dấu hiệu: GPT-5 đạt 49,7%.
- **Hạn chế:** Số ca lung lesion nhỏ (số cụ thể: không nêu trong nguồn đã đọc). Hit rate chỉ xét một điểm, không có IoU. Không có tinh chỉnh.
- **Liên quan đề tài:** Con số duy nhất về định vị lung lesion của MedGemma (14,3%). Là mốc trực tiếp cho §8.2.

### L49 — Choi et al. 2026, *arXiv* (CheXpercept) [AB]
- **Bối cảnh & luận điểm chính:** Benchmark nhận thức tổn thương ở nhiều mức: phát hiện thô, chỉnh đường viền, thuộc tính.
- **Phương pháp:** VLM y khoa và đa năng. Benchmark **không có nốt hay khối** (theo ghi chú toàn văn phiên trước). Cỡ mẫu: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Phát hiện thô khoảng 92%. Chỉnh đường viền tối đa 2,9%. Thuộc tính tối đa 25,7%. VLM y khoa không hơn VLM đa năng.
- **Hạn chế:** Không bao phủ nốt. Chỉ là preprint.
- **Liên quan đề tài:** Mẫu thiết kế kiểm toán nhiều mức ("có/không" → "ở đâu" → "hình dạng"). Áp dụng được cho §8.3.

### L42 — Pal et al. 2025, *arXiv* (ReXVQA) [FT]
- **Bối cảnh & luận điểm chính:** Benchmark VQA lớn nhất cho X-quang ngực, với câu hỏi đa dạng thay vì theo khuôn mẫu.
- **Phương pháp:** Khoảng 696.000 câu trắc nghiệm trên 160.000 nghiên cứu từ 4 hệ thống y tế Mỹ. 5 kỹ năng: negation 36,5%, presence 36,1%, differential 20,9%, location, geometric. 8 MLLM, có MedGemma-4B-it và Qwen2.5-VL. Nghiên cứu người đọc: 3 bác sĩ nội trú trên 200 ca.
- **Kết quả chính:** MedGemma đạt 83,24% chung. Trên 200 ca, MedGemma đạt 83,84% so với 77,27% của bác sĩ nội trú tốt nhất. Con số của Qwen2.5-VL: không trích trong phần đã đọc.
- **Hạn chế:** Câu hỏi trắc nghiệm sinh từ báo cáo nên kiểm tra hiểu ngôn ngữ nhiều hơn nhìn ảnh. Chỉ so với bác sĩ nội trú. Không có tầng riêng cho nốt.
- **Liên quan đề tài:** Giải thích nghịch lý: MedGemma mạnh ở VQA nhưng sensitivity với nốt chỉ 5,2% (L2). Benchmark tổng hợp che mất điểm yếu với nốt.

### L43 — Nguyen DH et al. 2025, *arXiv* (VinDr-CXR-VQA) [AB]
- **Bối cảnh & luận điểm chính:** Bộ VQA có giải thích và grounding dựng trên box của VinDr-CXR.
- **Phương pháp:** 17.597 cặp hỏi–đáp, 4.394 ảnh có box. Học đa tác vụ. Thử MedGemma-4B-it.
- **Kết quả chính:** MedGemma-4B-it đạt F1 0,624, tăng 11,8% so với baseline.
- **Hạn chế:** Abstract không có tầng riêng cho nốt. Chỉ là preprint.
- **Liên quan đề tài:** Nguồn dữ liệu có box để tinh chỉnh hoặc đánh giá grounding của MedGemma.

### L44 — Wei et al. 2026, *arXiv* (REVEAL-CXR, RSNA) [AB]
- **Bối cảnh & luận điểm chính:** Benchmark của RSNA cho LLM trên CXR bệnh tim–phổi.
- **Phương pháp:** 200 nghiên cứu, 12 nhãn, 17 bác sĩ.
- **Kết quả chính:** Mô tả benchmark. Kết quả mô hình: không nêu trong nguồn đã đọc.
- **Hạn chế:** Nhỏ. Chưa xác nhận có nhãn nốt hay khối.
- **Liên quan đề tài:** Tập đánh giá ngoài tiềm năng, nếu có nhãn nốt.

### L69 — Gao et al. 2026, *AAAI* (S2D-Align) [FT]
- **Bối cảnh & luận điểm chính:** SFT chuẩn chỉ căn chỉnh ở mức cả cặp ảnh–báo cáo, nên báo cáo thường theo khuôn và thiếu neo giải phẫu. S2D-Align căn chỉnh từ nông đến sâu: cặp ảnh–báo cáo → báo cáo tham chiếu → cụm từ khóa giải phẫu.
- **Phương pháp:** LLM Vicuna-7B-v1.5 **tinh chỉnh bằng LoRA** (mọi lớp tuyến tính, rank 16, alpha 16, dropout 0,1). Encoder văn bản BiomedVLP-CXR-BERT. Adapter có bộ nhớ (8-head cross-attention, MLP 3 lớp). MIMIC-CXR và IU X-Ray. Chỉ số: BLEU, ROUGE-L, chỉ số lâm sàng.
- **Kết quả chính:** Tác giả báo SOTA trên cả hai tập. Con số cụ thể không trích được chắc chắn từ bảng trong bản text, nên ghi: không nêu trong nguồn đã đọc.
- **Hạn chế:** Chỉ có chỉ số sinh văn bản, không đo định vị. Không riêng nốt.
- **Liên quan đề tài:** **Tiền lệ LoRA cho LLM trong sinh báo cáo CXR ở hội nghị A*.** Bài này làm giảm tính "duy nhất" của L64 mà `03` đã ghi. Cấu hình rank 16 tham khảo được.

### L70 — Zhang W et al. 2026, *IEEE TMI* (MedGrounder) [FT]
- **Bối cảnh & luận điểm chính:** Phrase grounding y khoa truyền thống giả định mỗi cụm từ ứng với đúng một box. Bài định nghĩa lại thành GMPG: mỗi câu ứng với 0, 1 hoặc nhiều vùng có điểm.
- **Phương pháp:** MedGrounder, huấn luyện hai giai đoạn. (1) Tiền huấn luyện trên GMPG-ImaGenome: câu báo cáo nối với box giải phẫu, làm sạch bằng LLM (loại khoảng 80% box). (2) Tinh chỉnh trên box do người gán (PadChest-GR, MS-CXR). Chỉ số: Precision@(F1=1, IoU ≥ 0,5), N-Acc, Center-Hit F1.
- **Kết quả chính:** Chuyển giao zero-shot tốt. SOTA trên PadChest-GR, cạnh tranh trên MS-CXR, lợi nhiều nhất ở cụm từ nhiều vùng, cần ít box người gán hơn. MDETR tinh chỉnh trực tiếp gần như bằng 0. Trong MS-CXR, khoảng 23% cụm từ có nhiều vùng.
- **Hạn chế:** Không có tầng riêng cho nốt. Nhãn yếu từ detector giải phẫu.
- **Liên quan đề tài:** Chỉ số cho trường hợp 0 hoặc nhiều nốt (N-Acc khi ảnh không có nốt). Rất phù hợp cho §8.2 vì một ảnh có thể có nhiều nốt.

### L71 — Xu et al. 2025, *MICCAI* (MedGround-R1) [FT]
- **Bối cảnh & luận điểm chính:** Dùng GRPO (kiểu DeepSeek-R1) cho grounding ảnh y khoa mà không cần chú thích chain-of-thought.
- **Phương pháp:** Tinh chỉnh **Qwen2.5-VL** bằng RL với phần thưởng không gian (IoU) cộng phần thưởng ngữ nghĩa (MedCLIP so ROI với câu mô tả). Template Chain-of-Box. 5.000 bước, lr 1e-6, G = 4, β = 0,04. MS-CXR, ChestX-ray8, M3D-RefSeg. Chỉ số: Acc (IoU > 0,5) và mIoU.
- **Kết quả chính:** MS-CXR mIoU 79,02, so với các baseline 52,7–62,8 (ví dụ CausalCLIPSeg 62,77). SOTA trên cả ba tập.
- **Hạn chế:** Phần đã đọc không có mục hạn chế riêng. Phê bình riêng: MS-CXR cho trước cụm từ, nên đây là grounding có điều kiện, không phải phát hiện. Không riêng nốt. Cỡ mô hình Qwen2.5-VL: không nêu trong nguồn đã đọc.
- **Liên quan đề tài:** Qwen2.5-VL (VLM phụ của đề tài) học được box tốt sau RL. Gợi ý thêm hướng GRPO bên cạnh LoRA-SFT.

---

## T5. LLM chỉ đọc văn bản báo cáo X-quang

### L55 — Mukherjee et al. 2023, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** Dùng LLM cục bộ Vicuna để gán nhãn báo cáo CXR mà vẫn bảo vệ quyền riêng tư.
- **Phương pháp:** 13 dấu hiệu, so với CheXpert labeler và CheXbert. Cỡ mẫu: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Kappa trung vị 0,57–0,64. AUC trung vị 0,84.
- **Hạn chế:** Abstract không có số riêng cho lung lesion. Chuẩn so sánh cũng là công cụ tự động.
- **Liên quan đề tài:** Cách trích nhãn nốt từ báo cáo MIMIC bằng LLM mở.

### L56 — Savage et al. 2024, *AJR* [AB]
- **Bối cảnh & luận điểm chính:** So LLM đa năng với công cụ NLP chuyên ngành trong trích nhãn báo cáo CXR.
- **Phương pháp:** Chi tiết: không nêu trong nguồn đã đọc.
- **Kết quả chính:** GPT-4 vượt công cụ NLP chuyên ngành. Con số: không nêu trong nguồn đã đọc.
- **Hạn chế:** Không có số liệu trong nguồn đã đọc.
- **Liên quan đề tài:** Ủng hộ dùng LLM để dựng nhãn yếu.

### L57 — Park et al. 2024, *PLOS ONE* [AB]
- **Bối cảnh & luận điểm chính:** Trích dữ liệu 7 bệnh phổi lớn, có ung thư phổi, từ báo cáo tự do bằng GPT-4.
- **Phương pháp:** 1.800 báo cáo CXR và CT.
- **Kết quả chính:** Sensitivity 0,71–1,00, specificity 0,89–1,00.
- **Hạn chế:** Abstract không tách CXR với CT. Một trung tâm.
- **Liên quan đề tài:** Bằng chứng LLM trích được "lung cancer" từ báo cáo CXR.

### L58 — Yasaka et al. 2024, *Journal of Imaging Informatics in Medicine* [AB]
- **Bối cảnh & luận điểm chính:** LLM tinh chỉnh lọc ra bệnh nhân ung thư phổi chờ điều trị từ báo cáo trên PACS.
- **Phương pháp:** Modality: không nêu trong nguồn đã đọc. So với bác sĩ.
- **Kết quả chính:** Accuracy 0,983 so với 0,969 của bác sĩ. Thời gian 46 giây so với 1.538–2.539 giây.
- **Hạn chế:** Không rõ modality. Một cơ sở.
- **Liên quan đề tài:** Tiền lệ tinh chỉnh LLM cho ung thư phổi, ở nhánh văn bản.

### L15 — Gamble et al. 2024, *Canadian Association of Radiologists Journal* [AB]
- **Bối cảnh & luận điểm chính:** GPT-3.5 và GPT-4 áp dụng guideline Fleischner cho nốt phổi tình cờ phát hiện.
- **Phương pháp:** 120 mô tả nốt hư cấu. So có và không có guideline trong prompt, và một bản GPT-3.5 tinh chỉnh.
- **Kết quả chính:** Không có guideline: GPT-3.5 đạt 0,058, GPT-4 đạt 0,15. Có guideline: 0,42 và 0,66. GPT-3.5 tinh chỉnh: 0,46.
- **Hạn chế:** Dữ liệu tổng hợp, không phải CXR, mô hình đã cũ.
- **Liên quan đề tài:** Kiến thức guideline cần đưa vào prompt. Liên quan §9 (khuyến nghị theo dõi).

### L16 — Gulati et al. 2026, *Clinical Radiology* [AB]
- **Bối cảnh & luận điểm chính:** GPT-4o phân loại Lung-RADS, đánh giá theo từng nhóm và theo chiến lược prompt.
- **Phương pháp:** 327 báo cáo LDCT. So prompt có cấu trúc với prompt tự do.
- **Kết quả chính:** Prompt có cấu trúc: accuracy 68,8%, kappa có trọng số 0,758. Prompt tự do: 58,1% và 0,695.
- **Hạn chế:** CT, không phải CXR. Một trung tâm.
- **Liên quan đề tài:** Prompt có cấu trúc cải thiện đầu ra. Áp dụng cho thiết kế prompt đánh giá của đề tài.

### L17 — Zhou T et al. 2024, *kỷ yếu (PubMed 41726431)* [AB]
- **Bối cảnh & luận điểm chính:** Tự động hóa phân loại Lung-RADS và khuyến nghị theo dõi bằng in-context learning với LLM.
- **Phương pháp:** Báo cáo LDCT. Chi tiết: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Abstract không có số liệu.
- **Hạn chế:** Thiếu số liệu, là bài kỷ yếu.
- **Liên quan đề tài:** Bối cảnh cho in-context learning. Liên quan thấp.

### L18 — Wen et al. 2025, *European Journal of Radiology Open* [AB]
- **Bối cảnh & luận điểm chính:** LLM sinh khuyến nghị theo dõi nốt phổi.
- **Phương pháp:** 1.009 báo cáo CT theo dõi. So ERNIE-4.0-Turbo-8K với GPT-4o-mini.
- **Kết quả chính:** Đúng 94,6% so với 92,8% (P = 0,07). Tỷ lệ khuyến nghị có hại 2,9% so với 3,5%.
- **Hạn chế:** CT, không phải CXR. Nguồn đã đọc không mô tả tiêu chuẩn chấm.
- **Liên quan đề tài:** Mốc an toàn (tỷ lệ có hại) cho nhánh mở rộng §9.

---

## T6. CAD học sâu phát hiện nốt và ung thư phổi, hiệu năng độc lập

### L19 — Nam et al. 2019, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** Phát triển và kiểm định DLAD phát hiện nốt ác tính trên CXR, tiền thân của Lunit.
- **Phương pháp:** 43.292 CXR. Một tập test nội bộ và 4 tập ngoài. So với 18 bác sĩ. Chỉ số: AUROC, JAFROC FOM.
- **Kết quả chính:** AUROC 0,92–0,99, JAFROC FOM 0,831–0,924. Cao hơn 17/18 bác sĩ.
- **Hạn chế:** Hồi cứu, tập test giàu ca bệnh.
- **Liên quan đề tài:** Mốc CAD kinh điển, đồng thời mẫu cho chỉ số định vị (JAFROC).

### L21 — Yoo et al. 2020, *JAMA Network Open* [AB]
- **Bối cảnh & luận điểm chính:** Kiểm định DL (Lunit INSIGHT CXR) trên dữ liệu sàng lọc NLST.
- **Phương pháp:** NLST, 5.485 người tham gia.
- **Kết quả chính:** Nốt: sensitivity 86,2%, specificity 85,0%. Mọi ca ung thư: sensitivity 75,0%, PPV 3,8%.
- **Hạn chế:** PPV rất thấp vì tỷ lệ bệnh thấp. Ảnh NLST là ảnh sàng lọc.
- **Liên quan đề tài:** Minh họa PPV khi ca dương hiếm (liên quan L84). Nên báo cáo PPV cho VLM.

### L25 — Hwang et al. 2019, *JAMA Network Open* [AB]
- **Bối cảnh & luận điểm chính:** DL phát hiện 4 bệnh lồng ngực chính, có u ác tính.
- **Phương pháp:** 89.834 CXR để phát triển, kiểm định ngoài ở 5 cơ sở.
- **Kết quả chính:** AUC mức ảnh 0,979. Định vị mức tổn thương 0,972.
- **Hạn chế:** Gộp nhiều bệnh. Abstract không tách số riêng cho nốt nhỏ.
- **Liên quan đề tài:** Mốc định vị CAD đa bệnh.

### L26 — Majkowska et al. 2020, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** Google đánh giá DL với chuẩn tham chiếu do bác sĩ thẩm định và hiệu chỉnh theo quần thể.
- **Phương pháp:** DS1 759.611 ảnh và ChestX-ray14. Nhãn do bác sĩ thẩm định.
- **Kết quả chính:** Nốt hoặc khối: AUC hiệu chỉnh theo quần thể 0,72. Trên ChestX-ray14: 0,91.
- **Hạn chế:** Chênh lệch lớn giữa hai tập cho thấy nhãn và quần thể ảnh hưởng mạnh.
- **Liên quan đề tài:** Cảnh báo AUC nốt phụ thuộc chuẩn tham chiếu. Nhãn ChestX-ray14 đã thẩm định là tài nguyên dùng được.

### L29 — Shimazaki et al. 2022, *Scientific Reports* [AB]
- **Bối cảnh & luận điểm chính:** DL dạng phân đoạn phát hiện ung thư phổi, phân tích vùng mù.
- **Phương pháp:** 629 ảnh train, 151 ảnh test, ung thư xác nhận giải phẫu bệnh.
- **Kết quả chính:** Sensitivity 0,73 với 0,13 dương tính giả mỗi ảnh. Ở vùng mù (sau tim, rốn phổi...) sensitivity 0,50–0,64, so với 0,87 ở vùng khác.
- **Hạn chế:** Mẫu nhỏ, một trung tâm.
- **Liên quan đề tài:** Gợi ý phân tầng lỗi theo vị trí giải phẫu khi kiểm toán VLM.

### L31 — Ohlmann-Knafo et al. 2022, *arXiv* [AB]
- **Bối cảnh & luận điểm chính:** Phần mềm AI thương mại làm người đọc thứ hai cho nốt trên CXR.
- **Phương pháp:** JSRT (n = 247), 8 bác sĩ.
- **Kết quả chính:** AUC trung bình của bác sĩ 0,77 ± 0,07. Các chế độ AI tăng sensitivity 14% và 12%.
- **Hạn chế:** Chỉ là preprint, JSRT cũ.
- **Liên quan đề tài:** Mốc bác sĩ và CAD trên **cùng tập JSRT** với L2. Có thể đặt VLM, CAD và bác sĩ cạnh nhau.

### L45 — Arzamasov et al. 2024, *Quantitative Imaging in Medicine and Surgery* [AB]
- **Bối cảnh & luận điểm chính:** Đánh giá độc lập 5 phần mềm AI thương mại phát hiện nốt trên CXR.
- **Phương pháp:** MosMedData VII, 100 CXR (50/50), cùng nguồn với L1. Các sản phẩm gồm Celsus, Lunit INSIGHT CXR, qXR...
- **Kết quả chính:** AUC tốt nhất 0,956 ở bước xác suất. Khi bác sĩ kiểm tra cả phân đoạn và phân loại, AUC tốt nhất còn 0,812.
- **Hạn chế:** Mẫu nhỏ. L1 không xác nhận dùng đúng cùng mẫu.
- **Liên quan đề tài:** Điểm so sánh gần nhất giữa CNN và VLM: CAD 0,956 so với BiomedCLIP 0,743 ở L1.

### L72 — Cid et al. 2024, *Lancet Digital Health* (X-Raydar) [FT]
- **Bối cảnh & luận điểm chính:** Thiếu hệ thống AI đọc CXR toàn diện, mở và huấn luyện trên dữ liệu lớn. Bài phát triển X-Raydar (ảnh) và X-Raydar-NLP (gán nhãn báo cáo).
- **Phương pháp:** 6 bệnh viện NHS (3 Trust), 2.513.546 nghiên cứu (2006–2019), 1.896.034 ảnh thẳng, 37 dấu hiệu, có "parenchymal mass or nodule". NLP huấn luyện trên 23.230 báo cáo gán tay. Test: tập tự gán nhãn n = 103.328; tập đồng thuận 3 chuyên gia n = 1.427; MIMIC-CXR n = 252.374.
- **Kết quả chính:** AUC trung bình 0,919 (tự gán nhãn), 0,864 (đồng thuận), 0,842 (MIMIC). Vượt balanced accuracy của bác sĩ đã đọc gốc ở 27/37 dấu hiệu, trung bình +13,3%. Tương đương bác sĩ với mass/nodule.
- **Hạn chế:** Hồi cứu. Nhãn chủ yếu từ NLP. AUC riêng cho mass/nodule không trích trong phần đã đọc. So với "bác sĩ đã đọc gốc" chứ không phải nghiên cứu người đọc có kiểm soát.
- **Liên quan đề tài:** Baseline CNN mở, dùng được làm B1 hoặc B2. Quy trình NLP gán nhãn quy mô lớn.

### L73 — Pesce et al. 2019, *Medical Image Analysis* [FT]
- **Bối cảnh & luận điểm chính:** Kết hợp nhiều ảnh nhãn yếu (từ NLP) với ít ảnh có box, để phát hiện CXR có tổn thương phổi.
- **Phương pháp:** Hơn 430.000 CXR (Guy's & St Thomas'). 23.132 ảnh có tổn thương, trong đó 2.196 ảnh gán tay với 3.253 box. Hai kiến trúc: CONAF (phản hồi định vị qua saliency map) và RAMAF (recurrent attention học bằng RL). Đánh giá định vị: box trùng ít nhất 25% với ground truth.
- **Kết quả chính:** Định vị tốt lên khi có phản hồi attention. Accuracy tăng tuyến tính theo kích thước: nốt dưới 10 mm có accuracy dưới 0,2, khối trên 100 mm khoảng 0,7. RAMAF tốt hơn RAM.
- **Hạn chế:** Nhãn "tổn thương" rộng, không xác nhận CT. Ngưỡng trùng 25% là dễ. Phần đã đọc không có AUC tổng thể chắc chắn.
- **Liên quan đề tài:** Bằng chứng mạnh rằng nốt nhỏ cực khó. Ý tưởng dùng một ít box để giám sát attention khi tinh chỉnh.

### L75 — Lee JH et al. 2020, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** DL thương mại so với bác sĩ trong phát hiện ung thư phổi ở quần thể khám sức khỏe.
- **Phương pháp:** Validation: 10.285 ảnh, 10 ca ung thư. Screening: 100.525 ảnh, 47 ca.
- **Kết quả chính:** Validation: AUC 0,99, sensitivity 90% (bác sĩ 60%), tỷ lệ dương tính giả 3,1% (bác sĩ 0,3%). Screening: AUC 0,97, sensitivity 83%, tỷ lệ dương tính giả 3%.
- **Hạn chế:** Rất ít ca dương nên khoảng tin cậy sensitivity rộng. Tỷ lệ dương tính giả gấp 10 lần bác sĩ.
- **Liên quan đề tài:** Mốc CAD khi tỷ lệ bệnh thấp. Cần báo cáo FPR và PPV.

### L77 — Hwang EJ et al. 2021, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** Triển khai thực tế CAD nốt cho bệnh nhân ung thư, nhằm phát hiện di căn phổi.
- **Phương pháp:** 2.916 ảnh đọc có CAD so với 5.681 ảnh không có CAD.
- **Kết quả chính:** Tỷ lệ phát hiện di căn mới tăng từ 0,32% lên 0,86%. Tỷ lệ chuyển tuyến sai không kém hơn (0,34% so với 0,25%).
- **Hạn chế:** Không ngẫu nhiên, so sánh theo giai đoạn thời gian.
- **Liên quan đề tài:** Bằng chứng giá trị thực tế của CAD nốt. Liên quan trung bình.

### L78 — Shen et al. 2023, *Medical Image Analysis* [FT]
- **Bối cảnh & luận điểm chính:** Thiếu dữ liệu nốt có nhãn. Bài tổng hợp nốt với thuộc tính tách rời (hình dạng, kích thước, kết cấu) để tăng cường dữ liệu.
- **Phương pháp:** Shape Generator dựa trên GAN, Size Modulation ở mức pixel, Texture Generator dạng gated convolution thô đến mịn. Tăng cường có chọn lọc ca khó (HEM) theo phân bố kích thước của nốt bị bỏ sót. 2.420 ảnh nốt thật để tinh chỉnh, 604 ảnh để test. Chỉ số: FROC AUC và NODE21 score = 0,75·AUC + 0,25·Sen@0,25 FP.
- **Kết quả chính:** Tác giả báo chất lượng, đa dạng và khả năng điều khiển tốt hơn, và phát hiện cải thiện rõ. Con số cụ thể: không nêu trong nguồn đã đọc (không trích được bảng).
- **Hạn chế:** Tác giả nêu: chưa điều khiển được mật độ và dạng xuất hiện; khó đo định lượng chất lượng ảnh. Phê bình riêng: dữ liệu tổng hợp có thể tạo shortcut.
- **Liên quan đề tài:** Hướng tăng cường dữ liệu nốt nhỏ cho LoRA (C3). Chỉ số NODE21 có thể dùng ở §8.2.

### L79 — Kim C et al. 2023, *European Radiology* [FT]
- **Bối cảnh & luận điểm chính:** Kiểm định ngoài AI thương mại đọc CXR ở cơ sở chăm sóc ban đầu với tỷ lệ bệnh thấp.
- **Phương pháp:** 3.047 CXR liên tiếp từ 2 trung tâm (2018), nhãn từ CT. 4 bác sĩ nội trú đọc hai lần, có và không có AI.
- **Kết quả chính:** Tỷ lệ tổn thương có ý nghĩa 2,2% (68 ca). AI đạt AUROC 0,648, sensitivity 35,3%, specificity 94,2%. AI phát hiện 9/22 khối u. Tổn thương bị AI bỏ sót nhỏ hơn. Người đọc: AUROC 0,534–0,676, có AI 0,571–0,688. Thời gian đọc tăng 2,96–10,27 giây.
- **Hạn chế:** Tác giả nêu: người đọc là bác sĩ nội trú, xem trên web viewer không có màn hình độ phân giải cao. Phê bình riêng: chỉ một sản phẩm; số ca u nhỏ.
- **Liên quan đề tài:** Hiệu năng AI giảm mạnh khi tỷ lệ bệnh thấp. Spec cần đánh giá cả ở tỷ lệ bệnh thực tế.

### L80 — van Leeuwen et al. 2024, *Radiology* [FT]
- **Bối cảnh & luận điểm chính:** Kiểm định độc lập AI thương mại (Project AIR) cho phát hiện nốt trên CXR và tuổi xương.
- **Phương pháp:** 9/17 sản phẩm đủ điều kiện tham gia, dữ liệu 7 bệnh viện Hà Lan. Nốt: 7 thuật toán, 386 bệnh nhân, chuẩn từ CT do bác sĩ lồng ngực đặt. Tập con 140 CXR do 17 người đọc. Chỉ số: AUC.
- **Kết quả chính:** Người đọc đạt AUC trung bình 0,81 (95% CI 0,77–0,85). 4 thuật toán tốt hơn (AUC 0,86–0,93). 3 thuật toán còn lại không khác biệt.
- **Hạn chế:** Tác giả nêu: một số nhà cung cấp từ chối tham gia; chỉ đo hiệu năng độc lập, trong khi sản phẩm dùng để hỗ trợ quyết định. Phê bình riêng: người đọc chỉ xem 140 ca.
- **Liên quan đề tài:** Mốc CAD độc lập (AUC 0,86–0,93) và mốc người đọc (0,81). VLM sau LoRA nên được đặt cạnh hai mốc này.

### L81 — Nam et al. 2020, *Radiology: Cardiothoracic Imaging* [FT]
- **Bối cảnh & luận điểm chính:** DL phát hiện ung thư phổi đã bị bỏ sót trên CXR thẳng PA trong thực hành.
- **Phương pháp:** 168 CXR với 187 ung thư bị bỏ sót (kích thước trung bình 2,3 ± 1,2 cm) và 50 CXR bình thường, chuẩn từ CT. 4 bác sĩ lồng ngực đọc có và không có thuật toán. Chỉ số: AUROC, AUFROC.
- **Kết quả chính:** Thuật toán: AUROC 0,899, AUFROC 0,744; bác sĩ 0,634–0,663 và 0,619–0,651. Sensitivity 69,6% so với 47,0%, specificity 94,0% so với 78,0%. Có AI, AUROC của bác sĩ tăng lên 0,685–0,724. FPR tương đương (21,1% so với 19,0%).
- **Hạn chế:** Tập giàu ca bệnh, một trung tâm, ít ca bình thường. Phần đã đọc không có mục hạn chế riêng.
- **Liên quan đề tài:** Cho thấy ung thư bị bỏ sót vẫn khá lớn (trung bình 2,3 cm) và bác sĩ chỉ đạt sensitivity 47%. Là mốc độ khó, kèm chỉ số AUFROC.

---

## T7. Tương tác người–AI và tác động lâm sàng của CAD

### L20 — Nam et al. 2023, *Radiology* (RCT) [AB]
- **Bối cảnh & luận điểm chính:** RCT về AI phát hiện nốt trên CXR ở quần thể khám sức khỏe.
- **Phương pháp:** 10.476 người, ngẫu nhiên đọc có hoặc không có AI.
- **Kết quả chính:** Tỷ lệ phát hiện nốt cần xử trí 0,59% (31/5.238) so với 0,25% (13/5.238). OR 2,4.
- **Hạn chế:** Một trung tâm. Kết cục là phát hiện, không phải tử vong.
- **Liên quan đề tài:** Mức bằng chứng cao nhất cho CAD nốt, làm chuẩn lâm sàng.

### L22 — Yoo et al. 2021, *European Radiology* [AB]
- **Bối cảnh & luận điểm chính:** AI cải thiện phát hiện ung thư phổi trên CXR, nghiên cứu nhiều người đọc trên NLST.
- **Phương pháp:** NLST, 519 ảnh, 8 người đọc.
- **Kết quả chính:** Sensitivity của bác sĩ nội trú 0,61 → 0,72. FPPI của bác sĩ chẩn đoán hình ảnh 0,24 → 0,17.
- **Hạn chế:** Ảnh sàng lọc, ít người đọc.
- **Liên quan đề tài:** Mốc lợi ích khi AI làm người đọc thứ hai.

### L27 — Dissez et al. 2022, *arXiv* [AB]
- **Bối cảnh & luận điểm chính:** AI thương mại (Behold.ai) hỗ trợ phát hiện sớm ung thư phổi, nghiên cứu nhiều người đọc ở Anh.
- **Phương pháp:** 400 CXR của NHS, 11 bác sĩ.
- **Kết quả chính:** Số ca ung thư được phát hiện tăng 17,4%.
- **Hạn chế:** Chỉ là preprint. Có xung đột lợi ích tiềm năng với nhà cung cấp.
- **Liên quan đề tài:** Mốc lợi ích của AI riêng cho ung thư phổi.

### L28 — Topff et al. 2024, *European Radiology* [AB]
- **Bối cảnh & luận điểm chính:** AI đọc kép CXR (đối chiếu ảnh với báo cáo) để tìm dấu hiệu có ý nghĩa bị bỏ sót.
- **Phương pháp:** 25.104 CXR, 2 trung tâm. CNN kết hợp NLP.
- **Kết quả chính:** 71,4% dấu hiệu bị bỏ sót là nốt (25/35). Chiếm 0,1% tổng số ca.
- **Hạn chế:** Không có tỷ lệ dương tính giả trong nguồn đã đọc.
- **Liên quan đề tài:** Nốt là loại dấu hiệu bị bỏ sót nhiều nhất. Tiền thân của việc dùng LLM kiểm tra chất lượng báo cáo.

### L32 — Tang et al. 2023, *Radiology: Artificial Intelligence* [AB]
- **Bối cảnh & luận điểm chính:** Ảnh hưởng của các giao diện AI khác nhau đến phát hiện nốt và khối.
- **Phương pháp:** 140 CXR, 10 bác sĩ, nhiều kiểu giao diện.
- **Kết quả chính:** Giao diện chỉ có văn bản tốt hơn không có AI (0,87 so với 0,82; P < 0,001).
- **Hạn chế:** Mẫu nhỏ.
- **Liên quan đề tài:** Đầu ra văn bản kiểu VLM có thể hữu ích lâm sàng nếu đúng. Liên quan đến thiết kế đầu ra.

### L74 — Jang et al. 2020, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** DLAD giúp giảm ung thư phổi bị bỏ sót trên CXR.
- **Phương pháp:** 117 bệnh nhân ung thư thấy được trên ảnh trước (105 ca bị bỏ sót) và 234 chứng. 9 người đọc.
- **Kết quả chính:** AUC (alternative FROC) 0,67 → 0,76. Sensitivity 40% → 53%. Tỷ lệ chỉ định CT không tăng (10% so với 8%).
- **Hạn chế:** Hồi cứu, tập giàu ca bệnh.
- **Liên quan đề tài:** Mốc độ khó: bác sĩ chỉ đạt 40–53% với ung thư dễ bị bỏ sót.

### L76 — Sung et al. 2021, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** Thử nghiệm ngẫu nhiên bắt chéo về hệ thống DL phát hiện nhiều dấu hiệu, có nốt.
- **Phương pháp:** 228 ảnh, 6 người đọc, 5 loại dấu hiệu.
- **Kết quả chính:** JAFROC 0,90 → 0,95. AUC 0,93 → 0,98. Sensitivity theo tổn thương 83% → 89,1%. Specificity 89,3% → 96,6%. Riêng DL: JAFROC 0,96.
- **Hạn chế:** Mẫu nhỏ, ít người đọc.
- **Liên quan đề tài:** Mẫu chỉ số định vị JAFROC/FROC cho §8.2.

### L85 — Sim et al. 2020, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** Phần mềm DCNN cải thiện khả năng bác sĩ phát hiện nốt ác tính.
- **Phương pháp:** 4 trung tâm. 600 ảnh ung thư (xác nhận CT và giải phẫu bệnh) và 200 ảnh bình thường. 12 bác sĩ.
- **Kết quả chính:** Sensitivity 65,1% → 70,3%. Dương tính giả mỗi ảnh 0,20 → 0,18.
- **Hạn chế:** Tập giàu ca bệnh.
- **Liên quan đề tài:** Mốc sensitivity của bác sĩ (khoảng 65%), gần ngưỡng 0,60 của spec.

### L86 — Homayounieh et al. 2021, *JAMA Network Open* [FT]
- **Bối cảnh & luận điểm chính:** AI (AI-Rad Companion Chest X-ray) có giúp phát hiện nốt ở các mức độ khó khác nhau không.
- **Phương pháp:** 100 CXR thẳng PA từ Đức và LIDC (Mỹ), làm giàu nốt ở mọi mức độ khó, có chứng bình thường và bất thường khác. 9 bác sĩ đọc hai phiên cách nhau ít nhất 1 tháng (1.800 lượt đọc). Ground truth do 2 bác sĩ lồng ngực đặt.
- **Kết quả chính:** Accuracy trung bình tăng 6,4% (95% CI 2,3–10,6%). pAUC (FPR 0–0,2) tăng 5,6% (CI −1,4–12,0%). Sensitivity tăng 12% ở bác sĩ trẻ, 9% ở bác sĩ lâu năm.
- **Hạn chế:** Tác giả nêu: mẫu 100 ảnh là nhỏ cho phân tích theo độ khó. Phê bình riêng: tập làm giàu; pAUC không có ý nghĩa thống kê.
- **Liên quan đề tài:** Phân tầng theo độ khó của nốt là thiết kế nên mượn cho phần kiểm toán.

### L87 — Ahn et al. 2022, *JAMA Network Open* [FT]
- **Bối cảnh & luận điểm chính:** AI dùng đồng thời khi đọc có cải thiện hiệu năng và tốc độ đọc CXR không.
- **Phương pháp:** 497 CXR thẳng (247 MIMIC-CXR, 250 MGH), 4 dấu hiệu: viêm phổi, nốt, tràn khí, tràn dịch. 6 bác sĩ, bắt chéo ngẫu nhiên, washout 4 tuần. AI đưa ra heat map và xác suất.
- **Kết quả chính:** Sensitivity với nốt: AI 0,816 so với người đọc 0,567. Có AI, sensitivity của người đọc tăng ở mọi dấu hiệu, AUROC tăng có ý nghĩa với nốt và tràn khí. Thời gian báo cáo giảm 10% (40,8 → 36,9 giây).
- **Hạn chế:** Tác giả nêu: ground truth chỉ từ ảnh thẳng, không từ CT; không mô phỏng được quy trình thực tế. Phê bình riêng: ít người đọc.
- **Liên quan đề tài:** Mốc người đọc với nốt (0,567) trên một phần ảnh MIMIC-CXR, cùng nguồn dữ liệu đề tài có thể dùng.

### L88 — Bernstein et al. 2023, *European Radiology* [FT]
- **Bối cảnh & luận điểm chính:** Kết quả AI sai có ảnh hưởng đến bác sĩ không, và yếu tố con người nào giảm được tác hại.
- **Phương pháp:** 6 bác sĩ đọc 90 CXR qua 4 phiên (quyết định cần CT hay không). Dùng AI giả: 12 ca bị cố ý làm sai (8 FP, 4 FN), ROC-AUC 0,87. Ba điều kiện: Keep AI (No Box), Delete AI (No Box), Keep AI (Box).
- **Kết quả chính:** Không có AI: FN 2,7%, FP 51,4%. Keep AI (No Box): FN 33,0%, FP 86,0%. Delete AI: FN 26,7%, FP 80,5%. Keep AI (Box): FN 20,7%, FP 80,5%. Có box giảm FN (P = 0,04).
- **Hạn chế:** Tác giả nêu: thiếu tính sinh thái, không có hậu quả thật; chỉ ảnh thẳng; ít bác sĩ, một cơ sở.
- **Liên quan đề tài:** Luận cứ an toàn cho phần kiểm toán: VLM sai mà tự tin sẽ kéo bác sĩ theo. Đầu ra có box giảm tác hại, ủng hộ yêu cầu định vị.

### L89 — Bennani et al. 2023, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** AI cải thiện khả năng bác sĩ phát hiện bất thường trên CXR, với nhãn chuẩn từ CT.
- **Phương pháp:** 500 ảnh, CT trong vòng 72 giờ, 5 loại bất thường, 12 người đọc ở 3 mức kinh nghiệm.
- **Kết quả chính:** Sensitivity với nốt tăng tuyệt đối 12%. Specificity với nốt tăng 2,1%. Thời gian đọc giảm 31%.
- **Hạn chế:** Tập có chọn lọc.
- **Liên quan đề tài:** Mốc lợi ích của AI riêng cho nốt, với chuẩn CT.

### L90 — Lee JH et al. 2023, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** Tương tác người–AI khi phát hiện nốt ác tính, với AI độ chính xác cao và thấp.
- **Phương pháp:** 120 ảnh (60 ca ung thư), 30 người đọc, chia ngẫu nhiên dùng AI tốt hoặc AI kém, không biết mình dùng loại nào.
- **Kết quả chính:** AI tốt: AUC 0,77 → 0,82. AI kém: AUC không đổi (0,75). Người đọc đổi theo AI 67% (AI tốt) và 59% (AI kém).
- **Hạn chế:** Tập nhỏ và giàu ca bệnh.
- **Liên quan đề tài:** Cùng hướng với L88: AI kém vẫn được nghe theo. Luận cứ cho kiểm toán trước khi dùng.

---

## T8. Dự đoán nguy cơ và sàng lọc ung thư phổi từ CXR

### L23 — Lu et al. 2020, *Annals of Internal Medicine* (CXR-LC) [AB]
- **Bối cảnh & luận điểm chính:** Dùng CXR để xác định người hút thuốc nguy cơ cao cần chụp CT sàng lọc.
- **Phương pháp:** Phát triển trên PLCO (n = 41.856), kiểm định trên NLST. So với PLCOm2012.
- **Kết quả chính:** AUC trên PLCO 0,755 so với 0,634. Trên NLST 0,659 so với 0,650.
- **Hạn chế:** Lợi thế trên NLST rất nhỏ. Ảnh là ảnh thử nghiệm sàng lọc.
- **Liên quan đề tài:** Tiền lệ CXR → nguy cơ dài hạn. Dùng cho hướng mở rộng §9.

### L24 — Lee JH et al. 2022, *Radiology* [AB]
- **Bối cảnh & luận điểm chính:** DL tối ưu việc chọn người chụp CT sàng lọc, bổ sung khuyến nghị USPSTF 2021.
- **Phương pháp:** 19.488 người hút thuốc khám sức khỏe tại Hàn Quốc.
- **Kết quả chính:** AUC với ung thư mới mắc 0,68 và 0,75 (hai cấu hình; chi tiết không nêu trong nguồn đã đọc).
- **Hạn chế:** Quần thể chủ yếu nam giới, theo dõi ngắn (theo L91).
- **Liên quan đề tài:** Kiểm định ngoài của CXR-LC. Liên quan thấp.

### L30 — Kotevska et al. 2026, *arXiv* [AB]
- **Bối cảnh & luận điểm chính:** ViT dự đoán ung thư phổi 1–2 năm trước chẩn đoán từ CXR.
- **Phương pháp:** 259.361 CXR của VA. So mô hình tiền huấn luyện với mô hình huấn luyện từ đầu.
- **Kết quả chính:** Mô hình tiền huấn luyện vượt mô hình từ đầu 6–10 điểm AUC.
- **Hạn chế:** Chỉ là preprint. Quần thể VA đặc thù.
- **Liên quan đề tài:** Tiền huấn luyện có lợi, ủng hộ việc thích nghi mô hình nền thay vì học từ đầu.

### L13 — Yang C et al. 2026, *arXiv* (LUNG-KGMM) [AB]
- **Bối cảnh & luận điểm chính:** Học đa phương thức có dẫn dắt bằng tri thức để dự đoán ung thư phổi: biểu diễn CXR, báo cáo, EHR và đồ thị tri thức guideline.
- **Phương pháp:** Cohort MIMIC, kiểm định ngoài ở Hạ Môn. Abstract không nói rõ có dùng LLM không.
- **Kết quả chính:** Abstract chỉ nói "vượt SOTA", không có số liệu.
- **Hạn chế:** Thiếu số liệu, chỉ là preprint.
- **Liên quan đề tài:** Mô hình đa phương thức CXR cộng báo cáo cho nguy cơ ung thư. Hướng mở rộng.

### L61 — Kim et al. 2026, *Journal of Thoracic Oncology* (tóm tắt hội nghị P1.079) [T]
- **Bối cảnh & luận điểm chính:** Dự đoán nguy cơ ung thư phổi từ CXR kết hợp đặc trưng lâm sàng nhúng bằng LLM. Chỉ có tiêu đề.
- **Phương pháp:** Không nêu trong nguồn đã đọc.
- **Kết quả chính:** Không nêu trong nguồn đã đọc.
- **Hạn chế:** Chưa có bài đầy đủ.
- **Liên quan đề tài:** Cần theo dõi xem có bài đầy đủ không. Đây là một trong hai mô hình CXR kết hợp LLM cho nguy cơ (cùng L13).

### L67 — Hayeso et al. 2026, *Translational Lung Cancer Research* [FT]
- **Bối cảnh & luận điểm chính:** Mô hình nền lai nhẹ (LHFM) kết hợp CXR với dữ liệu lâm sàng để tiên lượng ung thư phổi ở nơi ít nguồn lực.
- **Phương pháp:** Encoder SAM-Med2D cho ảnh, BioGPT sinh prompt ngữ nghĩa, kèm metadata lâm sàng. Gộp bằng transformer hai nhánh để dự đoán sống còn. Huấn luyện và kiểm định trên JSRT và PadChest, kiểm định ngoài trên NIH CXR và các tập khác. Chỉ số: C-index, AUROC, Kaplan–Meier.
- **Kết quả chính:** C-index 0,910 (95% CI 0,898–0,922), AUROC 0,935 (0,927–0,943). Baseline lâm sàng: C-index 0,696. FGCN: 0,769.
- **Hạn chế:** Phần đã đọc không có mục hạn chế rõ ràng. Phê bình riêng (quan trọng): JSRT, PadChest và NIH CXR vốn không nổi tiếng là có dữ liệu thời gian sống còn, và **nguồn nhãn sống còn không rõ trong phần đã đọc**. Con số cao bất thường cần kiểm tra kỹ trước khi trích dẫn. Cụm từ "X-quang liều thấp" cũng mơ hồ.
- **Liên quan đề tài:** Liên quan thấp. Nếu trích, cần ghi rõ rủi ro về độ tin cậy.

### L91 — Raghu et al. 2022, *JAMA Network Open* [FT]
- **Bối cảnh & luận điểm chính:** Kiểm định CXR-LC (CXR kèm tuổi, giới, tình trạng hút thuốc lấy từ EMR) để xác định người nguy cơ cao, bổ sung tiêu chí CMS 2022.
- **Phương pháp:** Nghiên cứu tiên lượng trên 14.737 người đang hoặc từng hút thuốc, có CXR PA ngoại trú 2013–2014 tại Mass General Brigham. Kết cục: ung thư phổi mới mắc trong 6 năm (2,4%, 361 ca). Ngưỡng CXR-LC: nguy cơ 12 năm ≥ 3,297%.
- **Kết quả chính:** Chỉ xác định được tiêu chí CMS ở 42,6% người. Đủ điều kiện theo cả hai tiêu chí: 8,5% mắc ung thư, so với 2,8% nếu chỉ theo CMS. Chỉ theo CXR-LC: 3,3%. Khi không rõ tiêu chí CMS: 2,5% so với 0,5% (gấp 5 lần).
- **Hạn chế:** Tác giả nêu, cho các nghiên cứu trước: quần thể chủ yếu người da trắng; ảnh từ thử nghiệm sàng lọc. Phê bình riêng: một hệ thống y tế; dữ liệu hút thuốc trong EMR không đầy đủ.
- **Liên quan đề tài:** Tận dụng CXR thường quy cho sàng lọc. Hướng mở rộng §9.

---

## T9. Dataset và benchmark có nhãn nốt/khối

### L35 — Sogancioglu et al. 2024, *IEEE TMI* (NODE21, trước là arXiv) [AB]
- **Bối cảnh & luận điểm chính:** Thử thách NODE21 về phát hiện và tổng hợp nốt trên CXR.
- **Phương pháp:** Bộ NODE21 công khai, có box nốt. Cỡ mẫu chi tiết: không nêu trong nguồn đã đọc.
- **Kết quả chính:** Abstract không có số liệu.
- **Hạn chế:** Chưa đọc toàn văn bản TMI.
- **Liên quan đề tài:** Benchmark nốt công khai chính, ứng viên tập train và test có box cho LoRA. Chỉ số NODE21 score (xem L78).

### L36 — Mustafic et al. 2026, *arXiv* (NoduLoCC2026) [AB]
- **Bối cảnh & luận điểm chính:** Cuộc thi định vị và phân loại nốt trên CXR.
- **Phương pháp:** Tập thi cộng tập test ngoài, 5 đội.
- **Kết quả chính:** Balanced accuracy 0,72, AUC-ROC 0,79. Định vị: đúng số nốt ở 53% ảnh, khoảng cách trung vị 12,83 mm.
- **Hạn chế:** Ít đội, chỉ là preprint.
- **Liên quan đề tài:** Xác nhận định vị là phần khó. Chỉ số khoảng cách (mm) dùng được cho §8.2.

### L60 — Mustafic et al. 2026, *kỷ yếu ICCVIT* (dataset NoduLoCC) [T]
- **Bối cảnh & luận điểm chính:** Bản bình duyệt của dataset NoduLoCC (xem L36). Chỉ có metadata.
- **Phương pháp:** Không nêu trong nguồn đã đọc.
- **Kết quả chính:** Không nêu trong nguồn đã đọc.
- **Hạn chế:** Chưa có abstract hay DOI.
- **Liên quan đề tài:** Trích kèm L36 khi dùng dataset.

### L37 — Nguyen HQ et al. 2022, *Scientific Data* (VinDr-CXR) [FT]
- **Bối cảnh & luận điểm chính:** Phần lớn dataset CXR không có vị trí tổn thương. VinDr-CXR cung cấp box do bác sĩ vẽ.
- **Phương pháp:** Hơn 100.000 ảnh từ 2 bệnh viện Việt Nam. Công bố 18.000 ảnh với 22 nhãn cục bộ (box) và 6 nhãn toàn cục, do 17 bác sĩ gán. Train 15.000 ảnh, mỗi ảnh 3 bác sĩ đọc độc lập. Test 3.000 ảnh, đồng thuận của 5 bác sĩ. Định dạng DICOM.
- **Kết quả chính:** Nodule/Mass: 410 ảnh train (2,73%), 176 ảnh test (5,87%). Lung tumor (nhãn toàn cục): 132 và 80.
- **Hạn chế:** Ít ca nốt. Ở tập train, ba người đọc có thể không thống nhất box. Quần thể chỉ một quốc gia.
- **Liên quan đề tài:** Nguồn box Nodule/Mass chính cho LoRA và đánh giá định vị. Cần xử lý box bất đồng ở tập train và mất cân bằng lớp (khoảng 3%).

### L38 — Wang X et al. 2017, *CVPR* (ChestX-ray8/14; corpus ghi arXiv) [AB]
- **Bối cảnh & luận điểm chính:** CSDL CXR quy mô bệnh viện, kèm benchmark phân loại và định vị yếu.
- **Phương pháp:** 108.948 ảnh, 8 nhãn ban đầu (sau mở rộng thành 14, có Nodule và Mass), nhãn trích bằng NLP.
- **Kết quả chính:** Abstract không liệt kê nhãn hay con số riêng cho nốt.
- **Hạn chế:** Nhãn NLP nhiễu (xem L26 về nhãn đã thẩm định). Chỉ có box cho một tập con nhỏ.
- **Liên quan đề tài:** Nguồn nhãn Nodule/Mass và box trong ChestX-ray8. Được L71 dùng để đánh giá grounding.

### L39 — Irvin et al. 2019, *AAAI* (CheXpert; corpus ghi arXiv) [AB]
- **Bối cảnh & luận điểm chính:** Dataset CXR lớn với nhãn không chắc chắn, có so sánh với chuyên gia.
- **Phương pháp:** 224.316 CXR, 14 quan sát, gồm nhãn "Lung Lesion" (abstract không nêu tên nhãn này).
- **Kết quả chính:** Không nêu trong nguồn đã đọc.
- **Hạn chế:** **PDF cục bộ bị nhầm file** (chứa bài LVQ của Saralajew et al.), nên chưa đọc được toàn văn.
- **Liên quan đề tài:** Nhãn Lung Lesion được dùng trong MedSigLIP (L7) và các đánh giá zero-shot.

### L40 — Johnson et al. 2019, *arXiv* (MIMIC-CXR-JPG) [AB]
- **Bối cảnh & luận điểm chính:** Phiên bản JPG có nhãn của MIMIC-CXR.
- **Phương pháp:** 377.110 ảnh, 14 nhãn. Bài Sci Data cùng nhóm (đọc toàn văn trong lượt này, file `[15]_Johnson2019`) cho biết thêm: 227.835 nghiên cứu, 65.379 bệnh nhân, khoa Cấp cứu BIDMC 2011–2016, báo cáo tự do đã khử định danh.
- **Kết quả chính:** Không áp dụng (bài mô tả dataset).
- **Hạn chế:** Nhãn trích tự động. Quần thể cấp cứu, nhiều ảnh AP chụp tại giường. Không có box.
- **Liên quan đề tài:** Nguồn huấn luyện của hầu hết VLM CXR, nên có rủi ro rò rỉ dữ liệu khi đánh giá MedGemma trên MIMIC. Báo cáo MIMIC có thể trích nhãn nốt (L55, L65).

### L41 — Liu J et al. 2020, *arXiv* (ChestX-Det10) [AB]
- **Bối cảnh & luận điểm chính:** Dataset phát hiện bất thường lồng ngực với box mức tổn thương, là tập con của NIH.
- **Phương pháp:** Khoảng 3.500 ảnh, 10 nhóm (abstract không liệt kê).
- **Kết quả chính:** Không nêu trong nguồn đã đọc.
- **Hạn chế:** Nhỏ, chỉ là preprint.
- **Liên quan đề tài:** Được L47 dùng để chứng minh cơ chế mất tín hiệu nốt nhỏ, nên nên dùng lại để tái lập kết quả.

---

## T10. Tổng quan hệ thống và phân tích gộp

### L33 — Essa 2025, *European Journal of Radiology Open* [AB]
- **Bối cảnh & luận điểm chính:** Phân tích gộp độ chính xác của AI trên CXR cho viêm phổi và ung thư phổi.
- **Phương pháp:** 15 nghiên cứu, khoảng 12.000 CXR.
- **Kết quả chính:** Với nốt: sensitivity gộp khoảng 72%, specificity khoảng 95%. AI làm người đọc thứ hai tăng 9–10 điểm phần trăm.
- **Hạn chế:** Ít nghiên cứu. Tác giả một mình. Độ không đồng nhất giữa các nghiên cứu: không nêu trong nguồn đã đọc.
- **Liên quan đề tài:** Số gộp làm mốc cho CAD.

### L34 — Megat Ramli et al. 2025, *Diagnostics* [AB]
- **Bối cảnh & luận điểm chính:** Tổng quan hệ thống về AI phát hiện nốt trên CXR trong sàng lọc ung thư phổi.
- **Phương pháp:** 34 nghiên cứu (2020–2024).
- **Kết quả chính:** Sensitivity 56,4–95,7%, specificity 71,9–97,5%. AUROC 0,89–0,99, so với trung bình 0,81 của bác sĩ. AI gặp khó với GGO và vùng rốn phổi.
- **Hạn chế:** Không gộp số liệu. Phạm vi rất rộng.
- **Liên quan đề tài:** Khoảng hiệu năng CAD. Các kiểu thất bại (GGO, rốn phổi) nên phân tầng khi kiểm toán VLM.

### L92 — Aggarwal et al. 2021, *npj Digital Medicine* [FT]
- **Bối cảnh & luận điểm chính:** Độ chính xác chẩn đoán của DL trong hình ảnh y khoa chưa chắc chắn. Bài tổng quan hệ thống và phân tích gộp.
- **Phương pháp:** Medline và EMBASE đến 01/2020. 11.921 bài, 503 được đưa vào. Phân tích gộp: 82 bài nhãn khoa, 82 bài vú, 115 bài hô hấp. Gộp bằng mô hình hiệu ứng ngẫu nhiên.
- **Kết quả chính:** Hô hấp: AUC 0,864–0,937 cho nốt hoặc ung thư phổi trên CXR hoặc CT. Riêng khối hoặc ung thư phổi trên CXR (8 nghiên cứu, 10 cohort): AUC 0,864 (95% CI 0,827–0,901), so với CT 0,887. Chỉ 2 nghiên cứu hô hấp có dữ liệu tiến cứu, 13 có kiểm định ngoài.
- **Hạn chế:** Tác giả nêu: độ không đồng nhất cao, thuật ngữ và chỉ số không thống nhất, nên có thể đánh giá quá cao DL; cần guideline STARD-AI. Phê bình riêng: dữ liệu đến 2020, trước thời VLM.
- **Liên quan đề tài:** Mốc gộp AUC 0,864 cho khối hoặc ung thư trên CXR. Nhắc đề tài báo cáo theo chuẩn STARD.

---

## M. Phương pháp: độ phân giải, mất cân bằng lớp, đánh giá

### L82 — Sabottke & Spieler 2020, *Radiology: Artificial Intelligence* [FT]
- **Bối cảnh & luận điểm chính:** Độ phân giải đầu vào ảnh hưởng thế nào đến CNN cho từng chẩn đoán trên CXR.
- **Phương pháp:** NIH ChestX-ray14, 112.120 ảnh, 30.805 bệnh nhân. ResNet34 và DenseNet121. Độ phân giải từ 32×32 đến 600×600. Chia 80/20. Mạng nhị phân riêng cho từng nhãn (8/14 nhãn). Chỉ số: AUC.
- **Kết quả chính:** AUC tối đa ở 256–448 px. Nốt có AUC ở 64 px bằng 80,7% ± 1,5% AUC ở 320 px (0,689 so với 0,854). Khối: 86,7% (0,767 so với 0,886). Nốt nằm trong nhóm hưởng lợi nhiều nhất từ độ phân giải cao.
- **Hạn chế:** Tác giả nêu: có đánh đổi giữa độ phân giải và batch size do bộ nhớ GPU. Phê bình riêng: nhãn NLP của NIH nhiễu; chia 80/20 không chắc theo bệnh nhân; chỉ là CNN phân loại, không phải VLM.
- **Liên quan đề tài:** Cơ sở cho thí nghiệm độ phân giải và chia ô. Cảnh báo encoder 384–512 px.

### L83 — Kim YG et al. 2020, *European Radiology* [FT]
- **Bối cảnh & luận điểm chính:** Kích thước ma trận ảnh tối ưu cho CAD DL phát hiện nốt và khối trên CXR.
- **Phương pháp:** 2.088 CXR bất thường và 352 bình thường từ 2 cơ sở. 2.758 vùng do 3 bác sĩ lồng ngực vẽ. Train và tune trên 1.736 ảnh. Test 352 ảnh bất thường và 352 bình thường. Mask R-CNN và RetinaNet ở 256, 448, 896, 1.344 và 1.792 px. Chỉ số: JAFROC FOM, sensitivity.
- **Kết quả chính:** Mask R-CNN: FOM 0,869 (896) và 0,856 (1.344), so với 0,667–0,820 ở các cỡ khác. RetinaNet: 0,906 ở 896, so với 0,329–0,832. Sensitivity với nốt nhỏ dưới 10 mm chỉ 0,418 và 0,409; với khối 0,937 và 0,956.
- **Hạn chế:** Tác giả nêu: dữ liệu nhỏ; không kiểm định ngoài hay theo thời gian; chỉ 2 CNN.
- **Liên quan đề tài:** 896 px là tối ưu, **trùng đúng độ phân giải đầu vào của MedGemma (896×896, L7)**. Nốt dưới 10 mm vẫn khó ngay cả với CAD.

### L84 — Mosquera et al. 2024, *European Radiology* [FT]
- **Bối cảnh & luận điểm chính:** Thực hành đánh giá bộ phân loại ảnh y khoa khi mất cân bằng lớp, xét cả phân biệt (discrimination) lẫn hiệu chỉnh (calibration).
- **Phương pháp:** Tổng quan ngắn tài liệu về phân loại CXR. Một ca được coi là "mất cân bằng" khi lớp thiểu số dưới 33%. Thí nghiệm DenseNet-121 trên CheXpert (11 nhãn) và ChestX-ray14 (14 nhãn) với nhiều tỷ lệ lớp.
- **Kết quả chính:** Cộng đồng chủ yếu dùng chỉ số bị lớp đa số chi phối (AUC-ROC) và hiếm khi báo cáo hiệu chỉnh. AUC-ROC che mất hiệu năng của lớp thiểu số. Khuyến nghị dùng AUC-PR, AUC-PR điều chỉnh và balanced Brier score.
- **Hạn chế:** Phần đã đọc không có hạn chế tác giả tự nêu. Phê bình riêng: thí nghiệm dùng một kiến trúc; không xét bài toán định vị.
- **Liên quan đề tài:** Quyết định chỉ số cho §8.1. Báo cáo AUC-PR, PPV và hiệu chỉnh (ví dụ ECE hoặc Brier) bên cạnh AUC, vì tỷ lệ nốt khoảng 3% (L37).

---

## Bảng tổng hợp

| ID | Cluster | Read level | Design | n | Main metric |
|---|---|---|---|---|---|
| L1 | T1 | FT | Hồi cứu, so sánh 9 mô hình zero-shot | 100 CXR (83 phân tích) | AUC 0,743 (BiomedCLIP); CXR-LLaVA sens 0,079 |
| L2 (+L46) | T1 | FT (preprint) | Hồi cứu, 6 VLM zero-shot | 247 (JSRT) | Sens 0,6–27,3%; MedGemma 5,2% (áp chót), spec 100% |
| L3 (=L54) | T1 | FT | Hồi cứu, 4 MLLM | 247 (NIH) | Ác tính đơn ổ: phát hiện 10,2% (gộp); đa ổ 55,0% |
| L4 | T1 | FT | Pilot có/không ngữ cảnh | 50 | Acc ung thư phổi 20% ở cả hai điều kiện |
| L51 | T1 | FT (tóm tắt hội nghị) | Pilot sinh báo cáo, 2 VLM | 100 (25 khối) | Acc dấu hiệu chính 62% / 51% |
| L52 | T1 | FT | Hồi cứu, ChatGPT-4o | 257 CXR (10 khối > 3 cm) | Acc CXR 66,15%; khối 9/10 |
| L53 | T1 | AB | Tóm tắt hội nghị, chatbot | 180 | Sens nốt ≤ 41,3% |
| L62 | T1 | T | Bình luận | — | — |
| L5 | T2 | AB | Phát triển MLLM | 592.580 | F1 0,81 / 0,56 |
| L6 | T2 | FT | Mô hình nền và benchmark | 1,05 triệu cặp (encoder) | Tiết kiệm 36% thời gian |
| L7 | T2 | FT | Mô hình nền, báo cáo kỹ thuật | đa miền | Macro F1 MIMIC 88,9 |
| L8 | T2 | FT | Sinh báo cáo có grounding | MIMIC, PadChest, USMix | RadFact (SOTA) |
| L9 | T2 | AB | VLM hội thoại | > 1 triệu cặp | Không nêu |
| L10 | T2 | FT | Mô hình nhỏ và chỉ số | 697.435 cặp | ROUGE-L +12,1% so với Med-PaLM M |
| L11 | T2 | AB | Agent | 2.500 truy vấn | Không nêu |
| L12 | T2 | AB | VLM tóm tắt | khoảng 217.000 | Không nêu |
| L59 | T2 | AB | LVLM định vị và phân loại | VinDr-CXR | Không nêu |
| L63 | T2 | AB | Mô hình tạo sinh, đánh giá bởi bác sĩ | 8,8 triệu / 2.145 test | Chấp nhận 70,5% |
| L64 | T2 | FT | LoRA (QLoRA) cho VLM nhỏ | MultiCaRe (không nêu n) | ROUGE-L 0,128 → 0,580 |
| L65 | T2 | FT | Nhãn từ LLM rồi huấn luyện CNN | 2.097 nghiên cứu | AUC 0,832 |
| L14 | T3 | AB | Mô hình nền CLIP | 15 triệu cặp | AUC 0,743 (ở L1) |
| L47 | T3 | AB | Probe cơ chế | ChestX-Det10 | AUC 0,688 → 0,899 |
| L50 | T3 | AB | Kết quả âm tính | ChestX-ray14 | F1 0,203 so với 0,811 |
| L66 | T3 | FT | Embedding đông cứng (CT, NLST) | 1.984 (cân bằng) | PR-AUC 0,705 |
| L68 | T3 | FT | Tiền huấn luyện VL theo thời gian | MIMIC-CXR; MS-CXR-T 1.326 | mIoU 0,243 / CNR 1,33 |
| L48 | T4 | AB | Đánh giá định vị MLLM | 9 dấu hiệu | Hit rate lesion: MedGemma 14,3% |
| L49 | T4 | AB | Benchmark nhận thức | không nêu | Chỉnh đường viền ≤ 2,9% |
| L42 | T4 | FT | Benchmark VQA | 696.000 câu / 160.000 nghiên cứu | Acc 83,24% (MedGemma) |
| L43 | T4 | AB | VQA có grounding | 17.597 câu / 4.394 ảnh | F1 0,624 |
| L44 | T4 | AB | Benchmark RSNA | 200 nghiên cứu | — |
| L69 | T4 | FT | SFT có LoRA cho sinh báo cáo | MIMIC-CXR, IU X-Ray | BLEU/ROUGE (SOTA) |
| L70 | T4 | FT | Phrase grounding tổng quát | PadChest-GR, MS-CXR | Precision@F1=1, N-Acc |
| L71 | T4 | FT | RL (GRPO) cho grounding | MS-CXR, CXR8, M3D | mIoU 79,02 |
| L55 | T5 | AB | LLM gán nhãn báo cáo | không nêu | AUC trung vị 0,84 |
| L56 | T5 | AB | So LLM với NLP | không nêu | Không nêu |
| L57 | T5 | AB | Trích xuất bằng LLM | 1.800 báo cáo | Sens 0,71–1,00 |
| L58 | T5 | AB | LLM tinh chỉnh | không nêu | Acc 0,983 |
| L15 | T5 | AB | Thực nghiệm prompt | 120 ca hư cấu | Acc 0,15 → 0,66 |
| L16 | T5 | AB | Chiến lược prompt | 327 báo cáo LDCT | Acc 68,8%, κ 0,758 |
| L17 | T5 | AB | In-context learning | không nêu | Không nêu |
| L18 | T5 | AB | So 2 LLM | 1.009 báo cáo CT | Đúng 94,6% / 92,8% |
| L19 | T6 | AB | Phát triển và kiểm định ngoài | 43.292 | AUROC 0,92–0,99 |
| L21 | T6 | AB | Kiểm định NLST | 5.485 | Sens 86,2%, spec 85,0% |
| L25 | T6 | AB | Phát triển và kiểm định ngoài | 89.834 | AUC 0,979 |
| L26 | T6 | AB | Đánh giá với chuẩn thẩm định | 759.611 | AUC nốt/khối 0,72 |
| L29 | T6 | AB | Phân đoạn | 629 / 151 | Sens 0,73 @ 0,13 mFPI |
| L31 | T6 | AB | Đọc với AI thứ hai | 247 (JSRT) | AUC bác sĩ 0,77 |
| L45 | T6 | AB | Kiểm định độc lập 5 phần mềm | 100 | AUC 0,956 → 0,812 |
| L72 | T6 | FT | Hồi cứu đa trung tâm | 2,5 triệu nghiên cứu; test 1.427 | AUC TB 0,864 (đồng thuận) |
| L73 | T6 | FT | Nhãn yếu kèm attention | > 430.000 | Acc theo kích thước (< 0,2 nếu < 10 mm) |
| L75 | T6 | AB | Kiểm định ở quần thể sàng lọc | 100.525 | AUC 0,97, FPR 3% |
| L77 | T6 | AB | Triển khai thực tế | 2.916 so với 5.681 | Tỷ lệ phát hiện 0,32% → 0,86% |
| L78 | T6 | FT | Tổng hợp ảnh để tăng cường | 2.420 / 604 | FROC AUC, NODE21 score |
| L79 | T6 | FT | Kiểm định ngoài, tỷ lệ bệnh thấp | 3.047 | AUROC 0,648 |
| L80 | T6 | FT | Kiểm định độc lập 7 sản phẩm | 386 | AUC 0,86–0,93 so với 0,81 |
| L81 | T6 | FT | Hồi cứu, ung thư bị bỏ sót | 218 | AUROC 0,899, AUFROC 0,744 |
| L20 | T7 | AB | RCT | 10.476 | Phát hiện 0,59% so với 0,25% |
| L22 | T7 | AB | Nhiều người đọc | 519 | Sens 0,61 → 0,72 |
| L27 | T7 | AB | Nhiều người đọc | 400 | +17,4% ca ung thư |
| L28 | T7 | AB | AI đọc kép | 25.104 | 71,4% ca bỏ sót là nốt |
| L32 | T7 | AB | So giao diện AI | 140 | AUC 0,87 so với 0,82 |
| L74 | T7 | AB | Nhiều người đọc | 351 | Sens 40% → 53% |
| L76 | T7 | AB | RCT bắt chéo | 228 | JAFROC 0,90 → 0,95 |
| L85 | T7 | AB | Nhiều người đọc, đa trung tâm | 800 | Sens 65,1% → 70,3% |
| L86 | T7 | FT | Nhiều người đọc, đa trung tâm | 100 | Acc +6,4% |
| L87 | T7 | FT | Nhiều người đọc, bắt chéo | 497 | Sens nốt: AI 0,816, người 0,567 |
| L88 | T7 | FT | Thực nghiệm với AI giả | 90 | FN 2,7% → 33,0% |
| L89 | T7 | AB | Nhiều người đọc, chuẩn CT | 500 | Sens nốt +12% |
| L90 | T7 | AB | Thực nghiệm ngẫu nhiên | 120 | AUC 0,77 → 0,82; đổi theo AI 59–67% |
| L23 | T8 | AB | Phát triển và kiểm định | 41.856 | AUC 0,755 |
| L24 | T8 | AB | Kiểm định ngoài | 19.488 | AUC 0,68–0,75 |
| L30 | T8 | AB | ViT tiên đoán | 259.361 | +6–10 điểm AUC |
| L13 | T8 | AB | Mô hình đa phương thức | MIMIC + Hạ Môn | Không nêu |
| L61 | T8 | T | Tóm tắt hội nghị | — | — |
| L67 | T8 | FT | Tiên lượng đa phương thức | JSRT, PadChest, NIH | C-index 0,910 (cần kiểm tra) |
| L91 | T8 | FT | Tiên lượng, cohort hồi cứu | 14.737 | Tỷ lệ ung thư 8,5% so với 2,8% |
| L35 | T9 | AB | Thử thách | NODE21 | — |
| L36 | T9 | AB | Cuộc thi | 5 đội | AUC 0,79; 53% đúng số nốt |
| L60 | T9 | T | Dataset | — | — |
| L37 | T9 | FT | Dataset có box | 18.000 | Nodule/Mass 2,73% / 5,87% |
| L38 | T9 | AB | Dataset | 108.948 | — |
| L39 | T9 | AB | Dataset | 224.316 | — |
| L40 | T9 | AB | Dataset | 377.110 ảnh | — |
| L41 | T9 | AB | Dataset có box | khoảng 3.500 | — |
| L33 | T10 | AB | Phân tích gộp | 15 nghiên cứu | Sens khoảng 72%, spec khoảng 95% |
| L34 | T10 | AB | Tổng quan hệ thống | 34 nghiên cứu | Sens 56,4–95,7% |
| L92 | T10 | FT | Tổng quan hệ thống và phân tích gộp | 503 nghiên cứu | AUC khối/ung thư trên CXR 0,864 |
| L82 | M | FT | Thực nghiệm độ phân giải | 112.120 | AUC nốt 0,689 (64 px) so với 0,854 (320 px) |
| L83 | M | FT | Thực nghiệm kích thước ma trận | 2.440 | FOM 0,906 ở 896 px |
| L84 | M | FT | Tổng quan và thực nghiệm chỉ số | CheXpert, CXR14 | AUC-PR, balanced Brier |

### Đếm mức đọc (90 bài duy nhất)

| Mức | Số bài | Danh sách |
|---|---|---|
| **FT** | **34** | L1, L2/L46, L3, L4, L51 (tóm tắt hội nghị), L52, L6, L7, L8, L10, L37, L42, L64, L65, L66, L67, L68, L69, L70, L71, L72, L73, L78, L79, L80, L81, L82, L83, L84, L86, L87, L88, L91, L92 |
| **AB** | **53** | Các bài còn lại, gồm L47, L48, L49 (có ghi chú toàn văn từ phiên trước) và L53 (tóm tắt hội nghị) |
| **T** | **3** | L60, L61, L62 |

### Phát hiện cần cập nhật vào các file trước

0. **(Bổ sung 2026-11-12 sau khi đọc toàn văn T1.)** Trong L2, MedGemma-4b-it có sensitivity 5,2%, đứng **áp chót**; thấp nhất là CheXpert Plus với 0,6%. Mô hình Qwen trong L2 là Qwen3-VL-8B, không phải Qwen2.5-VL-3B. L1 không thử MedGemma. Trong L3, ác tính đơn ổ chỉ được phát hiện khoảng 10%. DOI của L3 là 10.3390/diagnostics16030376.

1. **L66 là nghiên cứu CT liều thấp (NLST), không phải CXR.** Trong `03-thematic-clusters.md`, "bằng chứng trực tiếp rằng độ phân giải là một biến số" trên CXR cần sửa. Bằng chứng độ phân giải trên CXR còn lại là L82 và L83.
2. **L69 (S2D-Align, AAAI) đã dùng LoRA (rank 16) cho LLM sinh báo cáo CXR.** Vì vậy L64 không còn là "tiền lệ duy nhất" về LoRA. Điểm mới của đề tài phải phát biểu chính xác hơn: LoRA cho **phát hiện và định vị nốt** của VLM nhỏ.
3. **L83 cho thấy 896 px là tối ưu cho CAD nốt**, trùng với đầu vào 896×896 của MedGemma (L7). Lập luận về độ phân giải vì vậy chuyển từ "MedGemma thiếu độ phân giải" sang "độ phân giải đủ, mà vấn đề nằm ở cách gộp đặc trưng (L47) và dữ liệu".
4. **L71 đã tinh chỉnh Qwen2.5-VL cho grounding CXR (mIoU 79,02 trên MS-CXR)**. Đây là mốc trực tiếp cho VLM phụ của đề tài.
5. **L67 có rủi ro về độ tin cậy** (nguồn nhãn sống còn không rõ). Không nên dùng làm bằng chứng chính.
6. **File PDF của L39 bị nhầm**, cần tải lại bài CheXpert.
