# Báo cáo tổng quan kiểu hệ thống (systematic-review-like): VLM nhỏ cho phát hiện và định vị nốt phổi trên X-quang ngực

*Phiên bản 1, ngày 2026-11-12. Đây là **tổng hợp tường thuật theo khung PRISMA (PRISMA-informed narrative synthesis)**, không phải systematic review đã đăng ký (không có protocol PROSPERO, không có hai người sàng lọc độc lập, không có meta-analysis). Nguồn chính: `05-detailed-analysis.md` (v1). Các nguồn phụ: `03-thematic-clusters.md` (v2), `corpus.md`, `01-literature-review.md`, `02-literature-synthesis.md`, `04-scopus-prestige-search.md`, `papers/_RSNA-abstract-notes.md`. Khi 03 và 05 khác nhau, theo 05.*

## Quy ước trích dẫn và mức đọc

- Mỗi nhận định đều kèm mã bài trong `corpus.md`, theo dạng **[Lxx, mức đọc]**.
- **FT**: đã đọc toàn văn PDF cục bộ trong lượt phân tích 05.
- **AB**: chỉ dựa trên abstract (gồm trích dẫn nguyên văn trong `corpus.md` và `papers/_RSNA-abstract-notes.md`).
- **AB\***: 05 gắn AB, nhưng số liệu lấy từ ghi chú toàn văn của một phiên trước (qua Undermind `read_pdfs`) và chưa được đọc lại. Áp dụng cho L1, L47, L48, L49.
- **T**: chỉ có tiêu đề hoặc metadata (L60, L61, L62).
- ⚠ đứng trước một nhận định nghĩa là nhận định đó **chỉ dựa trên AB hoặc T**, không có bài FT nào xác nhận.
- "(suy luận)" là nhận định của người viết tổng quan, không phải kết quả của bài gốc.
- Con số nào không có trong nguồn đã đọc thì ghi "không ghi nhận".

---

> *Cập nhật 2026-11-12: cụm T1 đã được đọc toàn văn sau khi viết báo cáo này. Gồm L1, L2/L46 (qua preprint), L3, L4, L52, cùng L51 (bản gốc chỉ là tóm tắt hội nghị). Các nhãn mức đọc đã đổi sang FT; số liệu không đổi. Số liệu mới bổ sung xem `05` mục T1: MedGemma đứng áp chót về sensitivity trong L2 và bị CheXpert Plus (0,6%) vượt xuống; L3 cho thấy ác tính đơn ổ chỉ được phát hiện khoảng 10%. Một số dấu ⚠ ở các bài này nay không còn cần.*

## 1. Tóm tắt (structured abstract)

**Bối cảnh.** VLM nhỏ và mở như MedGemma-4B và Qwen2.5-VL-3B đang được đề xuất cho đọc X-quang ngực (CXR). Nhưng chưa rõ chúng phát hiện và định vị nốt phổi tốt đến đâu, và chưa rõ tinh chỉnh LoRA có thu hẹp được khoảng cách với CAD học sâu chuyên dụng (DenseNet-121, YOLOv8) hay không.

**Mục tiêu.** Tổng hợp bằng chứng về: (i) hiệu năng zero-shot của VLM/MLLM với nốt phổi trên CXR; (ii) tiền lệ tinh chỉnh VLM trên CXR; (iii) cơ chế thất bại; (iv) định vị và grounding; (v) mốc CAD và mốc bác sĩ; (vi) rủi ro tương tác người–AI; (vii) dữ liệu dùng được.

**Phương pháp.** Tìm kiếm ba vòng ngày 2026-11-12:
- Vòng 1: arXiv, PubMed, Semantic Scholar, OpenAlex, Europe PMC, Consensus, WebSearch (khoảng 85–90 lần gọi).
- Vòng 2: Undermind deep search (101 bài xếp hạng).
- Vòng 3: Scopus (3 truy vấn, 175 bản ghi).

Một người sàng lọc duy nhất, có AI hỗ trợ, ở mức tiêu đề và abstract. Kết quả: 92 mục, tức 90 bài duy nhất, chia thành 11 cụm. Mức đọc: FT 28 bài, AB 59 bài, T 3 bài. Chất lượng được đánh giá theo khung QUADAS-2 rút gọn ở mức cụm. Không gộp số liệu định lượng (không meta-analysis).

**Kết quả.**
- VLM zero-shot có sensitivity với nốt rất thấp: 0,6–27,3% trên JSRT, riêng MedGemma-4b-it là 5,2% [L2, FT].
- Trên MosMedData, MLLM đa năng tốt nhất đạt AUC 0,625 [L1, FT]. Cùng nguồn dữ liệu, CAD thương mại đạt AUC tốt nhất 0,956 [L45, AB].
- Định vị "lung lesion": MedGemma có hit rate 14,3%, so với 85,0% của bác sĩ [L48, AB\*].
- CAD độc lập đạt AUC 0,86–0,93 trong kiểm định độc lập đa sản phẩm, so với 0,81 của người đọc [L80, FT].
- Bác sĩ có sensitivity với nốt khoảng 0,47–0,65 trên các tập giàu ca bệnh [L81, FT; L85, AB; L87, FT].
- Cơ chế thất bại có bằng chứng hội tụ: gộp đặc trưng toàn cục (CLS) làm mất tín hiệu nốt nhỏ [L47, AB\*]. Độ phân giải thấp làm giảm phát hiện nốt [L82, FT]. Nhưng 896 px, đúng cỡ đầu vào của MedGemma, là tối ưu cho CAD nốt [L83, FT; L7, FT].
- Đã có tiền lệ LoRA trên CXR, nhưng cho sinh báo cáo [L64, FT; L69, FT]. Đã có tiền lệ tinh chỉnh Qwen2.5-VL cho grounding tổng quát [L71, FT]. **Chưa có bài nào tinh chỉnh VLM để phát hiện hoặc định vị nốt.**
- AI sai làm tăng lỗi của bác sĩ [L88, FT; L90, AB].

**Kết luận.** Bằng chứng nhất quán rằng VLM zero-shot chưa đạt mức dùng được cho nốt phổi trên CXR. Ngưỡng của spec (sensitivity ≥ 0,60 ở specificity 0,90) gần mức bác sĩ trên tập giàu ca bệnh, nên là mục tiêu tham vọng nhưng có cơ sở. Độ tin cậy của bằng chứng bị giới hạn bởi: mẫu nhỏ, một nguồn dữ liệu, nhiều preprint, và 2/3 số bài chỉ đọc ở mức abstract.

---

## 2. Giới thiệu và câu hỏi tổng quan

### 2.1. Bối cảnh

- Nốt phổi là loại dấu hiệu bị bỏ sót nhiều nhất khi đọc CXR. Trong một nghiên cứu đọc kép bằng AI, 71,4% (25/35) dấu hiệu có ý nghĩa bị bỏ sót là nốt [L28, AB].
- CAD học sâu đã có lịch sử kiểm định dài, gồm cả RCT [L20, AB] và kiểm định độc lập đa sản phẩm [L80, FT].
- VLM y khoa mở (MedGemma [L7, FT], CheXagent [L6, FT], LLaVA-Rad [L10, FT], MAIRA-2 [L8, FT]) báo kết quả tốt trên benchmark tổng hợp. Tuy vậy, khi được kiểm tra độc lập riêng với nốt, các mô hình này cho sensitivity rất thấp [L1, FT; L2, FT].
- Đề tài CS315 đặt MedGemma-4B làm mô hình chính, Qwen2.5-VL-3B làm mô hình so sánh, và DenseNet-121, YOLOv8 làm baseline. Mục tiêu là kiểm toán năng lực zero-shot, rồi tinh chỉnh LoRA cho phát hiện và định vị nốt.

### 2.2. Câu hỏi tổng quan

**Câu hỏi chính.** Bằng chứng hiện có nói gì về khả năng của VLM (đặc biệt VLM nhỏ, mở) trong phát hiện và định vị nốt phổi trên CXR? So với CAD chuyên dụng và với bác sĩ thì chúng đứng ở đâu? Có tiền lệ nào cho thấy tinh chỉnh (LoRA hoặc tương đương) cải thiện được năng lực này không?

**Các câu hỏi con** (ánh xạ vào §5):

| Mã | Câu hỏi con | Mục |
|---|---|---|
| Q-a | VLM zero-shot đạt sensitivity, specificity, AUC bao nhiêu với nốt? | §5(a) |
| Q-b | VLM chuyên CXR và các tiền lệ tinh chỉnh (LoRA, RL) đã làm được gì? | §5(b) |
| Q-c | Vì sao VLM bỏ sót nốt (độ phân giải, cách gộp đặc trưng)? | §5(c) |
| Q-d | VLM định vị hoặc grounding tổn thương tốt đến đâu, và đo bằng chỉ số gì? | §5(d) |
| Q-e | CAD độc lập đạt mốc nào? | §5(e) |
| Q-f | AI sai ảnh hưởng thế nào đến bác sĩ? | §5(f) |
| Q-g | Bằng chứng về dự đoán nguy cơ và sàng lọc từ CXR? | §5(g) |
| Q-h | Dataset nào có nhãn hoặc box nốt? | §5(h) |

**Khung PICO (điều chỉnh cho bài toán độ chính xác chẩn đoán).**
- **P (quần thể/ảnh):** CXR người lớn, có hoặc không có nốt, khối hay ung thư phổi.
- **I (index test):** VLM/MLLM (zero-shot hoặc tinh chỉnh), mô hình nền thị giác–ngôn ngữ, agent.
- **C (so sánh):** CAD CNN/ViT, bác sĩ.
- **O (kết cục):** sensitivity, specificity, AUC, chỉ số định vị (hit rate, IoU, JAFROC/FROC, khoảng cách tâm).

---

## 3. Phương pháp

### 3.1. Nguồn và chiến lược tìm kiếm

Mọi tìm kiếm diễn ra ngày 2026-11-12, chia ba vòng.

| Vòng | Nguồn | Số lần gọi / truy vấn | Số bản ghi | Số bài giữ (mới) | Tài liệu gốc |
|---|---|---|---|---|---|
| 1 | arXiv, PubMed, Semantic Scholar, OpenAlex, Europe PMC, Consensus, WebSearch | Khoảng 85–90 lần gọi; 63 truy vấn được ghi lại | Khoảng 470 cho 63 truy vấn được ghi; khoảng 600 trở lên trên mọi lần gọi, kể cả trùng | 44 (L1–L44) | `corpus.md` §1; `01` §1 |
| 2 | Undermind deep search (workspace fbc55738) | 1 deep search | 101 bài xếp hạng | 18 (L45–L62) | `corpus.md` §6 |
| 3 | Scopus (tài khoản trường), DOI kiểm qua Crossref | 3 truy vấn TITLE-ABS-KEY (Q1–Q3) | 36 + 116 + 23 = 175 | 30 (L63–L92), cộng 2 bài nâng cấp (L9, L35) | `04` §1 |

Chi tiết:
- Bảng nhật ký vòng 1 được dựng lại sau khi phiên bị nén ngữ cảnh, nên **không đầy đủ** [corpus §1]. Các lần gọi không được ghi chủ yếu là tra cứu bài đã biết và các biến thể truy vấn trả về 0 kết quả.
- Scopus Q2 lọc theo 13 tạp chí (Radiology, Radiology AI, Lancet Digital Health, Nature Medicine, Nature Communications, npj Digital Medicine, MedIA, IEEE TMI, European Radiology, JAMA Netw Open, Lancet Oncology, JTO, Chest). Trong 116 kết quả, chỉ 50 bài trích dẫn nhiều nhất được xem [04 §1].
- Vòng 2 so 101 bài với L1–L44. Các bài không được thêm là bài trùng hoặc bài CAD CNN có điểm liên quan thấp (r < 0,55) [corpus §6].

### 3.2. Tiêu chí chọn và loại

**Tiêu chí chọn** [01 §1; corpus đầu file]:
1. Modality là CXR.
2. Mục tiêu là nốt, khối, ung thư phổi, hoặc sàng lọc và nguy cơ ung thư phổi.
3. Mô hình thuộc một trong các nhóm: VLM/MLLM đọc ảnh; LLM chỉ đọc văn bản báo cáo; mô hình nền thị giác–ngôn ngữ kiểu CLIP; CNN/ViT kinh điển làm mốc (có giới hạn số lượng).
4. Bài toán: phát hiện/phân loại, sinh báo cáo, VQA/grounding, sàng lọc/triage.
5. Ngoại lệ có chủ đích:
   - Giữ khoảng 6 VLM nền cho CXR làm bối cảnh dù không đánh giá riêng nốt (CheXagent, MAIRA-2, RadVLM, LLaVA-Rad, MedGemma, CXR-LLaVA).
   - Vòng 3 bổ sung các bài phương pháp (độ phân giải, mất cân bằng lớp) và các bài grounding/LoRA tổng quát (L64, L68–L71), vì chúng trực tiếp quyết định thiết kế spec.

**Tiêu chí loại** [01 §1; corpus §4; 04 §1]:
- Chỉ dùng CT. *Ngoại lệ:* nhánh LLM văn bản (L16, L17, L18) dùng báo cáo CT, giữ làm bằng chứng lân cận. L66 bị xếp nhầm là CXR ở 03, xem §6.
- Bệnh học (pathology).
- VLM cho CXR nói chung, không có kết quả riêng cho nốt hay khối (ví dụ CheXzero, ELIXR, M4CXR so với GPT-4o).
- Bài tổng quan tường thuật.
- X-quang nhi, bụi phổi silic, tăng áp động mạch phổi, tràn khí màng phổi đơn thuần, NLP không liên quan phổi, mục lục kỷ yếu, bài trùng [04 §1].

### 3.3. Lưu đồ kiểu PRISMA

Các con số dưới đây lấy nguyên từ tài liệu nguồn. Con số "suy ra" là phép trừ đơn giản từ hai số đã ghi. Ô nào tài liệu nguồn không có số thì ghi "không ghi nhận".

```
┌───────────────────────────── NHẬN DIỆN (Identification) ─────────────────────────────┐
│ Vòng 1: 7 nguồn, ~85–90 lần gọi        Vòng 2: Undermind        Vòng 3: Scopus Q1–Q3 │
│ ~470 bản ghi (63 truy vấn ghi lại);    101 bài xếp hạng         36 + 116 + 23 = 175  │
│ ≥ ~600 kể cả trùng (mọi lần gọi)                                                     │
│ Tổng số bản ghi sau khi bỏ trùng giữa các vòng: không ghi nhận                       │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           ▼
┌────────────────────────────── SÀNG LỌC (Screening) ──────────────────────────────────┐
│ Vòng 1: số bản ghi sàng lọc theo tiêu đề/abstract: không ghi nhận                    │
│ Vòng 2: 101 so với L1–L44                                                            │
│ Vòng 3: Q1 36 và Q3 23 xem theo tiêu đề; Q2 chỉ xem 50/116 bài trích dẫn cao nhất    │
│                                                                                      │
│ Loại:                                                                                │
│ • Vòng 1: tổng số bị loại không ghi nhận; 15 bài "loại nhưng đáng chú ý" có lý do    │
│   (corpus §4); 3 bài chỉ thấy tiêu đề, chưa sàng lọc (corpus §3)                     │
│ • Vòng 2: 83 không thêm (suy ra 101 − 18): trùng hoặc điểm liên quan r < 0,55        │
│ • Vòng 3: Q1 28 (suy ra 36 − 6 − 2 trùng); Q2 29 trong 50 bài đã xem                 │
│   (suy ra 50 − 20 − 1 nâng cấp), 66 không xem; Q3 18 (suy ra 23 − 4 − 1 nâng cấp)    │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           ▼
┌──────────────────────── LẤY TOÀN VĂN (Retrieval / Eligibility) ──────────────────────┐
│ Có PDF cục bộ: 22/30 bài vòng 3 (8 bài RSNA không lấy được: L63, L74–L77, L85,       │
│ L89, L90); thêm PDF của L6, L7, L8, L10, L37, L42 từ thư mục papers/ có sẵn          │
│ PDF lỗi: L39 (nhầm bài khác), L40 (khác phiên bản, xem §8)                           │
│ L1, L47, L48, L49: đọc toàn văn ở phiên trước qua Undermind, không có PDF cục bộ     │
│ Không đánh giá toàn văn để loại thêm: mọi bài đã chọn ở mức abstract đều được giữ    │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           ▼
┌──────────────────────────────── ĐƯA VÀO (Included) ──────────────────────────────────┐
│ 44 + 18 + 30 = 92 mục (L1–L92); 2 cặp trùng (L46 = L2, L54 = L3) → 90 bài duy nhất   │
│ 11 cụm: T1 8 · T2 12 · T3 5 · T4 8 · T5 8 · T6 15 · T7 13 · T8 7 · T9 8 · T10 3 · M 3│
│ Mức đọc: FT 28 · AB 59 (gồm 4 AB*) · T 3                                             │
│ Tổng hợp định lượng (meta-analysis): 0 — chỉ tổng hợp tường thuật và bảng khoảng số  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.4. Quy trình sàng lọc: một người, có AI hỗ trợ

- Sàng lọc và trích dữ liệu do **một người duy nhất**, với trợ lý AI (Claude, qua các MCP tìm kiếm và Undermind). **Không có người thứ hai** sàng lọc độc lập, nên không có chỉ số đồng thuận (kappa).
- Vòng 1 sàng lọc chỉ ở mức abstract [corpus đầu file]. Vòng 3 chỉ đọc tiêu đề, nguồn, năm và số trích dẫn trên Scopus khi chọn bài [04 đầu file].
- Hệ quả:
  - Có thể sót bài (lỗi recall), nhất là ở vòng 1 vì nhật ký không đầy đủ, và ở Scopus Q2 vì 66/116 kết quả không được xem.
  - Có thể có lỗi gán cụm, đã thấy ít nhất một lần: L66 bị coi là bằng chứng độ phân giải trên CXR trong 03, nhưng thực ra là CT [L66, FT].
  - Trích số liệu bởi AI có thể sai. Để giảm rủi ro, 05 chỉ ghi số có trong nguồn đã đọc, và ghi "không nêu" nếu không có.
- Hạng nguồn (Q1/Q2, A*) trong 04 là "theo hiểu biết chung", **chưa tra lại từng bài** [04 §3].

### 3.5. Trích dữ liệu và mức đọc

Trích theo mẫu cố định trong 05 cho mọi bài: bối cảnh, phương pháp, kết quả chính, hạn chế (tác giả nêu và phê bình riêng), liên quan đề tài.

| Mức đọc | Số bài | Tỷ lệ (trên 90) | Danh sách / ghi chú |
|---|---|---|---|
| **FT** | 28 | 31% | L6, L7, L8, L10, L37, L42, L64, L65, L66, L67, L68, L69, L70, L71, L72, L73, L78, L79, L80, L81, L82, L83, L84, L86, L87, L88, L91, L92 |
| **AB** | 59 | 66% | Các bài còn lại. Trong đó L1, L47, L48, L49 là AB\* (có ghi chú toàn văn từ phiên trước); 8 bài RSNA (L63, L74–L77, L85, L89, L90) chỉ có abstract qua Undermind |
| **T** | 3 | 3% | L60, L61, L62 |

Phân bố mức đọc theo cụm (đếm từ bảng tổng hợp trong 05):

| Cụm | FT | AB | T | Tổng |
|---|---|---|---|---|
| T1 Zero-shot VLM | 0 | 7 (L1 là AB\*) | 1 (L62) | 8 |
| T2 VLM chuyên CXR | 6 (L6, L7, L8, L10, L64, L65) | 6 | 0 | 12 |
| T3 Encoder thị giác–ngôn ngữ | 2 (L66, L68) | 3 (L47 là AB\*) | 0 | 5 |
| T4 Định vị / VQA | 4 (L42, L69, L70, L71) | 4 (L48, L49 là AB\*) | 0 | 8 |
| T5 LLM văn bản | 0 | 8 | 0 | 8 |
| T6 CAD độc lập | 6 (L72, L73, L78, L79, L80, L81) | 9 | 0 | 15 |
| T7 Người–AI | 3 (L86, L87, L88) | 10 | 0 | 13 |
| T8 Nguy cơ / sàng lọc | 2 (L67, L91) | 4 | 1 (L61) | 7 |
| T9 Dataset | 1 (L37) | 6 | 1 (L60) | 8 |
| T10 Tổng quan | 1 (L92) | 2 | 0 | 3 |
| M Phương pháp | 3 (L82, L83, L84) | 0 | 0 | 3 |
| **Tổng** | **28** | **59** | **3** | **90** |

**Điểm yếu quan trọng nhất:** cụm lõi T1 **không có bài nào FT** trong lượt 05. Hai bài trụ cột là L1 (AB\*) và L2 (AB).

### 3.6. Đánh giá chất lượng và tổng hợp

- Dùng khung **QUADAS-2 rút gọn** ở mức cụm, không chấm từng bài. Bốn miền: chọn bệnh nhân, index test, tiêu chuẩn tham chiếu, luồng và thời gian. Cộng thêm bốn cờ rủi ro theo yêu cầu: mẫu nhỏ, dữ liệu một nguồn, preprint, chỉ đọc abstract (§6).
- Tổng hợp tường thuật theo 8 chủ đề (§5), kèm bảng khoảng số liệu. **Không gộp thống kê**, vì thiết kế, ngưỡng, tập dữ liệu và chỉ số quá khác nhau.

---

## 4. Đặc điểm các nghiên cứu

### 4.1. Tổng quan theo cụm

| Cụm | Số bài | Giai đoạn [03] | Thiết kế chủ đạo | Vai trò với đề tài |
|---|---|---|---|---|
| T1 Đánh giá zero-shot VLM cho nốt | 8 | 2025–2026 | Hồi cứu, so sánh mô hình, mẫu 50–247 | Xác lập vấn đề |
| T2 VLM, agent, mô hình tạo sinh chuyên CXR | 12 | 2023–2026 | Phát triển mô hình, báo cáo kỹ thuật | Mô hình ứng viên; tiền lệ tinh chỉnh |
| T3 Encoder thị giác–ngôn ngữ, giới hạn biểu diễn | 5 | 2023–2026 | Mô hình nền, probe cơ chế | Cơ chế thất bại |
| T4 Định vị, grounding, VQA | 8 | 2025–2026 | Benchmark, phương pháp grounding | Phương pháp đo định vị |
| T5 LLM chỉ đọc văn bản | 8 | 2023–2026 | Trích xuất, phân loại báo cáo | Hướng mở rộng |
| T6 CAD độc lập | 15 | 2019–2024 | Phát triển, kiểm định ngoài, kiểm định độc lập | Mốc so sánh |
| T7 Người–AI và tác động lâm sàng | 13 | 2019–2024 | Nhiều người đọc, RCT, thực nghiệm | Mốc lâm sàng; rủi ro AI sai |
| T8 Nguy cơ và sàng lọc | 7 | 2020–2026 | Cohort tiên lượng | Hướng mở rộng |
| T9 Dataset có nhãn nốt/khối | 8 | 2017–2026 | Mô tả dataset, challenge | Nguồn dữ liệu |
| T10 Tổng quan hệ thống, phân tích gộp | 3 | 2021–2025 | SR / MA | Số gộp làm mốc |
| M Phương pháp | 3 | 2020–2024 | Thực nghiệm phương pháp | Quyết định thiết kế spec |

**Loại nguồn trong toàn bộ corpus** (theo cột PR? trong `corpus.md`, có cập nhật từ 04 và 05):
- 21 mục ghi là preprint (PP) ở corpus: L6, L8, L11, L12, L13, L14, L27, L30, L31, L36, L38, L39, L40, L41, L42, L43, L44, L47, L49, L50, L59. Theo 05, L38 (CVPR) và L39 (AAAI) có bản đã đăng, nên còn 19 preprint thực sự.
- L9 (nay ở *Scientific Reports*) và L35 (nay ở *IEEE TMI*) đã được nâng từ PP lên bình duyệt [04 §2].
- Tóm tắt hội nghị: L53, L61. Bình luận: L62. Kỷ yếu: L17, L60, L64 (SCM, hội nghị hạng thấp theo 04).
- **Cụm LLM/VLM (T1–T4)** có tỷ lệ preprint cao. **Cụm CAD/lâm sàng (T6, T7)** gần như toàn bộ ở tạp chí bình duyệt, đa số là *Radiology*, *European Radiology*, *JAMA Netw Open*.

### 4.2. Bảng đặc điểm theo cụm

Cột "Nguồn / hạng": tên tạp chí hoặc hội nghị; PR = bình duyệt, PP = preprint. Hạng Q1/Q2/A\* chỉ ghi cho L63–L92 và lấy theo 04 (chưa tra lại). "n" là cỡ mẫu chính theo 05; "không ghi nhận" nếu nguồn không nêu.

#### T1. Đánh giá zero-shot MLLM/VLM cho nốt và ung thư phổi

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L1 | 2026 | Front Digit Health, PR | Hồi cứu, so 9 hệ thống | MosMedData VII, 100 CXR (83 phân tích) | MLLM đa năng, VLM CXR, CLIP, agent | FT |
| L2 (+L46) | 2026 | AJR, PR (L46: medRxiv) | Hồi cứu, 6 VLM | JSRT, 247 | VLM (MedGemma-4b-it, RadVLM, LLaVA-Rad...) | FT |
| L3 (=L54) | 2026 | Diagnostics, PR | Hồi cứu, 4 MLLM | 247 CXR, 13 nhóm | MLLM đóng (Gemini, GPT-4) | FT |
| L4 | 2026 | Cureus, PR | Pilot có/không ngữ cảnh | 50 | MLLM (ChatGPT) | FT |
| L51 | 2026 | RöFo, PR | Pilot sinh báo cáo | 100 (25 khối) | MLLM (Claude 3.5, GPT-4) | FT |
| L52 | 2025 | J Pers Med, PR | Hồi cứu | không ghi nhận | MLLM (GPT-4o) | FT |
| L53 | 2025 | ERS Congress, tóm tắt hội nghị | Hồi cứu | 180 | Chatbot (có Mistral) | AB |
| L62 | 2026 | AJR, bình luận | Bình luận về L2 | — | — | T |

#### T2. VLM, agent và mô hình tạo sinh chuyên CXR

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L5 | 2025 | Eur Radiol, PR | Phát triển MLLM | 592.580 CXR | VLM CXR (CXR-LLaVA) | AB |
| L6 | 2024 | arXiv, PP | Mô hình nền và benchmark | 1.052.257 cặp (encoder) | VLM CXR (CheXagent) | FT |
| L7 | 2026 | Nature Medicine, PR | Báo cáo kỹ thuật mô hình nền | Đa miền | VLM y khoa (MedGemma 4B/27B, MedSigLIP) | FT |
| L8 | 2024 | arXiv, PP | Sinh báo cáo có grounding | MIMIC-CXR, PadChest, USMix | VLM CXR (MAIRA-2) | FT |
| L9 | 2026 | Sci Rep, PR | VLM hội thoại | > 1 triệu cặp | VLM CXR có grounding (RadVLM) | AB |
| L10 | 2025 | Nat Commun, PR | Mô hình nhỏ và chỉ số | 697.435 cặp | VLM 7B (LLaVA-Rad) | FT |
| L11 | 2025 | arXiv, PP | Agent | 2.500 truy vấn | Agent (MedRAX) | AB |
| L12 | 2023 | arXiv, PP | VLM tóm tắt | khoảng 217.000 tóm tắt | VLM (XrayGPT) | AB |
| L59 | 2024 | arXiv, PP | LVLM định vị và phân loại | VinDr-CXR | LVLM nhỏ (LiteGPT) | AB |
| L63 | 2025 | Radiology, Q1 | Mô hình tạo sinh, đánh giá bởi bác sĩ | 8,8 triệu cặp; test 2.145 | VLM chuyên ngành | AB |
| L64 | 2025 | SCM 2025 (IEEE), hội nghị hạng thấp | QLoRA cho VLM nhỏ | MultiCaRe (n không ghi nhận) | SmolVLM khoảng 2,26 tỷ tham số | FT |
| L65 | 2026 | Front Digit Health, Q2 | Nhãn từ LLM, rồi huấn luyện CNN | 2.097 nghiên cứu MIMIC | GPT-4o + CNN (có DenseNet-121) | FT |

#### T3. Bộ mã hóa thị giác–ngôn ngữ và giới hạn biểu diễn

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L14 | 2023 | arXiv, PP | Mô hình nền CLIP | PMC-15M | CLIP y sinh (BiomedCLIP) | AB |
| L47 | 2026 | arXiv, PP | Probe cơ chế | ChestX-Det10 | BiomedCLIP, MedSigLIP (đông cứng) | AB\* |
| L50 | 2025 | medRxiv, PP | Kết quả âm tính | ChestX-ray14 | VLM nền so với CNN | AB |
| L66 | 2025 | J Imaging Inform Med, Q1/Q2 | Embedding đông cứng **trên CT** | NLST CT; 1.984 (tập cân bằng) | RAD-DINO + MLP | FT |
| L68 | 2023 | CVPR, A\* | Tiền huấn luyện VL theo thời gian | MIMIC-CXR; MS-CXR-T 1.326 cặp | CNN–Transformer + CXR-BERT (BioViL-T) | FT |

#### T4. Định vị, grounding và benchmark VQA

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L48 | 2025 | PMLR (ML4H), PR | Đánh giá định vị | 9 dấu hiệu; số ca lung lesion không ghi nhận | GPT-4, GPT-5, MedGemma, CNN, bác sĩ | AB\* |
| L49 | 2026 | arXiv, PP | Benchmark nhận thức nhiều mức | không ghi nhận (không có nốt/khối) | VLM y khoa và đa năng | AB\* |
| L42 | 2025 | arXiv, PP | Benchmark VQA | khoảng 696.000 câu / 160.000 nghiên cứu | 8 MLLM (có MedGemma-4B-it, Qwen2.5-VL) | FT |
| L43 | 2025 | arXiv, PP | VQA có grounding | 17.597 câu / 4.394 ảnh | MedGemma-4B-it | AB |
| L44 | 2026 | arXiv, PP | Benchmark RSNA | 200 nghiên cứu | LLM đa phương thức | AB |
| L69 | 2026 | AAAI, A\* | SFT có LoRA cho sinh báo cáo | MIMIC-CXR, IU X-Ray | Vicuna-7B + LoRA r16 | FT |
| L70 | 2026 | IEEE TMI, Q1 | Phrase grounding tổng quát | PadChest-GR, MS-CXR | MedGrounder | FT |
| L71 | 2025 | MICCAI, A | RL (GRPO) cho grounding | MS-CXR, ChestX-ray8, M3D-RefSeg | Qwen2.5-VL (cỡ không ghi nhận) | FT |

#### T5. LLM chỉ đọc văn bản báo cáo

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L55 | 2023 | Radiology, PR | LLM gán nhãn báo cáo CXR | không ghi nhận | Vicuna (cục bộ) | AB |
| L56 | 2024 | AJR, PR | So LLM với NLP | không ghi nhận | GPT-4 | AB |
| L57 | 2024 | PLOS ONE, PR | Trích xuất | 1.800 báo cáo CXR + CT | GPT-4 | AB |
| L58 | 2024 | J Imaging Inform Med, PR | LLM tinh chỉnh | không ghi nhận (modality không rõ) | LLM tinh chỉnh | AB |
| L15 | 2024 | Can Assoc Radiol J, PR | Thực nghiệm prompt | 120 mô tả hư cấu | GPT-3.5, GPT-4 | AB |
| L16 | 2026 | Clin Radiol, PR | Chiến lược prompt | 327 báo cáo LDCT | GPT-4o | AB |
| L17 | 2024 | Kỷ yếu (PubMed 41726431) | In-context learning | Báo cáo LDCT, n không ghi nhận | LLM | AB |
| L18 | 2025 | Eur J Radiol Open, PR | So 2 LLM | 1.009 báo cáo CT | ERNIE-4.0, GPT-4o-mini | AB |

#### T6. CAD học sâu, hiệu năng độc lập

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L19 | 2019 | Radiology, PR | Phát triển và kiểm định ngoài | 43.292 | CNN (DLAD) | AB |
| L21 | 2020 | JAMA Netw Open, PR | Kiểm định trên NLST | 5.485 người | CNN thương mại (Lunit) | AB |
| L25 | 2019 | JAMA Netw Open, PR | Phát triển và kiểm định ngoài | 89.834 | CNN đa bệnh | AB |
| L26 | 2020 | Radiology, PR | Đánh giá với chuẩn đã thẩm định | 759.611 + ChestX-ray14 | CNN (Google) | AB |
| L29 | 2022 | Sci Rep, PR | Phân đoạn | 629 train / 151 test | CNN phân đoạn | AB |
| L31 | 2022 | arXiv, PP | AI làm người đọc thứ hai | JSRT 247; 8 bác sĩ | CNN thương mại | AB |
| L45 | 2024 | Quant Imaging Med Surg, PR | Kiểm định độc lập 5 phần mềm | MosMedData VII, 100 | CNN thương mại | AB |
| L72 | 2024 | Lancet Digit Health, Q1 | Hồi cứu đa trung tâm | 2,5 triệu nghiên cứu; test đồng thuận 1.427 | CNN mở (X-Raydar) | FT |
| L73 | 2019 | Med Image Anal, Q1 | Nhãn yếu kèm attention | > 430.000 | CNN + attention (CONAF, RAMAF) | FT |
| L75 | 2020 | Radiology, Q1 | Kiểm định ở quần thể sàng lọc | 100.525 | CNN thương mại | AB |
| L77 | 2021 | Radiology, Q1 | Triển khai thực tế | 2.916 so với 5.681 | CNN thương mại | AB |
| L78 | 2023 | Med Image Anal, Q1 | Tổng hợp ảnh để tăng cường | 2.420 / 604 | GAN + detector | FT |
| L79 | 2023 | Eur Radiol, Q1 | Kiểm định ngoài, tỷ lệ bệnh thấp | 3.047 | CNN thương mại | FT |
| L80 | 2024 | Radiology, Q1 | Kiểm định độc lập 7 sản phẩm | 386 bệnh nhân | CNN thương mại | FT |
| L81 | 2020 | Radiol Cardiothorac Imaging, Q1/Q2 | Hồi cứu, ung thư bị bỏ sót | 218 CXR | CNN (DLAD) | FT |

#### T7. Tương tác người–AI và tác động lâm sàng

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L20 | 2023 | Radiology, PR | RCT | 10.476 người | CNN thương mại | AB |
| L22 | 2021 | Eur Radiol, PR | Nhiều người đọc | 519 ảnh NLST; 8 người đọc | CNN | AB |
| L27 | 2022 | arXiv, PP | Nhiều người đọc | 400; 11 bác sĩ | CNN thương mại (Behold.ai) | AB |
| L28 | 2024 | Eur Radiol, PR | AI đọc kép | 25.104 | CNN + NLP | AB |
| L32 | 2023 | Radiol AI, PR | So giao diện AI | 140; 10 bác sĩ | CNN | AB |
| L74 | 2020 | Radiology, Q1 | Nhiều người đọc | 351 (117 ung thư, 234 chứng); 9 người đọc | CNN (DLAD) | AB |
| L76 | 2021 | Radiology, Q1 | RCT bắt chéo | 228; 6 người đọc | CNN đa dấu hiệu | AB |
| L85 | 2020 | Radiology, Q1 | Nhiều người đọc, 4 trung tâm | 800 (600 ung thư); 12 bác sĩ | DCNN | AB |
| L86 | 2021 | JAMA Netw Open, Q1 | Nhiều người đọc, đa trung tâm | 100; 9 bác sĩ | CNN thương mại | FT |
| L87 | 2022 | JAMA Netw Open, Q1 | Nhiều người đọc, bắt chéo | 497; 6 bác sĩ | CNN (heat map) | FT |
| L88 | 2023 | Eur Radiol, Q1 | Thực nghiệm với AI giả | 90; 6 bác sĩ | AI giả lập | FT |
| L89 | 2023 | Radiology, Q1 | Nhiều người đọc, chuẩn CT | 500; 12 người đọc | CNN | AB |
| L90 | 2023 | Radiology, Q1 | Thực nghiệm ngẫu nhiên | 120; 30 người đọc | CNN chính xác cao và thấp | AB |

#### T8. Dự đoán nguy cơ và sàng lọc

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L23 | 2020 | Ann Intern Med, PR | Phát triển và kiểm định | PLCO 41.856; NLST | CNN (CXR-LC) | AB |
| L24 | 2022 | Radiology, PR | Kiểm định ngoài | 19.488 | CNN (CXR-LC) | AB |
| L30 | 2026 | arXiv, PP | ViT tiên đoán | 259.361 | ViT | AB |
| L13 | 2026 | arXiv, PP | Mô hình đa phương thức | MIMIC + cohort Hạ Môn | Đa phương thức + đồ thị tri thức | AB |
| L61 | 2026 | J Thorac Oncol, tóm tắt hội nghị | không ghi nhận | không ghi nhận | CXR + embedding LLM | T |
| L67 | 2026 | Transl Lung Cancer Res, Q1/Q2 | Tiên lượng đa phương thức | JSRT, PadChest, NIH | SAM-Med2D + BioGPT | FT |
| L91 | 2022 | JAMA Netw Open, Q1 | Tiên lượng, cohort hồi cứu | 14.737 | CNN + EMR (CXR-LC) | FT |

#### T9. Dataset và benchmark có nhãn nốt/khối

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Nhãn nốt | Mức đọc |
|---|---|---|---|---|---|---|
| L35 | 2024 | IEEE TMI, PR | Challenge (NODE21) | không ghi nhận | Box nốt | AB |
| L36 | 2026 | arXiv, PP | Cuộc thi (NoduLoCC2026) | 5 đội | Vị trí nốt | AB |
| L60 | 2026 | Kỷ yếu ICCVIT | Dataset NoduLoCC | không ghi nhận | không ghi nhận | T |
| L37 | 2022 | Sci Data, PR | Dataset có box | 18.000 ảnh | Box Nodule/Mass | FT |
| L38 | 2017 | CVPR (corpus ghi arXiv) | Dataset | 108.948 | Nhãn NLP; box cho tập con | AB |
| L39 | 2019 | AAAI (corpus ghi arXiv) | Dataset | 224.316 | "Lung Lesion" | AB |
| L40 | 2019 | arXiv, PP | Dataset | 377.110 ảnh | 14 nhãn tự động; không box | AB |
| L41 | 2020 | arXiv, PP | Dataset có box | khoảng 3.500 | Box (10 nhóm) | AB |

#### T10. Tổng quan hệ thống và phân tích gộp

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L33 | 2025 | Eur J Radiol Open, PR | Phân tích gộp | 15 nghiên cứu, khoảng 12.000 CXR | CAD AI | AB |
| L34 | 2025 | Diagnostics, PR | Tổng quan hệ thống, không gộp | 34 nghiên cứu | CAD AI | AB |
| L92 | 2021 | npj Digit Med, Q1 | SR và MA | 503 nghiên cứu (115 hô hấp) | DL hình ảnh | FT |

#### M. Phương pháp

| ID | Năm | Nguồn / hạng | Thiết kế | Dữ liệu (n) | Loại mô hình | Mức đọc |
|---|---|---|---|---|---|---|
| L82 | 2020 | Radiol AI, PR | Thực nghiệm độ phân giải | NIH ChestX-ray14, 112.120 | ResNet34, DenseNet121 | FT |
| L83 | 2020 | Eur Radiol, PR | Thực nghiệm kích thước ma trận | 2.440 CXR | Mask R-CNN, RetinaNet | FT |
| L84 | 2024 | Eur Radiol, PR | Tổng quan ngắn + thực nghiệm chỉ số | CheXpert, ChestX-ray14 | DenseNet-121 | FT |

---

## 5. Tổng hợp kết quả theo chủ đề

### (a) Hiệu năng zero-shot của VLM với nốt phổi

**Bằng chứng chính** (toàn bộ ở mức AB hoặc AB\*):
- ⚠ **L2 (JSRT, 247 ảnh, 6 VLM):** accuracy 36,4–44,5%, sensitivity 0,6–27,3%, specificity 71,0–100% [L2, FT].
  - RadVLM có accuracy cao nhất (44,5%) nhưng sensitivity chỉ 11,0%.
  - LLaVA-Rad có sensitivity cao nhất (27,3%).
  - **MedGemma-4b-it: sensitivity 5,2%** [L2, FT].
  - Bản preprint thêm nhận xét: mô hình phát hiện được nốt rõ, kém với nốt mờ [L46, AB].
- ⚠ **L1 (MosMedData VII, 83/100 ảnh phân tích, 9 hệ thống):** [L1, FT]
  - BiomedCLIP có AUC tốt nhất 0,743, rồi MedRAX 0,704. Accuracy của cả hai là 0,711.
  - MLLM đa năng tốt nhất: Claude 3.7 Sonnet, AUC 0,625, sensitivity 0,316, specificity 0,933.
  - CXR-LLaVA: sensitivity 0,079.
- ⚠ **Các nghiên cứu đa bệnh, cùng hướng:**
  - Tổn thương nhỏ bị bỏ sót [L3, FT].
  - ChatGPT đúng 20% với ung thư phổi [L4, FT].
  - Khối phổi "ít được nhận đúng hơn" [L51, FT].
  - Sensitivity với nốt tối đa 41,3% [L53, AB, tóm tắt hội nghị].
- ⚠ **Phản ví dụ:** "tumor" thuộc nhóm phát hiện tốt nhất với GPT-4o [L52, FT]. Có thể do chọn ca khối lớn (suy luận theo 05).
- ⚠ Có bình luận của biên tập AJR kêu gọi thận trọng [L62, T].

**Bằng chứng gián tiếp (FT):**
- MedGemma đạt 83,24% trên ReXVQA, và 83,84% so với 77,27% của bác sĩ nội trú tốt nhất trên 200 ca. Nhưng benchmark này không có tầng riêng cho nốt [L42, FT]. Đây là **nghịch lý benchmark tổng hợp**: điểm VQA cao che mất điểm yếu với nốt (suy luận, nối L42 với L2).
- MedGemma không có nhãn nốt hay khối trong các tác vụ phân loại CXR được đánh giá [L7, FT].

**Nhận định tổng hợp.**
- Kiểu sai nhất quán: sensitivity thấp, specificity cao. Các VLM chuyên CXR nghiêng về trả lời "bình thường" [L2, FT; L1, FT].
- Mức chắc chắn **trung bình–thấp**. Kết quả cùng hướng ở ít nhất 6 bài độc lập. Nhưng cả 6 đều mẫu nhỏ, một nguồn, và không bài nào được đọc FT trong lượt 05.

### (b) VLM chuyên CXR và các tiền lệ tinh chỉnh

**VLM chuyên CXR vẫn yếu với nốt:**
- CXR-LLaVA: F1 0,81 nội bộ, 0,56 ngoài phân phối [L5, AB]. Sensitivity với nốt 0,079 [L1, FT].
- RadVLM có grounding nhưng sensitivity với nốt chỉ 11,0% [L9, AB; L2, FT].
- LLaVA-Rad mạnh về sinh báo cáo [L10, FT] nhưng sensitivity với nốt chỉ 27,3% [L2, FT].
- MedGemma cải thiện phân loại CXR 15,5–18,1% so với mô hình gốc. Macro F1 trên MIMIC 88,9 so với 81,2 của Gemma 3 4B [L7, FT]. Phần tinh chỉnh dùng SFT toàn tham số, **không dùng LoRA**, và không có nhãn nốt [L7, FT].
- ⚠ Một mô hình tạo sinh chuyên ngành (8,8 triệu cặp) có 70,5% báo cáo được chấp nhận không sửa, so với 29,6% của GPT-4V. Không có số riêng cho nốt [L63, AB].

**Tiền lệ tinh chỉnh (toàn bộ FT):**

| ID | Mô hình | Kỹ thuật | Tác vụ | Kết quả chính | Khác với đề tài |
|---|---|---|---|---|---|
| L64 | SmolVLM khoảng 2,26 tỷ tham số | QLoRA 4-bit, 0,47% tham số | Sinh báo cáo cấu trúc (MultiCaRe) | ROUGE-L 0,128 → 0,580; BERTScore F1 0,820 → 0,938 | Chỉ số NLG; ví dụ minh họa sai lâm sàng; không phát hiện hay định vị nốt [L64, FT] |
| L69 | Vicuna-7B (LLM) | LoRA r16, alpha 16, mọi lớp tuyến tính | Sinh báo cáo CXR | SOTA BLEU/ROUGE (số cụ thể không ghi nhận) | LoRA áp vào LLM, không vào encoder ảnh; không định vị [L69, FT] |
| L71 | Qwen2.5-VL (cỡ không ghi nhận) | RL (GRPO), phần thưởng IoU và ngữ nghĩa | Phrase grounding | mIoU 79,02 trên MS-CXR, so với 52,7–62,8 của baseline | Grounding có cho trước cụm từ, không phải phát hiện; không riêng nốt [L71, FT] |
| L6 | CheXagent | Huấn luyện lại encoder, nâng độ phân giải 384 → 512 | Đa tác vụ | Bác sĩ nội trú tiết kiệm 36% thời gian | Không có kết quả riêng cho nốt [L6, FT] |

**Rủi ro khi tinh chỉnh:**
- ⚠ Tinh chỉnh VLM nền trên ChestX-ray14 làm sensitivity sụp xuống dưới 15%. F1 của VLM tốt nhất là 0,203, so với 0,811 của CNN [L50, AB, preprint].
- Chỉ số NLG không đo được độ đúng lâm sàng [L64, FT].

**Nhận định tổng hợp.**
- Chắc chắn (FT, 3 bài): đã có LoRA trên CXR [L64, L69] và tinh chỉnh Qwen2.5-VL cho grounding [L71].
- Trong phạm vi tìm kiếm (04 Q1, Q3; 05), **không có bài nào tinh chỉnh VLM để phát hiện hoặc định vị nốt**.
- Điểm mới của đề tài vì vậy phải phát biểu chính xác: "LoRA cho phát hiện và định vị nốt của VLM nhỏ", không phải "LoRA đầu tiên trên CXR" [05, phát hiện 2].

### (c) Cơ chế thất bại: độ phân giải và cách gộp đặc trưng

**1. Gộp toàn cục làm mất tín hiệu nốt nhỏ.** ⚠ [L47, AB\*, preprint]
- AUC mức box với probe trên CLS so với gộp patch-local:
  - BiomedCLIP 0,688 → 0,899.
  - MedSigLIP 0,805 → 0,955.
- CLS mức ảnh của MedSigLIP: nốt nhỏ 0,627, nốt lớn 0,844.
- MedSigLIP là encoder của MedGemma [L7, FT]. Vậy thông tin nốt **còn trong patch token** nhưng mất khi gộp.
- Hạn chế: cần ROI biết trước (oracle), chỉ một dataset.
- Gián tiếp: MAIRA-2 không dùng token CLS mà đưa toàn bộ 1.369 patch token vào LLM [L8, FT] (suy luận: một thiết kế tránh được cơ chế này).

**2. Độ phân giải đầu vào.**
- Trên NIH (CNN): AUC tối đa ở 256–448 px. AUC nốt ở 64 px chỉ bằng 80,7% ± 1,5% so với ở 320 px (0,689 so với 0,854). Nốt thuộc nhóm hưởng lợi nhiều nhất từ độ phân giải cao [L82, FT].
- Trên CAD phát hiện nốt và khối:
  - Mask R-CNN: JAFROC FOM 0,869 ở 896 px.
  - RetinaNet: 0,906 ở 896 px, so với 0,329–0,832 ở các cỡ khác.
  - **896 px là tối ưu** [L83, FT].
- MedGemma resize ảnh về 896×896 [L7, FT]. CheXagent dùng 512 px [L6, FT]. MAIRA-2 dùng 518 px [L8, FT]. L64 dùng SigLIP 384 px [L64, FT].
- Ngay cả CAD ở cỡ tối ưu, sensitivity với nốt dưới 10 mm chỉ 0,418 và 0,409. Với khối là 0,937 và 0,956 [L83, FT].
- Accuracy định vị tăng theo kích thước: dưới 0,2 với nốt dưới 10 mm, khoảng 0,7 với khối trên 100 mm [L73, FT].

**3. Loại bỏ bằng chứng sai modality.** L66 (RAD-DINO, embedding ở độ phân giải gốc) là nghiên cứu **CT liều thấp trên NLST**, không phải CXR [L66, FT]. Vì vậy L66 **không được dùng** làm bằng chứng độ phân giải trên CXR, ngược với 03. Giá trị còn lại của L66 là về phương pháp: PR-AUC và hiệu chỉnh isotonic ở tỷ lệ bệnh khoảng 6% [L66, FT].

**Nhận định tổng hợp.**
- Với MedGemma, giả thuyết "thiếu độ phân giải" **yếu đi**, vì 896 px đã trùng mức tối ưu của CAD [L83, FT; L7, FT].
- Giả thuyết "gộp đặc trưng và dữ liệu" **mạnh lên** [L47, AB\*; 05, phát hiện 3].
- Với Qwen2.5-VL, độ phân giải đầu vào không có trong các nguồn đã đọc. Cần xác định từ technical report (PDF `[10]`, ngoài corpus).
- Kích thước nốt là biến điều chỉnh mạnh nhất trên mọi họ mô hình [L83, FT; L73, FT; L47, AB\*; L3, FT].

### (d) Định vị và grounding

**Hiệu năng định vị của MLLM:** ⚠ [L48, AB\*]
- Hit rate với lung lesion: GPT-4 35,7%, GPT-5 35,7%, **MedGemma 14,3%**, CNN 29,0%, bác sĩ 85,0%.
- Trung bình 9 dấu hiệu, GPT-5 đạt 49,7%.
- Số ca lung lesion không ghi nhận. Hit rate chỉ xét một điểm, không có IoU.

**Benchmark nhận thức:** ⚠ phát hiện thô khoảng 92%, chỉnh đường viền tối đa 2,9%, thuộc tính tối đa 25,7%. VLM y khoa không hơn VLM đa năng. **Benchmark này không có nốt hay khối** [L49, AB\*].

**Hệ chuyên biệt cũng khó:** ⚠ NoduLoCC2026 có AUC 0,79. Đúng số nốt ở 53% ảnh, khoảng cách trung vị 12,83 mm [L36, AB].

**Grounding tổng quát (FT):**
- Qwen2.5-VL sau GRPO: mIoU 79,02 trên MS-CXR [L71, FT].
- BioViL-T: mIoU 0,243, CNR 1,33 [L68, FT].
- MedGrounder định nghĩa bài toán 0, 1 hoặc nhiều vùng cho mỗi câu, với chỉ số N-Acc và Precision@(F1 = 1, IoU ≥ 0,5). Khoảng 23% cụm từ trong MS-CXR ứng với nhiều vùng [L70, FT].
- MAIRA-2 mã hóa tọa độ box thành token [L8, FT].
- ⚠ VinDr-CXR-VQA: MedGemma-4B-it đạt F1 0,624 [L43, AB].

**Chỉ số định vị trong nhánh CAD:**
- JAFROC FOM 0,831–0,924 [L19, AB].
- AUFROC 0,744 [L81, FT].
- NODE21 score = 0,75·AUC + 0,25·Sen@0,25 FP [L78, FT].
- Định vị mức tổn thương 0,972 [L25, AB].

**Nhận định tổng hợp.**
- Con số định vị duy nhất cho MedGemma là 14,3% [L48, AB\*].
- Grounding đã cải thiện mạnh khi tinh chỉnh Qwen2.5-VL [L71, FT]. Nhưng chưa ai làm định vị nốt **không cho trước cụm từ**.
- Đề tài nên báo cáo bộ chỉ số kết hợp: hit rate (so được với L48), IoU/mIoU (so với L71), FROC hoặc JAFROC (so với CAD ở L19, L81), và N-Acc cho ảnh không có nốt (theo L70).

### (e) Mốc CAD độc lập

| Bối cảnh | Kết quả | Nguồn |
|---|---|---|
| Phát triển và kiểm định ngoài | AUROC 0,92–0,99; JAFROC 0,831–0,924; cao hơn 17/18 bác sĩ | ⚠ [L19, AB] |
| Kiểm định độc lập 7 sản phẩm (Hà Lan) | 4 thuật toán AUC 0,86–0,93, tốt hơn người đọc (0,81); 3 thuật toán không khác biệt | [L80, FT] |
| MosMedData VII (cùng nguồn với L1) | AUC tốt nhất 0,956 ở bước xác suất; 0,812 khi bác sĩ kiểm tra lại | ⚠ [L45, AB] |
| NLST | Nốt: sensitivity 86,2%, specificity 85,0%; mọi ca ung thư: sensitivity 75,0%, PPV 3,8% | ⚠ [L21, AB] |
| Ung thư bị bỏ sót (trung bình 2,3 cm) | AUROC 0,899; sensitivity 69,6% / specificity 94,0% | [L81, FT] |
| Tỷ lệ bệnh thấp (2,2%), chăm sóc ban đầu | AUROC 0,648; sensitivity 35,3%; specificity 94,2% | [L79, FT] |
| Quần thể sàng lọc | AUC 0,97–0,99; sensitivity 83–90%; FPR khoảng 3% (gấp 10 lần bác sĩ ở tập validation) | ⚠ [L75, AB] |
| Chuẩn tham chiếu đã thẩm định | AUC nốt/khối 0,72 (hiệu chỉnh theo quần thể) so với 0,91 trên ChestX-ray14 | ⚠ [L26, AB] |
| CNN mở đa dấu hiệu | AUC trung bình 0,864 (đồng thuận); tương đương bác sĩ với mass/nodule | [L72, FT] |
| Vùng mù | Sensitivity 0,50–0,64 ở vùng mù, so với 0,87 ở vùng khác | ⚠ [L29, AB] |
| Phân tích gộp | Nốt: sensitivity khoảng 72%, specificity khoảng 95% | ⚠ [L33, AB] |
| Tổng quan hệ thống | Sensitivity 56,4–95,7%; specificity 71,9–97,5%; AUROC 0,89–0,99 | ⚠ [L34, AB] |
| Phân tích gộp DL | Khối hoặc ung thư trên CXR: AUC 0,864 (95% CI 0,827–0,901), 8 nghiên cứu | [L92, FT] |

**Nhận định tổng hợp.**
- CAD độc lập ổn định ở AUC khoảng 0,86–0,99 trên tập giàu ca bệnh [L80, FT; L92, FT; L19, AB].
- Nhưng CAD **giảm mạnh** khi tỷ lệ bệnh thấp (AUROC 0,648 [L79, FT]) và khi chuẩn tham chiếu nghiêm hơn (0,72 [L26, AB]).
- PPV rất thấp khi ca dương hiếm (3,8% [L21, AB]).
- Phép so sánh VLM–CAD gần nhất trên cùng nguồn dữ liệu là 0,956 [L45, AB] so với 0,743 [L1, FT]. Tuy vậy, L1 không xác nhận dùng đúng cùng mẫu với L45.

### (f) Tương tác người–AI và an toàn

**AI sai kéo bác sĩ theo:**
- Thực nghiệm với AI giả (12/90 ca bị cố ý làm sai) [L88, FT]:
  - Không có AI: FN 2,7%, FP 51,4%.
  - Keep AI (No Box): FN 33,0%, FP 86,0%.
  - Keep AI (Box): FN 20,7%, FP 80,5%. **Có box giảm FN** (P = 0,04).
- ⚠ Với AI chính xác cao, AUC của người đọc tăng 0,77 → 0,82. Với AI kém, AUC không đổi (0,75). Người đọc đổi theo AI 67% (AI tốt) và **59% (AI kém)** [L90, AB].

**AI tốt giúp bác sĩ:**
- ⚠ RCT: tỷ lệ phát hiện nốt cần xử trí 0,59% so với 0,25%, OR 2,4 [L20, AB].
- Sensitivity với nốt: AI 0,816 so với người đọc 0,567. Có AI, AUROC của người đọc tăng có ý nghĩa với nốt [L87, FT].
- Accuracy tăng 6,4% (95% CI 2,3–10,6%). Sensitivity tăng 12% ở bác sĩ trẻ, 9% ở bác sĩ lâu năm [L86, FT].
- ⚠ Sensitivity với nốt tăng tuyệt đối 12% [L89, AB]. Sensitivity bác sĩ 65,1% → 70,3% [L85, AB]. Sensitivity 40% → 53% với ung thư dễ bị bỏ sót [L74, AB].
- ⚠ Giao diện chỉ có văn bản tốt hơn không có AI (0,87 so với 0,82) [L32, AB]. Gợi ý rằng đầu ra văn bản kiểu VLM có thể hữu ích, *nếu đúng*.

**Nhận định tổng hợp.**
- Bằng chứng nhất quán (1 bài FT, 1 bài AB) rằng AI sai làm tăng lỗi của bác sĩ, và đầu ra có box giảm phần nào tác hại [L88, FT; L90, AB].
- Với VLM có sensitivity zero-shot 5,2% [L2, FT], rủi ro FN do người đọc nghe theo là đáng kể (suy luận). Đây là luận cứ an toàn cho việc **kiểm toán trước khi dùng**, và cho yêu cầu đầu ra có định vị.

### (g) Nguy cơ và sàng lọc

- ⚠ CXR-LC: AUC 0,755 so với 0,634 trên PLCO; 0,659 so với 0,650 trên NLST [L23, AB]. Kiểm định ngoài ở Hàn Quốc: AUC 0,68 và 0,75 [L24, AB].
- CXR-LC kết hợp EMR, ở người hút thuốc [L91, FT]:
  - Đủ điều kiện theo cả hai tiêu chí: 8,5% mắc ung thư, so với 2,8% nếu chỉ theo CMS.
  - Chỉ xác định được tiêu chí CMS ở 42,6% người.
- ⚠ ViT tiền huấn luyện vượt mô hình huấn luyện từ đầu 6–10 điểm AUC khi dự đoán 1–2 năm trước chẩn đoán [L30, AB, preprint].
- ⚠ Mô hình đa phương thức có LLM: LUNG-KGMM không có số liệu [L13, AB, preprint]. Kim 2026 chỉ có tiêu đề [L61, T].
- Mô hình lai SAM-Med2D + BioGPT báo C-index 0,910 [L67, FT], nhưng **nguồn nhãn sống còn không rõ** trên JSRT, PadChest và NIH. Không dùng làm bằng chứng chính [05, phát hiện 5].
- Nhánh LLM văn bản ⚠: guideline Fleischner chỉ áp dụng đúng 0,15 với GPT-4, tăng lên 0,66 khi đưa guideline vào prompt [L15, AB]. Prompt có cấu trúc cải thiện phân loại Lung-RADS (68,8% so với 58,1%) [L16, AB].

**Nhận định tổng hợp.** Cụm này là hướng mở rộng (§9 spec), không trực tiếp phục vụ câu hỏi chính. Mô hình nguy cơ CXR có LLM chưa có bản bình duyệt có số liệu.

### (h) Dataset

| Dataset | Nhãn nốt | Quy mô và đặc điểm | Ghi chú cho đề tài | Nguồn |
|---|---|---|---|---|
| VinDr-CXR | Box Nodule/Mass | 18.000 ảnh; Nodule/Mass 410 ảnh train (2,73%), 176 ảnh test (5,87%); test đồng thuận 5 bác sĩ | Nguồn box chính cho LoRA; mất cân bằng khoảng 3%; box train có thể bất đồng | [L37, FT] |
| NODE21 | Box nốt | không ghi nhận | Benchmark công khai chính; chỉ số NODE21 score | ⚠ [L35, AB]; [L78, FT] |
| NoduLoCC | Vị trí nốt | không ghi nhận | Chỉ số khoảng cách (mm) | ⚠ [L36, AB]; [L60, T] |
| ChestX-ray8/14 | Nhãn Nodule/Mass (NLP); box cho tập con | 108.948 ảnh | Nhãn nhiễu; có nhãn đã thẩm định [L26] | ⚠ [L38, AB] |
| ChestX-Det10 | Box | khoảng 3.500 ảnh | Dùng để tái lập L47 | ⚠ [L41, AB] |
| CheXpert | "Lung Lesion" | 224.316 ảnh | PDF cục bộ sai | ⚠ [L39, AB] |
| MIMIC-CXR(-JPG) | 14 nhãn tự động; không box | 377.110 ảnh; 227.835 nghiên cứu; 65.379 bệnh nhân (Sci Data) | Nguồn huấn luyện của hầu hết VLM CXR, nên có rủi ro rò rỉ khi đánh giá | ⚠ [L40, AB] |
| JSRT | Có nốt | 247 ảnh (154 có nốt) | Dùng chung ở L2 và L31, nên đặt được VLM, CAD và bác sĩ cạnh nhau | ⚠ [L2, FT; L31, AB] |
| MosMedData VII | Nốt 6–30 mm, xác nhận CT | 100 ảnh (50/50) | Dùng chung ở L1 và L45 | ⚠ [L1, FT; L45, AB] |
| MS-CXR | Box theo cụm từ | — | Benchmark grounding (L68, L70, L71) | [L71, FT] |

**Nhận định tổng hợp.**
- Dữ liệu có box nốt công khai **nhỏ và mất cân bằng**: VinDr khoảng 3% ở train [L37, FT].
- Tỷ lệ bệnh thấp như vậy buộc phải báo cáo AUC-PR, PPV và hiệu chỉnh xác suất, không chỉ AUC-ROC [L84, FT].
- Rủi ro rò rỉ khi đánh giá trên MIMIC là đáng kể [L40, AB].

### 5.9. Bảng so sánh khoảng định lượng: VLM zero-shot, CAD, bác sĩ

*Các con số đến từ những tập dữ liệu, tỷ lệ bệnh và định nghĩa ca dương khác nhau, nên **không so sánh trực tiếp được**. Bảng chỉ cho biết thứ bậc độ lớn.*

| Chỉ số | VLM / MLLM zero-shot | CAD học sâu | Bác sĩ |
|---|---|---|---|
| **Sensitivity với nốt** | 0,6–27,3% (6 VLM, JSRT) ⚠ [L2]; MedGemma 5,2% ⚠ [L2]; Claude 3.7 0,316 ⚠ [L1]; CXR-LLaVA 0,079 ⚠ [L1]; tối đa 41,3% ⚠ [L53] | 86,2% (NLST) ⚠ [L21]; 69,6% (ung thư bỏ sót) [L81]; 81,6% [L87]; 35,3% (tỷ lệ bệnh 2,2%) [L79]; gộp khoảng 72% ⚠ [L33]; 56,4–95,7% ⚠ [L34]; nốt < 10 mm 0,409–0,418 [L83] | 47,0% (ung thư bỏ sót) [L81]; 56,7% [L87]; 65,1% ⚠ [L85]; 40% (ung thư dễ bỏ sót) ⚠ [L74]; bác sĩ nội trú 0,61 ⚠ [L22] |
| **Specificity với nốt** | 71,0–100% ⚠ [L2]; Claude 3.7 0,933 ⚠ [L1] | 85,0% ⚠ [L21]; 94,0% [L81]; 94,2% [L79]; gộp khoảng 95% ⚠ [L33]; 71,9–97,5% ⚠ [L34] | 78,0% [L81] |
| **AUC (ROC)** | BiomedCLIP 0,743 (CLIP, không sinh văn bản) ⚠ [L1]; MedRAX 0,704 ⚠ [L1]; Claude 3.7 0,625 ⚠ [L1] | 0,92–0,99 ⚠ [L19]; 0,86–0,93 [L80]; 0,956 → 0,812 ⚠ [L45]; 0,899 [L81]; 0,648 [L79]; gộp 0,864 [L92]; 0,89–0,99 ⚠ [L34] | 0,81 [L80]; 0,77 ± 0,07 (JSRT) ⚠ [L31]; 0,634–0,663 [L81]; trung bình 0,81 ⚠ [L34]; 0,534–0,676 (bác sĩ nội trú, tỷ lệ bệnh thấp) [L79] |
| **Định vị** | Hit rate lung lesion: MedGemma 14,3%, GPT-4/GPT-5 35,7% ⚠ [L48] | Hit rate lung lesion 29,0% ⚠ [L48]; JAFROC 0,831–0,924 ⚠ [L19]; AUFROC 0,744 [L81]; FOM 0,906 ở 896 px [L83]; NoduLoCC: đúng số nốt 53%, khoảng cách 12,83 mm ⚠ [L36] | Hit rate lung lesion 85,0% ⚠ [L48]; AUFROC 0,619–0,651 [L81] |
| **Sau tinh chỉnh (VLM, không riêng nốt)** | mIoU 79,02 (Qwen2.5-VL, MS-CXR, grounding có cho trước cụm từ) [L71]; F1 VLM 0,203 so với CNN 0,811, sensitivity sụp < 15% sau tinh chỉnh ⚠ [L50] | — | — |

---

## 6. Đánh giá chất lượng và nguy cơ sai lệch (QUADAS-2 rút gọn)

**Cách chấm.**
- Chấm ở mức cụm, dựa trên phần "Hạn chế" của từng bài trong 05. Đây là phán đoán của một người, không có người thứ hai chấm độc lập.
- Mức: **Thấp** / **Không rõ** / **Cao** (nguy cơ sai lệch). "N/A" khi miền không áp dụng, ví dụ với dataset hay bài phương pháp.
- Bốn cờ ghi số bài có vấn đề trên tổng số bài của cụm:
  - **Mẫu nhỏ:** test dưới khoảng 300 ảnh, hoặc 05 tự nêu "mẫu nhỏ".
  - **Một nguồn:** dữ liệu từ một trung tâm hoặc một dataset.
  - **PP:** preprint, tóm tắt hội nghị hoặc bình luận.
  - **Chỉ AB/T:** mức đọc AB, AB\* hoặc T.

### 6.1. Bảng đánh giá theo cụm

| Cụm | Chọn bệnh nhân | Index test | Chuẩn tham chiếu | Luồng và thời gian | Mẫu nhỏ | Một nguồn | PP | Chỉ AB/T | Nhận xét chính |
|---|---|---|---|---|---|---|---|---|---|
| **T1** | **Cao** | **Cao** | Không rõ | **Cao** | 6/7 (trừ L52 không rõ n; L62 không áp dụng) | ≥ 3/8 (L1, L2, L51; các bài khác không ghi nhận) | 2/8 (L53, L62) | **8/8** | Tập giàu ca bệnh, JSRT có tỷ lệ nốt 62% [L2]. Prompt không thống nhất, đầu ra nhị phân không có ngưỡng [L2]. Ngưỡng Youden lấy trên chính tập test [01 về L1]. Loại 17/100 ảnh do lỗi xử lý [L1]. Mô hình đóng đổi phiên bản [L3]. Ngữ cảnh có thể lộ đáp án [L4] |
| **T2** | Không rõ | Không rõ | **Cao** | Không rõ | 1/12 (L65) | 2/12 (L65, L64) | 5/12 (L6, L8, L11, L12, L59) | 6/12 | Benchmark do chính nhóm tác giả xây [L6, L11]. Nhãn tự động hoặc báo cáo tổng hợp bằng GPT-4 [L10, L65]. Chỉ số NLG thay cho độ đúng lâm sàng [L64]. Không có tầng riêng cho nốt ở mọi bài |
| **T3** | Không rõ | **Cao** | Không rõ | Không rõ | 0/5 | 3/5 (L47, L50, L66) | 3/5 | 3/5 | Probe cần ROI oracle [L47]. Chi tiết tinh chỉnh không rõ [L50]. **L66 sai modality (CT)**, loại khỏi luận cứ CXR |
| **T4** | Không rõ | Không rõ | Không rõ | Không rõ | 2/8 (L48, L44) | không ghi nhận | 4/8 (L42, L43, L44, L49) | 4/8 | Hit rate chỉ xét một điểm, ít ca lung lesion [L48]. CheXpercept không có nốt [L49]. VQA sinh từ báo cáo, thiên về hiểu ngôn ngữ [L42]. Grounding có cho trước cụm từ [L71] |
| **T5** | Không rõ | Thấp | **Cao** | Không rõ | 1/8 (L15) | 3/8 (L16, L57, L58) | 0/8 (L17 là kỷ yếu) | **8/8** | Chuẩn so sánh cũng là công cụ tự động [L55]. Dữ liệu hư cấu [L15]. 3/8 bài là báo cáo CT [L16–L18] |
| **T6** | **Cao** (tập giàu ca bệnh) | Thấp | Thấp–Không rõ | Không rõ | 4/15 (L29, L31, L45, L81) | 4/15 (L29, L31, L45, L81) | 1/15 (L31) | 9/15 | Chuẩn CT ở nhiều bài [L79, L80, L81]. Nhưng nhãn NLP ở L72, L73. Hiệu năng giảm mạnh khi tỷ lệ bệnh thấp [L79]. Có trường hợp nhà cung cấp từ chối tham gia [L80] |
| **T7** | **Cao** (tập giàu ca bệnh) | Thấp | Thấp–Không rõ | Thấp | 6/13 (L32, L74, L76, L86, L88, L90) | ≥ 2/13 (L20, L88) | 1/13 (L27) | 10/13 | Thiết kế tốt nhất corpus: RCT [L20, L76], bắt chéo có washout [L87]. Nhưng ít người đọc, thiếu tính sinh thái [L88]. Chuẩn chỉ từ ảnh thẳng [L87] |
| **T8** | Không rõ | Không rõ | **Cao** (L67) | Không rõ | 0/7 | ≥ 2/7 (L30, L91) | 3/7 (L13, L30, L61) | 5/7 | Nguồn nhãn sống còn không rõ [L67]. Ảnh lấy từ thử nghiệm sàng lọc [L23, L91]. Quần thể đặc thù (VA, chủ yếu nam giới) [L30, L24] |
| **T9** | N/A | N/A | Không rõ | N/A | 2/8 (L41, L36) | 1/8 (L37, một quốc gia) | 3/8 (L36, L40, L41) | 7/8 | Nhãn NLP nhiễu [L38, L40]. Box train có thể bất đồng, một quốc gia [L37]. Hai PDF sai (L39, L40) |
| **T10** | N/A | N/A | Không rõ | N/A | 1/3 (L33: 15 nghiên cứu) | N/A | 0/3 | 2/3 | Độ không đồng nhất cao, dễ đánh giá quá cao DL [L92]. L33 một tác giả. L34 không gộp số liệu |
| **M** | Không rõ | Thấp | Không rõ | Thấp | 1/3 (L83) | 2/3 (L82 một dataset; L83 không kiểm định ngoài) | 0/3 | 0/3 | Nhãn NIH nhiễu, chia 80/20 không chắc theo bệnh nhân [L82]. Không kiểm định ngoài [L83]. Một kiến trúc [L84] |

### 6.2. Tính áp dụng (applicability) cho đề tài

| Cụm | Lo ngại về áp dụng | Lý do |
|---|---|---|
| T1 | Thấp–Trung bình | Đúng bài toán (nốt, CXR, VLM). Chỉ L2 có MedGemma-4b-it; **không bài nào thử Qwen2.5-VL-3B trên nốt** |
| T2 | Trung bình | Các tiền lệ tinh chỉnh khác tác vụ (sinh báo cáo, grounding tổng quát) |
| T3 | Trung bình | Probe trên encoder đông cứng, chưa kiểm chứng trên VLM sinh văn bản end-to-end [L47] |
| T4 | Trung bình | Không bài nào định vị riêng nốt bằng VLM, trừ "lung lesion" ở L48 |
| T6, T7 | Thấp | Đúng bài toán nốt trên CXR, nhưng là CNN thương mại, không phải DenseNet-121 hay YOLOv8 tự huấn luyện |
| T5, T8 | Cao | Hướng mở rộng, không phục vụ câu hỏi chính |
| T9, M | Thấp | Trực tiếp quyết định dữ liệu, tiền xử lý và chỉ số |

### 6.3. Tóm tắt mức chắc chắn (kiểu GRADE, không chính thức)

| Kết luận | Mức chắc chắn | Lý do hạ mức |
|---|---|---|
| VLM zero-shot có sensitivity thấp với nốt | Trung bình | Nhất quán qua nhiều bài, nhưng mẫu nhỏ, một nguồn, chỉ AB |
| CAD độc lập đạt AUC khoảng 0,86–0,99 trên tập giàu ca bệnh | Trung bình–Cao | Có FT và kiểm định độc lập [L80, L92]; tập giàu ca bệnh |
| Gộp CLS làm mất tín hiệu nốt nhỏ | Thấp | Một preprint, AB\*, ROI oracle |
| 896 px tối ưu cho CAD nốt | Trung bình | FT, nhưng một nghiên cứu, không kiểm định ngoài [L83] |
| AI sai làm tăng lỗi của bác sĩ | Trung bình | Một FT, một AB, cùng hướng; thiếu tính sinh thái |
| Chưa có VLM tinh chỉnh cho phát hiện hay định vị nốt | Trung bình | Vắng bằng chứng trong 3 vòng tìm kiếm; sàng lọc một người; vòng 1 ghi nhật ký không đủ |

---

## 7. Thảo luận

### 7.1. Tính nhất quán và mâu thuẫn trong bằng chứng

**Các điểm nhất quán:**
1. **VLM zero-shot bỏ sót nốt.** Ít nhất 6 nghiên cứu độc lập, trên 3 nguồn dữ liệu khác nhau (JSRT, MosMedData, tập riêng), đều cho sensitivity thấp [L1, FT; L2, FT; L3, FT; L4, FT; L51, FT; L53, AB].
2. **Kích thước nốt chi phối mọi họ mô hình.**
   - VLM: tổn thương nhỏ bị bỏ sót [L3, FT].
   - Encoder: CLS cho nốt nhỏ 0,627, nốt lớn 0,844 [L47, AB\*].
   - CAD: sensitivity nốt dưới 10 mm khoảng 0,41 [L83, FT]. Accuracy định vị dưới 0,2 với nốt dưới 10 mm [L73, FT].
3. **Khoảng cách VLM–CAD khoảng 0,2 AUC** trên cùng nguồn MosMedData: 0,743 so với 0,956 [L1, FT; L45, AB; 02 §2].
4. **Benchmark tổng hợp che điểm yếu với nốt.** MedGemma đạt 83,24% trên VQA [L42, FT] và macro F1 88,9 [L7, FT], nhưng sensitivity với nốt chỉ 5,2% [L2, FT].

**Các mâu thuẫn và cách giải thích:**

| Mâu thuẫn | Bằng chứng | Giải thích khả dĩ |
|---|---|---|
| Mô hình y khoa có hơn mô hình đa năng không? | Có: MedGemma > Gemma 3 [L7, FT]; mô hình chuyên ngành > GPT-4V [L63, AB]. Không: CXR-LLaVA < Claude [L1, FT]; MedGemma < GPT-5 về định vị [L48, AB\*]; VLM y khoa không hơn [L49, AB\*] | Khác tác vụ. Lợi thế miền thể hiện ở benchmark tổng hợp, không ở nốt (suy luận) |
| Khối u có dễ phát hiện không? | Dễ [L52, FT] so với khó [L51, FT] | Khác định nghĩa (khối lớn so với nốt 6–30 mm) và cách chọn ca [01 §4] |
| CLIP hơn MLLM sinh văn bản? | BiomedCLIP dẫn đầu ở L1 [AB\*], nhưng VLM nền rất kém trên ChestX-ray14 [L50, AB] | Ngưỡng Youden trên tập test ở L1; khác tác vụ (nhị phân so với đa nhãn) |
| Tinh chỉnh có giúp không? | Grounding tốt lên mạnh [L71, FT]; LoRA cải thiện chỉ số NLG [L64, FT] so với sensitivity sụp [L50, AB] | Khác tác vụ và cách tinh chỉnh. L50 không rõ chi tiết (có thể do mất cân bằng lớp hoặc ngưỡng) |
| Độ phân giải có phải nút thắt? | Có: L82 [FT]; 03 dẫn L66. Không với MedGemma: 896 px đã tối ưu [L83, FT; L7, FT] | L66 là CT nên bị loại. Với MedGemma, nút thắt nhiều khả năng là gộp đặc trưng [L47] |
| Hiệu năng CAD | 0,86–0,99 [L80, L19] so với 0,648 [L79] và 0,72 [L26] | Tỷ lệ bệnh, chuẩn tham chiếu và bối cảnh lâm sàng khác nhau |

### 7.2. Hàm ý cho spec: sensitivity ≥ 0,60 ở specificity 0,90

**1. Ngưỡng đặt ở đâu so với các mốc.**
- Bác sĩ trên tập giàu ca bệnh có sensitivity với nốt 0,47–0,65 [L81, FT; L87, FT; L85, AB]. Mốc 0,60 vì vậy **xấp xỉ mức bác sĩ** [03, nhận xét xuyên cụm 2].
- Specificity 0,90 cao hơn specificity của bác sĩ ở L81 (78,0%) [L81, FT].
- CAD ở vùng specificity tương tự:
  - 69,6% ở specificity 94,0% (ung thư bị bỏ sót) [L81, FT].
  - 35,3% ở 94,2% (tỷ lệ bệnh thấp) [L79, FT].
  - Gộp khoảng 72% ở khoảng 95% [L33, AB].
- Tức là một CAD tốt có thể vượt ngưỡng trên tập giàu ca bệnh, nhưng có thể **không đạt** ở tỷ lệ bệnh thực tế.

**2. Khoảng cách phải lấp.**
- MedGemma-4b-it zero-shot có sensitivity 5,2% [L2, FT]. Khoảng cách tới mốc khoảng 55 điểm phần trăm.
- Lưu ý: L2 không có đường ROC (đầu ra là văn bản nhị phân), và specificity ở L2 nằm trong khoảng 71,0–100% chứ không cố định ở 0,90. Vì vậy, để báo "sensitivity ở specificity 0,90" cho VLM, đề tài **phải trích điểm liên tục** (ví dụ xác suất của token "yes" hoặc logit) thay vì dùng câu trả lời nhị phân (suy luận, nối với 02 §3).
- Không có tiền lệ nào cho biết LoRA đóng góp được bao nhiêu điểm sensitivity với nốt. Tiền lệ gần nhất chỉ đo NLG [L64, FT; L69, FT] hoặc grounding tổng quát [L71, FT].

**3. Rủi ro kỹ thuật cho ngưỡng.**
- Sensitivity có thể sụp sau tinh chỉnh [L50, AB]. Cần cân bằng lớp, theo dõi sensitivity trong khi huấn luyện, và chọn ngưỡng trên tập validation, **không phải tập test** (tránh lỗi Youden-trên-test như ở L1 [01 §4; 02 §3]).
- Tỷ lệ Nodule/Mass trong VinDr khoảng 2,73–5,87% [L37, FT]. Ở mức này, PPV sẽ thấp dù đạt ngưỡng (CAD: PPV 3,8% [L21, AB]). Spec nên báo thêm AUC-PR, PPV và hiệu chỉnh (Brier hoặc ECE) [L84, FT; L66, FT về PR-AUC].
- Thành phần kích thước nốt trong tập test sẽ quyết định việc đạt hay không đạt ngưỡng. Nốt dưới 10 mm cho CAD sensitivity khoảng 0,41 [L83, FT]. Vì vậy spec **cần phân tầng theo kích thước** (dưới 10 mm, 10–30 mm, khối), như 05 đề xuất cho L52 và L83.
- Kiểm định ngoài: CAD giảm từ khoảng 0,9 xuống 0,648 khi tỷ lệ bệnh thấp [L79, FT]. Nên đánh giá ngưỡng trên ít nhất một tập ngoài, ví dụ JSRT để so được với L2 và L31.

**4. Thiết kế mô hình.**
- Với MedGemma, 896 px đã là mức tối ưu của CAD [L83, FT; L7, FT]. Thí nghiệm chia ô (tiling) nên được định vị là kiểm tra giả thuyết gộp đặc trưng [L47, AB\*], không phải giả thuyết độ phân giải.
- Với Qwen2.5-VL-3B: có tiền lệ học box tốt bằng GRPO [L71, FT]. Có thể thêm nhánh RL bên cạnh LoRA-SFT. Độ phân giải đầu vào cần xác minh từ technical report (ngoài corpus).
- Baseline DenseNet-121 có tiền lệ trên CXR: thực nghiệm độ phân giải [L82, FT], mất cân bằng lớp [L84, FT], nhãn từ LLM [L65, FT]. YOLOv8 **không xuất hiện** trong corpus. Mốc detector gần nhất là Mask R-CNN và RetinaNet [L83, FT].
- Chỉ số định vị: hit rate (so với L48), mIoU (so với L71), FROC/JAFROC (so với L19, L81, L83), và N-Acc cho ảnh không có nốt (L70).

**5. An toàn.** Nếu VLM sau LoRA vẫn dưới ngưỡng, kết quả kiểm toán vẫn có giá trị an toàn, vì AI kém vẫn được nghe theo ở 59% trường hợp [L90, AB] và làm tăng FN của bác sĩ [L88, FT]. Đầu ra có box giảm phần nào tác hại [L88, FT].

---

## 8. Hạn chế của tổng quan này

1. **Không phải systematic review đã đăng ký.** Không có protocol trước, không đăng ký PROSPERO, không có tiêu chí được khóa trước khi tìm kiếm. Tiêu chí được mở rộng giữa chừng (vòng 3 thêm bài phương pháp và grounding tổng quát).
2. **Một người sàng lọc, có AI hỗ trợ.** Không có người thứ hai độc lập, không có kappa. Trích số liệu bởi AI có thể sai. Hạng nguồn chưa tra lại [04 §3].
3. **Nhật ký tìm kiếm không đầy đủ.** Bảng vòng 1 dựng lại sau khi phiên bị nén ngữ cảnh. Số bản ghi chỉ là xấp xỉ. Không có tổng số sau bỏ trùng giữa các vòng. Scopus Q2 chỉ xem 50/116 kết quả.
4. **Đọc nông.** 66% số bài chỉ đọc abstract, 3% chỉ có tiêu đề. **Cụm lõi T1 không có bài nào FT** trong lượt 05. L1, L47, L48, L49 dựa trên ghi chú toàn văn từ phiên trước, chưa đọc lại.
5. **Lỗi file và phiên bản:**
   - **L39:** file PDF cục bộ thực ra là bài khác (Saralajew et al. 2019, *Robustness of Generalized LVQ Models against Adversarial Attacks*, arXiv 1902.00577). Cần thay file [05].
   - **L40:** PDF cục bộ là bài MIMIC-CXR ở *Scientific Data* (bản DICOM kèm báo cáo), không phải bài MIMIC-CXR-JPG trên arXiv được liệt kê [05]. Các số 227.835 nghiên cứu và 65.379 bệnh nhân lấy từ bài Sci Data.
   - 8 bài RSNA (L63, L74–L77, L85, L89, L90) không lấy được PDF [RSNA notes].
6. **Lỗi phân cụm đã phát hiện và sửa trong báo cáo này:** L66 là CT, không phải CXR. L64 không phải tiền lệ LoRA duy nhất (còn L69, L71). Có thể còn lỗi tương tự trong các bài chỉ đọc AB.
7. **Thiên lệch công bố và ngôn ngữ.** Kết quả âm tính của VLM chủ yếu nằm ở preprint (L50) và tóm tắt hội nghị (L53). Chỉ tìm bằng tiếng Anh (L51 tiếng Đức được thấy qua abstract tiếng Anh). Phạm vi thời gian không giới hạn rõ, nhưng thiên về 2019–2026.
8. **Không gộp số liệu.** Các khoảng số trong §5.9 đến từ quần thể, tỷ lệ bệnh, ngưỡng và định nghĩa ca dương khác nhau, nên không so sánh trực tiếp được.
9. **Khoảng trống của corpus:**
   - Không có bài đánh giá Qwen2.5-VL-3B trên nốt.
   - Không có YOLOv8 làm baseline nốt.
   - Không có bài gốc JSRT (Shiraishi) [corpus §3].
   - Không có PadChest dưới dạng bài đã sàng lọc [corpus §3].
   - Nhiều tài liệu phương pháp có PDF nhưng ngoài corpus (LoRA, Qwen2.5-VL, shortcut learning, ROAR...) [05 lưu ý nguồn].
10. **Thời điểm.** Mọi tìm kiếm diễn ra trong một ngày (2026-11-12). Lĩnh vực đang thay đổi nhanh: nhiều bài 2026 vẫn là preprint.

---

## 9. Kết luận

1. **VLM zero-shot chưa dùng được để phát hiện nốt phổi trên CXR.** Sensitivity chỉ 0,6–27,3%, riêng MedGemma-4b-it 5,2% [L2, FT]. MLLM đa năng tốt nhất đạt AUC 0,625 [L1, FT]. Mức chắc chắn trung bình, vì mọi bằng chứng lõi có mẫu nhỏ và chỉ đọc ở mức abstract.
2. **Hai mốc so sánh đã rõ.** CAD độc lập đạt AUC 0,86–0,93 và vượt người đọc (0,81) trong kiểm định độc lập [L80, FT]. Bác sĩ có sensitivity với nốt 0,47–0,65 trên tập giàu ca bệnh [L81, FT; L87, FT; L85, AB]. Ngưỡng của spec (sensitivity ≥ 0,60 ở specificity 0,90) gần mức bác sĩ, và cách MedGemma zero-shot khoảng 55 điểm phần trăm.
3. **Cơ chế thất bại khả dĩ nhất với MedGemma là cách gộp đặc trưng và dữ liệu, không phải độ phân giải.** 896 px là tối ưu cho CAD nốt [L83, FT], trùng đầu vào của MedGemma [L7, FT]. Tín hiệu nốt còn trong patch token của MedSigLIP nhưng mất khi gộp CLS [L47, AB\*]. Nốt dưới 10 mm khó với mọi mô hình [L83, FT; L73, FT].
4. **Khoảng trống của đề tài được xác nhận, nhưng phải phát biểu hẹp lại.** Đã có LoRA trên CXR cho sinh báo cáo [L64, FT; L69, FT] và tinh chỉnh Qwen2.5-VL cho grounding tổng quát [L71, FT]. Chưa có bài nào tinh chỉnh VLM để **phát hiện và định vị nốt**.
5. **Kiểm toán có ý nghĩa an toàn.** AI sai làm tăng FN của bác sĩ từ 2,7% lên 20,7–33,0% [L88, FT]. Người đọc vẫn nghe theo AI kém ở 59% trường hợp [L90, AB]. Đầu ra có box giảm phần nào tác hại [L88, FT], nên yêu cầu định vị trong spec có thêm luận cứ.

**Khuyến nghị cho bước tiếp theo:**
- (i) Đọc toàn văn L1, L2 (cả L46) và L47 để nâng mức chắc chắn của cụm lõi.
- (ii) Thay PDF của L39.
- (iii) Bổ sung bài gốc JSRT, PadChest và technical report Qwen2.5-VL vào corpus.
- (iv) Trong spec: trích điểm liên tục từ VLM, chọn ngưỡng trên tập validation, phân tầng theo kích thước nốt, báo AUC-PR, PPV và hiệu chỉnh, và đánh giá ngoài trên JSRT.

---

## 10. Tài liệu tham khảo

DOI và mã arXiv lấy từ `corpus.md` (§2, §6, §7) và các cập nhật trong `04` §2. "DOI không ghi nhận" nghĩa là corpus ghi n/s. Thứ tự theo cụm, như §4.

**T1. Đánh giá zero-shot**
- **L1** Khovanova D et al. 2026. *A comparative accuracy study of multimodal LLMs, VLM and agent-based framework for pulmonary nodule detection on chest radiographs.* Frontiers in Digital Health. DOI 10.3389/fdgth.2025.1674835
- **L2** Nishio M et al. 2026. *Zero-Shot Performance of Six Vision-Language Models for Lung Nodule Detection on Chest Radiographs.* AJR. DOI 10.2214/AJR.26.35421
- **L46** (preprint của L2) Nishio et al. 2026. medRxiv. DOI 10.64898/2026.05.31.26354565
- **L3** (= L54) Chen CH et al. 2026. *Limitations in Chest X-Ray Interpretation by Vision-Capable LLMs...* Diagnostics. DOI không ghi nhận
- **L4** Cusick A et al. 2026. *The Effect of Structured Context on CXR Interpretation by a Multimodal LLM.* Cureus. DOI 10.7759/cureus.108715
- **L51** Abou-Zeid et al. 2026. *Automated reporting of chest radiographs: pilot study with two VLMs.* RöFo. DOI 10.1055/s-0046-1817587
- **L52** Lacaita et al. 2025. *Accuracy of ChatGPT-4o in Interpreting Chest and Abdominal X-Rays.* J Pers Med. DOI 10.3390/jpm15050194
- **L53** Starychenko et al. 2025. *Sensitivity/specificity of generative AI chatbots in CXR interpretation.* ERS Congress (Eur Respir J suppl.). DOI 10.1183/13993003.congress-2025.pa4975
- **L62** Chu 2026. *Caution Needed When Applying VLMs for CXR Nodule Screening.* AJR (commentary). DOI 10.2214/AJR.26.35708

**T2. VLM chuyên CXR**
- **L5** Lee S et al. 2025. *CXR-LLaVA.* European Radiology. DOI 10.1007/s00330-024-11339-6; arXiv 2310.18341
- **L6** Chen Z et al. 2024. *CheXagent.* arXiv 2401.12208
- **L7** Sellergren A et al. 2026. *MedGemma.* Nature Medicine. DOI 10.1038/s41591-026-04626-w
- **L8** Bannur S et al. 2024. *MAIRA-2: Grounded Radiology Report Generation.* arXiv 2406.04449
- **L9** Deperrois N et al. 2026. *RadVLM.* Scientific Reports. DOI 10.1038/s41598-026-66181-1 (trước là arXiv 2502.03333)
- **L10** Zambrano Chaves JM et al. 2025. *LLaVA-Rad.* Nature Communications. DOI 10.1038/s41467-025-58344-x
- **L11** Fallahpour A et al. 2025. *MedRAX.* arXiv 2502.02673
- **L12** Thawakar O et al. 2023. *XrayGPT.* arXiv 2306.07971
- **L59** Le-Duc et al. 2024. *LiteGPT.* arXiv 2407.12064
- **L63** Hong EK et al. 2025. *Domain-specific multimodal generative AI for CXR report generation.* Radiology. DOI 10.1148/radiol.241476
- **L64** Volkov E et al. 2025. *VLM fine-tuning via LoRA for structured reports on lung X-ray.* SCM 2025 (IEEE). DOI 10.1109/scm66446.2025.11060140
- **L65** Zhang Q et al. 2026. *CXR screening using LLMs and deep CNNs on MIMIC-CXR.* Front Digit Health. DOI 10.3389/fdgth.2026.1883337

**T3. Encoder thị giác–ngôn ngữ**
- **L14** Zhang S et al. 2023. *BiomedCLIP.* arXiv 2303.00915
- **L47** Muthyala et al. 2026. *Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest Radiography.* arXiv 2606.11606
- **L50** Fisher 2025. *VL Foundation Models Do Not Transfer to Medical Imaging Classification.* medRxiv. DOI 10.64898/2025.12.06.25341759
- **L66** Hoq ME et al. 2025. *Native-resolution 2D embeddings, RAD-DINO, lung cancer classification* (CT). J Imaging Inform Med. DOI 10.1007/s10278-025-01748-4
- **L68** Bannur S et al. 2023. *BioViL-T.* CVPR. DOI 10.1109/cvpr52729.2023.01442

**T4. Định vị, grounding, VQA**
- **L42** Pal A et al. 2025. *ReXVQA.* arXiv 2506.04353
- **L43** Nguyen DH et al. 2025. *VinDr-CXR-VQA.* arXiv 2511.00504
- **L44** Wei Y et al. 2026. *REVEAL-CXR (RSNA LLM Benchmark).* arXiv 2601.15129
- **L48** Gosai et al. 2025. *Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in CXR.* PMLR (ML4H). arXiv 2509.18015
- **L49** Choi et al. 2026. *CheXpercept.* arXiv 2606.21020
- **L69** Gao J et al. 2026. *S2D-Align.* AAAI. DOI 10.1609/aaai.v40i36.40335
- **L70** Zhang W et al. 2026. *Generalized medical phrase grounding (MedGrounder).* IEEE TMI. DOI 10.1109/tmi.2026.3707568
- **L71** Xu et al. 2025. *MedGround-R1.* MICCAI (LNCS). DOI 10.1007/978-3-032-04971-1_37

**T5. LLM văn bản**
- **L15** Gamble JL et al. 2024. *Limitations of GPT-3.5 and GPT-4 in Applying Fleischner Society Guidelines.* Can Assoc Radiol J. DOI 10.1177/08465371231218250
- **L16** Gulati V et al. 2026. *GPT-4o in automated Lung-RADS classification.* Clinical Radiology. DOI 10.1016/j.crad.2026.107466
- **L17** Zhou T et al. 2024. *Automating Lung-RADS Categorization ... In-Context Learning.* Kỷ yếu (PubMed 41726431). DOI không ghi nhận
- **L18** Wen J et al. 2025. *LLMs in generating pulmonary nodule follow-up recommendations.* Eur J Radiol Open. DOI không ghi nhận
- **L55** Mukherjee et al. 2023. *Privacy-preserving LLM Vicuna for labeling radiology reports.* Radiology. DOI 10.1148/radiol.231147
- **L56** Savage et al. 2024. *General-purpose LLMs vs domain-specific NLP for CXR label extraction.* AJR. DOI 10.2214/AJR.23.30573
- **L57** Park et al. 2024. *Extraction of clinical data on major pulmonary diseases using an LLM.* PLOS ONE. DOI 10.1371/journal.pone.0314136
- **L58** Yasaka et al. 2024. *Fine-tuned LLM for extracting lung cancer patients from PACS reports.* J Imaging Inform Med. DOI 10.1007/s10278-024-01186-8

**T6. CAD độc lập**
- **L19** Nam JG et al. 2019. Radiology. DOI 10.1148/radiol.2018180237
- **L21** Yoo H et al. 2020. JAMA Netw Open. DOI 10.1001/jamanetworkopen.2020.17135
- **L25** Hwang EJ et al. 2019. JAMA Netw Open. DOI 10.1001/jamanetworkopen.2019.1095
- **L26** Majkowska A et al. 2020. Radiology. DOI 10.1148/radiol.2019191293
- **L29** Shimazaki A et al. 2022. Scientific Reports. DOI 10.1038/s41598-021-04667-w
- **L31** Ohlmann-Knafo S et al. 2022. arXiv 2206.10912
- **L45** Arzamasov et al. 2024. Quant Imaging Med Surg. DOI 10.21037/qims-24-160
- **L72** Cid YD et al. 2024. *X-Raydar.* Lancet Digital Health. DOI 10.1016/s2589-7500(23)00218-2
- **L73** Pesce E et al. 2019. Medical Image Analysis. DOI 10.1016/j.media.2018.12.007
- **L75** Lee JH et al. 2020. Radiology. DOI 10.1148/radiol.2020201240
- **L77** Hwang EJ et al. 2021. Radiology. DOI 10.1148/radiol.2021210578
- **L78** Shen et al. 2023. Medical Image Analysis. DOI 10.1016/j.media.2022.102708
- **L79** Kim et al. 2023. European Radiology. DOI 10.1007/s00330-022-09315-z
- **L80** van Leeuwen KG et al. 2024. Radiology. DOI 10.1148/radiol.230981
- **L81** Nam JG et al. 2020. Radiology: Cardiothoracic Imaging. DOI 10.1148/ryct.2020190222

**T7. Người–AI**
- **L20** Nam JG et al. 2023 (RCT). Radiology. DOI 10.1148/radiol.221894
- **L22** Yoo H et al. 2021. European Radiology. DOI 10.1007/s00330-021-08074-7
- **L27** Dissez G et al. 2022. arXiv 2208.14742
- **L28** Topff L et al. 2024. European Radiology. DOI 10.1007/s00330-024-10676-w
- **L32** Tang JSN et al. 2023. Radiology: Artificial Intelligence. DOI 10.1148/ryai.220079
- **L74** Jang S et al. 2020. Radiology. DOI 10.1148/radiol.2020200165
- **L76** Sung J et al. 2021. Radiology. DOI 10.1148/radiol.2021202818
- **L85** Sim Y et al. 2020. Radiology. DOI 10.1148/radiol.2019182465
- **L86** Homayounieh F et al. 2021. JAMA Netw Open. DOI 10.1001/jamanetworkopen.2021.41096
- **L87** Ahn JS et al. 2022. JAMA Netw Open. DOI 10.1001/jamanetworkopen.2022.29289
- **L88** Bernstein MH et al. 2023. European Radiology. DOI 10.1007/s00330-023-09747-1
- **L89** Bennani S et al. 2023. Radiology. DOI 10.1148/radiol.230860
- **L90** Lee JH et al. 2023. Radiology. DOI 10.1148/radiol.222976

**T8. Nguy cơ và sàng lọc**
- **L13** Yang C et al. 2026. *LUNG-KGMM.* arXiv 2608.14657
- **L23** Lu MT et al. 2020. *CXR-LC.* Annals of Internal Medicine. DOI 10.7326/M20-1868
- **L24** Lee JH et al. 2022. Radiology. DOI 10.1148/radiol.212877
- **L30** Kotevska O et al. 2026. arXiv 2608.21571
- **L61** Kim et al. 2026. J Thorac Oncol (conference abstract P1.079). DOI 10.1016/j.jtho.2026.104482
- **L67** Hayeso et al. 2026. Transl Lung Cancer Res. DOI 10.21037/tlcr-2025-aw-1299
- **L91** Raghu VK et al. 2022. JAMA Netw Open. DOI 10.1001/jamanetworkopen.2022.48793

**T9. Dataset**
- **L35** Sogancioglu E et al. 2024. *NODE21.* IEEE TMI. DOI 10.1109/tmi.2024.3382042 (trước là arXiv 2401.02192)
- **L36** Mustafic A et al. 2026. *NoduLoCC2026.* arXiv 2606.21290
- **L60** Mustafic et al. 2026. *NoduLoCC dataset.* Kỷ yếu ICCVIT. DOI không ghi nhận
- **L37** Nguyen HQ et al. 2022. *VinDr-CXR.* Scientific Data. DOI 10.1038/s41597-022-01498-w
- **L38** Wang X et al. 2017. *ChestX-ray8.* arXiv 1705.02315 (bản CVPR chưa xác minh trong corpus)
- **L39** Irvin J et al. 2019. *CheXpert.* arXiv 1901.07031 (bản AAAI chưa xác minh trong corpus)
- **L40** Johnson AEW et al. 2019. *MIMIC-CXR-JPG.* arXiv 1901.07042
- **L41** Liu J et al. 2020. *ChestX-Det10.* arXiv 2006.10550

**T10. Tổng quan và phân tích gộp**
- **L33** Essa MEA 2025. Eur J Radiol Open. DOI không ghi nhận
- **L34** Megat Ramli PN et al. 2025. Diagnostics. DOI 10.3390/diagnostics15030246
- **L92** Aggarwal R et al. 2021. npj Digital Medicine. DOI 10.1038/s41746-021-00438-z

**M. Phương pháp**
- **L82** Sabottke CF, Spieler BM 2020. Radiology: Artificial Intelligence. DOI 10.1148/ryai.2019190015
- **L83** Kim YG et al. 2020. European Radiology. DOI 10.1007/s00330-020-06892-9
- **L84** Mosquera C et al. 2024. European Radiology. DOI 10.1007/s00330-024-10834-0
