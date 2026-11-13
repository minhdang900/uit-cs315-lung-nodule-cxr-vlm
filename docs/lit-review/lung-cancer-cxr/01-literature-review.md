# Tổng quan tài liệu: Mô hình ngôn ngữ lớn cho ung thư phổi trên X-quang ngực

*Ngày 2026-11-12. Báo cáo này thực hiện bước `literature_review` của arxiv-mcp-server: phạm vi, các nhóm chủ đề, xu hướng phương pháp, đồng thuận và bất đồng, khoảng trống, hướng tương lai. Bộ tài liệu là `corpus.md`, gồm 60 bài khác nhau (L1–L62).*

**Mức độ đọc.**
- Đọc toàn văn 4 bài: L1 Khovanova 2026, L47 Muthyala 2026, L48 Gosai 2025, L49 Choi 2026.
- Các bài còn lại chỉ đọc tóm tắt. Mọi con số lấy nguyên văn từ tóm tắt hoặc toàn văn.
- Câu nào ghi "(suy luận)" là nhận định của người viết, không phải kết quả của bài gốc.

---

## 1. Phạm vi và tiêu chí chọn bài

**Câu hỏi tổng quan.** LLM, VLM và mô hình nền thị giác–ngôn ngữ đã được dùng và đánh giá thế nào cho ung thư phổi, nốt phổi và khối phổi trên X-quang ngực (CXR)? So với CAD học sâu chuyên biệt thì chúng đứng ở đâu?

**Tiêu chí chọn.**
- Ảnh là CXR.
- Mục tiêu là nốt, khối, ung thư phổi, hoặc sàng lọc/nguy cơ ung thư phổi.
- Mô hình thuộc một trong các nhóm: (a) VLM/MLLM đọc ảnh; (b) LLM chỉ đọc văn bản báo cáo; (c) mô hình nền kiểu CLIP; (d) một số CNN/ViT kinh điển để làm mốc so sánh.
- Bài toán: phát hiện/phân loại, sinh báo cáo, VQA/grounding, sàng lọc/triage.

**Tiêu chí loại.**
- Nghiên cứu chỉ dùng CT.
- Bệnh học (pathology).
- VLM cho CXR nói chung mà không đánh giá riêng nốt/khối. Riêng nhóm này vẫn giữ khoảng 6 mô hình nền tảng để giới thiệu bối cảnh: CheXagent, MAIRA-2, RadVLM, LLaVA-Rad, MedGemma, CXR-LLaVA.

**Nguồn tìm bài.**
- arXiv, PubMed, Semantic Scholar, OpenAlex, Europe PMC, Consensus, WebSearch: khoảng 85–90 lần gọi.
- Undermind deep search: 101 bài được xếp hạng.
- Bỏ trùng còn 60 bài.

**Thống kê sơ bộ** (theo `corpus.md`; các bài thêm từ Undermind được tính theo cùng cách phân loại):
- VLM/MLLM: khoảng 20 bài.
- LLM chỉ đọc văn bản: 8 bài, gồm 4 bài trên báo cáo CXR hoặc có CXR, 4 bài trên báo cáo CT.
- Mô hình nền thị giác–ngôn ngữ (VL-foundation): 4 bài.
- CNN/ViT làm mốc: 15 bài.
- Dataset/benchmark: 12 bài.
- Bài tổng quan: 2 bài.
- Phần lớn là bài 2024–2026.

---

## 2. Các nhóm chủ đề

### Nhóm A: Đánh giá zero-shot VLM/MLLM trên nốt phổi CXR

Đây là nhóm sát chủ đề nhất. Nhóm còn nhỏ, các bài đều mới (2025–2026), và kết quả nhất quán theo hướng tiêu cực.

- **Nishio et al. (2026, AJR)**, L2/L46:
  - Thử sáu VLM trên 247 ảnh JSRT.
  - Sensitivity 0,6–27,3%, specificity 71,0–100%, accuracy 36,4–44,5%.
  - MedGemma-4b-it có sensitivity 5,2% dù specificity 100%. LLaVA-Rad có sensitivity cao nhất, 27,3%.
  - Mô hình phát hiện được nốt rõ; nốt mờ (subtle) thì gần như bỏ sót.
- **Khovanova et al. (2026, Front Digit Health)**, L1, đọc toàn văn:
  - 9 hệ thống trên MosMedData loại VII. Nốt đặc 6–30 mm, có CT xác nhận. Chỉ 83/100 ảnh được phân tích.
  - BiomedCLIP đạt AUC 0,743, sensitivity 0,737.
  - Agent MedRAX đạt AUC 0,704.
  - MLLM đa năng tốt nhất là Claude 3.7 Sonnet: AUC 0,625, sensitivity 0,316.
  - CXR-LLaVA chỉ đạt sensitivity 0,079.
- **Các nghiên cứu đa bệnh có nhóm nốt/khối hoặc u:**
  - Chen CH 2026 (L3): VLM nhạy với tổn thương lớn và nhiều ổ, có hạn chế với tổn thương nhỏ.
  - Cusick 2026 (L4): ChatGPT đúng 20% ca ung thư phổi.
  - Abou-Zeid 2026 (L51): khối phổi "ít được nhận diện đúng hơn".
  - Starychenko 2025 (L53): nốt có sensitivity ≤ 41,3%.
  - Lacaita 2025 (L52): là phản ví dụ, vì "tumor" thuộc nhóm phát hiện tốt nhất với GPT-4o.

### Nhóm B: VLM chuyên cho CXR làm mô hình nền

Gồm CXR-LLaVA (L5), CheXagent (L6), MedGemma (L7), MAIRA-2 (L8), RadVLM (L9), LLaVA-Rad (L10), MedRAX (L11), XrayGPT (L12) và LiteGPT (L59).

- Các bài này báo kết quả tốt trên benchmark tổng hợp. Ví dụ:
  - CXR-LLaVA: F1 0,81 trên 6 bệnh nội bộ, 0,56 ở tập ngoài.
  - MedGemma: cải thiện phân loại CXR 15,5–18,1%.
- Không bài nào báo kết quả riêng cho nốt hoặc ung thư trong tóm tắt.
- Khi được kiểm tra độc lập trên nốt (Nhóm A), chính các mô hình này cho sensitivity rất thấp.

### Nhóm C: Cơ chế, vì sao mô hình nền bỏ sót tổn thương nhỏ

- **Muthyala et al. (2026)**, L47, đọc toàn văn:
  - Embedding đông cứng của ViT, gồm BiomedCLIP và MedSigLIP, làm mất tín hiệu tổn thương nhỏ ở bước gộp toàn cục (token CLS).
  - Nốt trên ChestX-Det10, mức bbox: BiomedCLIP tăng AUC từ 0,688 (CLS) lên 0,899 khi gộp patch-local. MedSigLIP tăng từ 0,805 lên 0,955.
  - Tín hiệu vẫn còn trong các patch token, nhưng chỉ lấy lại được khi biết trước vùng tổn thương (oracle ROI).
- **Fisher (2025)**, L50: VLM đông cứng đạt F1 0,203 trên ChestX-ray14, so với 0,811 của CNN. Fine-tune làm sensitivity sụp xuống dưới 15%.
- *(Suy luận)* Đây là giả thuyết cơ chế hợp lý cho Nhóm A. Nhưng chưa bài nào kiểm chứng trực tiếp trên pipeline VLM sinh văn bản.

### Nhóm D: Định vị và grounding

- **Gosai et al. (2025, PMLR)**, L48, đọc toàn văn:
  - Với "lung lesion" trên CheXlocalize, tỷ lệ trúng: GPT-4 35,7%, GPT-5 35,7%, MedGemma 14,3%, CNN 29,0%, bác sĩ 85,0%.
  - Trung bình 9 bệnh, GPT-5 đạt 49,7%, so với CNN 59,9% và bác sĩ 80,1%.
- **NoduLoCC2026** (L36/L60): ngay cả hệ chuyên biệt cũng chỉ đếm đúng số nốt ở 53% ca, sai số trung vị 12,83 mm.
- **CheXpercept** (L49): VLM đạt khoảng 92% ở mức phát hiện thô, nhưng ≤ 2,9% ở mức chỉnh sửa đường viền. Benchmark này **không có nốt/khối**.
- Các mô hình hỗ trợ grounding (MAIRA-2, RadVLM, LiteGPT, VinDr-CXR-VQA) đều không báo kết quả riêng cho nốt.

### Nhóm E: LLM chỉ đọc văn bản trên báo cáo X-quang

- **Báo cáo CXR:**
  - Mukherjee 2023 (L55, Radiology): Vicuna gán nhãn 13 dấu hiệu, AUC trung vị 0,84.
  - Savage 2024 (L56, AJR): GPT-4 tốt hơn công cụ NLP chuyên ngành.
  - Park 2024 (L57): báo cáo CXR + CT, có ung thư phổi; GPT-4 sensitivity 0,71–1,00.
- **Ung thư phổi, báo cáo CT hoặc không rõ loại ảnh:**
  - Yasaka 2024 (L58): accuracy 0,983.
  - Gulati 2026 (L16): Lung-RADS 68,8%.
  - Wen 2025 (L18): khuyến nghị theo dõi 92,8–94,6%.
- **Suy luận theo guideline còn yếu:** Gamble 2024 (L15) cho thấy GPT-4 chỉ áp dụng đúng Fleischner 0,15. Con số tăng lên 0,66 khi đưa guideline vào prompt.

### Nhóm F: CAD học sâu làm mốc lâm sàng

- **Nam 2019** (L19): AUROC 0,92–0,99.
- **Yoo 2020** (L21, NLST): sensitivity nốt 86,2%.
- **Nam 2023** (L20): RCT với 10.476 người. AI giúp tăng tỷ lệ phát hiện nốt cần xử trí từ 0,25% lên 0,59% (OR 2,4).
- **Arzamasov 2024** (L45):
  - 5 phần mềm thương mại trên cùng nguồn MosMedData VII với Khovanova.
  - AUC tốt nhất 0,956 ở mức xác suất. Còn 0,812 khi bác sĩ kiểm tra lại phần phân đoạn và phân loại.
- **Nguy cơ ung thư từ CXR:** CXR-LC (L23) đạt AUC 0,755, so với 0,634 của tiêu chí CMS.
- **Hướng mới nhất của nhóm nguy cơ:** mô hình đa phương thức CXR + báo cáo + EHR (LUNG-KGMM, L13) và CXR + embedding LLM (Kim 2026, L61, mới có tiêu đề).

---

## 3. Xu hướng phương pháp theo thời gian

| Giai đoạn | Phương pháp chủ đạo | Cách đánh giá |
|---|---|---|
| 2004–2014 | CAD cổ điển, bone suppression (Kak04, Li08, Sch14 trong deep search) | Nghiên cứu nhiều bác sĩ đọc (multi-reader) trên JSRT |
| 2019–2021 | CNN có giám sát (DLAD/Lunit, Google), dùng NLST/PLCO | AUROC, JAFROC, kiểm định ngoài nhiều trung tâm, mô hình nguy cơ (CXR-LC) |
| 2022–2023 | Tương tác người–AI, giao diện, RCT, sinh/tăng cường nốt (NODE21) | RCT (Nam 2023), so sánh UI (Tang 2023), challenge |
| 2023–2024 | Mô hình nền thị giác–ngôn ngữ (BiomedCLIP, CheXzero, ELIXR) và VLM chuyên CXR (CXR-LLaVA, CheXagent, MAIRA-2) | Benchmark tổng hợp (CheXbench, F1 nhiều nhãn, RadGraph); không có đánh giá riêng cho nốt |
| 2024–2025 | LLM văn bản gán nhãn và trích xuất báo cáo; MLLM đa năng (GPT-4V/4o) zero-shot | κ so với CheXpert labeler; accuracy trên ca tự chọn |
| 2025–2026 | Đánh giá độc lập zero-shot VLM riêng cho nốt (Nishio, Khovanova); agent (MedRAX); định vị (Gosai); probe cơ chế (Muthyala); đa phương thức cho nguy cơ (LUNG-KGMM) | Sens/spec có McNemar; phân tầng theo độ mờ của nốt; hit rate định vị; probe tuyến tính theo kiểu gộp |

Ba nhận xét:
1. Kỳ vọng dịch từ "VLM làm được mọi thứ" sang kiểm tra độc lập, có phân tầng.
2. Cách đánh giá dịch từ chỉ số sinh văn bản (BLEU, RadGraph) sang chỉ số chẩn đoán và định vị.
3. Các bài mới bắt đầu tách riêng nốt mờ và nốt nhỏ khi báo kết quả.

---

## 4. Đồng thuận và bất đồng

### Đã tương đối được xác lập (nhiều nguồn độc lập cùng hướng)

1. **VLM/MLLM zero-shot chưa đủ để phát hiện nốt phổi độc lập trên CXR.** Bằng chứng: L1, L2, L3, L4, L51, L53, L62. Hai bài có thiết kế tốt nhất (L1, L2) đều kết luận "chưa đủ cho lâm sàng".
2. **Sensitivity là điểm yếu chính, đặc biệt với nốt nhỏ hoặc mờ.**
   - Các mô hình chuyên CXR thường có specificity cao nhưng sensitivity rất thấp (RadVLM, MedGemma, CXR-LLaVA).
   - *(Suy luận)* Có thể vì báo cáo huấn luyện phần lớn là bình thường, nên mô hình nghiêng về trả lời "không có bất thường".
3. **CAD chuyên biệt vẫn vượt xa.** Mốc tham chiếu là AUC 0,92–0,99 (L19), 0,956 (L45) và RCT có lợi ích lâm sàng (L20). Phép so sánh gần nhất trên cùng nguồn dữ liệu: AUC 0,956 (CAD, L45) so với 0,743 (BiomedCLIP, L1).
4. **LLM văn bản trích xuất thông tin từ báo cáo tốt**, ngang hoặc hơn NLP chuyên ngành (L55, L56, L57, L58). Nhưng **suy luận theo guideline** thì yếu nếu không đưa guideline vào prompt (L15).

### Còn mở hoặc chưa ngã ngũ

1. **Mô hình y khoa có tốt hơn mô hình đa năng không?**
   - Ở L1, các mô hình chuyên miền (CXR-LLaVA, XrayGPT) kém hơn Claude. Ở L49, VLM y khoa không có lợi thế. Ở L48, MedGemma kém GPT-5.
   - Ngược lại, MedGemma báo cải thiện so với Gemma gốc (L7).
   - Có thể do khác bài toán hoặc khác prompt; cần so sánh có kiểm soát.
2. **Agent hoặc mô hình nền thay cho MLLM sinh văn bản.** MedRAX và BiomedCLIP dẫn đầu ở L1. Nhưng L50 cho thấy BiomedCLIP đông cứng rất kém trên ChestX-ray14. Kết quả phụ thuộc cách lấy ngưỡng (Youden trên chính tập test ở L1) và cỡ mẫu.
3. **Khối u lớn có dễ hơn không?** L52 xếp "tumor" vào nhóm tốt nhất; L51 lại thấy khối ít được nhận đúng. *(Suy luận)* Có thể do khác định nghĩa (khối lớn so với nốt 6–30 mm) và khác cách chọn ca.
4. **Cơ chế thất bại.** Giả thuyết gộp toàn cục (L47) có bằng chứng probe, nhưng chưa được kiểm chứng trên VLM sinh văn bản end-to-end.

---

## 5. Khoảng trống và câu hỏi mở

| # | Khoảng trống | Bằng chứng còn thiếu |
|---|---|---|
| K1 | **Chưa có đánh giá VLM trên nốt CXR ở quy mô lớn, nhiều trung tâm.** Mọi nghiên cứu đều 83–247 ảnh, một nguồn, hồi cứu | L1, L2 tự nêu hạn chế |
| K2 | **Chưa có fine-tuning hay PEFT (LoRA) VLM riêng cho nốt hoặc ung thư trên CXR** có báo cáo sens/spec theo nốt | Không thấy bài nào. L50 cảnh báo fine-tune toàn phần làm sụp sensitivity |
| K3 | **Chưa có đánh giá grounding hay định vị riêng cho nốt hoặc khối bằng VLM.** CheXpercept bỏ nốt/khối; Gosai gộp chung thành "lung lesion" | L48, L49 |
| K4 | **Chưa có phân tầng theo kích thước và độ mờ của nốt** ngoài phân tích theo grade của JSRT ở L2 | L2 có một phần; L1 không có phân bố kích thước |
| K5 | **Chưa nối cơ chế với hành vi.** Mất tín hiệu khi gộp (L47) chưa được kiểm chứng trên VLM sinh văn bản; chưa có can thiệp kiểu patch-local hay ROI cho VLM | L47 tự nêu |
| K6 | **LLM văn bản chưa được đánh giá cho khuyến nghị theo dõi nốt trên báo cáo CXR.** Mọi bài về Fleischner/Lung-RADS đều dùng báo cáo CT | L15–L18 |
| K7 | **Chưa có mô hình nguy cơ CXR + LLM đã qua bình duyệt.** LUNG-KGMM là preprint, Kim 2026 mới có tiêu đề | L13, L61 |
| K8 | **Chưa có benchmark VQA với câu hỏi riêng về nốt** (có/không, vị trí, kích thước, tính chất). ReXVQA và REVEAL-CXR chưa xác nhận có nhãn nốt | L42, L44 |
| K9 | **Đánh giá chưa chuẩn hóa.** Prompt khác nhau, chuyển văn bản thành nhãn nhị phân theo cách khác nhau, ảnh PNG thay vì DICOM, lấy ngưỡng Youden trên tập test | L1 tự nêu |

---

## 6. Hướng tương lai đề xuất

1. **Benchmark nốt CXR chuẩn hóa cho VLM.** Gộp JSRT, NODE21, NoduLoCC, VinDr-CXR (nốt/khối) và MosMedData VII. Dùng chung prompt và chung quy tắc chuyển thành nhãn. Báo sens/spec có phân tầng theo kích thước và độ mờ, kèm định vị (giải quyết K1, K4, K8, K9).
2. **Tinh chỉnh hiệu quả tham số (LoRA/QLoRA) VLM cho nốt**, so với CAD CNN trên cùng tập. Theo dõi riêng sensitivity, vì L50 cho thấy sensitivity có thể sụp. Có thể dùng loss cân bằng hoặc focal để giữ sensitivity (K2).
3. **Kiến trúc nhận biết vùng (region-aware).** Theo L47, đưa patch token cục bộ hoặc đề xuất ROI từ một detector vào LLM, kiểu "detector → VLM giải thích" hoặc agent như MedRAX (K5).
4. **Grounding và VQA riêng cho nốt.** Đánh giá định vị bằng IoU hoặc khoảng cách tâm (theo NoduLoCC), mở rộng CheXpercept sang nốt/khối (K3).
5. **LLM văn bản trên báo cáo CXR**: trích xuất nốt và gợi ý chuyển chụp CT, có đưa guideline vào prompt hoặc dùng RAG (K6).
6. **Nguy cơ đa phương thức**: CXR + embedding LLM từ báo cáo và EHR, so với CXR-LC (K7).

---

*Ghi chú về nguồn: danh mục đầy đủ có DOI hoặc mã arXiv nằm trong `corpus.md` (L1–L62). Workspace Undermind: https://app.undermind.ai/projects/fbc55738-03be-48ab-b376-2937981a6356*
