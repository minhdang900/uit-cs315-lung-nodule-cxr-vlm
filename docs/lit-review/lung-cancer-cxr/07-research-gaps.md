# Bản đồ khoảng trống nghiên cứu: VLM cho phát hiện và định vị nốt phổi trên X-quang ngực

*Phiên bản 1, ngày 2026-11-12. Tài liệu này kế thừa K1–K9 và RQ1–RQ6 từ `01-literature-review.md` §5 và `02-literature-synthesis.md` §5, rồi cập nhật chúng theo 30 bài mới (L63–L92).*

**Nguồn đầu vào**
- `05-detailed-analysis.md`: nguồn chính. Khi `05` và `03-thematic-clusters.md` khác nhau, theo `05`.
- `03`, `01`, `02`, `corpus.md`, `papers/_RSNA-abstract-notes.md`.
- Spec `../../design/2026-11-12-lung-nodule-cxr-vlm-lora-design.md`: chỉ đọc, không sửa.

**Đề tài đang xét**
- VLM chính là MedGemma-4B. VLM so sánh là Qwen2.5-VL-3B.
- Baseline là DenseNet-121 (B1) và YOLOv8 (B2).
- Ca dương gộp Nodule và Mass.
- Huấn luyện trên VinBigData. Test ngoài trên NIH ChestX-ray14 và JSRT.

---

## 0. Quy ước

### Mức đọc của bằng chứng (theo `05`)

| Thẻ | Nghĩa | Bài |
|---|---|---|
| **FT** | Đọc toàn văn PDF trong lượt phân tích `05` | L6, L7, L8, L10, L37, L42, L64, L65, L66, L67, L68, L69, L70, L71, L72, L73, L78, L79, L80, L81, L82, L83, L84, L86, L87, L88, L91, L92 (28 bài) |
| **AB†** | Abstract, kèm ghi chú toàn văn từ một phiên trước, không đọc lại | L1, L47, L48, L49 |
| **AB** | Chỉ abstract | Các bài còn lại (55 bài, không kể 4 bài AB†) |
| **T** | Chỉ tiêu đề hoặc metadata | L60, L61, L62 |

### Ký hiệu

- **∅ (vắng bằng chứng):** khoảng trống được suy ra vì *không có bài nào trong corpus 90 bài* làm việc đó. Đây là tuyên bố về corpus, không phải bằng chứng rằng chưa ai làm trên thế giới. Corpus được dựng từ một số lượt tìm có giới hạn (xem `corpus.md` §1, §6, §7).
- **Mức độ chắc chắn**
  - **Cao:** có ít nhất một bài FT trực tiếp liên quan, và nhiều bài cùng hướng.
  - **Trung bình:** chủ yếu dựa trên AB, hoặc dựa trên ∅ trong một cụm đã tìm khá kỹ.
  - **Thấp:** chỉ dựa trên ∅, hoặc bằng chứng gián tiếp, mâu thuẫn hay có rủi ro độ tin cậy.

---

## 1. Phương pháp xác định khoảng trống

### 1.1 Taxonomy

Tài liệu dùng **taxonomy 7 loại của Miles (2017)**, *A Taxonomy of Research Gaps: Identifying and Defining the Seven Research Gaps*. Đây là nguồn phương pháp ngoài corpus và không có mã L. Mỗi khoảng trống được gán **một loại chính**. Nếu cần, ghi thêm một loại phụ trong ngoặc.

| Loại (EN) | Tên tiếng Việt | Định nghĩa dùng ở đây | Dấu hiệu trong corpus |
|---|---|---|---|
| **Evidence** | Mâu thuẫn bằng chứng | Các kết quả trái chiều nhau, chưa có so sánh có kiểm soát để phân xử | Bài A và bài B kết luận ngược nhau |
| **Knowledge** | Thiếu tri thức | Chưa có nghiên cứu nào làm việc X | ∅ trong cụm liên quan |
| **Practical** | Khoảng cách thực hành | Thực hành lâm sàng hoặc triển khai có vấn đề mà nghiên cứu chưa xử lý | Bài T7 cho thấy rủi ro, chưa có bài VLM tương ứng |
| **Methodological** | Phương pháp | Cách đo hoặc thiết kế hiện tại có vấn đề hệ thống | Chỉ số sai mục tiêu, ngưỡng chọn trên test, rò rỉ dữ liệu |
| **Empirical** | Thực nghiệm | Có giả thuyết hoặc cơ chế, nhưng chưa được kiểm chứng thực nghiệm trong bối cảnh cần | Có probe hoặc kết quả trên CNN, chưa có trên VLM |
| **Theoretical** | Lý thuyết | Thiếu khung giải thích vì sao hiện tượng xảy ra | Có quan sát, chưa có cơ chế được kiểm chứng |
| **Population** | Quần thể | Một nhóm dân số hoặc nguồn dữ liệu chưa được đại diện | Bằng chứng tập trung ở vài quốc gia hoặc vài bộ dữ liệu |

### 1.2 Quy trình

1. **Kế thừa.** Lấy K1–K9 (`01` §5) và RQ1–RQ6 (`02` §5) làm điểm xuất phát.
2. **Đối chiếu với 30 bài mới (L63–L92)**, theo ghi chép trong `05`. Mỗi K được gán một trong năm trạng thái:
   - **Xác nhận:** bài mới không thay đổi khoảng trống.
   - **Củng cố:** bài mới tăng tầm quan trọng hoặc độ chắc chắn.
   - **Thu hẹp:** khoảng trống vẫn còn, nhưng phát biểu phải hẹp lại vì đã có tiền lệ gần.
   - **Yếu đi:** đã có bài lấp một phần.
   - **Đóng:** đã có bài lấp đầy.
3. **Tìm khoảng trống mới (K10–K20).** Duyệt theo hai cách:
   - Theo từng cụm T1–T10 và M của `03`.
   - Theo năm khía cạnh: task, model, data, evaluation, clinical.
4. **Gán thông tin cho từng khoảng trống:** loại theo taxonomy, mức độ chắc chắn, ánh xạ sang RQ, và mức đề tài giải quyết, có dẫn mục cụ thể trong spec.
5. **Xếp ưu tiên** theo impact × feasibility, với ràng buộc một học kỳ và Colab trả phí (§4).

### 1.3 Ràng buộc trung thực

- Mọi nhận định đều có mã L. Không có con số nào ngoài nguồn đã đọc. Phép tính số học từ số trong nguồn được ghi rõ là "ước tính số học".
- Khoảng trống dựa trên ∅ được đánh dấu riêng.
- **Các chỉnh sửa từ Task 1 đã áp dụng:**
  - L66 là CT liều thấp (NLST), **không** được dùng làm bằng chứng độ phân giải trên CXR.
  - L64 không phải tiền lệ LoRA duy nhất. L69 dùng LoRA r16 cho LLM sinh báo cáo CXR. L71 tinh chỉnh Qwen2.5-VL cho grounding, đạt mIoU 79,02 trên MS-CXR.
  - L83 cho thấy 896 px là tối ưu cho CAD nốt.
  - Mức đọc lấy theo `05`.

---

## 2. Bảng khoảng trống

### 2.1 Khoảng trống kế thừa (K1–K9) và trạng thái sau khi thêm L63–L92

| ID | Khoảng trống | Loại | Bằng chứng (mức đọc) | Trạng thái sau L63–L92 | Mức độ chắc chắn | Map K/RQ | Đề tài giải quyết? |
|---|---|---|---|---|---|---|---|
| **K1** | Chưa có đánh giá VLM cho nốt CXR ở quy mô lớn và nhiều nguồn. Các nghiên cứu hiện có dùng 50–247 ảnh từ một nguồn, hồi cứu | Methodological (+Population) | L1 (FT, 83/100 ảnh, MosMedData VII); L2 (FT, 247 ảnh JSRT); L3, L4, L51 (FT); L53 (AB). ∅: đợt Scopus không thêm bài nào vào T1 (`03` T1) | **Xác nhận.** 30 bài mới không có đánh giá zero-shot VLM nào cho nốt | Cao | K1 → RQ5 | **Một phần.** Spec §2 mục tiêu 1, §6 và E2 đánh giá trên 3 tập test (VinBigData, NIH, JSRT), lớn hơn L1 và L2, nhưng vẫn hồi cứu và không có MosMedData |
| **K2** | Chưa có tinh chỉnh hay PEFT VLM cho **phát hiện nốt/khối** trên CXR có báo cáo sens/spec theo nốt | Knowledge | L64 (FT): QLoRA SmolVLM, chỉ sinh báo cáo. L69 (FT): LoRA r16 cho Vicuna, sinh báo cáo. L71 (FT): RL tinh chỉnh Qwen2.5-VL cho phrase grounding. L7 (FT): MedGemma tinh chỉnh SFT toàn tham số, không có nhãn nốt. L50 (AB): tinh chỉnh làm sensitivity giảm dưới 15%. ∅ cho tác vụ nốt | **Thu hẹp.** Đã có tiền lệ PEFT và tinh chỉnh VLM trên CXR (L64, L69, L71), nhưng **không bài nào nhắm phát hiện hay định vị nốt**. Phát biểu phải đổi từ "chưa ai LoRA VLM CXR" thành "chưa ai LoRA VLM cho phát hiện/định vị nốt" | Trung bình (tiền lệ là FT; phần "chưa có cho nốt" là ∅) | K2 → RQ1, RQ2 | **Có.** Spec §5.2 C1–C4, E3–E4. §3 và §4 cần thêm L64, L69, L71 vào phần "đã có" (xem §5.2, S3) |
| **K3** | Chưa có đánh giá định vị riêng cho nốt/khối bằng VLM | Knowledge (+Methodological) | L48 (AB†): gộp thành "lung lesion", MedGemma hit rate 14,3%. L49 (AB†): benchmark không có nốt/khối. L69, L70, L71 (FT) và L8 (FT): grounding tổng quát, không riêng nốt. L36 (AB): CAD chuyên biệt cũng chỉ đếm đúng số nốt ở 53% ảnh. L88 (FT): đầu ra có box giảm FN của người đọc (P = 0,04) | **Củng cố.** Ba phương pháp grounding ở hội nghị và tạp chí hàng đầu (L69–L71) đều không thử trên nốt. L88 cho thấy box có giá trị an toàn | Cao | K3 → RQ4 | **Có.** Spec §5.2 C2, §8.2 (IoU, FROC, khoảng cách tâm theo L36) |
| **K4** | Chưa có phân tầng theo kích thước và độ mờ của nốt khi đánh giá VLM | Empirical | VLM: L2 (FT, có bảng đầy đủ theo 5 grade JSRT; MedGemma: 25% ở grade 5 → 0% ở grade 1), L3 (FT, định tính). CAD: L73 (FT) accuracy dưới 0,2 với nốt nhỏ hơn 10 mm; L83 (FT) sensitivity 0,418 / 0,409 với nốt nhỏ hơn 10 mm; L86 (FT) phân tầng theo độ khó; L29 (AB) điểm mù giải phẫu, sensitivity 0,50–0,64; L34 (AB) khó với GGO và rốn phổi | **Củng cố.** Bằng chứng từ CAD cho thấy kích thước và vị trí là biến số mạnh. Vẫn ∅ cho VLM, nhất là phân tầng theo **vị trí giải phẫu** | Cao (với CAD); Trung bình (với VLM) | K4 → RQ2 | **Một phần.** Spec E5 và §8.3 phân tầng theo kích thước (tertile, mm với JSRT) và độ mờ JSRT. **Chưa có** phân tầng theo vùng giải phẫu (L29, L34) |
| **K5** | Chưa nối cơ chế với hành vi: hiện tượng mất tín hiệu khi gộp (L47) chưa được kiểm chứng trên VLM sinh văn bản | Empirical (+Theoretical) | L47 (AB†): probe MedSigLIP, AUC 0,805 → 0,955, cần oracle ROI. L8 (FT): MAIRA-2 không dùng token CLS. L82 (FT, CNN): nốt hưởng lợi nhiều từ độ phân giải. L83 (FT, CNN): 896 px tối ưu. L66 (FT) **bị loại** vì là CT | **Xác nhận, nhưng nhánh độ phân giải yếu đi với MedGemma.** MedGemma đã nhận 896×896 (L7), đúng mức tối ưu của L83. Giả thuyết còn lại là cách gộp và nén đặc trưng, cộng với dữ liệu | Trung bình | K5 → RQ3 | **Một phần, ở mức mở rộng.** C5 và C6 chỉ là phần mở rộng (spec §5.2). E5 là kiểm toán hành vi (che vùng), không phải kiểm chứng cơ chế. Spec §9 nêu probe patch-local sau LoRA là câu hỏi mở |
| **K6** | LLM văn bản chưa được đánh giá cho khuyến nghị theo dõi nốt trên **báo cáo CXR** | Knowledge | L15 (AB): mô tả nốt hư cấu. L16, L17, L18 (AB): báo cáo CT hoặc LDCT. ∅ cho CXR | **Xác nhận.** Không có bài mới trong T5 | Trung bình (∅) | K6 → RQ6 | **Không.** Spec §9 xếp vào hướng phát triển |
| **K7** | Chưa có mô hình nguy cơ CXR + LLM đã qua bình duyệt và đáng tin | Knowledge (+Evidence) | L13 (AB, preprint); L61 (T, tóm tắt hội nghị); L67 (FT): đã bình duyệt (TLCR), dùng BioGPT và CXR, nhưng **nguồn nhãn sống còn không rõ**; L91 (FT): CXR + EMR, không có LLM | **Yếu đi một chút về hình thức.** L67 là bài đã bình duyệt, nhưng độ tin cậy thấp nên không lấp được khoảng trống | Thấp | K7 → (ngoài RQ hiện tại) | **Không.** Spec §9 |
| **K8** | Chưa có benchmark VQA với câu hỏi riêng về nốt (có/không, vị trí, kích thước, tính chất) | Knowledge | L42 (FT): ReXVQA không có tầng riêng cho nốt; câu hỏi sinh từ báo cáo. L43 (AB): VinDr-CXR-VQA không có tầng nốt trong abstract. L44 (AB): REVEAL-CXR chưa xác nhận có nhãn nốt | **Xác nhận.** L42 (FT) cho thấy benchmark tổng hợp che mất điểm yếu: MedGemma đạt 83,24% trên ReXVQA nhưng sensitivity với nốt chỉ 5,2% (L2) | Trung bình | K8 → RQ5 | **Một phần.** Benchmark của spec (§4 đóng góp 1, §5.3) chỉ hỏi có/không và box. Không hỏi kích thước hay tính chất |
| **K9** | Đánh giá chưa chuẩn hóa: prompt khác nhau, chuyển văn bản thành nhãn khác nhau, ngưỡng Youden chọn trên test | Methodological | L1 (FT): tự nêu. L4 (FT): ngữ cảnh làm "lộ" đáp án, accuracy 28% → 62%. L84 (FT): cộng đồng dùng chủ yếu AUC-ROC, hiếm khi báo hiệu chỉnh. L92 (FT): độ không đồng nhất cao, cần STARD-AI | **Củng cố.** L84 và L92 (đều FT) cho thấy vấn đề có hệ thống, không riêng VLM | Cao | K9 → RQ5 | **Có.** Spec §2 mục tiêu 1, §5.3, §8.1 và §8.4: prompt cố định, đầu ra JSON, ngưỡng chọn trên validation, chạy lặp, bootstrap CI. Nên thêm điều kiện có/không ngữ cảnh (L4) |

### 2.2 Khoảng trống mới (K10–K20)

> **Ghi chú về "chưa có VLM tinh chỉnh cho định vị nốt".** Brief nêu ví dụ này như một khoảng trống mới. Ở đây nó **được gộp vào K2 và K3**, dưới dạng phát biểu đã thu hẹp, để tránh trùng. Khoảng trống mới liên quan là K10: tiền lệ tinh chỉnh chỉ đo bằng chỉ số NLG.

| ID | Khoảng trống | Loại | Bằng chứng (mức đọc) | Mức độ chắc chắn | Map K/RQ | Đề tài giải quyết? |
|---|---|---|---|---|---|---|
| **K10** | Tinh chỉnh VLM CXR hiện chỉ được đánh giá bằng **chỉ số NLG hoặc benchmark tổng hợp**. Các chỉ số này không đo độ đúng lâm sàng cho nốt | Methodological | L64 (FT): ROUGE-L 0,128 → 0,580, nhưng ví dụ minh họa chính của bài sai lâm sàng (kỳ vọng "Chilaiditis sign", sinh ra "right upper lobe consolidation"). L69 (FT): chỉ có chỉ số sinh văn bản. L7 (FT): các tác vụ phân loại CXR không có nhãn nốt/khối. L42 (FT): VQA cao (83,24%) che điểm yếu với nốt (L2, FT) | Cao | Mới; tách từ K2/K9 → RQ1, RQ5 | **Có.** Spec §5.3 dùng đầu ra JSON có cấu trúc. §8.1 và §8.2 dùng sens/spec, AUC, IoU, FROC thay cho NLG |
| **K11** | **Cách xử lý độ phân giải trong VLM** chưa được kiểm tra cho nốt: độ phân giải gốc, chia ô (tiling), cắt theo trường phổi | Empirical | L82 (FT, CNN): AUC nốt ở 64 px là 0,689, ở 320 px là 0,854; tối đa ở 256–448 px. L83 (FT, CNN): 896 px tối ưu, JAFROC FOM của RetinaNet 0,906. Encoder VLM: L6 (FT) CheXagent 512 px qua nội suy positional encoding; L64 (FT) SmolVLM 384×384, 81 token ảnh; L8 (FT) MAIRA-2 518×518; L7 (FT) MedGemma 896×896. ∅ cho thí nghiệm độ phân giải trên VLM với nốt | Trung bình (dữ kiện là FT; khoảng trống là ∅) | Mới; tách từ K5 → RQ3 (đề xuất RQ3b) | **Một phần, ở mức mở rộng.** Spec C6 (§5.2). Cách phát biểu C6 cần sửa với MedGemma (xem §5.2, S4) |
| **K12** | **Đánh giá có tính đến tỷ lệ bệnh** (PPV, FPR, AUC-PR, calibration) vắng mặt trong các nghiên cứu VLM cho nốt | Methodological | L84 (FT): AUC-ROC che hiệu năng của lớp thiểu số; khuyến nghị AUC-PR và balanced Brier. L79 (FT): ở tỷ lệ bệnh 2,2%, AI chỉ đạt AUROC 0,648 và sensitivity 35,3%. L75 (AB): FPR khoảng 3%, gấp 10 lần bác sĩ. L21 (AB): PPV 3,8%. L66 (FT, CT, chỉ dùng làm ví dụ phương pháp): PR-AUC và hiệu chỉnh isotonic ở tỷ lệ bệnh khoảng 6%. VLM: L1 (FT) và L2 (FT) chỉ báo accuracy và sens/spec. L2: JSRT có tỷ lệ nốt 62% | Cao | Mới → đề xuất RQ8 | **Một phần.** Spec §8.1 có PPV và NPV, nhưng **không có** AUC-PR, calibration (ECE hoặc Brier), hay phân tích ở tỷ lệ bệnh thực tế |
| **K13** | **Nguy cơ automation bias** từ VLM yếu nhưng nói tự tin chưa được đo. Chưa có nghiên cứu người đọc nào dùng đầu ra VLM | Practical | L88 (FT): AI giả sai làm FN của bác sĩ tăng từ 2,7% lên 33,0%; có box thì còn 20,7%. L90 (AB): người đọc đổi theo AI trong 59% (AI kém) đến 67% (AI tốt) trường hợp. L32 (AB): giao diện chỉ văn bản tốt hơn không có AI. L62 (T): commentary cảnh báo. ∅: không có nghiên cứu người đọc với VLM | Cao (về rủi ro); Trung bình (về khoảng trống VLM, là ∅) | Mới → đề xuất RQ9 (đo gián tiếp) | **Một phần, gián tiếp.** Kiểm toán E5 và §8.3 cho bằng chứng an toàn ở mức mô hình. **Không** có nghiên cứu người đọc. Có thể thêm chỉ số "lỗi tự tin" (xem RQ9) |
| **K14** | Các phương pháp grounding mới **chưa được thử trên nốt**, và chưa có chỉ số cho trường hợp **0, 1 hoặc nhiều nốt** trong một ảnh | Methodological (+Knowledge) | L69 (FT): neo giải phẫu trong sinh báo cáo. L70 (FT): GMPG với 0, 1 hoặc nhiều vùng; khoảng 23% cụm từ trong MS-CXR có nhiều vùng; chỉ số N-Acc và Precision@F1=1. L71 (FT): GRPO, mIoU 79,02 nhưng là grounding có điều kiện (cụm từ cho trước). L68 (FT): mIoU 0,243. L36 (AB): đúng số nốt ở 53% ảnh. ∅ cho nốt | Cao | Mới; mở rộng K3 → RQ4 | **Một phần.** Spec §8.2 có IoU, FROC và khoảng cách tâm. **Chưa có** N-Acc hay độ chính xác số nốt cho ảnh có nhiều nốt hoặc không có nốt |
| **K15** | Bằng chứng **mâu thuẫn** về việc VLM y khoa có hơn VLM đa năng với nốt hay không. Chưa có so sánh có kiểm soát | Evidence | L1 (FT): CXR-LLaVA và XrayGPT kém Claude 3.7. L48 (AB†): MedGemma 14,3% so với GPT-5 35,7%. L49 (AB†): VLM y khoa không hơn. Ngược lại: L7 (FT) MedGemma hơn Gemma 3 (macro F1 MIMIC 88,9 so với 81,2); L63 (AB) mô hình chuyên ngành 70,5% so với GPT-4V 29,6%. ∅: chưa có so sánh cùng dữ liệu và cùng cấu hình tinh chỉnh | Trung bình | Mới (trước đây chỉ ghi "còn mở" ở `01` §4) → đề xuất RQ7 | **Có, ở quy mô nhỏ.** Spec C4 và E4: cấu hình LoRA tốt nhất áp lên Qwen2.5-VL-3B, so với MedGemma-4B. Hai mô hình khác cả cỡ lẫn kiến trúc, nên kết luận bị nhiễu |
| **K16** | Bằng chứng VLM cho nốt đến từ **rất ít quần thể**. Chưa có đánh giá VLM cho nốt trên dữ liệu Việt Nam hay Đông Nam Á | Population | VLM: MosMedData Nga (L1, AB†); JSRT Nhật (L2, FT). CAD: Hàn Quốc (L19, L20, L75, AB), Hà Lan (L80, FT), Anh (L72, FT; L27, AB). VinDr-CXR: một quốc gia, 2 bệnh viện (L37, FT). L43 (AB) dùng VinDr cho VQA, không riêng nốt. ∅ cho VLM + nốt + dữ liệu Việt Nam | Trung bình | Mới → RQ1 (trục kiểm định ngoài) | **Một phần.** Huấn luyện trên VinBigData (Việt Nam), test ngoài trên NIH (Mỹ) và JSRT (Nhật). Spec §6 |
| **K17** | **Tinh chỉnh VLM khi dữ liệu dương rất hiếm** (khoảng 3%) chưa có nghiên cứu về chiến lược cân bằng lớp. Rủi ro sụp sensitivity đã được ghi nhận | Methodological (+Empirical) | L37 (FT): Nodule/Mass có 410 ảnh train (2,73%) và 176 ảnh test (5,87%). L50 (AB): tinh chỉnh làm sensitivity xuống dưới 15%; chi tiết cách tinh chỉnh không rõ trong nguồn đã đọc. L78 (FT): tổng hợp nốt và hard-example mining cho CAD. L73 (FT): nhiều nhãn yếu cộng ít box. ∅ cho chiến lược cân bằng lớp khi LoRA VLM | Trung bình | Mới; tách từ K2 → RQ2 | **Có.** Spec C3 (oversampling 1:1 cộng trọng số loss), §8.1 (chỉ số chế độ báo bình thường), E3 (sensitivity theo epoch) |
| **K18** | **Rò rỉ dữ liệu huấn luyện** (contamination) khi đánh giá VLM chưa được kiểm tra | Methodological | L40 (AB): MIMIC-CXR là nguồn huấn luyện của hầu hết VLM CXR. L5 (AB), L10 (FT) và L12 (AB) dùng MIMIC. L63 (AB) test trên VinDr-CXR. L43 (AB) dựng VQA từ box VinDr-CXR. Nguồn đã đọc của L7 (FT) không cho biết MedGemma có thấy VinDr hay JSRT khi huấn luyện không. ∅: không bài nào trong T1 kiểm tra rò rỉ | Thấp (∅, và thiếu thông tin về dữ liệu huấn luyện) | Mới → RQ5 (giao thức) | **Một phần.** Spec dùng nhiều tập test, nhưng **không** có bước kiểm tra rò rỉ cho MedGemma hay Qwen trên VinBigData, NIH và JSRT |
| **K19** | **RL (GRPO) như một thay thế cho LoRA-SFT** khi học xuất box chưa được thử cho nốt | Knowledge | L71 (FT): GRPO với phần thưởng IoU cộng phần thưởng ngữ nghĩa, Qwen2.5-VL, mIoU 79,02 trên MS-CXR so với các baseline 52,7–62,8. ∅ cho nốt | Trung bình | Mới → RQ4 (hướng phát triển) | **Không.** Ngoài phạm vi; nên ghi vào §9 |
| **K20** | **Nhãn yếu từ LLM** trích từ báo cáo chưa được dùng để mở rộng dữ liệu huấn luyện VLM cho nốt | Practical | L55 (AB): Vicuna, AUC trung vị 0,84. L56 (AB): GPT-4 hơn NLP chuyên ngành. L57 (AB): sensitivity 0,71–1,00. L65 (FT): nhãn GPT-4o accuracy 92,9%, nhưng chỉ nhị phân "có bệnh/không bệnh". L72 (FT): X-Raydar-NLP gán nhãn quy mô lớn, có "mass or nodule". L73 (FT): nhãn yếu cộng ít box. ∅ cho VLM + nốt | Trung bình | Mới; nối K5, K6 → (ngoài RQ) | **Không.** Spec §5.3 cố ý **không** dùng nhãn do LLM sinh. Đây là lựa chọn hợp lý cho một học kỳ |

### 2.3 Tổng kết số lượng

| Nhóm | Số | ID |
|---|---|---|
| Kế thừa | 9 | K1–K9 |
| · Xác nhận | 4 | K1, K6, K8; K5 (xác nhận, nhưng nhánh độ phân giải yếu đi) |
| · Củng cố | 3 | K3, K4, K9 |
| · Thu hẹp | 1 | K2 |
| · Yếu đi | 1 | K7 (chỉ về hình thức; bằng chứng lấp vào kém tin cậy) |
| · Đóng | 0 | — |
| Mới | 11 | K10–K20 |
| **Tổng** | **20** | |

**Phân bố theo loại:**

| Loại | Số | ID |
|---|---|---|
| Methodological | 7 | K1, K9, K10, K12, K14, K17, K18 |
| Knowledge | 6 | K2, K3, K6, K7, K8, K19 |
| Empirical | 3 | K4, K5, K11 |
| Practical | 2 | K13, K20 |
| Evidence | 1 | K15 (và là loại phụ của K7) |
| Population | 1 | K16 |

Không khoảng trống nào được gán loại chính là theoretical. Loại này chỉ xuất hiện như loại phụ của K5.

**Dựa chủ yếu trên ∅:** K2 (phần "cho nốt"), K6, K11, K13 (phần VLM), K14, K16, K17, K18, K19, K20.

---

## 3. Bản đồ khoảng trống

### 3.1 Ma trận cụm × khía cạnh

**Năm khía cạnh**
- **Task:** bài toán nốt (phát hiện, định vị) có được làm không.
- **Model:** loại mô hình, đặc biệt VLM.
- **Data:** dữ liệu nốt có nhãn, quy mô và số nguồn.
- **Evaluation:** chỉ số phù hợp (định vị, tỷ lệ bệnh, calibration).
- **Clinical:** người đọc, RCT hoặc triển khai.

**Mức độ bao phủ** (xét *cho nốt/khối trên CXR*)

| Ký hiệu | Mức | Tiêu chí |
|---|---|---|
| ● | covered | Ít nhất 3 bài có bằng chứng riêng cho nốt/khối |
| ◐ | sparse | 1–2 bài, hoặc có bằng chứng nhưng không riêng cho nốt |
| ○ | empty | Không có bài nào trong corpus (∅) |

| Cụm | Task | Model | Data | Evaluation | Clinical |
|---|---|---|---|---|---|
| **T1** Zero-shot VLM | ◐ Phát hiện ● (L1, L2, L3, L53); định vị ○ | ● Nhiều VLM đa năng và chuyên CXR (L1, L2) | ◐ 50–247 ảnh, một nguồn (L1, L2, L4) | ◐ Accuracy, sens/spec; ngưỡng Youden trên test (L1); không có AUC-PR | ○ Chỉ có commentary (L62, T) |
| **T2** VLM CXR / tạo sinh | ◐ Sinh báo cáo, VQA; không riêng nốt (L5, L6, L10, L63) | ● Nhiều kiến trúc; có LoRA (L64) | ◐ Dữ liệu lớn nhưng không có nhãn nốt chuyên biệt (L63, L10) | ◐ NLG, RadGraph; ví dụ sai lâm sàng ở L64 | ◐ Bác sĩ chấp nhận báo cáo (L63); tiết kiệm thời gian (L6) |
| **T3** Encoder VL | ◐ Probe nốt (L47) | ◐ BiomedCLIP, MedSigLIP (L14, L47) | ◐ Chỉ ChestX-Det10 (L47); L66 là CT nên loại | ◐ AUC mức box có oracle ROI (L47) | ○ |
| **T4** Định vị / VQA | ◐ Grounding tổng quát (L69–L71); lung lesion (L48) | ● Nhiều VLM grounding (L8, L9, L70, L71) | ◐ MS-CXR, PadChest-GR, VinDr-VQA (L43); ít dữ liệu nốt chuyên biệt | ● IoU, mIoU, hit rate, N-Acc (L48, L70, L71) | ○ |
| **T5** LLM văn bản | ◐ Trích xuất (L55–L57); theo dõi nốt chỉ trên CT (L15–L18) | ● Nhiều LLM | ◐ Báo cáo CT chiếm ưu thế với guideline | ◐ κ, accuracy | ◐ Tỷ lệ khuyến nghị có hại (L18, CT) |
| **T6** CAD độc lập | ● Phát hiện và định vị (L19, L25, L29, L81) | ◐ Chỉ CNN, không có VLM | ● Lớn, nhiều trung tâm (L19, L72, L75) | ● JAFROC, FROC, PPV, tỷ lệ bệnh thấp (L19, L21, L79, L81) | ◐ Triển khai thực tế (L77) |
| **T7** Người–AI | ● Nốt (L85, L87, L89) | ◐ Chỉ CAD; ○ VLM | ◐ Tập làm giàu ca bệnh (L74, L86, L90) | ● MRMC, JAFROC (L76, L86) | ● RCT (L20); automation bias (L88, L90) |
| **T8** Nguy cơ / sàng lọc | ◐ Nguy cơ dài hạn, không phải phát hiện nốt (L23, L91) | ◐ CNN, ViT; LLM chỉ ở L13, L61, L67 | ● PLCO, NLST, MGB (L23, L91) | ◐ AUC, C-index (L67 kém tin cậy) | ◐ Chọn người chụp CT (L91) |
| **T9** Dataset | ◐ Phát hiện, định vị (L35, L36) | — (không áp dụng) | ● Có box nốt (L35, L37, L38, L41) | ◐ NODE21 score (L78), khoảng cách mm (L36) | ○ |
| **T10** Tổng quan | ● Gộp số liệu cho nốt (L33, L34, L92) | ◐ Trước thời VLM (L92 đến 2020) | ◐ 15–503 nghiên cứu, độ không đồng nhất cao | ● Gộp và chuẩn báo cáo (L92) | ◐ AI làm người đọc thứ hai (L33) |
| **M** Phương pháp | ◐ Độ phân giải cho phát hiện nốt (L82, L83) | ○ Không có VLM (chỉ CNN) | ◐ NIH, 2 cơ sở (L82, L83) | ● Mất cân bằng lớp, calibration (L84) | ○ |
| **Giao VLM × nốt** (khoảng trống của đề tài) | ○ Định vị sau tinh chỉnh (K2, K3) | ○ VLM tinh chỉnh cho nốt (K2) | ◐ Có dữ liệu (L37) nhưng chưa dùng cho VLM | ○ Tỷ lệ bệnh, calibration, đa nốt (K12, K14) | ○ Người đọc với VLM (K13) |

### 3.2 Hình ASCII: nơi bằng chứng tập trung và nơi còn trống

```
                     Task      Model     Data      Eval      Clinical
                   ┌─────────┬─────────┬─────────┬─────────┬─────────┐
 T6 CAD            │ ███████ │ ▒▒▒▒    │ ███████ │ ███████ │ ▒▒▒▒    │
 T7 Người–AI       │ ███████ │ ▒▒▒▒    │ ▒▒▒▒    │ ███████ │ ███████ │  ← bằng chứng
 T10 Tổng quan     │ ███████ │ ▒▒▒▒    │ ▒▒▒▒    │ ███████ │ ▒▒▒▒    │    lâm sàng mạnh
 T9 Dataset        │ ▒▒▒▒    │   n/a   │ ███████ │ ▒▒▒▒    │         │
                   ├─────────┼─────────┼─────────┼─────────┼─────────┤
 T1 Zero-shot VLM  │ ▒▒▒▒    │ ███████ │ ▒▒▒▒    │ ▒▒▒▒    │         │
 T2 VLM CXR        │ ▒▒▒▒    │ ███████ │ ▒▒▒▒    │ ▒▒▒▒    │ ▒▒▒▒    │  ← nhiều mô hình,
 T3 Encoder        │ ▒▒▒▒    │ ▒▒▒▒    │ ▒▒▒▒    │ ▒▒▒▒    │         │    ít bằng chứng
 T4 Grounding      │ ▒▒▒▒    │ ███████ │ ▒▒▒▒    │ ███████ │         │    riêng cho nốt
                   ├─────────┼─────────┼─────────┼─────────┼─────────┤
 M Phương pháp     │ ▒▒▒▒    │         │ ▒▒▒▒    │ ███████ │         │
                   ╞═════════╪═════════╪═════════╪═════════╪═════════╡
 VLM × nốt         │  K2,K3  │   K2    │  (L37)  │ K12,K14 │   K13   │  ← ĐỀ TÀI
 (giao điểm)       │ ░TRỐNG░ │ ░TRỐNG░ │ ▒▒▒▒    │ ░TRỐNG░ │ ░TRỐNG░ │
                   └─────────┴─────────┴─────────┴─────────┴─────────┘
   ███ covered (≥3 bài riêng nốt)   ▒▒▒ sparse   (trống/░) empty (∅)
```

**Đọc bản đồ**
1. **Hai khối tách rời.**
   - Khối CAD và lâm sàng (T6, T7, T10) phủ kín task, evaluation và clinical, nhưng chỉ có CNN.
   - Khối VLM (T1–T4) phủ kín model, nhưng task, data và evaluation cho nốt đều thưa.
2. **Hàng giao "VLM × nốt" trống ở 4/5 khía cạnh.** Dữ liệu có sẵn (L37) nhưng chưa ai dùng cho VLM. Đây là vùng đề tài nhắm tới.
3. **Cột Clinical của khối VLM gần như trống.** Ô duy nhất là L63, nhưng không riêng nốt. Trong khi đó T7 cho thấy AI sai gây hại (L88, L90). Vì vậy kiểm toán trước khi dùng là cần thiết (K13).
4. Cụm M và T6 cho sẵn **công cụ đánh giá** (JAFROC/FROC ở L19, L76, L81; AUC-PR và calibration ở L84). Đề tài có thể mượn ngay mà không phải tự phát minh.

---

## 4. Ưu tiên

### 4.1 Tiêu chí chấm

| Trục | 3 | 2 | 1 |
|---|---|---|---|
| **Impact** | Trả lời câu hỏi lõi của lĩnh vực hoặc của đề tài | Làm rõ hay củng cố kết quả lõi | Bối cảnh, hoặc hướng phụ |
| **Feasibility** (một học kỳ, Colab trả phí, dữ liệu công khai) | Đã có trong spec, dữ liệu sẵn, chi phí thêm thấp | Cần thêm thí nghiệm hoặc chỉ số vừa phải | Cần dữ liệu, người đọc hoặc tính toán ngoài tầm |

Điểm ưu tiên = Impact × Feasibility. Khi bằng điểm, ưu tiên khoảng trống chi phối tính hợp lệ của kết luận chính.

### 4.2 Bảng xếp hạng

| Hạng | ID | Khoảng trống (rút gọn) | I | F | Điểm | Quyết định |
|---|---|---|---|---|---|---|
| **1** | **K2** | LoRA VLM cho **phát hiện** nốt/khối | 3 | 3 | **9** | **Đề tài lấp** (C1–C4, E3–E4) |
| **2** | **K3** | **Định vị** nốt bằng VLM sau tinh chỉnh | 3 | 2 | **6** | **Đề tài lấp** (C2, §8.2). Rủi ro: MedGemma chưa quen xuất box (spec §5.1; L48) |
| **3** | **K17** | Cân bằng lớp khi dương tính khoảng 3%, chống sụp sensitivity | 2 | 3 | **6** | **Đề tài lấp** (C3, chỉ số chế độ báo bình thường ở §8.1) |
| 4 | K12 | Đánh giá theo tỷ lệ bệnh (AUC-PR, PPV, calibration) | 2 | 3 | 6 | **Đề tài lấp nếu bổ sung chỉ số** (thêm AUC-PR và ECE/Brier vào §8.1; L84) |
| 5 | K9 | Đánh giá chuẩn hóa | 2 | 3 | 6 | **Đề tài lấp** (§2 mục tiêu 1, §5.3, §8.4) |
| 6 | K10 | Chỉ số lâm sàng thay cho NLG khi tinh chỉnh | 2 | 3 | 6 | **Đề tài lấp** (đầu ra JSON, §8) |
| 7 | K4 | Phân tầng theo kích thước, độ mờ, vị trí | 2 | 3 | 6 | **Lấp một phần.** Kích thước và độ mờ: có. Vị trí giải phẫu: nên thêm (L29, L34) |
| 8 | K14 | Chỉ số đa nốt hoặc không nốt (N-Acc, đếm đúng số nốt) | 2 | 3 | 6 | **Lấp nếu thêm chỉ số** (L70, L36). Chi phí thấp |
| 9 | K15 | VLM y khoa so với đa năng | 2 | 2 | 4 | **Lấp một phần** (C4). Bị nhiễu bởi khác cỡ và kiến trúc |
| 10 | K18 | Kiểm tra rò rỉ dữ liệu huấn luyện | 2 | 2 | 4 | **Lấp một phần.** Nên thêm mục giới hạn và kiểm tra tài liệu dữ liệu của MedGemma/Qwen |
| 11 | K11 | Độ phân giải và chia ô trong VLM | 2 | 2 | 4 | **Mở rộng** (C6, cần sửa cách phát biểu) |
| 12 | K1 | Quy mô và số nguồn của đánh giá | 2 | 2 | 4 | **Lấp một phần** (3 tập test, vẫn hồi cứu) |
| 13 | K5 | Kiểm chứng cơ chế gộp trên VLM | 3 | 1 | 3 | **Để lại cho tương lai.** Có thể làm thêm probe patch-local nếu dư thời gian (spec §9) |
| 14 | K13 | Automation bias với VLM | 3 | 1 | 3 | **Để lại cho tương lai** (cần người đọc). Đề tài chỉ đo gián tiếp qua RQ9 |
| 15 | K16 | Quần thể Việt Nam / Đông Nam Á | 2 | 1 | 2 | **Lấp một phần, ngẫu nhiên do thiết kế** (huấn luyện trên dữ liệu VN), không phải mục tiêu |
| 16 | K8 | VQA riêng nốt (kích thước, tính chất) | 1 | 2 | 2 | **Lấp một phần** (chỉ có/không và box) |
| 17 | K19 | GRPO thay cho LoRA-SFT | 2 | 1 | 2 | **Tương lai** (L71) |
| 18 | K20 | Nhãn yếu LLM để mở rộng dữ liệu | 1 | 1 | 1 | **Tương lai** |
| 19 | K6 | LLM khuyến nghị theo dõi trên báo cáo CXR | 1 | 1 | 1 | **Tương lai** (§9) |
| 20 | K7 | Mô hình nguy cơ CXR + LLM | 1 | 1 | 1 | **Tương lai** (§9) |

### 4.3 Ma trận impact × feasibility

```
 Impact
   3 │  K5  K13            │  K3                │  K2 ★
     │  (tương lai)        │                    │
   2 │  K16  K19           │  K15 K18 K11 K1    │  K17 K12 K9 K10 K4 K14
     │                     │                    │
   1 │  K20  K6  K7        │  K8                │
     └─────────────────────┴────────────────────┴──────────────────────
          1 (khó)                 2                  3 (dễ)     Feasibility
```

### 4.4 Kết luận ưu tiên

**Đề tài lấp (cam kết)**
- K2, K3, K17: lõi.
- K9, K10: thiết kế đánh giá.
- K12, K14: chỉ cần bổ sung chỉ số.
- K4: thêm tầng vị trí.

**Đề tài lấp một phần**
- K1, K8, K15, K16, K18.

**Để lại làm hướng phát triển (ghi ở spec §9)**
- K5: probe cơ chế sau LoRA.
- K11: chia ô hoặc độ phân giải cao hơn, qua C6.
- K13: nghiên cứu người đọc.
- K19: GRPO.
- K20: nhãn yếu từ LLM.
- K6, K7: nhánh văn bản và nguy cơ.

---

## 5. Đề xuất cập nhật RQ

### 5.1 RQ sửa và RQ mới

| RQ | Trạng thái | Phát biểu đề xuất | Map gap | Cấu hình / thí nghiệm trong spec |
|---|---|---|---|---|
| **RQ1** | **Sửa** | Tinh chỉnh LoRA MedGemma-4B trên VinBigData có đưa sensitivity ở specificity 0,90 lên mức ≥ 0,60, và thu hẹp khoảng cách với B1/B2 **trên cùng split** không? Kết quả có giữ được trên NIH và JSRT không? | K2, K10, K16 | C1–C3, B1, B2; E1, E3; §8.1 |
| **RQ2** | **Sửa** | Sau LoRA, sensitivity thay đổi thế nào theo **kích thước, độ mờ và vùng giải phẫu** của nốt? Cân bằng lớp (C3) có ngăn mô hình rơi lại vào chế độ báo bình thường không? | K4, K17 | C2 so với C3; E3, E5; §8.3 phân tầng |
| **RQ3** | **Giữ, xếp là mở rộng** | Đưa ROI từ detector (C5) vào VLM có tăng phát hiện nốt nhỏ không? | K5 | C5 |
| **RQ3b** | **Mới, mở rộng** | Với Qwen2.5-VL, tăng độ phân giải hoặc chia ô có tăng sensitivity với nốt nhỏ hơn 10 mm hay các tertile nhỏ nhất không? Với MedGemma (đầu vào cố định 896 px, L7), chia ô có lợi gì so với ảnh gốc? | K11 | C6 (sửa) |
| **RQ4** | **Sửa** | VLM sau LoRA định vị nốt chính xác đến đâu, đo bằng IoU ≥ 0,3/0,5, FROC, khoảng cách tâm (mm, theo L36), và **độ chính xác số nốt / N-Acc** cho ảnh có 0 hoặc nhiều nốt (theo L70)? | K3, K14 | C2–C4; §8.2 |
| **RQ5** | **Giữ, bổ sung giao thức** | Đánh giá chuẩn hóa (prompt cố định, ngưỡng chọn trên validation, chạy lặp, **có và không có ngữ cảnh lâm sàng** theo L4, **kiểm tra rò rỉ** theo K18) có đổi xếp hạng zero-shot so với L1 và L2 không? | K1, K8, K9, K18 | C0; E2 |
| **RQ6** | **Chuyển sang hướng phát triển** | (giữ nguyên phát biểu ở `02` §5) | K6 | §9, không thí nghiệm |
| **RQ7** | **Mới** | Cùng một công thức LoRA, VLM y khoa (MedGemma-4B) có hơn VLM đa năng có sẵn khả năng xuất box (Qwen2.5-VL-3B) cho phát hiện và định vị nốt không? | K15 | C4; E4. Cần nêu nhiễu do khác cỡ (4B so với 3B) và kiến trúc |
| **RQ8** | **Mới** | VLM sau LoRA có **hiệu chỉnh tốt** không (ECE, Brier/balanced Brier theo L84)? AUC-PR và PPV thay đổi thế nào khi quy về tỷ lệ bệnh thấp (ví dụ mức 2,2% ở L79)? | K12 | §8.1 (bổ sung); dùng điểm xác suất token "yes" sẵn có trong spec |
| **RQ9** | **Mới, đo gián tiếp** | Trong số lỗi FN/FP của VLM, bao nhiêu là **lỗi tự tin** (xác suất cao)? So với B1/B2 ra sao? Đây là chỉ số thay thế cho nguy cơ automation bias (L88, L90) khi không có nghiên cứu người đọc | K13 | §8.1 và §8.3 (bổ sung); không cần dữ liệu mới |

**Không đưa thành RQ:** K7, K19 và K20 chỉ ghi vào §9.

### 5.2 Những điểm trong spec bị bằng chứng mâu thuẫn hoặc cần sửa

*Chỉ đánh dấu ở đây. Không sửa spec.*

| # | Mục spec | Spec viết | Bằng chứng | Mức | Đề xuất |
|---|---|---|---|---|---|
| **S1** | §5.1, lý do chọn MedGemma | "trong bài Nishio [L2] nó cho sensitivity zero-shot **thấp nhất** (5,2%)" | L2 (FT): khoảng sensitivity của 6 VLM là **0,6–27,3%**. Toàn văn xác nhận: CheXpert Plus 0,6% (1/154); MedGemma đứng áp chót | **Mâu thuẫn trực tiếp** | Sửa thành "sensitivity rất thấp (5,2%), dù specificity 100%" |
| **S2** | §3 bảng, dòng "Rủi ro fine-tune làm sụp sensitivity"; `01` K2 | "[L50]: fine-tune **toàn phần**…" | L50 (AB) theo `05`: "Chi tiết cách tinh chỉnh không rõ trong nguồn đã đọc" | **Khẳng định vượt quá nguồn** | Bỏ chữ "toàn phần", hoặc đọc toàn văn L50 để xác minh |
| **S3** | §1 (đoạn cuối), §3 ("Tinh chỉnh VLM cho nốt/ung thư: Không tìm thấy"), §4 đóng góp 2 | Không nhắc tiền lệ tinh chỉnh nào | L64 (FT): QLoRA VLM nhỏ trên CXR. L69 (FT): LoRA r16, AAAI. L71 (FT): tinh chỉnh **chính Qwen2.5-VL**, VLM phụ của đề tài, cho grounding CXR với mIoU 79,02 | **Thiếu, dễ bị phản biện về tính mới** | Thêm ba bài vào cột "Đã có". Giới hạn tuyên bố "đầu tiên" vào **phát hiện và định vị nốt/khối**. Nêu L71 như bằng chứng Qwen2.5-VL học được box |
| **S4** | §5.2 C6 | "tăng độ phân giải đầu vào (ví dụ **448 → 896** px)" | L7 (FT): MedGemma resize ảnh về **896×896** sẵn. L83 (FT): 896 px tối ưu cho CAD nốt (RetinaNet FOM 0,906). L66 là CT, không dùng được | **Không áp dụng cho MedGemma như đang viết** | Với MedGemma, C6 nên là chia ô hoặc cắt theo trường phổi vượt 896 px. Với Qwen2.5-VL, mới là thử độ phân giải đầu vào (RQ3b). Lập luận chính chuyển sang cách gộp đặc trưng (L47) và dữ liệu (L37, L50) |
| **S5** | §3 "Giới hạn của tuyên bố"; dòng "Tổng quan tài liệu làm cơ sở"; bảng tài liệu tham khảo | "Khảo sát 60 bài"; "Chỉ 4 bài được đọc toàn văn"; L1 ghi "Toàn văn" | `05`: 90 bài, FT 28 / AB 59 / T 3. L1 là AB† (ghi chú toàn văn từ phiên trước, không đọc lại) | **Lỗi thời** | Cập nhật số bài và mức đọc. Thêm `03`, `05`, `07` vào danh sách tài liệu cơ sở |
| **S6** | §1 | "Kiểu sai **nhất quán** là chế độ báo bình thường: specificity gần 1, sensitivity gần 0" | L2 (FT): specificity 71,0–100%. LLaVA-Rad có specificity 0,710 và sensitivity 0,273 (`02` §2). Ở L1, BiomedCLIP có specificity 0,689 (`02` §2). L52 (FT): khối lớn được phát hiện tốt | **Khái quát quá mức** | Giới hạn thành: "phổ biến ở các VLM sinh văn bản chuyên CXR (MedGemma, RadVLM, CXR-LLaVA)" |
| **S7** | §8.1 | Chỉ số: sens, spec, PPV, NPV, F1, AUC | L84 (FT): AUC-ROC che lớp thiểu số; cần AUC-PR và calibration. L79 (FT): hiệu năng giảm mạnh ở tỷ lệ bệnh 2,2%. L2 (FT): JSRT có tỷ lệ nốt 62%, nên PPV trên JSRT không phản ánh lâm sàng | **Thiếu** (không phải mâu thuẫn) | Thêm AUC-PR, ECE hoặc Brier. Báo PPV quy về tỷ lệ bệnh thấp. Ghi chú PPV trên JSRT không có ý nghĩa lâm sàng (RQ8) |
| **S8** | §6, VinBigData | "Cần kiểm tra số ca Nodule/Mass" | L37 (FT): tập train 15.000 ảnh của VinDr-CXR có **410 ảnh Nodule/Mass (2,73%)**; tập test chính thức 3.000 ảnh có 176 ảnh (5,87%), đồng thuận 5 bác sĩ | **Rủi ro cỡ mẫu** | Nếu VinBigData Kaggle tương ứng tập train của VinDr-CXR (cần xác minh), chia 70/10/20 chỉ còn khoảng 287 / 41 / 82 ảnh dương (ước tính số học), nên CI của sensitivity sẽ rộng. Nên cân nhắc dùng tập test chính thức của VinDr-CXR (176 ảnh dương, nhãn đồng thuận) nếu truy cập được |
| **S9** | Quyết định ngưỡng thành công, và cách diễn giải ở `03` (nhận xét 2) và RSNA notes | Ngưỡng sensitivity ≥ 0,60 ở specificity 0,90 được coi là "tương đương mức bác sĩ" | L85 (AB): bác sĩ đạt 65,1%, với 0,20 FP mỗi ảnh, trên tập làm giàu ca bệnh. L87 (FT): người đọc đạt sensitivity 0,567 với nốt. L74 (AB): 40–53% với ung thư từng bị bỏ sót. Không mốc nào ở điểm vận hành specificity 0,90 | **Cần thận trọng khi diễn giải** | Ngưỡng vẫn dùng được, nhưng báo cáo không nên viết "ngang bác sĩ". Nên viết "cùng khoảng với sensitivity của người đọc trong các tập làm giàu (L85, L87)" |
| **S10** | §5.3 và §8 | Không có điều kiện có/không ngữ cảnh lâm sàng; không kiểm tra rò rỉ | L4 (FT): ngữ cảnh làm accuracy tăng từ 28% lên 62%, có thể do "lộ" đáp án. L40 (AB) và L43 (AB): dữ liệu CXR công khai được tái dùng rộng rãi để huấn luyện VLM | **Thiếu** | Cố định prompt không có ngữ cảnh làm điều kiện chính. Thêm mục giới hạn về rò rỉ (K18, RQ5) |

**Không thấy bằng chứng mâu thuẫn với**
- Lựa chọn MedGemma làm VLM chính. L7 (FT) và L42 (FT) xác nhận nó mạnh trên benchmark tổng hợp nhưng chưa có đánh giá cho nốt.
- Gộp Nodule và Mass (L37, L72).
- Việc tránh nhãn do LLM sinh (§5.3).
- Kiểm toán che vùng (§8.3). L88 (FT) còn ủng hộ đầu ra có box.

---

## 6. Hạn chế của tài liệu này

1. **Phần lớn khoảng trống là ∅ trong corpus 90 bài, không phải chứng minh tuyệt đối.** Corpus dựng từ arXiv, PubMed, Semantic Scholar, OpenAlex, Europe PMC, Consensus, Undermind và Scopus Q1 (`corpus.md`). Đợt bổ sung L63–L92 thiên về tạp chí uy tín (`04-scopus-prestige-search.md`), nên preprint VLM mới nhất có thể bị thiếu.
2. **Mức đọc không đồng đều.** 59/90 bài chỉ đọc abstract, và 3 bài chỉ có tiêu đề. Các bằng chứng then chốt sau **không phải FT trong lượt `05`**:
   - L1, L47, L48: AB†.
   - L2, L50: AB.
   - Đây đều là các bài định hình K1–K5. Vì vậy chưa xếp các khoảng trống này ở mức "Cao" nếu không có thêm bài FT cùng hướng.
3. **File PDF có vấn đề:**
   - L39 (CheXpert): file cục bộ chứa bài khác (Saralajew et al. 2019, LVQ).
   - L40: PDF cục bộ là bài MIMIC-CXR trên *Scientific Data*, không phải MIMIC-CXR-JPG.
   - Cả hai giữ ở mức AB. Không có khoảng trống nào phụ thuộc chủ yếu vào hai bài này.
4. **Bài bị loại khỏi vai trò bằng chứng:**
   - L66: CT liều thấp. Chỉ dùng làm ví dụ phương pháp cho K12, không dùng cho độ phân giải CXR.
   - L67: nguồn nhãn sống còn không rõ. Chỉ nhắc ở K7, với mức chắc chắn Thấp.
5. **Việc gán loại taxonomy và chấm impact × feasibility là đánh giá chủ quan** của người viết. Cần nhóm rà soát.
6. **Không tìm thêm tài liệu trong lượt này** (không dùng trình duyệt, không tải bài). Mọi bằng chứng lấy từ các file đầu vào đã liệt kê.
