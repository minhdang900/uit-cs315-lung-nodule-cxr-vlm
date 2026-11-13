# Đề cương đồ án cuối kỳ CS315: Machine Learning nâng cao

**Ngày lập:** 2026-11-12
**Trạng thái:** v2, nhóm đã duyệt ngày 2026-11-12. Ngày 2026-11-12 đã sửa theo các điểm S1–S10 trong `07-research-gaps.md` §5, sau khi tổng quan mở rộng lên 90 bài. Ngày 2026-11-13 đã triển khai pipeline (repo `code/`); các điểm làm rõ phương pháp khi triển khai ghi ở §8.5.
**Hiệu lực:** đây là spec duy nhất còn hiệu lực (đề cương đề tài cũ ngày 2026-11-10 đã hủy).
**Phạm vi:** bản gọn: hai VLM mở cỡ nhỏ, hai baseline CNN, ngân sách Colab trả phí
**Quyết định của nhóm (2026-11-12):**
- VLM chính là MedGemma-4B.
- C5 chỉ là phần mở rộng.
- Ca dương gộp Nodule/Mass.
- Ngưỡng thành công: sensitivity ≥ 0,60 ở specificity 0,90.
**Tổng quan tài liệu làm cơ sở:**
- `docs/lit-review/lung-cancer-cxr/00-review-report.md`
- `01-literature-review.md`
- `02-literature-synthesis.md`
- `corpus.md` (L1–L92, 90 bài duy nhất)
- `03-thematic-clusters.md` (11 cụm)
- `05-detailed-analysis.md` (phân tích từng bài, có ghi mức đọc)
- `06-systematic-review-report.md`
- `07-research-gaps.md` (khoảng trống K1–K20, các điểm sửa spec S1–S10)

## Tên đề tài

**Trợ lý AI phát hiện và định vị nốt phổi trên X-quang ngực: tinh chỉnh LoRA mô hình ngôn ngữ–thị giác cỡ nhỏ**

Tên tiếng Anh: *An AI Assistant for Lung Nodule Detection and Localization on Chest X-rays via LoRA-Tuned Small Vision-Language Models*

Ghi chú (2026-11-13): nhóm đổi sang tên mang tính ứng dụng (phương án 1). Nội dung, phạm vi và tiêu chí thành công không đổi. Tên cũ: *Can Vision-Language Models See Lung Nodules? Auditing and LoRA-Adapting Small VLMs for Lung Nodule Detection and Localization on Chest X-rays*. "Trợ lý" ở đây là nguyên mẫu nghiên cứu, không phải công cụ lâm sàng (xem §9).

### Tóm tắt (bản đề cương, chưa có kết quả)

X-quang ngực (CXR) là xét nghiệm hình ảnh phổ biến nhất; nhiều ca ung thư phổi lần đầu lộ ra dưới dạng nốt/khối trên CXR, nhưng nốt nhỏ và mờ dễ bị bỏ sót. VLM đang được giới thiệu như trợ lý đọc CXR, song đánh giá độc lập năm 2026 cho thấy sensitivity zero-shot với nốt phổi chỉ 0,6–27,3% [L2]; MedGemma-4B đạt 5,2% ở specificity 100% (chế độ báo bình thường), trong khi CAD CNN đạt AUROC 0,92–0,99 [L19]. Đồ án xây dựng và đánh giá một nguyên mẫu trợ lý AI cho biết ảnh có nốt/khối hay không và chỉ ra vị trí bằng bounding box, dựa trên VLM cỡ nhỏ tinh chỉnh LoRA, gồm ba phần: (i) benchmark chuẩn hóa MedGemma-4B và Qwen2.5-VL-3B so với DenseNet-121 và YOLOv8 trên VinBigData, NIH ChestX-ray14 và JSRT; (ii) tinh chỉnh LoRA MedGemma-4B cho phân loại và phân loại kèm định vị, có/không cân bằng lớp; (iii) kiểm tra trợ lý có nhìn đúng chỗ không bằng IoU, phép che nốt so với che vùng ngẫu nhiên cùng diện tích, phân tầng theo kích thước và độ mờ. Chỉ số chính: sensitivity ở specificity 0,90 (mục tiêu 0,60); chỉ số phụ: AUC-ROC, AUC-PR, calibration, FROC, CI 95% bootstrap. Kết quả và kết luận: [điền sau E1–E5]. Bản LaTeX: `report/chapters/00_tom_tat.tex`.

---

## 1. Vấn đề

Ung thư phổi là nguyên nhân tử vong do ung thư hàng đầu. X-quang ngực (CXR) là xét nghiệm hình ảnh phổ biến nhất, và nhiều ca ung thư phổi lần đầu lộ ra dưới dạng nốt hoặc khối trên CXR. Các mô hình ngôn ngữ–thị giác (VLM) và LLM đa phương thức đang được giới thiệu như trợ lý đọc CXR. Tuy vậy, các đánh giá độc lập năm 2026 cho thấy chúng gần như không thấy nốt phổi:

- Sáu VLM zero-shot trên 247 ảnh JSRT chỉ đạt sensitivity **0,6–27,3%** [L2]:
  - MedGemma-4b-it đạt sensitivity 5,2% (8/154) với specificity 100%.
  - Theo độ mờ, MedGemma phát hiện 25% nốt grade 5 (rõ) và 0% nốt grade 1. Ở grade 1, mọi mô hình đều ≤ 12%.
- Chín hệ thống trên MosMedData VII (83 ảnh, nốt đặc 6–30 mm có CT xác nhận):
  - AUC tốt nhất 0,743 (BiomedCLIP).
  - MLLM đa năng tốt nhất là Claude 3.7 Sonnet, AUC 0,625, sensitivity 0,316.
  - CXR-LLaVA có sensitivity 0,079 [L1].
- Bốn MLLM đa năng chỉ phát hiện khoảng 10% ca ác tính đơn ổ, nhưng khoảng 55% ca đa ổ [L3]. Khi thêm ngữ cảnh có cấu trúc, accuracy chung tăng, nhưng ung thư phổi vẫn chỉ đúng 20% [L4].
- CAD CNN thương mại trên cùng nguồn MosMedData VII đạt AUC **0,956** [L45]. Trên các tập ngoài, CAD chuyên biệt đạt AUROC 0,92–0,99 [L19]. Đã có RCT cho thấy CAD tăng tỷ lệ phát hiện nốt cần xử trí từ 0,25% lên 0,59% [L20].

Ở các VLM sinh văn bản chuyên CXR (MedGemma, RadVLM, CXR-LLaVA), kiểu sai phổ biến là **chế độ báo bình thường**: specificity gần 1, sensitivity gần 0 [L1, L2]. Kiểu sai này không phổ quát. LLaVA-Rad đổi lấy sensitivity 27,3% bằng specificity 71,0% [L2], còn khối lớn trên 3 cm được phát hiện tốt [L52]. Có hai giả thuyết bổ trợ nhau:
- **Thị giác:** bộ mã hóa ViT làm mất tín hiệu tổn thương nhỏ ở bước gộp toàn cục. Probe tuyến tính lấy lại được tín hiệu từ patch cục bộ: AUC nốt của MedSigLIP tăng từ 0,805 lên 0,955 [L47].
- **Dữ liệu và ngôn ngữ:** báo cáo huấn luyện phần lớn là bình thường, nên mô hình nghiêng về câu trả lời "không có bất thường". Giả thuyết này là suy luận, chưa được kiểm chứng riêng cho nốt.

Đã có tiền lệ tinh chỉnh trên CXR, nhưng đều cho tác vụ khác:
- QLoRA một VLM nhỏ để sinh báo cáo, chỉ đo bằng chỉ số văn bản [L64].
- LoRA rank 16 cho LLM sinh báo cáo (AAAI) [L69].
- Tinh chỉnh Qwen2.5-VL cho grounding cụm từ tổng quát trên MS-CXR, đạt mIoU 79,02 [L71].

Trong phạm vi 90 bài đã khảo sát, chưa có nghiên cứu nào tinh chỉnh VLM cỡ nhỏ để **phát hiện và định vị nốt/khối** trên CXR rồi so với CNN trên cùng split. Cũng chưa có nghiên cứu nào kiểm tra mô hình sau tinh chỉnh có thật sự nhìn đúng vị trí nốt hay không.

## 2. Mục tiêu

1. **Kiểm toán zero-shot có chuẩn hóa:** đánh giá 2–3 VLM mở cỡ nhỏ trên các tập nốt CXR công khai. Dùng chung prompt, đầu ra có cấu trúc, chọn ngưỡng trên tập validation, chạy lặp, rồi đặt cạnh baseline CNN trên cùng split.
2. **Tinh chỉnh LoRA** VLM cho hai bài toán: (a) có hay không có nốt/khối; (b) có nốt/khối kèm bounding box. Đo mức thu hẹp khoảng cách sensitivity so với CNN.
3. **Chống chế độ báo bình thường:** so sánh LoRA chuẩn với LoRA cân bằng lớp (oversampling ca dương tính và trọng số loss).
4. **Kiểm tra mô hình có nhìn đúng chỗ không:** định vị (IoU, hit rate) và phép thử che nốt so với che vùng ngẫu nhiên cùng diện tích. Phân tích phân tầng theo kích thước và độ mờ của nốt.

**Tiêu chí thành công.**
- Có bảng so sánh 7 cấu hình × 3 tập test (mục 7–8), kèm khoảng tin cậy 95%.
- Kết luận được, dù theo hướng nào, về hai câu hỏi:
  - LoRA có đưa sensitivity của VLM (ở specificity 0,90) lên tối thiểu 0,60 trên tập test trong phân phối không?
  - Cải thiện có còn khi kiểm tra ngoài phân phối (JSRT) và khi che nốt không?
- **Cách diễn giải ngưỡng 0,60.** Ngưỡng này nằm cùng khoảng với sensitivity của người đọc trên các tập làm giàu ca bệnh: 65,1% ở [L85], 0,567 ở [L87], và 40–53% với ung thư từng bị bỏ sót ở [L74]. Tuy vậy, không mốc nào đo ở điểm vận hành specificity 0,90. Vì vậy báo cáo sẽ **không** viết "ngang bác sĩ". Ngưỡng này cũng thấp hơn rõ so với CAD (AUC 0,86–0,99 [L80, L75]).

## 3. Khoảng trống nghiên cứu

Khảo sát 90 bài (`corpus.md`) từ:
- arXiv, PubMed, Semantic Scholar, OpenAlex, Europe PMC, Consensus;
- Undermind deep search;
- Scopus, giới hạn ở các tạp chí uy tín (`04`).

Chi tiết khoảng trống K1–K20 nằm trong `07-research-gaps.md`.

| Việc | Đã có | Còn trống |
|---|---|---|
| Đánh giá VLM zero-shot cho nốt CXR | [L1], [L2]: 83–247 ảnh, một nguồn, prompt khác nhau. [L1] chọn ngưỡng Youden trên tập test; [L2] chỉ có đầu ra nhị phân, không có ROC | Đánh giá chuẩn hóa, nhiều tập, có CNN trên cùng split |
| Tinh chỉnh VLM/LLM trên CXR | Có, nhưng cho tác vụ khác:<br>– [L64]: QLoRA VLM nhỏ, sinh báo cáo, chỉ số văn bản<br>– [L69]: LoRA r16, sinh báo cáo, AAAI<br>– [L71]: tinh chỉnh Qwen2.5-VL, grounding tổng quát | **LoRA VLM nhỏ cho phát hiện và định vị nốt/khối, báo sens/spec theo nốt, so với CNN** |
| Định vị nốt bằng VLM | – [L48]: gộp chung thành "lung lesion" (GPT-5 35,7%, MedGemma 14,3%)<br>– [L49]: không có nốt/khối<br>– [L69]–[L71]: grounding không riêng nốt | Định vị riêng cho nốt/khối, sau tinh chỉnh |
| Phân tầng theo kích thước và độ mờ | – [L2]: có theo grade JSRT, nhưng chỉ zero-shot<br>– [L3]: đơn ổ và đa ổ<br>– CAD: [L73], [L83] | Phân tầng có hệ thống, trước và sau tinh chỉnh |
| Cơ chế bỏ sót nốt nhỏ | – [L47]: probe trên embedding đông cứng, có oracle ROI<br>– [L82], [L83]: ảnh hưởng của độ phân giải với CAD | Kiểm chứng trên VLM sinh văn bản và sau tinh chỉnh |
| Rủi ro tinh chỉnh làm sụp sensitivity | [L50]: tinh chỉnh trên ChestX-ray14 làm sensitivity giảm dưới 15%. Cách tinh chỉnh chưa rõ trong nguồn đã đọc | LoRA kết hợp cân bằng lớp, theo dõi sensitivity |
| Đánh giá có tính đến tỷ lệ bệnh | Các nghiên cứu VLM chỉ báo accuracy và sens/spec. [L84] khuyến nghị AUC-PR và calibration khi lớp dương hiếm | AUC-PR, calibration và PPV ở tỷ lệ bệnh thực tế |

**Giới hạn của tuyên bố.**
- Các khoảng trống trên đúng trong phạm vi 90 bài đã khảo sát. Đó không phải bằng chứng tuyệt đối rằng chưa ai làm.
- Mức đọc: 34 bài toàn văn, 53 bài tóm tắt, 3 bài chỉ tiêu đề (`05`).
- Cụm T1 đã được đọc toàn văn, trừ L62 và L53. L53 vốn chỉ là tóm tắt hội nghị.

## 4. Đóng góp

1. **Benchmark nốt CXR chuẩn hóa cho VLM:** prompt và parser công khai, chọn ngưỡng trên validation, ba tập test (trong phân phối, ngoài phân phối, có độ mờ). Đây là bản tái lập có kiểm soát và quy mô lớn hơn của [L1], [L2].
2. **Bằng chứng đầu tiên, trong phạm vi khảo sát, về LoRA VLM cỡ nhỏ cho phát hiện và định vị nốt/khối phổi trên CXR**, so trực tiếp với CNN classifier và detector trên cùng split. Các tiền lệ LoRA hoặc tinh chỉnh trên CXR [L64, L69, L71] chỉ làm sinh báo cáo hoặc grounding tổng quát.
3. **Phân tích chế độ báo bình thường và tác dụng của cân bằng lớp** khi tinh chỉnh VLM cho tổn thương hiếm và nhỏ.
4. **Kiểm toán "đúng vì đúng lý do":** kết hợp IoU và phép che nốt so với che vùng ngẫu nhiên cùng diện tích, phân tầng theo kích thước và độ mờ của nốt.

## 5. Phương pháp

### 5.1 Mô hình

- **VLM chính (nhóm chọn ngày 2026-11-12):** MedGemma-4B-it, mô hình y khoa, bộ mã hóa MedSigLIP.
  - LoRA r = 16, α = 32, dropout 0,05, gắn vào các lớp attention và MLP của LLM; bộ mã hóa ảnh đông cứng.
  - Lý do chọn:
    - Đây là mô hình y khoa mở mới nhất [L7].
    - Trong bài Nishio [L2], nó cho sensitivity zero-shot rất thấp (5,2%, đứng áp chót sau CheXpert Plus 0,6%) dù specificity 100%. Đây là ví dụ điển hình của chế độ báo bình thường, nên câu hỏi "LoRA có kéo được mô hình này ra khỏi chế độ đó không" có tính thông tin cao.
    - MedGemma nhận ảnh 896×896 [L7]. Đây cũng là độ phân giải tối ưu cho CAD nốt theo [L83], nên nút thắt nhiều khả năng nằm ở cách gộp đặc trưng [L47] và dữ liệu, chứ không ở độ phân giải đầu vào.
  - Rủi ro: MedGemma không được huấn luyện mạnh cho xuất tọa độ box, nên khả năng định vị phải học hoàn toàn qua LoRA. Phải kiểm tra ở bước chạy thử.
  - Cần chấp nhận điều khoản Health AI Developer Foundations trên Hugging Face trước khi tải.
- **VLM thứ hai:** Qwen2.5-VL-3B-Instruct, mô hình đa năng, có sẵn khả năng xuất tọa độ box.
  - Dùng cho zero-shot và một cấu hình LoRA tốt nhất, để so mô hình y khoa với mô hình đa năng.
  - [L71] đã cho thấy Qwen2.5-VL học được box trên CXR sau khi tinh chỉnh.
  - Lưu ý: [L2] thử Qwen3-VL-8B (sensitivity 7,8%), không phải Qwen2.5-VL-3B. Khi so sánh với [L2] phải ghi rõ.
- **VLM chỉ chạy zero-shot (nếu còn thời gian):** LLaVA-Rad hoặc CheXagent, để so với [L2].
- **Baseline CNN:**
  - **B1, classifier:** DenseNet-121 tiền huấn luyện trên CXR (`torchxrayvision`), fine-tune nhị phân nốt/khối.
  - **B2, detector:** YOLOv8 (hoặc Faster R-CNN) huấn luyện trên box nốt/khối.

### 5.2 Các cấu hình

| Mã | Cấu hình | Đầu ra |
|---|---|---|
| C0 | VLM zero-shot (MedGemma-4B, Qwen2.5-VL-3B) | JSON `{"nodule": yes/no, "boxes": [...]}` |
| C1 | LoRA MedGemma-4B, chỉ phân loại | yes/no |
| C2 | LoRA MedGemma-4B, phân loại và định vị | yes/no kèm box |
| C3 | C2 + cân bằng lớp (oversampling ca dương 1:1, trọng số loss cho token trả lời) | yes/no kèm box |
| C4 | Cấu hình tốt nhất trong C1–C3, áp lên Qwen2.5-VL-3B | như trên |
| B1 | DenseNet-121 classifier | xác suất |
| B2 | YOLOv8 detector | box và điểm |

**Mở rộng, không thuộc cam kết:**
- **C5, VLM có hỗ trợ vùng:** đưa box đề xuất của B2, hoặc ảnh cắt theo box, vào prompt để VLM xác nhận và mô tả. Đây là hướng xử lý giả thuyết gộp toàn cục [L47].
- **C6, độ phân giải và chia ô:**
  - Với MedGemma, ảnh vốn đã được đưa về 896 px, nên C6 là **chia ô hoặc cắt theo trường phổi** để vượt quá 896 px hiệu dụng.
  - Với Qwen2.5-VL (độ phân giải động), C6 là thử nhiều mức độ phân giải đầu vào, theo [L82, L83].

### 5.3 Định dạng huấn luyện

- **Câu hỏi cố định:** "Is there a lung nodule or mass on this chest X-ray? If yes, give the bounding box(es) in pixel coordinates. Answer in JSON."
- **Đáp án đích:** sinh tự động từ nhãn gốc (box hợp nhất của các bác sĩ bằng weighted boxes fusion). Ca âm tính có đáp án `{"nodule":"no","boxes":[]}`.
- Không dùng câu hỏi–đáp do LLM sinh, để tránh nhiễu nhãn đã gặp ở đề tài cũ.
- **Không đưa ngữ cảnh lâm sàng vào prompt** trong điều kiện chính. Theo [L4], ngữ cảnh làm accuracy tăng từ 28% lên 62%, có thể vì "lộ" đáp án. [L51] cũng đưa bệnh sử vào prompt. Nếu còn thời gian, thử thêm điều kiện có ngữ cảnh như một phân tích phụ.

### 5.4 Cơ sở lý thuyết

- **Mất cân bằng lớp và học lối tắt:** khi ca dương hiếm, nghiệm ERM có lợi khi luôn trả lời "không". Nối với nội dung về cost-sensitive learning và hiệu chỉnh (calibration).
- **Thích ứng hạng thấp (LoRA):** cập nhật ΔW = BA với hạng r rất nhỏ. Câu hỏi đặt ra là không gian hạng thấp có đủ để đưa tín hiệu cục bộ vào đầu ra ngôn ngữ khi bộ mã hóa ảnh đông cứng hay không. Nối với [L47].
- **Đánh giá chẩn đoán:** ROC, FROC, chọn ngưỡng, so sánh bắt cặp.

## 6. Dữ liệu

| Vai trò | Bộ dữ liệu | Nhãn dùng | Trạng thái |
|---|---|---|---|
| Huấn luyện, validation, test trong phân phối | **VinBigData Chest X-ray Abnormalities Detection** (Kaggle): 15.000 ảnh huấn luyện, box theo từng bác sĩ | Lớp "Nodule/Mass" (dương), "No finding" (âm); có thể thêm ca có bệnh khác làm âm khó | Đã dùng ở đề tài cũ. Cần kiểm tra điều khoản cuộc thi. **Rủi ro cỡ mẫu (theo [L37]):** tập train của VinDr-CXR chỉ có 410/15.000 ảnh Nodule/Mass (2,73%). Chia 70/10/20 sẽ còn khoảng 287 / 41 / 82 ảnh dương (ước tính), nên khoảng tin cậy của sensitivity sẽ rộng. Nếu truy cập được, ưu tiên dùng tập test chính thức của VinDr-CXR: 3.000 ảnh, 176 ảnh Nodule/Mass, nhãn đồng thuận 5 bác sĩ |
| Test ngoài phân phối 1 | **NIH ChestX-ray14** (Kaggle hoặc NIH): nhãn Nodule, Mass; khoảng 1.000 ảnh có box cho 8 bệnh | Nhãn ảnh (rút từ báo cáo, nhiễu) và tập con có box | Cần xác minh số ảnh có box Nodule/Mass |
| Test ngoài phân phối 2 (độ mờ) | **JSRT**: 247 ảnh, 154 có nốt kèm vị trí, kích thước, độ mờ 1–5, lành/ác | Nốt có hoặc không, tâm nốt, độ mờ | Cần kiểm tra giấy phép và nguồn tải. Cho phép so trực tiếp với [L2] và [L31] |
| Tham khảo | NODE21 [L35], MosMedData VII [L1, L45] | — | NODE21 trùng nguồn với JSRT và NIH, nên không dùng làm test để tránh rò rỉ |

**Chia dữ liệu:** VinBigData chia theo `image_id`, tỷ lệ 70/10/20, phân tầng theo nhãn Nodule/Mass. Ngưỡng và siêu tham số chọn trên validation. Mỗi tập test chỉ dùng một lần.

**Tiền xử lý:**
- DICOM/PNG đổi kích thước về độ phân giải đầu vào của từng mô hình, ghi lại hệ số quy đổi tọa độ.
- Kích thước nốt quy đổi theo kích thước box tương đối. Riêng JSRT có kích thước nốt tính bằng mm.

## 7. Thí nghiệm

| Mã | Nội dung | Kết quả cần có |
|---|---|---|
| E0 | Chuẩn bị dữ liệu: thống kê số ca Nodule/Mass, phân bố kích thước box, độ đồng thuận giữa bác sĩ; hợp nhất box | Bảng mô tả dữ liệu |
| E1 | Baseline CNN B1, B2 trên VinBigData; đánh giá trên 3 tập test | AUC, sens@spec0,9, FROC |
| E2 | Zero-shot C0 cho 2–3 VLM, chạy 3 lần (nhiệt độ 0 và 0,7) | Bảng zero-shot; tỷ lệ parse lỗi; độ ổn định giữa các lần |
| E3 | LoRA C1, C2, C3 trên MedGemma-4B, mỗi cấu hình 3 seed | Bảng so sánh, đường học, sensitivity theo epoch |
| E4 | C4: cấu hình tốt nhất áp lên Qwen2.5-VL-3B, 1–3 seed | So sánh mô hình y khoa với mô hình đa năng |
| E5 | Kiểm toán "nhìn đúng chỗ" cho C0, cấu hình LoRA tốt nhất và B2: che nốt so với che vùng ngẫu nhiên cùng diện tích; phân tầng theo kích thước và độ mờ | Bảng đổi đáp án; biểu đồ sensitivity theo tầng |

**Số lượt chạy:** C1–C3 × 3 seed = 9 lượt; C4 1–3 lượt; B1, B2 mỗi loại 3 seed.

**Ngân sách tính toán:**
- Lượt LoRA đầu tiên dùng để đo thời gian.
- Nếu một lượt MedGemma-4B vượt 3 giờ, cắt giảm theo thứ tự: giảm số ảnh âm (giữ toàn bộ ảnh dương); giảm độ phân giải; giảm còn 2 epoch; giảm C1 còn 1 seed.

## 8. Đánh giá

### 8.1 Chỉ số phát hiện (mức ảnh)

- Sensitivity, specificity, PPV, NPV, F1.
- **AUC:**
  - Với VLM, lấy điểm liên tục từ xác suất token "yes" ở vị trí trả lời.
  - Nếu không lấy được, dùng tỷ lệ trả lời "yes" qua nhiều lần lấy mẫu.
- **Sensitivity ở specificity 0,90 và 0,95.** Ngưỡng chọn trên validation.
- **Chỉ số chế độ báo bình thường:** tỷ lệ trả lời "no" trên ca dương; tỷ lệ "yes" trên ca âm.
- **Chỉ số có tính đến tỷ lệ bệnh [L84, L79, L75]:**
  - AUC-PR, đặt cạnh AUC-ROC.
  - Calibration: ECE và Brier (balanced Brier nếu lớp dương hiếm).
  - PPV quy về tỷ lệ bệnh thấp (ví dụ 1–3%).
  - Trên JSRT, tỷ lệ nốt là 62%, nên PPV ở đó không có ý nghĩa lâm sàng. Chỉ báo kèm chú thích.

### 8.2 Chỉ số định vị

- IoU trung bình; tỷ lệ "đúng và IoU ≥ 0,3", tức đúng vì đúng lý do. Ngưỡng 0,3 vì nốt nhỏ, và sẽ báo thêm ở 0,5.
- **FROC:** sensitivity theo số false positive mỗi ảnh. Áp cho B2 và VLM có box.
- **JSRT:** "trúng" khi tâm box dự đoán nằm trong bán kính nốt cộng biên. Báo thêm khoảng cách tâm trung vị (mm), theo cách của NoduLoCC [L36].

### 8.3 Kiểm toán phụ thuộc thị giác

Giữ lại từ đề tài cũ.

| Mã | Phép thử | Hành vi mong đợi |
|---|---|---|
| T1 | Che nốt (box nới 10%, tô giá trị trung bình) | Đáp án đổi từ "yes" sang "no" |
| T2 | Che vùng ngẫu nhiên cùng diện tích, không giao nốt, trong trường phổi | Đáp án giữ nguyên |
| T3 | Tráo sang ảnh âm tính của bệnh nhân khác | "no" |

- **Chỉ số chính:** tỷ lệ đổi đáp án ở T1 trừ tỷ lệ đổi ở T2.
- **Phân tầng:** theo kích thước nốt (tertile của diện tích box; mm với JSRT) và theo độ mờ JSRT 1–5.

### 8.4 Thống kê

- Trung bình ± độ lệch chuẩn qua các seed.
- Khoảng tin cậy 95% bằng bootstrap theo ảnh.
- So sánh bắt cặp: McNemar cho sens/spec, DeLong cho AUC, điều chỉnh Holm khi so nhiều cặp.

### 8.5 Ghi chú triển khai (2026-11-13)

Các điểm dưới đây làm rõ cách đo trong code (`code/`), không đổi mục tiêu hay tiêu chí thành công. Báo cáo nên mô tả phương pháp theo đúng các điểm này.

- **Điểm liên tục của VLM:** so log-prob của hai chuỗi trả lời ép buộc `{"nodule": "yes` và `{"nodule": "no` (không có dấu `"` đóng), tính từ token đầu tiên hai chuỗi khác nhau.
- **Trọng số loss của C3:** $w_{+}$ áp lên mọi token đáp án của ca dương (cả token box, không chỉ token yes/no). Loss được chuẩn hóa theo tổng số token nhãn của cả cửa sổ tích lũy gradient, để $w_{+}$ có tác dụng khi batch = 1.
- **Câu trả lời không parse được:**
  - `raw_sens`/`raw_spec` vẫn tính câu không parse được là "no".
  - Chỉ số chế độ báo bình thường (tỷ lệ "no" trên ca dương, "yes" trên ca âm) chỉ tính trên câu trả lời parse được, và báo riêng tỷ lệ `unparsed`.
  - Báo cáo được tính lại từ `raw_text` bằng parser hiện hành.
- **Định vị:** IoU trung bình và tỷ lệ "đúng và IoU ≥ 0,3/0,5" chỉ dùng box có điểm ≥ ngưỡng t90 (box của VLM mang điểm của ảnh). FROC dùng mọi box. JSRT dùng box điểm cao nhất, biên 5 mm (12,5 px trên ảnh 896), khoảng cách đổi ra mm với 0,4 mm/px.
- **C0:** dòng chính trong bảng dùng lần chạy nhiệt độ 0. Hai lần chạy nhiệt độ 0,7 chỉ dùng để đo độ ổn định (tỷ lệ đồng thuận trên ảnh parse được ở mọi lần chạy). C0 sinh tối đa 256 token.
- **Chọn C4 và cấu hình cho E5:** theo `val_sens90` trung bình qua seed trong `val_selection.csv` (validation), không dùng số trên test.
- **Kiểm toán che vùng:**
  - T1 che box nới 10% (5% mỗi cạnh).
  - T2 dùng vùng cùng kích thước với box đã nới, không giao vùng T1, ≥ 90% nằm trong mặt nạ phổi (PSPNet của torchxrayvision, dự phòng bằng mặt nạ heuristic).
  - $\Delta_{\text{occ}}$ tính trên các ảnh lấy mẫu được T2 (bắt cặp), báo kèm `n_t2_ok` và tỷ lệ đổi ở T1 trên toàn bộ ca dương.
- **Phân tầng kích thước:** VinBigData/NIH theo tertile diện tích box tương đối (ngưỡng từ ca dương train VinBigData). JSRT theo mm: ≤10, 10–20, >20 mm.
- **Khoảng tin cậy:** bootstrap phân tầng theo nhãn, theo ảnh. Không gộp theo bệnh nhân, nên CI có thể hẹp hơn thực tế nếu một bệnh nhân có nhiều ảnh; đây là một hạn chế. CI theo từng seed nằm trong `summary_per_seed.csv`.
- **Phần cứng:** MedGemma chỉ chạy trên GPU có compute capability ≥ 8 (L4/A100, bf16 thật), vì fp16 bị tràn số. "Giảm độ phân giải" trong quy tắc ngân sách không áp dụng cho MedGemma (processor cố định 896 px).

## 9. Thảo luận và hướng phát triển

- **VLM có hỗ trợ vùng (C5):** pipeline "detector đề xuất, VLM xác nhận và giải thích", theo kiểu agent MedRAX [L11] dẫn đầu ở [L1].
- **Từ phát hiện đến báo cáo và theo dõi:** sinh câu mô tả nốt, gồm vị trí, kích thước, tính chất, và gợi ý chụp CT theo guideline (đưa guideline vào prompt hoặc dùng RAG). Hướng này dựa trên [L15].
- **Nguy cơ ung thư đa phương thức:** CXR + embedding LLM từ báo cáo và EHR, so với CXR-LC [L23]. Tham khảo [L13], [L61].
- **Câu hỏi mở:** bộ mã hóa ảnh đông cứng có phải nút thắt không? Cần thí nghiệm mở khóa một phần bộ mã hóa ảnh, hoặc probe patch-local trên mô hình sau LoRA.

**Hạn chế cần nêu trong báo cáo:**
- Nhãn VinBigData và NIH có nhiễu; nhãn NIH rút từ báo cáo.
- VinBigData không có kích thước nốt tính bằng mm.
- JSRT là phim scan cũ, ngoài phân phối mạnh.
- Mô hình cỡ nhỏ.
- Không có xác nhận mô bệnh học cho nốt "ác tính" ngoài JSRT.
- Rủi ro rò rỉ dữ liệu: các bộ CXR công khai (MIMIC-CXR, VinDr-CXR, NIH) được tái dùng rộng rãi để huấn luyện VLM [L40, L43]. Kết quả zero-shot trên các tập này có thể bị thổi phồng. JSRT ít bị dùng huấn luyện hơn, nhưng không đảm bảo hoàn toàn.
- Một VLM yếu nhưng trả lời tự tin có thể kéo người đọc theo hướng sai [L88, L90]. Vì vậy kiểm toán ở §8.3 có ý nghĩa an toàn, và kết quả không được trình bày như một công cụ sẵn sàng dùng.
- Kết quả không dùng cho quyết định lâm sàng.

---

## Sản phẩm bàn giao

1. Báo cáo theo 9 mục trên.
2. Mã nguồn: chuẩn bị dữ liệu, prompt và parser, huấn luyện LoRA, baseline CNN, đánh giá, kiểm toán che vùng.
3. Bảng benchmark và file dự đoán (chỉ nhãn và dự đoán, không phân phối lại ảnh).
4. Slide thuyết trình.

## Tài liệu tham khảo chính và mức độ kiểm chứng

Mã L# khớp với `docs/lit-review/lung-cancer-cxr/corpus.md`.

| # | Tài liệu | Nơi công bố | Mức đã đọc |
|---|---|---|---|
| L1 | Khovanova et al., *A comparative accuracy study of multimodal LLMs, VLM and agent-based framework for pulmonary nodule detection on chest radiographs* | Front. Digit. Health 2026, doi 10.3389/fdgth.2025.1674835 | Toàn văn (đọc lại qua Undermind ngày 2026-11-12) |
| L2 | Nishio et al., *Zero-Shot Performance of Six VLMs for Lung Nodule Detection on Chest Radiographs* | AJR 2026, doi 10.2214/AJR.26.35421 (bản preprint medRxiv 10.64898/2026.05.31.26354565) | Toàn văn qua bản preprint, gồm Bảng 2–3; bản AJR chỉ đọc tóm tắt |
| L3 | Chen CH et al., *Limitations in CXR Interpretation by Vision-Capable LLMs* | Diagnostics 2026, doi 10.3390/diagnostics16030376 | Toàn văn |
| L4 | Cusick et al., *Effect of Structured Context on CXR Interpretation by a Multimodal LLM* | Cureus 2026, doi 10.7759/cureus.108715 | Toàn văn |
| L64 | Volkov et al., *SmolVLM + LoRA cho báo cáo CXR có cấu trúc* | SCM (IEEE) 2025 | Toàn văn |
| L69 | Gao et al., *S2D-Align* | AAAI 2026 | Toàn văn |
| L71 | Xu et al., *MedGround-R1* | MICCAI 2025 | Toàn văn |
| L83 | Kim et al., *Optimal matrix size for CAD of nodules/masses on CXR* | Eur Radiol 2020 | Toàn văn |
| L84 | Mosquera et al., *Class imbalance in medical imaging evaluation* | Eur Radiol 2024 | Toàn văn |
| L11 | Fallahpour et al., *MedRAX* | arXiv 2502.02673 (preprint) | Tóm tắt |
| L15 | Gamble et al., *Limitations of GPT-3.5 and GPT-4 in Applying Fleischner Guidelines* | Can Assoc Radiol J 2024 (theo DOI SAGE), doi 10.1177/08465371231218250 | Tóm tắt |
| L19 | Nam et al., *DL-based Automatic Detection Algorithm for Malignant Pulmonary Nodules on CXR* | Radiology 2019, doi 10.1148/radiol.2018180237 | Tóm tắt |
| L20 | Nam et al., *AI Improves Nodule Detection on CXR: an RCT* | Radiology 2023, doi 10.1148/radiol.221894 | Tóm tắt |
| L23 | Lu et al., *CXR-LC* | Ann Intern Med 2020, doi 10.7326/M20-1868 | Tóm tắt |
| L31 | Ohlmann-Knafo et al., *AI-based software for lung nodule detection in CXR (JSRT)* | arXiv 2206.10912 (preprint) | Tóm tắt |
| L35 | Sogancioglu et al., *NODE21 Challenge* | IEEE TMI 2024 | Tóm tắt |
| L36 | Mustafic et al., *NoduLoCC2026* | arXiv 2606.21290 (preprint) | Tóm tắt |
| L45 | Arzamasov et al., *Independent evaluation of 5 AI software for lung nodules on CXR* | Quant Imaging Med Surg 2024, doi 10.21037/qims-24-160 | Tóm tắt |
| L47 | Muthyala et al., *Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in CXR* | arXiv 2606.11606 (preprint) | Toàn văn (qua Undermind) |
| L48 | Gosai et al., *Beyond Diagnosis: Evaluating MLLMs for Pathology Localization in CXR* | PMLR (ML4H) 2025, arXiv 2509.18015 | Toàn văn (qua Undermind) |
| L49 | Choi et al., *CheXpercept* | arXiv 2606.21020 (preprint) | Toàn văn (qua Undermind) |
| L50 | Fisher, *VL Foundation Models Do Not Transfer to Medical Imaging Classification* | medRxiv 2025 (preprint) | Tóm tắt |

Hai bài cốt lõi [L1], [L2] đã qua bình duyệt. Xếp hạng Q1/Q2 của từng tạp chí cần đối chiếu trên Scimago trước khi ghi vào báo cáo.

## Việc phải làm đầu tiên

1. Đọc điều khoản dữ liệu của VinBigData (Kaggle), NIH ChestX-ray14 và JSRT. Xác nhận được phép dùng cho học thuật. Chấp nhận điều khoản MedGemma (HAI-DEF) trên Hugging Face.
2. Thống kê số ca Nodule/Mass trong VinBigData và số ảnh có box Nodule/Mass trong NIH. Xác minh VinBigData trên Kaggle có tương ứng tập train của VinDr-CXR không, và có lấy được tập test chính thức của VinDr-CXR không (S8 trong `07`).
3. Chạy thử zero-shot MedGemma-4B trên 50 ảnh để chốt prompt, parser và cách lấy xác suất token "yes". Kiểm tra MedGemma có xuất được box dạng JSON không.
4. Chạy một lượt LoRA thử để đo thời gian và chốt độ phân giải.
5. Đọc toàn văn [L45] và [L50] để xác minh cách tinh chỉnh dẫn đến sensitivity sụp (S2). Cách chuyển đầu ra thành nhãn của [L2] đã có trong bản preprint: hai bác sĩ đọc báo cáo, không biết mô hình nào, quy về nhị phân, và cách diễn đạt mơ hồ được tính là dương.
