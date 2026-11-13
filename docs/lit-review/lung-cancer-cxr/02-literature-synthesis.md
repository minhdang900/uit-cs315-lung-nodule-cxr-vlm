# Tổng hợp tài liệu (comprehensive): LLM cho ung thư phổi trên X-quang ngực

*Ngày 2026-11-12. Đây là bước `literature-synthesis` của arxiv-mcp-server với `synthesis_type = comprehensive` và `domain = computer_science / medical imaging`.*

**Dữ liệu đầu vào.**
- 24 mã arXiv trong `corpus.md` §5, cộng các bài đã qua bình duyệt L1–L62, vì bằng chứng chính nằm ở PubMed chứ không ở arXiv.
- Trong code của server, prompt này chưa có nội dung riêng và dùng lại khung của `literature_review`. Tài liệu này mở rộng khung đó theo bốn trục: chủ đề, phương pháp, dòng thời gian, khoảng trống.
- Mức độ đọc giống tài liệu 01: 4 bài đọc toàn văn, còn lại đọc tóm tắt.

---

## 1. Ma trận bằng chứng: loại mô hình × bài toán

Ô ghi "Ít" nghĩa là có dưới 2 nghiên cứu đánh giá riêng nốt/khối/ung thư. Ô ghi "Không" nghĩa là không có nghiên cứu nào.

| | Phát hiện/phân loại nốt | Định vị/grounding | Sinh báo cáo | VQA | Sàng lọc/nguy cơ |
|---|---|---|---|---|---|
| **MLLM đa năng** (GPT-4o/5, Claude, Gemini, Llama-V, Qwen-VL) | **Có, kết quả tiêu cực**: L1, L2, L3, L4, L52, L53 | Ít: L48 (lung lesion) | Ít: L51 (khối) | Không | Không |
| **VLM chuyên CXR** (CXR-LLaVA, RadVLM, LLaVA-Rad, MedGemma, XrayGPT, CheXagent, MAIRA-2) | **Có, sensitivity rất thấp**: L1, L2 | Không riêng cho nốt: L8, L9, L59 | Không riêng cho nốt: L5, L10 | Không riêng cho nốt: L42, L43 | Không |
| **Mô hình nền/CLIP** (BiomedCLIP, MedSigLIP, RAD-DINO) | Ít: L1 (AUC 0,743); probe cơ chế: L47 | Probe patch-local có oracle ROI: L47 | — | — | Không |
| **Agent** (MedRAX) | Ít: L1 (AUC 0,704) | Không | Không | Không | Không |
| **LLM văn bản** | Gián tiếp: gán nhãn báo cáo CXR (L55, L56, L57) | — | — | — | Ít: L58 (lọc cohort); theo dõi nốt chỉ trên CT (L15–L18) |
| **Đa phương thức CXR + LLM/văn bản** | — | — | — | — | **Preprint hoặc chỉ có tiêu đề**: L13, L61 |
| **CAD CNN/ViT** (mốc) | **Mạnh**: L19, L21, L45, meta-analysis L33/L34 | Có: L25, L36 | — | — | **Mạnh**: L23, L24, L30 |

**Đọc ma trận.**
- Bằng chứng về LLM dồn vào một ô duy nhất: phát hiện zero-shot. Ô đó cho kết quả tiêu cực.
- Mọi ô "LLM × định vị / VQA / nguy cơ" cho nốt đều gần như trống.
- Hàng CAD CNN thì đầy. Vì vậy câu hỏi đáng hỏi không còn là "LLM có thay CAD được không". Câu hỏi đúng hơn là LLM bổ sung được gì cho CAD, và vì sao LLM thất bại.

---

## 2. Tổng hợp theo chủ đề

### Chủ đề 1: Khoảng cách hiệu năng giữa VLM và CAD chuyên biệt

| Hệ thống | Dữ liệu | AUC | Sens | Spec | Nguồn |
|---|---|---|---|---|---|
| CAD thương mại tốt nhất (Celsus, Lunit, qXR) | MosMedData VII, 100 CXR | **0,956** (0,812 khi bác sĩ kiểm tra phân đoạn) | — | 100% (giai đoạn 2–3) | L45 |
| BiomedCLIP (lấy ngưỡng Youden) | MosMedData VII, 83 CXR | 0,743 | 0,737 | 0,689 | L1 |
| MedRAX (agent, Qwen2.5-14B) | như trên | 0,704 | 0,553 | 0,844 | L1 |
| Claude 3.7 Sonnet | như trên | 0,625 | 0,316 | 0,933 | L1 |
| CXR-LLaVA | như trên | 0,517 | 0,079 | 0,956 | L1 |
| LLaVA-Rad (zero-shot) | JSRT, 247 | — | 0,273 | 0,710 | L2 |
| RadVLM | JSRT, 247 | — | 0,110 | 1,000 | L2 |
| MedGemma-4b-it | JSRT, 247 | — | 0,052 | 1,000 | L2 |
| Bác sĩ X-quang (trung bình 8 người) | JSRT, 247 | 0,77 | — | — | L31 |
| CNN nhiều trung tâm (DLAD) | 4 tập ngoài | 0,92–0,99 | — | — | L19 |

*Lưu ý: L45 và L1 dùng cùng nguồn MosMedData VII, nhưng L1 không xác nhận trùng từng ảnh và chỉ phân tích 83/100 ảnh. Đây là phép so sánh gần nhất hiện có, chưa phải so sánh đối đầu.*

**Kết luận tổng hợp.** Cách biệt AUC khoảng 0,2 giữa CAD và VLM tốt nhất là nhất quán. Kiểu sai cũng nhất quán: VLM chuyên CXR rơi vào chế độ "báo bình thường", với specificity gần 1 và sensitivity gần 0.

### Chủ đề 2: Vì sao VLM bỏ sót nốt (giả thuyết cơ chế)

Ba lớp giải thích, từ có bằng chứng nhất đến suy luận nhiều nhất:

1. **Mất tín hiệu khi gộp toàn cục** (L47, có bằng chứng probe). Token CLS hoặc trung bình patch của ViT đông cứng làm loãng tổn thương nhỏ. Patch-local lấy lại được AUC ≥ 0,899 cho nốt, với điều kiện có oracle ROI.
2. **Định vị yếu** (L48, L49, L36). VLM đặt dự đoán vào vùng giải phẫu hợp lý nhưng không chính xác. Năng lực giảm mạnh từ "có hay không" sang "ở đâu, đường viền thế nào".
3. **Thiên lệch nhãn và ngôn ngữ** *(suy luận, chưa có bằng chứng trực tiếp cho nốt)*. Báo cáo huấn luyện như MIMIC phần lớn bình thường, và nốt hiếm được nhắc tới, nên mô hình dễ đi đường tắt sang câu trả lời "không có bất thường". Điều này khớp với kiểu sai specificity ≈ 1 ở L2. Nó cũng nối được với dòng nghiên cứu về language prior và visual reliance trong đề tài cũ.

### Chủ đề 3: Vai trò thực tế của LLM là bổ trợ, không thay thế

- **LLM văn bản** trích xuất và gán nhãn báo cáo ở mức gần labeler chuyên dụng (L55, L56, L57). Đây là công cụ tạo nhãn yếu (weak label) cho dữ liệu nốt.
- **Agent** như MedRAX, vốn gọi công cụ CNN, đứng đầu nhóm sinh văn bản ở L1. Điều này ủng hộ kiến trúc "CAD phát hiện, LLM giải thích".
- **Giao diện chỉ có văn bản** giúp bác sĩ phát hiện nốt tốt nhất (L32). *(Suy luận)* LLM có thể là lớp giao diện cho CAD.

### Chủ đề 4: Nguy cơ và sàng lọc

- CXR-LC (L23) và kiểm định ngoài (L24) cho thấy CXR mang tín hiệu nguy cơ ung thư dài hạn.
- ViT tiền huấn luyện thêm được 6–10 điểm AUC (L30).
- LLM mới chỉ xuất hiện dưới dạng embedding đặc trưng lâm sàng (L61, chỉ có tiêu đề) hoặc kênh báo cáo + KG (L13, preprint).

---

## 3. So sánh phương pháp đánh giá

| Khía cạnh | Thực hành hiện tại | Vấn đề | Khuyến nghị |
|---|---|---|---|
| Dữ liệu | JSRT (2000, scan phim), MosMedData VII (100 ảnh), tập tự chọn 50–257 ảnh | Nhỏ, cũ, một nguồn | Gộp NODE21, VinDr-CXR (Nodule/Mass), ChestX-ray14 (Nodule/Mass có BBox), NoduLoCC |
| Định dạng ảnh | PNG/JPEG | L1 thừa nhận lệch so với DICOM | Ghi rõ tiền xử lý; thử nhiều độ phân giải |
| Prompt | Mỗi bài một kiểu; prompt cho mô hình chuyên miền khác prompt cho mô hình đa năng (L1) | Gây thiên lệch khi so sánh | Prompt chuẩn, chạy nhiều lần, báo độ biến thiên |
| Chuyển văn bản thành nhãn | Hai bác sĩ đồng thuận (L2); luật từ khóa (L1) | Không tái lập được | Đầu ra có cấu trúc (JSON); luật công khai |
| Ngưỡng | Youden trên chính tập test (L1) | Lạc quan quá mức | Chọn ngưỡng trên tập validation; báo sens ở spec cố định |
| Chỉ số | Accuracy, sens/spec, McNemar | Thiếu định vị, thiếu phân tầng | Thêm FROC/hit rate, phân tầng theo kích thước và độ mờ |
| So sánh | Hiếm khi so với CAD trên cùng tập | Không biết khoảng cách thật | Luôn có CNN baseline trên cùng split |

---

## 4. Dòng thời gian

```
2000  JSRT database (247 CXR, 154 nodule)
2017  ChestX-ray8/14 (Nodule, Mass labels)
2019  DLAD – Nam (Radiology) · CheXpert · MIMIC-CXR · PadChest
2020  Yoo (NLST) · CXR-LC (Ann Intern Med) · Majkowska (Google)
2022  VinDr-CXR · Lee (USPSTF + CXR-LC) · Dissez reader study
2023  RCT Nam (Radiology) · BiomedCLIP · XrayGPT · Vicuna labeling (Radiology)
2024  CheXagent · MAIRA-2 · NODE21 (IEEE TMI) · Arzamasov 5-CAD · GPT-4 labeling (AJR)
2025  CXR-LLaVA (Eur Radiol) · RadVLM · LLaVA-Rad (Nat Commun) · MedRAX · ReXVQA · Gosai localization · Fisher negative
2026  MedGemma (Nat Med) · Khovanova 9-model · Nishio 6-VLM (AJR) + commentary · Muthyala mechanism · CheXpercept · NoduLoCC2026 · LUNG-KGMM · Kim (JTO abstract)
```

Điểm chuyển quan trọng là năm 2026. Lần đầu tiên có hai đánh giá độc lập, riêng cho nốt và cùng hướng kết quả (L1, L2). Đi kèm là một commentary cảnh báo trên AJR (L62) và một bài cơ chế (L47). Lĩnh vực đang chuyển từ "trình diễn năng lực" sang "kiểm toán độc lập".

---

## 5. Từ khoảng trống đến câu hỏi nghiên cứu

| Khoảng trống (tài liệu 01, §5) | Câu hỏi nghiên cứu (RQ) | Khả thi cho đồ án CS315? |
|---|---|---|
| K2: không có LoRA VLM cho nốt | **RQ1.** Tinh chỉnh LoRA/QLoRA một VLM nhỏ (MedGemma-4B, Qwen-VL 3B/7B, LLaVA-Rad) trên dữ liệu nốt CXR có thu hẹp khoảng cách sensitivity so với CNN không? | **Cao.** Dữ liệu có trên Kaggle (NIH ChestX-ray14, VinBigData) và Hugging Face; LoRA chạy được trên 1 GPU |
| K4 + chế độ báo bình thường | **RQ2.** Sau LoRA, sensitivity theo kích thước nốt thay đổi thế nào, và có rơi lại vào chế độ báo bình thường không? | Cao (phần phân tích) |
| K5: cơ chế gộp | **RQ3.** Đưa đặc trưng patch-local hoặc ROI từ một detector vào prompt hay adapter có tăng phát hiện nốt nhỏ không? | Trung bình (cần detector) |
| K3/K8: grounding nốt | **RQ4.** VLM sau LoRA định vị nốt chính xác đến đâu (hit rate theo lưới kiểu Gosai, hoặc IoU)? | Trung bình–cao (VinBigData và NIH có BBox) |
| K9: đánh giá chưa chuẩn | **RQ5.** Đánh giá chuẩn hóa (prompt cố định, ngưỡng chọn trên validation, nhiều lần chạy) có thay đổi xếp hạng mô hình so với L1/L2 không? | Cao |
| K6: theo dõi nốt trên báo cáo CXR | **RQ6.** LLM có đưa guideline vào prompt (hoặc RAG) có xác định đúng báo cáo CXR cần chuyển chụp CT không? | Trung bình (thiếu nhãn tham chiếu công khai) |

---

## 6. Hàm ý cho đồ án (thay đề tài cũ)

*(Suy luận và đề xuất; cần bạn duyệt trước khi viết lại spec.)*

**Hướng đề xuất.** "Kiểm toán và tinh chỉnh LoRA VLM nhỏ cho phát hiện và định vị nốt phổi trên X-quang ngực, so sánh với CNN".
- Kết hợp RQ1, RQ2, RQ4 và RQ5.
- Tận dụng được hạ tầng LoRA và ý tưởng audit của đề tài cũ.

**Đóng góp dự kiến.**
1. Đánh giá chuẩn hóa zero-shot cho 3–5 VLM mở trên các tập nốt CXR công khai. Đây là bản tái lập có kiểm soát của L1/L2, ở quy mô lớn hơn.
2. Tinh chỉnh LoRA và báo sens/spec theo kích thước nốt, so với CNN baseline trên cùng split.
3. Phân tích chế độ báo bình thường và định vị: kiểm tra mô hình có thật sự "nhìn" vào nốt hay không. Phần này nối lại ý tưởng visual reliance của đề tài cũ.

**Dữ liệu ứng viên.**
- NIH ChestX-ray14 (Kaggle): nhãn Nodule/Mass, khoảng 1.000 ảnh có BBox.
- VinBigData CXR (Kaggle): lớp Nodule/Mass có box.
- NODE21.
- JSRT, làm tập test ngoài để so trực tiếp với L2/L31.

*Cần kiểm tra giấy phép và số lượng mẫu thực tế trước khi chốt.*

**Rủi ro.**
- Nhãn NIH được rút từ báo cáo nên nhiễu.
- Nốt nhỏ trên ảnh resize 224–448 px có thể mất hẳn; theo L47, nên thử độ phân giải cao hơn hoặc cắt theo vùng phổi.
- Fine-tune có thể làm sụp sensitivity (L50); cần theo dõi và cân bằng lớp.

---

*Nguồn chi tiết: `corpus.md`. Tổng quan theo 6 mục: `01-literature-review.md`. Workspace Undermind: https://app.undermind.ai/projects/fbc55738-03be-48ab-b376-2937981a6356*
