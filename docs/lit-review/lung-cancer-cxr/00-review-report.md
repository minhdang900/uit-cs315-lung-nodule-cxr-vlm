# Báo cáo rà soát: Đổi đề tài sang "LLM/VLM cho nốt phổi trên X-quang ngực"

**Ngày:** 2026-11-12
**Người đọc:** nhóm đồ án và giảng viên môn CS315.G12.LT.TTNT
**Mục đích:** tóm tắt bằng chứng từ tổng quan tài liệu, lý do đổi đề tài, đề tài đề xuất và các điểm cần nhóm quyết định.
**Tài liệu chi tiết:**
- `01-literature-review.md`: tổng quan theo 6 mục.
- `02-literature-synthesis.md`: tổng hợp và câu hỏi nghiên cứu.
- `corpus.md`: 60 bài, mã L1–L62.
- Spec mới: `docs/design/2026-11-12-lung-nodule-cxr-vlm-lora-design.md`.

---

## 1. Tóm tắt cho người bận

- **Câu hỏi:** LLM và VLM có giúp phát hiện ung thư phổi, cụ thể là nốt và khối, trên X-quang ngực không?
- **Trả lời từ tài liệu (2026):** ở chế độ zero-shot thì **chưa**.
  - Hai đánh giá độc lập cho sensitivity 0,6–27,3% [L2] và AUC tốt nhất 0,743 [L1].
  - CAD CNN thương mại trên cùng nguồn dữ liệu đạt AUC 0,956 [L45].
- **Kiểu sai:** VLM chuyên cho CXR gần như luôn trả lời "bình thường". Specificity khoảng 1, sensitivity khoảng 0 (MedGemma 5,2%).
- **Khoảng trống chính:** chưa ai tinh chỉnh LoRA VLM cỡ nhỏ riêng cho nốt CXR, và chưa ai đo định vị nốt sau tinh chỉnh.
- **Đề tài đề xuất:** *"Mô hình ngôn ngữ–thị giác có thấy nốt phổi không? Kiểm toán và tinh chỉnh LoRA VLM cỡ nhỏ cho phát hiện và định vị nốt phổi trên X-quang ngực"*. VLM chính là MedGemma-4B, Qwen2.5-VL-3B dùng để so sánh; đối chiếu với DenseNet và YOLOv8. Dữ liệu: VinBigData (Kaggle), NIH ChestX-ray14, JSRT.

---

## 2. Quy trình tìm kiếm

| Bước | Công cụ | Kết quả |
|---|---|---|
| Phỏng vấn phạm vi | AskUserQuestion | Thay đề tài cũ. Gồm 4 loại mô hình (VLM, LLM văn bản, mô hình nền/CLIP, CNN đối chứng) và 4 bài toán (phát hiện, sinh báo cáo, VQA/grounding, sàng lọc) |
| Tìm nhiều nguồn | arXiv MCP, PubMed, Semantic Scholar, OpenAlex, Europe PMC, Consensus, WebSearch (agent nền) | Khoảng 85–90 lần gọi, khoảng 600 bản ghi, giữ 44 bài |
| Deep search | Undermind | 101 bài xếp hạng, thêm 18 bài |
| Đọc toàn văn | Undermind `read_pdfs` | 4 bài: L1, L47, L48, L49 |
| Tổng quan và tổng hợp | Prompt `literature_review` và `literature-synthesis` của arxiv-mcp-server | Tài liệu 01, 02 |

**Bộ tài liệu cuối:** 60 bài khác nhau.

| Nhóm | Số bài |
|---|---|
| VLM/MLLM | khoảng 20 |
| LLM văn bản | 8 |
| Mô hình nền thị giác–ngôn ngữ | 4 |
| CNN/ViT | 15 |
| Dataset/benchmark | 12 |
| Tổng quan | 2 |

Phần lớn bài thuộc giai đoạn 2024–2026.

---

## 3. Bằng chứng then chốt

| Phát hiện | Con số | Nguồn | Độ tin cậy |
|---|---|---|---|
| VLM zero-shot bỏ sót nốt | 6 VLM: sens 0,6–27,3%, spec 71–100% (247 ảnh JSRT) | [L2] AJR 2026 | Bình duyệt; đọc tóm tắt |
| So sánh nhiều loại mô hình | AUC: BiomedCLIP 0,743; MedRAX 0,704; Claude 3.7 0,625; CXR-LLaVA sens 0,079 (83 ảnh) | [L1] Front Digit Health 2026 | Bình duyệt; **đọc toàn văn** |
| CAD CNN vượt xa | AUC 0,956 (cùng nguồn MosMedData VII); AUROC 0,92–0,99 (đa trung tâm) | [L45] QIMS 2024; [L19] Radiology 2019 | Bình duyệt |
| CAD có lợi ích lâm sàng | RCT 10.476 người: phát hiện nốt cần xử trí 0,25% → 0,59% | [L20] Radiology 2023 | Bình duyệt |
| Cơ chế: mất tín hiệu tổn thương nhỏ | AUC nốt, bbox-level, từ CLS lên patch-local: BiomedCLIP 0,688 → 0,899; MedSigLIP 0,805 → 0,955 | [L47] arXiv 2026 | Preprint; **đọc toàn văn**; cần oracle ROI |
| Định vị yếu | "Lung lesion": GPT-5 35,7%, MedGemma 14,3%, bác sĩ 85,0% | [L48] ML4H 2025 | Bình duyệt; **đọc toàn văn** |
| Fine-tune có thể làm sụp sensitivity | Fine-tune VLM toàn phần: sens > 65% → < 15% (ChestX-ray14) | [L50] medRxiv 2025 | Preprint |
| LLM văn bản đọc báo cáo tốt | Vicuna gán nhãn báo cáo CXR: AUC trung vị 0,84; GPT-4 tốt hơn NLP chuyên ngành | [L55] Radiology 2023; [L56] AJR 2024 | Bình duyệt |

---

## 4. Vì sao đổi đề tài

| Tiêu chí | Đề tài cũ (VQA, phụ thuộc thị giác) | Đề tài mới (nốt phổi CXR) |
|---|---|---|
| Ý nghĩa lâm sàng | Gián tiếp | Trực tiếp: ung thư phổi, phát hiện sớm |
| Khoảng trống | Bị thu hẹp sau khi tìm thấy Med-CAP (MICCAI 2026) | Rõ: chưa có LoRA VLM cho nốt CXR |
| Dữ liệu | VinDr-CXR-VQA có lỗi nhãn do LLM sinh | Nhãn box gốc của bác sĩ (VinBigData), có tập test ngoài (NIH, JSRT) |
| Tái sử dụng | — | Giữ hạ tầng LoRA, ảnh VinBigData và phép thử che vùng của đề tài cũ |
| Rủi ro | Phụ thuộc chất lượng câu hỏi–đáp | Nốt nhỏ có thể mất khi resize; ca dương hiếm |

---

## 5. Thiết kế đề xuất (tóm tắt spec)

**Mô hình**
- VLM chính: MedGemma-4B (y khoa). VLM so sánh: Qwen2.5-VL-3B (đa năng).
- Baseline: DenseNet-121 classifier (B1), YOLOv8 detector (B2).

**Cấu hình**

| Mã | Cấu hình |
|---|---|
| C0 | Zero-shot |
| C1 | LoRA, chỉ phân loại |
| C2 | LoRA, phân loại và box |
| C3 | C2 + cân bằng lớp |
| C4 | Cấu hình tốt nhất áp lên MedGemma |
| C5 (mở rộng) | VLM có hỗ trợ vùng từ detector |

**Dữ liệu**
- Huấn luyện và test trong phân phối: VinBigData, lớp Nodule/Mass.
- Test ngoài phân phối: NIH ChestX-ray14 và JSRT (có độ mờ 1–5 và kích thước mm).

**Đánh giá**
- AUC, sensitivity ở specificity 0,90/0,95.
- Chỉ số chế độ báo bình thường.
- IoU, FROC.
- Che nốt so với che vùng ngẫu nhiên cùng diện tích.
- Phân tầng theo kích thước và độ mờ.
- Thống kê: bootstrap, McNemar, DeLong.

**Tiêu chí thành công:** kết luận được LoRA có đưa sensitivity của VLM (ở specificity 0,90) lên ≥ 0,60 không, và cải thiện có còn khi kiểm tra ngoài phân phối và khi che nốt không.

---

## 6. Điểm cần nhóm quyết định

> **Đã quyết ngày 2026-11-12:**
> 1. VLM chính là **MedGemma-4B**; Qwen2.5-VL-3B dùng để so sánh.
> 2. C5 chỉ là phần mở rộng.
> 3. Ca dương gộp Nodule/Mass.
> 4. Giữ ngưỡng sensitivity ≥ 0,60 ở specificity 0,90.
>
> Spec đã cập nhật theo các quyết định này.

1. **Mô hình chính:** Qwen2.5-VL-3B (mặc định) hay MedGemma-4B? Bằng chứng hiện có [L1, L48, L49] chưa cho thấy mô hình y khoa tốt hơn mô hình đa năng.
2. **Phạm vi mở rộng:** có cam kết C5 (VLM có hỗ trợ vùng) không, hay chỉ để ở phần hướng phát triển?
3. **Định nghĩa dương tính:** gộp "Nodule/Mass" (mặc định, theo nhãn VinBigData) hay tách nốt < 3 cm và khối ≥ 3 cm? VinBigData không tách được, chỉ JSRT tách được.
4. **Ngưỡng thành công:** sensitivity ≥ 0,60 ở specificity 0,90 có hợp lý không? Mốc tham chiếu: CAD đạt khoảng 0,86 [L21], bác sĩ trên JSRT đạt AUC khoảng 0,77 [L31].

---

## 7. Hạn chế của tổng quan này

- **Mức độ đọc:** chỉ 4/60 bài đọc toàn văn; mọi con số khác lấy từ tóm tắt.
- **Nhật ký tìm kiếm:** bảng truy vấn của agent bị thiếu một phần (dựng lại sau khi agent nén ngữ cảnh). Bảng vẫn có mọi truy vấn đóng góp bài được giữ.
- **Bài chỉ có tiêu đề:** L61 (Kim 2026, JTO, mô hình nguy cơ CXR + LLM) và L62 (commentary trên AJR).
- **Nhãn chưa xác nhận:** chưa xác nhận từ tóm tắt rằng VinDr-CXR, ReXVQA và REVEAL-CXR có nhãn nốt/khối. Riêng VinBigData thì đội đã dùng ở đề tài cũ.
- **Giới hạn của Consensus:** gói miễn phí chỉ trả 3 bài mỗi lần tìm.
- **Phạm vi tuyên bố:** "chưa có ai làm" chỉ đúng trong phạm vi các nguồn đã tìm.

---

## 8. Bước tiếp theo

1. Nhóm duyệt báo cáo này và spec, trả lời 4 điểm ở mục 6.
2. Sau khi spec được duyệt, lập kế hoạch triển khai (writing-plans).
3. Song song: kiểm tra điều khoản dữ liệu (VinBigData, NIH, JSRT), thống kê số ca Nodule/Mass, chạy thử zero-shot trên 50 ảnh.

*Workspace Undermind (danh sách bài xếp hạng, có PDF): https://app.undermind.ai/projects/fbc55738-03be-48ab-b376-2937981a6356*
