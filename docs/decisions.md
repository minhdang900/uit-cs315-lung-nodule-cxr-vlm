# Nhật ký quyết định

File này ghi lại các quyết định thay đổi cách làm của đồ án, kèm lý do và nguồn.

**Quy tắc ghi**
- Mỗi mục có: mã (D-số), ngày (YYYY-MM-DD), quyết định, lý do, nguồn.
- Không sửa mục cũ. Nếu đổi ý, thêm mục mới và ghi "thay thế D-x".
- Ghi ngay khi quyết định, không đợi cuối kỳ.
- Spec có hiệu lực cao nhất: `docs/design/2026-11-12-lung-nodule-cxr-vlm-lora-design.md`. Mục nào ở đây mâu thuẫn với spec thì spec đúng, trừ khi mục đó ghi rõ là sửa spec.

Các mục D1–D27 do nhóm trước ghi lúc bàn giao (2026-11-13), lấy từ spec (§2, §5–§8, §8.5), RUNBOOK và code.

---

## Đề tài và phạm vi

### D1 – 2026-11-12 – Đổi đề tài sang phát hiện nốt phổi trên CXR
- **Quyết định:** bỏ đề cương cũ về VQA và mức phụ thuộc thị giác (2026-11-10); dùng đề tài nốt phổi trên X-quang ngực.
- **Lý do:** tổng quan 90 bài cho thấy khoảng trống rõ: chưa có nghiên cứu LoRA VLM nhỏ cho phát hiện và định vị nốt/khối, so với CNN trên cùng split.
- **Nguồn:** spec, phần đầu và §3.

### D2 – 2026-11-12 – VLM chính là MedGemma-4B-it
- **Quyết định:** MedGemma-4B-it là VLM chính; Qwen2.5-VL-3B-Instruct là VLM thứ hai (zero-shot và C4).
- **Lý do:** mô hình y khoa mở mới nhất; zero-shot sensitivity rất thấp (5,2%) với specificity 100%, là ví dụ điển hình của chế độ báo bình thường. Qwen2.5-VL vốn biết xuất box, dùng để so y khoa với đa năng.
- **Nguồn:** spec §5.1.

### D3 – 2026-11-12 – Gộp Nodule và Mass thành một lớp dương
- **Quyết định:** lớp dương là "Nodule/Mass".
- **Lý do:** ca dương rất hiếm (khoảng 2,73% tập train VinBigData); gộp để tăng cỡ mẫu.
- **Nguồn:** spec, phần "Quyết định của nhóm".

### D4 – 2026-11-12 – Tiêu chí thành công: sensitivity ≥ 0,60 ở specificity 0,90
- **Quyết định:** chỉ số chính là sensitivity ở specificity 0,90 (ngưỡng chọn trên validation); mục tiêu 0,60 trên VinBigData test.
- **Lý do:** cùng khoảng với sensitivity người đọc trong các nghiên cứu làm giàu ca bệnh, nhưng các nghiên cứu đó không đo ở spec 0,90. Vì vậy báo cáo **không** viết "ngang bác sĩ".
- **Nguồn:** spec §2.

### D5 – 2026-11-12 – C5, C6 chỉ là phần mở rộng
- **Quyết định:** C5 (VLM nhận box của B2) và C6 (chia ô ảnh) không thuộc cam kết.
- **Lý do:** ngân sách Colab và thời gian có hạn.
- **Nguồn:** spec §5.2.

### D6 – 2026-11-13 – Đổi tên đề tài sang dạng ứng dụng
- **Quyết định:** tên mới "Trợ lý AI phát hiện và định vị nốt phổi trên X-quang ngực: tinh chỉnh LoRA mô hình ngôn ngữ–thị giác cỡ nhỏ". Nội dung, phạm vi, tiêu chí không đổi.
- **Lý do:** tên mang tính ứng dụng dễ hiểu hơn. "Trợ lý" là nguyên mẫu nghiên cứu, không phải công cụ lâm sàng.
- **Nguồn:** spec, mục "Tên đề tài".

## Dữ liệu

### D7 – 2026-11-12 – Vai trò ba bộ dữ liệu
- **Quyết định:** VinBigData cho train/val/test trong phân phối; NIH ChestX-ray14 (tập con có box) là test ngoài phân phối 1; JSRT là test ngoài phân phối 2 (có độ mờ 1–5).
- **Lý do:** cần một tập lớn có box để huấn luyện, và hai tập ngoài phân phối để kiểm tra khả năng tổng quát.
- **Nguồn:** spec §6.

### D8 – 2026-11-12 – Không dùng NODE21 làm test
- **Quyết định:** NODE21 chỉ để tham khảo.
- **Lý do:** NODE21 trùng nguồn với JSRT và NIH, dễ rò rỉ.
- **Nguồn:** spec §6.

### D9 – 2026-11-12 – Chia VinBigData 70/10/20 theo image_id, phân tầng theo nhãn
- **Quyết định:** chia theo `image_id`, phân tầng theo Nodule/Mass, `--seed 0` trong `prepare_vinbig.py`. Nếu dùng tập test chính thức VinDr-CXR (`--official-test-csv`), chia VinBigData 85/15 (train/val) và dùng tập chính thức làm test.
- **Lý do:** ngưỡng và siêu tham số chọn trên val; mỗi tập test dùng một lần.
- **Hạn chế (rủi ro rò rỉ):** bản Kaggle không có mã bệnh nhân, nên đây **không** phải chia theo bệnh nhân. Nếu một bệnh nhân có nhiều ảnh, ảnh của họ có thể nằm ở cả train và test, làm kết quả trên VinBigData test đẹp hơn thực tế. Báo cáo phải nêu điều này thành hạn chế và không được mô tả cách chia là theo bệnh nhân.
- **Nguồn:** spec §6; `code/scripts/prepare_vinbig.py`.

### D10 – 2026-11-13 – Định nghĩa ca dương/âm trong code
- **Quyết định:**
  - VinBigData: ảnh dương nếu ít nhất một bác sĩ khoanh Nodule/Mass; box của các bác sĩ được hợp nhất bằng weighted boxes fusion (iou_thr 0,4). Mọi ảnh còn lại là âm (kể cả ảnh có bệnh khác).
  - NIH: ảnh dương là ảnh có box Nodule hoặc Mass; ảnh âm là ảnh "No Finding" của bệnh nhân không có ảnh dương, gấp 3 lần (`--neg-ratio 3`, seed 0).
  - JSRT: 154 ảnh có nốt là dương, 93 ảnh không nốt là âm; box vuông dựng từ tâm và đường kính nốt.
- **Lý do:** spec §6 cho phép thêm ca có bệnh khác làm "âm khó"; code chọn cách này. Số ca dương VinBigData thực tế có thể khác con số 410 trong spec; ghi số thật ở E0.
- **Nguồn:** `code/src/nodulecxr/data/{vinbig,nih,jsrt}.py`. **Nhóm mới cần xác nhận** điểm "mọi ảnh còn lại là âm" khi viết phần dữ liệu của báo cáo.

### D11 – 2026-11-12 – Tiền xử lý ảnh về 896×896
- **Quyết định:** mọi ảnh đổi về ảnh xám 896×896 (kéo giãn, không đệm), tọa độ box quy đổi theo hệ số từng chiều.
- **Lý do:** MedGemma nhận ảnh 896 px; đây cũng là độ phân giải tối ưu cho CAD nốt theo [L83].
- **Nguồn:** spec §5.1, §6; `code/src/nodulecxr/images.py`, `records.py`.

## Mô hình và huấn luyện

### D12 – 2026-11-12 – Cấu hình LoRA
- **Quyết định:** r = 16, α = 32, dropout 0,05; gắn vào q/k/v/o/gate/up/down_proj của LLM; bộ mã hóa ảnh và lớp chiếu đông cứng.
- **Lý do:** cấu hình phổ biến cho VLM cỡ nhỏ; câu hỏi nghiên cứu là liệu không gian hạng thấp có đủ khi bộ mã hóa ảnh đông cứng.
- **Nguồn:** spec §5.1, §5.4; `code/src/nodulecxr/train/lora.py`, `train/sft_data.py`.

### D13 – 2026-11-12 – Prompt cố định, đáp án sinh tự động, không ngữ cảnh lâm sàng
- **Quyết định:** một câu hỏi cố định; đáp án mẫu sinh từ nhãn gốc (`{"nodule":"no","boxes":[]}` cho ca âm); không dùng câu hỏi–đáp do LLM sinh; không đưa ngữ cảnh lâm sàng vào prompt trong điều kiện chính.
- **Lý do:** tránh nhiễu nhãn (đã gặp ở đề tài cũ) và tránh "lộ" đáp án qua ngữ cảnh [L4].
- **Nguồn:** spec §5.3.

### D14 – 2026-11-12 – C3 = C2 + cân bằng lớp
- **Quyết định:** oversampling ca dương đến 1:1 (`--balance`) và trọng số loss `--pos-weight 2.0` cho ca dương. Giá trị 2,0 cố định (RUNBOOK), **không** tinh chỉnh trên val.
- **Lý do:** chống chế độ báo bình thường khi ca dương hiếm.
- **Nguồn:** spec §5.2; `code/RUNBOOK.md` E3.

### D15 – 2026-11-13 – Trọng số loss C3 áp lên mọi token đáp án
- **Quyết định:** w₊ nhân vào mọi token đáp án của ca dương (cả token box). Loss chuẩn hóa theo tổng số token nhãn của cả cửa sổ tích lũy gradient.
- **Lý do:** để w₊ có tác dụng khi batch = 1.
- **Nguồn:** spec §8.5.

### D16 – 2026-11-13 – Siêu tham số mặc định của các script
- **Quyết định:** LoRA: batch 1, grad-accum 8, lr 1e-4, cosine, warmup 3%, 3 epoch, 4 ảnh âm mỗi ảnh dương (`--neg-per-pos 4.0`). DenseNet-121 (torchxrayvision `densenet121-res224-all`): 224 px, batch 32, Adam lr 1e-4, 10 epoch, giữ epoch có val AUC cao nhất. YOLOv8s: imgsz 896, batch 16, 50 epoch, `--neg-per-pos 4.0`, dùng `best.pt`.
- **Lý do:** giá trị mặc định khi triển khai; chưa tinh chỉnh vì chưa chạy GPU.
- **Nguồn:** `code/scripts/train_lora.py`, `train_densenet.py`, `train_yolo.py`.

### D17 – 2026-11-13 – Chỉ chạy MedGemma trên GPU có bf16 thật
- **Quyết định:** chỉ dùng L4 hoặc A100 (compute capability ≥ 8). Code từ chối tải MedGemma nếu không có bf16 thật.
- **Lý do:** fp16 làm MedGemma tràn số; T4 (7.5) chỉ giả lập bf16.
- **Nguồn:** spec §8.5; `code/src/nodulecxr/vlm/hf.py`.

## Đánh giá

### D18 – 2026-11-13 – Điểm liên tục của VLM từ log-prob
- **Quyết định:** so log-prob của hai chuỗi ép buộc `{"nodule": "yes` và `{"nodule": "no` (không có dấu `"` đóng), tính từ token đầu tiên hai chuỗi khác nhau, rồi chuẩn hóa thành P(yes).
- **Lý do:** cần điểm liên tục cho ROC, AUC và chọn ngưỡng ở spec 0,90.
- **Nguồn:** spec §8.1, §8.5; `code/src/nodulecxr/vlm/hf.py`, `vlm/scoring.py`.

### D19 – 2026-11-13 – Xử lý câu trả lời không parse được
- **Quyết định:** `raw_sens`/`raw_spec` coi câu không parse được là "no". Tỷ lệ "no" trên ca dương và "yes" trên ca âm chỉ tính trên câu parse được; báo riêng `unparsed`. Báo cáo tính lại từ `raw_text` bằng parser hiện hành.
- **Lý do:** tách lỗi định dạng khỏi lỗi nhận diện; sửa parser không cần chạy lại mô hình.
- **Nguồn:** spec §8.5; `code/src/nodulecxr/eval/report.py`.

### D20 – 2026-11-13 – Chỉ số định vị
- **Quyết định:** IoU trung bình và "đúng và IoU ≥ 0,3/0,5" chỉ dùng box có điểm ≥ t90 (box của VLM mang điểm của ảnh). FROC dùng mọi box, IoU ≥ 0,3. JSRT: box điểm cao nhất, "trúng" khi tâm nằm trong bán kính nốt + 5 mm (12,5 px trên ảnh 896); khoảng cách đổi ra mm với 0,4 mm/px.
- **Lý do:** ngưỡng IoU 0,3 vì nốt nhỏ; JSRT chỉ có tâm và đường kính.
- **Nguồn:** spec §8.2, §8.5.

### D21 – 2026-11-13 – C0: bảng chính dùng run 0 (nhiệt độ 0)
- **Quyết định:** dòng chính của C0 dùng lần chạy nhiệt độ 0. Hai lần nhiệt độ 0,7 chỉ dùng đo độ ổn định (`run_agreement.csv`). C0 sinh tối đa 256 token.
- **Lý do:** kết quả tất định cho bảng chính; 96 token dễ cắt cụt JSON của mô hình zero-shot.
- **Nguồn:** spec §8.5; `code/RUNBOOK.md` E2.

### D22 – 2026-11-13 – Chọn C4 và cấu hình cho E5 chỉ trên validation
- **Quyết định:** cấu hình LoRA tốt nhất = cấu hình C1–C3 có trung bình `val_sens90` qua seed cao nhất trong `outputs/report/val_selection.csv`. Không bao giờ chọn bằng số trên test. Các dòng đường học (`<cfg>-ep<k>`) không tham gia chọn.
- **Lý do:** tránh rò rỉ từ tập test.
- **Nguồn:** spec §8.5; `code/RUNBOOK.md` E4, E5.

### D23 – 2026-11-13 – Kiểm toán che vùng
- **Quyết định:** T1 che box nới 10% (5% mỗi cạnh) bằng giá trị xám trung bình. T2 che vùng cùng kích thước, không giao T1, ≥ 90% trong mặt nạ phổi (PSPNet của torchxrayvision; dự phòng mặt nạ heuristic). Δ_occ = flip_T1 − flip_T2 trên tập ảnh tạo được T2 (bắt cặp); báo kèm `n_t2_ok` và `flip_t1_all`. Chạy cho C0-medgemma, LoRA tốt nhất (seed 0), B2 (seed 0), trên `test` và `test_jsrt`.
- **Lý do:** so sánh công bằng giữa T1 và T2 trên cùng ảnh.
- **Nguồn:** spec §8.3, §8.5; `code/RUNBOOK.md` E5.

### D24 – 2026-11-13 – Phân tầng kích thước
- **Quyết định:** VinBigData/NIH theo tertile diện tích box tương đối (ngưỡng từ ca dương train VinBigData, lưu trong `e0_data_edges.json`). JSRT theo mm: ≤ 10, 10–20, > 20 mm.
- **Lý do:** VinBigData/NIH không có kích thước mm.
- **Nguồn:** spec §8.5.

### D25 – 2026-11-13 – Khoảng tin cậy và kiểm định
- **Quyết định:** bootstrap 1.000 lần, phân tầng theo nhãn, theo ảnh. McNemar (exact) cho sens/spec, DeLong cho AUC, Holm khi so nhiều cặp (seed 0, run 0). Mean ± sd qua seed.
- **Lý do:** tập test nhỏ; cần lượng hóa bất định. Hạn chế: không gộp theo bệnh nhân nên CI có thể hẹp hơn thực tế.
- **Nguồn:** spec §8.4, §8.5; `code/src/nodulecxr/eval/report.py`, `metrics/stats.py`.

## Ngân sách và quy trình

### D26 – 2026-11-12 – Quy tắc cắt giảm ngân sách
- **Quyết định:** lượt LoRA đầu tiên dùng để đo thời gian. Nếu một lượt vượt 3 giờ, cắt theo thứ tự: (1) `--neg-per-pos` 4 → 2; (2) giảm độ phân giải, chỉ cho Qwen qua `min_pixels`/`max_pixels` trong `vlm/hf.py` (không áp dụng cho MedGemma); (3) `--epochs 2`; (4) C1 chỉ 1 seed. Ghi mọi quyết định vào `outputs/budget.md` và tóm tắt vào file này.
- **Lý do:** ngân sách Colab trả phí có hạn.
- **Nguồn:** spec §7, §8.5; `code/RUNBOOK.md` mục "Ngân sách".

### D27 – 2026-11-13 – Repo và quy trình Git khi bàn giao
- **Quyết định:** cả thư mục môn học thành repo private `minhdang900/uit-cs315-lung-nodule-cxr-vlm`. Nhánh: `main` (ổn định, có tag; `v0.1.0-handoff` là trạng thái bàn giao), `develop` (tích hợp), `feature/*`, `exp/*`, `docs/*`. PR vào `develop`, review, rồi squash hoặc merge. Phát hành = PR `develop` → `main` + tag. Conventional commits. Không commit dữ liệu, trọng số, `outputs/`, token, PDF có bản quyền.
- **Lý do:** nhóm mới cần quy trình rõ ràng để làm song song và giữ lịch sử.
- **Nguồn:** `docs/handoff/so_tay_ban_giao.pdf` §7.

---

## Mẫu cho mục mới

```markdown
### D28 – YYYY-MM-DD – <quyết định ngắn>
- **Quyết định:** ...
- **Lý do:** ...
- **Nguồn:** <spec §..., PR #..., outputs/... (ngày chạy)>
```
