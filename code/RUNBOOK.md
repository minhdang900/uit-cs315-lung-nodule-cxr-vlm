# RUNBOOK (Colab, GPU L4 hoặc A100; không dùng T4 vì MedGemma cần bf16)

MedGemma từ chối tải model nếu không có GPU hỗ trợ bf16 (`load_base` ném `RuntimeError`), nên chỉ dùng L4 hoặc A100.

## Trạng thái và việc cần làm (cập nhật 2026-11-13, bản `v0.2.0`)

Đây là danh sách việc để nhóm làm tiếp. Làm theo thứ tự. Khi xong một mục, đổi `[ ]` thành `[x]`, ghi ngày, người làm và file kết quả. Commit theo quy trình nhánh: `exp/<E#-...>` hoặc `docs/...` → PR vào `develop`. Quyết định mới ghi vào `docs/decisions.md`.

**Chú thích:** ✅ xong · ⏳ việc tiếp theo · ⬜ chưa làm · 🔁 làm lại khi cần

### A. Đã xong (người bàn giao)
| # | Hạng mục | Trạng thái | Kết quả / nơi xem |
|---|---|---|---|
| A1 | Pipeline `nodulecxr` (dữ liệu, VLM, LoRA, baseline, chỉ số, kiểm toán, báo cáo) | ✅ | `code/`; `pytest -q` → 79 passed (CPU) |
| A2 | Repo GitHub, quy trình nhánh, báo cáo LaTeX, sổ tay bàn giao, nhật ký quyết định | ✅ | `report/main.pdf`, `docs/handoff/`, `docs/decisions.md` |
| A3 | E0: tải dữ liệu thô | ✅ | `data/raw/` (8,4 GB, trên Drive của GVHD, không vào git); nguồn: `docs/decisions.md` D28–D29 |
| A4 | E0: tạo ảnh 896 px + manifest, kiểm tra box bằng mắt | ✅ | `data/processed/` (4,9 GB): VinBigData 15.000 (826 dương), NIH 656 (164), JSRT 247 (154) |
| A5 | E0: thống kê dữ liệu | ✅ | `outputs/e0_data.md`, `outputs/e0_data_edges.json` (trên Drive; tạo lại được, xem B3) |

### B. Chuẩn bị (mỗi thành viên, làm một lần)
| # | Việc | Trạng thái | Cách làm |
|---|---|---|---|
| B1 | Có quyền đọc thư mục `data/` (và `outputs/`) do GVHD chia sẻ trên Google Drive | ⏳ | Mở link chia sẻ → *Organize → Add shortcut* vào `MyDrive`, đặt tên `cs315-data` (xem §0) |
| B2 | Clone repo vào Drive, tạo token Hugging Face, xin quyền `google/medgemma-4b-it`, lưu `HF_TOKEN` vào Colab Secrets | ⏳ | §0; sổ tay chương 5 |
| B3 | Tạo lại `$O/e0_data_edges.json` trong bản clone của mình (cần cho `make_report.py`, `run_audit.py`) | ⏳ | Chạy lệnh `data_stats.py` ở mục E0 (vài giây, CPU) |
| B4 | Colab Pro với GPU L4 hoặc A100 | ⏳ | Không dùng T4 |

### C. Thí nghiệm (theo thứ tự; xem từng mục bên dưới)
| # | Việc | Trạng thái | Đầu ra phải có | Điền vào báo cáo |
|---|---|---|---|---|
| C1 | 7 mục "Kiểm tra bắt buộc trên Colab" | ⏳ **làm đầu tiên** | ghi kết quả từng mục vào `$O/budget.md` | — |
| C2 | E1: B1 DenseNet, B2 YOLOv8 × 3 seed | ⬜ | `$O/preds/B1.jsonl`, `B2.jsonl`, `$O/yolo/seed*/weights/best.pt` | Bảng kết quả chính |
| C3 | E2: C0 zero-shot MedGemma, Qwen (run 0 ở nhiệt độ 0; run 1–2 ở 0,7) | ⬜ | `$O/preds/C0-medgemma.jsonl`, `C0-qwen.jsonl` | Bảng chính, độ ổn định |
| C4 | E3: pilot 160 mẫu (loss đầu ≈ 0,5–3, thời gian) | ⬜ | `timing.json`, log; ghi vào `$O/budget.md` | — |
| C5 | E3: LoRA MedGemma C1, C2, C3 × 3 seed + dự đoán | ⬜ | `$O/lora/C*-s*/final`, `$O/preds/C1.jsonl` … `C3.jsonl` | Bảng chính |
| C6 | E3: đường học seed 0 (dự đoán từng checkpoint) | ⬜ | `$O/preds/<cfg>-ep<k>.jsonl` (**để riêng**, xem lưu ý D22) | Mục chế độ báo bình thường |
| C7 | `make_report.py` lần 1 → chọn cấu hình tốt nhất theo trung bình `val_sens90` (chỉ C1–C3) | ⬜ | `$O/report/val_selection.csv`, `thresholds.json`; ghi lựa chọn vào `docs/decisions.md` | — |
| C8 | E4: C4 = cấu hình tốt nhất trên Qwen2.5-VL-3B (1–3 seed) | ⬜ | `$O/preds/C4.jsonl` | Bảng chính |
| C9 | E5: kiểm toán che vùng (C0-medgemma, LoRA tốt nhất, B2) trên `test` và `test_jsrt` | ⬜ | `$O/audit/*.json` (`delta`, `n_t2_ok`) | Bảng kiểm toán |
| C10 | `make_report.py` lần cuối, điền mọi ô `[--]` trong `report/`, viết kết quả và thảo luận | ⬜ | `$O/report/summary.md` …; `report/main.pdf` biên dịch không lỗi | Toàn bộ chương 4–5 |

### D. Phân tích thêm và việc tồn đọng
| # | Việc | Trạng thái | Ghi chú |
|---|---|---|---|
| D1 | Phân tích phụ: chỉ các ca dương VinBigData có ≥ 2 bác sĩ đồng ý (405/826 ca: 224 có 2 bác sĩ, 181 có 3 bác sĩ) | ⬜ | Nhãn 1 bác sĩ nhiễu hơn (D29). Báo như phân tích độ nhạy, không thay bảng chính |
| D2 | Nêu hạn chế: VinBigData chia theo `image_id` (không có mã bệnh nhân), bootstrap theo ảnh | ⬜ | Đã có trong báo cáo; kiểm lại khi viết thảo luận |
| D3 | Script huấn luyện không chạy tiếp khi Colab ngắt | 🔁 | Chạy lại cả lượt; `predict_vlm.py` thì tự bỏ qua ảnh đã xong |
| D4 | `draw_boxes.py` đặt tên ảnh NIH thành `*.png.png` | ⬜ | Lỗi hiển thị, không ảnh hưởng dữ liệu; sửa trên nhánh `feature/...` nếu muốn |

**Không bao giờ:** chọn cấu hình hay ngưỡng theo tập test; commit dữ liệu, trọng số, `outputs/` hay token; dùng GPU T4 cho MedGemma.

## Kiểm tra bắt buộc trên Colab trước khi chạy đủ
Làm lần lượt, trước E1–E5. Mục nào không đạt thì dừng và sửa, đừng chạy tiếp.

1. **Tokenizer cho điểm yes/no.** Với mỗi model (`medgemma`, `qwen`), chạy:
   ```python
   from nodulecxr.vlm.hf import load_base
   from nodulecxr.vlm.chat import build_chat_text
   from nodulecxr.prompts import QUESTION_LOC, ANSWER_PREFIX, build_target
   _, processor = load_base("medgemma")   # lặp lại với "qwen"
   tok = processor.tokenizer
   base = build_chat_text(processor, QUESTION_LOC) + ANSWER_PREFIX
   for w in ("yes", "no"):
       print(w, tok.convert_ids_to_tokens(tok(base + w, add_special_tokens=False)["input_ids"])[-8:])
   tgt = build_chat_text(processor, QUESTION_LOC, build_target(1, [[1, 2, 3, 4]]))
   print("target", tok.convert_ids_to_tokens(tok(tgt, add_special_tokens=False)["input_ids"]))
   ```
   Xác nhận: (a) các token sau điểm phân kỳ giữa `yes` và `no` chỉ là chính từ trả lời, không có dấu nháy bị gộp (ví dụ `"yes` hay `yes"` là sai); (b) các token của target huấn luyện, ngay sau `ANSWER_PREFIX`, bắt đầu bằng đúng token trả lời đó (`yes`).
2. **Một lần `build_example` cho mỗi model trên ảnh thật.**
   ```python
   from nodulecxr.records import read_manifest
   from nodulecxr.images import load_image
   from nodulecxr.train.sft_data import build_example
   df = read_manifest("/content/data/manifest_vinbig.csv"); r = df[df.label == 1].iloc[0]
   ex = build_example(processor, load_image("/content/data", r.path), QUESTION_LOC, build_target(1, [[1, 2, 3, 4]]))
   print({k: v.shape for k, v in ex.items()})
   ```
   Kiểm tra tiền tố của Gemma (prompt phải là tiền tố của prompt+answer) không ném `ValueError`.
3. **Loss đầu tiên của pilot 160 mẫu.** Chạy `train_lora.py ... --max-train 160 --epochs 1` (xem E3; 160 mẫu với grad-accum 8 cho 20 bước tối ưu nên `loss` được ghi ở bước 10 và 20). Dòng log đầu tiên (`logging_steps=10`) phải có `loss` khoảng 0,5–3. Nếu ~10–25 nghĩa là gradient accumulation bị chia hai lần: dừng và sửa. Đọc thêm `train_loss` cuối cùng trong log/`trainer_state.json`.
4. **Loại GPU.**
   ```python
   import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
   ```
   Phải là L4 hoặc A100, và số đầu của `get_device_capability()` phải ≥ 8 (bf16 thật). Không dựa vào `torch.cuda.is_bf16_supported()`: T4 (7.5) cũng trả `True` nhờ giả lập.
5. **Smoke YOLO 1 epoch.** Chạy:
   ```python
   from ultralytics import YOLO
   m = YOLO("yolov8s.pt"); m.train(data="<work>/yolo_data_seed0/data.yaml", imgsz=896, epochs=1, project="/content/yolo_smoke", name="s", exist_ok=True)
   print(m.trainer.best)   # file phải tồn tại
   ```
   Cách thay thế, dùng đủ cờ bắt buộc của `train_yolo.py`:
   ```bash
   %%bash
   export M=/content/data D=/content/data O=../outputs
   python scripts/train_yolo.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --data-root $D --epochs 1 --work /content/yolo_smoke --out /content/smoke.jsonl
   ls /content/yolo_smoke/seed0/weights/best.pt
   ```
   Lưu ý: lệnh thay thế này sau khi train còn dự đoán **toàn bộ** các split đánh giá (val, test, test_nih, test_jsrt), nên chậm; có thể dừng ô sau khi `best.pt` đã xuất hiện.
6. **Thời gian mỗi ảnh của mặt nạ phổi trong `run_audit.py`.** PSPNet của torchxrayvision chạy trên CPU. Đo trên 10 ảnh:
   ```python
   import sys, time; sys.path.insert(0, "scripts")
   from run_audit import lung_mask_xrv
   from nodulecxr.records import read_manifest
   from nodulecxr.images import load_image
   df = read_manifest("/content/data/manifest_jsrt.csv").head(10)
   lung_mask_xrv(load_image("/content/data", df.iloc[0].path))   # khởi động, không tính giờ
   t = time.time()
   for r in df.itertuples(): lung_mask_xrv(load_image("/content/data", r.path))
   print((time.time() - t) / len(df), "giây/ảnh")
   ```
   Nếu quá chậm, dùng `--lung-mask heuristic` và ghi quyết định đó vào `$O/budget.md`.
7. **Thời gian suy luận mỗi ảnh của VLM.** Với mỗi model, chạy 50 ảnh val (cấu hình `pilot`, ghi ra ngoài `$O/preds/` để không lẫn vào báo cáo):
   ```bash
   for m in medgemma qwen; do
     python scripts/predict_vlm.py --model $m --manifests $M/manifest_vinbig.csv --splits val --limit 50 --data-root $D --config pilot --max-new-tokens 256 --out /content/pilot_$m.jsonl
   done
   ```
   Đọc tốc độ ở thanh tqdm khi chạy xong (`s/it`, mỗi `it` là một ảnh, gồm cả tính điểm yes/no và sinh văn bản; không tính thời gian tải model) làm giây/ảnh, ghi vào `$O/budget.md`. Ngoại suy chi phí C0: giây/ảnh × số ảnh của `val test test_nih test_jsrt` × 3 lần chạy × 2 model. Nếu vượt ngân sách, xem mục "Ngân sách" bên dưới.

## 0. Mỗi phiên Colab
Lần đầu tiên: clone repo vào Google Drive của bạn (một lần duy nhất; repo public nên không cần token):
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone https://github.com/minhdang900/uit-cs315-lung-nodule-cxr-vlm.git
```
Mỗi phiên. Đổi `REPO` nếu bạn clone vào chỗ khác. `DATA` là thư mục chứa `processed/`: dùng shortcut `cs315-data` tới thư mục `data/` mà GVHD chia sẻ (mục B1), hoặc `f"{REPO}/data"` nếu bạn tự tạo dữ liệu trong bản clone. Dữ liệu không bao giờ vào git.
```python
from google.colab import drive, userdata
drive.mount('/content/drive')
REPO = "/content/drive/MyDrive/uit-cs315-lung-nodule-cxr-vlm"
DATA = "/content/drive/MyDrive/cs315-data"        # hoặc f"{REPO}/data"
%cd {REPO}/code
import os; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # lấy từ Colab Secrets
!pip install -q -e ".[gpu]"
!huggingface-cli login --token $HF_TOKEN
!mkdir -p /content/data ../outputs && rsync -a "{DATA}/processed/" /content/data/   # ảnh + manifest_*.csv sang ổ local (~4,9 GB)
```
Biến shell không giữ qua các dòng `!`, nên mọi ô chạy lệnh dưới đây phải bắt đầu bằng `%%bash` và `export` lại biến:
```bash
%%bash
export M=/content/data D=/content/data O=../outputs
# HF_TOKEN đã nằm trong os.environ (đặt ở ô Python trên), ô %%bash kế thừa nó
```
Biến dùng chung: `M=/content/data` (chứa `manifest_*.csv`), `D=/content/data` (ảnh), `O=../outputs` (kết quả, trên Drive trong bản clone, không vào git). Các khối lệnh bash bên dưới là nội dung ô `%%bash` (sau dòng `export` ở trên).

## E0. Dữ liệu — ✅ đã xong (2026-11-13)
Dữ liệu đã có trong `data/raw/` và `data/processed/` (xem bảng trạng thái và `docs/decisions.md` D28–D29). Mỗi bản clone mới chỉ cần chạy lại `data_stats.py` (dòng thứ hai của khối lệnh bên dưới) để có `$O/e0_data_edges.json`.

Nếu phải tạo lại từ đầu (chạy ở máy có dữ liệu thô, từ `code/`, với `M=../data/processed`): tải bằng `bash scripts/download_nih.sh ../data/raw/nih`, `bash scripts/download_jsrt.sh ../data/raw/jsrt`, còn VinBigData thì tải từ Kaggle (sổ tay §6.5). Sau đó chạy 3 script `prepare_*` (cờ bắt buộc: `prepare_vinbig.py --train-csv ... --out-root $M` cùng `--png-dir`/`--meta-csv` hoặc `--dicom-dir`; `prepare_nih.py --raw-dir ... --out-root $M`; `prepare_jsrt.py --raw-dir ... --out-root $M`), rồi:
```bash
for d in vinbig nih jsrt; do python scripts/draw_boxes.py --manifest $M/manifest_$d.csv --data-root $M --out $O/qa/$d -n 8; done
python scripts/data_stats.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --out $O/e0_data.md
```
Kiểm tra `outputs/qa/` bằng mắt.
Kết quả: `$O/e0_data.md`, `$O/e0_data_edges.json`.

## E1. Baseline CNN (3 seed mỗi loại)
```bash
for s in 0 1 2; do
  python scripts/train_densenet.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --data-root $D --seed $s --out $O/preds/B1.jsonl
  python scripts/train_yolo.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --data-root $D --seed $s --work /content/yolo --out $O/preds/B2.jsonl
  mkdir -p $O/yolo/seed$s/weights && cp /content/yolo/seed$s/weights/best.pt $O/yolo/seed$s/weights/best.pt
done
```
B2 làm việc trên ổ local `/content/yolo` (xuất ảnh YOLO lên Drive rất chậm): dữ liệu YOLO riêng cho từng seed nằm ở `/content/yolo/yolo_data_seed{seed}`, trọng số ở `/content/yolo/seed{seed}/weights/best.pt`. Dòng `cp` chép `best.pt` về `$O/yolo/seed{seed}/weights/best.pt` trên Drive (E5 dùng file này); `/content` mất khi phiên Colab kết thúc.

## E2. Zero-shot C0 (MedGemma, Qwen; 3 lần: nhiệt độ 0 một lần, 0,7 hai lần)
```bash
for m in medgemma qwen; do
  python scripts/predict_vlm.py --model $m --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --splits val test test_nih test_jsrt --data-root $D --config C0-$m --run 0 --temperature 0 --max-new-tokens 256 --out $O/preds/C0-$m.jsonl
  for r in 1 2; do
    python scripts/predict_vlm.py --model $m --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --splits val test test_nih test_jsrt --data-root $D --config C0-$m --run $r --temperature 0.7 --max-new-tokens 256 --out $O/preds/C0-$m.jsonl
  done
done
```
Script tự bỏ qua ảnh đã chạy, nên nếu Colab ngắt kết nối thì chỉ cần chạy lại cùng lệnh. `--max-new-tokens 256` vì model zero-shot hay trả lời dài, 96 token dễ cắt cụt JSON.
Bảng chính (`summary.csv`, `summary.md`) chỉ dùng run 0 (nhiệt độ 0); run 1–2 chỉ dùng cho `run_agreement.csv` (độ ổn định) và vẫn có trong `summary_per_seed.csv`.

## E3. LoRA MedGemma C1–C3 (3 seed)
Pilot đo thời gian và loss đầu tiên (kiểm tra mục 3 ở trên): thêm `--max-train 160 --epochs 1` vào một lệnh `train_lora.py` rồi đọc `timing.json`, log (`loss` ở bước 10 và 20) và `train_loss` cuối.
```bash
for s in 0 1 2; do
  python scripts/train_lora.py --model medgemma --mode cls --manifest $M/manifest_vinbig.csv --data-root $D --seed $s --out $O/lora/C1-s$s
  python scripts/train_lora.py --model medgemma --mode loc --manifest $M/manifest_vinbig.csv --data-root $D --seed $s --out $O/lora/C2-s$s
  python scripts/train_lora.py --model medgemma --mode loc --balance --pos-weight 2.0 --manifest $M/manifest_vinbig.csv --data-root $D --seed $s --out $O/lora/C3-s$s
done
```
Dự đoán:
- C1 dùng `--prompt cls`. Nên chạy `--generate all --max-new-tokens 16` (câu trả lời cls ngắn) để C1 có các tỷ lệ chế độ bình thường (`raw_sens`, `no_on_pos`, `yes_on_neg`, `unparsed`); `--generate none` chỉ cho điểm và để các cột đó trống (NaN).
- C2 và C3 dùng `--prompt loc --generate all`.
- Quy tắc chung: mọi lệnh dẫn xuất từ một cấu hình (C4, đường học, E5) dùng đúng `--prompt`/`--generate` của cấu hình nguồn (`--prompt cls` nếu nguồn là C1, `--prompt loc` nếu là C2/C3).
- Mỗi lệnh chạy trên `val test test_nih test_jsrt`, với `--adapter $O/lora/<cfg>-s$s/final --config <cfg> --seed $s`, ghi vào `$O/preds/<cfg>.jsonl`.

Ví dụ cho C3:
```bash
for s in 0 1 2; do
  python scripts/predict_vlm.py --model medgemma --adapter $O/lora/C3-s$s/final --prompt loc --generate all --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --splits val test test_nih test_jsrt --data-root $D --config C3 --seed $s --out $O/preds/C3.jsonl
done
```

Đường học (sensitivity theo epoch): chạy `predict_vlm.py --splits val --generate none` cho từng `checkpoint-*` của seed 0, với `--config <cfg>-ep<k>` và cùng `--prompt` của cấu hình nguồn (`--prompt cls` cho C1, `--prompt loc` cho C2/C3).

## E4. C4: cấu hình tốt nhất trên Qwen2.5-VL-3B (1–3 seed)
Chọn cấu hình tốt nhất trong C1–C3 theo **trung bình `val_sens90` qua các seed** trong `$O/report/val_selection.csv` (sensitivity trên val tại ngưỡng `t90` của đúng lần chạy đó; `val_auc` để tham khảo). Không bao giờ chọn từ các cột của tập test. Chạy `make_report.py` trước để có file này. Chạy lại `train_lora.py --model qwen` với đúng các cờ của cấu hình đó, rồi dự đoán với `--model qwen --config C4` và đúng `--prompt`/`--generate` (cùng `--max-new-tokens`) của cấu hình nguồn.

## Báo cáo
```bash
python scripts/make_report.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --preds "$O/preds/*.jsonl" --edges-json $O/e0_data_edges.json --out $O/report
```

## E5. Kiểm toán che vùng (C0-medgemma, cấu hình LoRA tốt nhất, B2), trên `test` và `test_jsrt`
"Cấu hình LoRA tốt nhất" = cấu hình C1–C3 có trung bình `val_sens90` qua seed cao nhất trong `$O/report/val_selection.csv` (giống E4; không chọn theo test). Dùng adapter seed 0 của cấu hình đó và `--prompt` của nó (`--prompt cls` nếu là C1; mặc định `loc` cho C2/C3).
Lấy ngưỡng `t90` từ `$O/report/thresholds.json` (khóa dạng `config|seed|run`), rồi chạy (`--manifest` theo split: `manifest_vinbig.csv` cho `test`, `manifest_jsrt.csv` cho `test_jsrt`):
```bash
# LoRA tốt nhất (ví dụ C3, seed 0; nếu là C1 thì thêm --prompt cls)
python scripts/run_audit.py --scorer vlm --model medgemma --adapter $O/lora/C3-s0/final --manifest $M/manifest_jsrt.csv --split test_jsrt --data-root $D --threshold <t90> --edges-json $O/e0_data_edges.json --out $O/audit/C3_jsrt.json
# C0 zero-shot (không có --adapter)
python scripts/run_audit.py --scorer vlm --model medgemma --manifest $M/manifest_jsrt.csv --split test_jsrt --data-root $D --threshold <t90 của C0-medgemma|0|0> --edges-json $O/e0_data_edges.json --out $O/audit/C0_jsrt.json
# B2 YOLO
python scripts/run_audit.py --scorer yolo --yolo-weights $O/yolo/seed0/weights/best.pt --manifest $M/manifest_jsrt.csv --split test_jsrt --data-root $D --threshold <t90 của B2|0|0> --edges-json $O/e0_data_edges.json --out $O/audit/B2_jsrt.json
```
Thêm `--lung-mask heuristic` nếu mặt nạ PSPNet trên CPU quá chậm (kiểm tra mục 6).

Tóm tắt trong file `.json` có `flip_t1`, `flip_t2`, `delta` tính trên tập con ghép cặp (những ảnh T2 tạo được), cùng `flip_t1_all` (toàn bộ ảnh) và `n_t2_ok` (cỡ tập con ghép cặp). **Số cần báo cáo là `delta` (= `flip_t1` − `flip_t2`)**, vì cả hai vế dùng cùng một tập ảnh; báo kèm `n_t2_ok`. `flip_t1_all` chỉ để tham khảo.

## Ngân sách (spec §7)
Đo thời gian huấn luyện bằng pilot ở E3 (`timing.json`) và thời gian suy luận ở kiểm tra mục 7. Nếu một lượt vượt 3 giờ, cắt theo thứ tự:
1. `--neg-per-pos` 4 → 2.
2. Giảm độ phân giải — không áp dụng cho MedGemma (processor cố định 896); chỉ áp dụng cho Qwen qua `min_pixels`/`max_pixels` — chỉnh tay trong `vlm/hf.py` (không có cờ dòng lệnh).
3. `--epochs 2`.
4. C1 chỉ 1 seed.

Ghi mọi quyết định vào `$O/budget.md`.
