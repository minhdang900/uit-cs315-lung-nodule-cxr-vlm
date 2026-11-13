# RUNBOOK (Colab, GPU L4 hoặc A100; không dùng T4 vì MedGemma cần bf16)

MedGemma từ chối tải model nếu không có GPU hỗ trợ bf16 (`load_base` ném `RuntimeError`), nên chỉ dùng L4 hoặc A100.

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
   df = read_manifest("../data/processed/manifest_vinbig.csv"); r = df[df.label == 1].iloc[0]
   ex = build_example(processor, load_image("/content/data", r.path), QUESTION_LOC, build_target(1, [[1, 2, 3, 4]]))
   print({k: v.shape for k, v in ex.items()})
   ```
   Kiểm tra tiền tố của Gemma (prompt phải là tiền tố của prompt+answer) không ném `ValueError`.
3. **Loss đầu tiên của pilot 64 mẫu.** Chạy `train_lora.py ... --max-train 64 --epochs 1` (xem E3). Dòng log đầu tiên (`logging_steps=10`) phải có `loss` khoảng 0,5–3. Nếu ~10–25 nghĩa là gradient accumulation bị chia hai lần: dừng và sửa.
4. **Loại GPU.**
   ```python
   import torch; print(torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())
   ```
   Phải là L4 hoặc A100 và `True`.
5. **Smoke YOLO 1 epoch.** Chạy:
   ```python
   from ultralytics import YOLO
   m = YOLO("yolov8s.pt"); m.train(data="<work>/yolo_data_seed0/data.yaml", imgsz=896, epochs=1, project="/content/yolo_smoke", name="s", exist_ok=True)
   print(m.trainer.best)   # file phải tồn tại
   ```
   (Đường dẫn `data.yaml`: lấy từ thư mục `yolo_data_seed0` do `train_yolo.py` xuất; có thể chạy `train_yolo.py --epochs 1 --work $O/yolo_smoke --out /tmp/x.jsonl` thay thế rồi kiểm tra `$O/yolo_smoke/seed0/weights/best.pt`.)
6. **Thời gian mỗi ảnh của mặt nạ phổi trong `run_audit.py`.** PSPNet của torchxrayvision chạy trên CPU. Chạy thử `run_audit.py` trên khoảng 10 ảnh và đo thời gian mỗi ảnh. Nếu quá chậm, dùng `--lung-mask heuristic` và ghi quyết định đó vào `$O/budget.md`.

## 0. Mỗi phiên Colab
```bash
from google.colab import drive; drive.mount('/content/drive')
%cd "/content/drive/MyDrive/ClaudeAI/Projects/UIT-VNU/CS315.G12.LT.TTNT - MachineLearningAdvance/code"
!pip install -q -e ".[gpu]"
!huggingface-cli login --token $HF_TOKEN
!mkdir -p /content/data && rsync -a ../data/processed/ /content/data/   # đọc ảnh từ ổ local
```
Biến dùng chung: `M="../data/processed"`, `D=/content/data`, `O=../outputs`.

## E0. Dữ liệu (Task 5)
Chạy 3 script `prepare_*` (cờ bắt buộc: `prepare_vinbig.py --train-csv ... --out-root $M` cùng `--png-dir`/`--meta-csv` hoặc `--dicom-dir`; `prepare_nih.py --raw-dir ... --out-root $M`; `prepare_jsrt.py --raw-dir ... --out-root $M`), rồi:
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
  python scripts/train_yolo.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --data-root $D --seed $s --work $O/yolo --out $O/preds/B2.jsonl
done
```
B2 xuất dữ liệu YOLO riêng cho từng seed (`$O/yolo/yolo_data_seed{seed}`) và lưu trọng số tại `$O/yolo/seed{seed}/weights/best.pt`.

## E2. Zero-shot C0 (MedGemma, Qwen; 3 lần: nhiệt độ 0 một lần, 0,7 hai lần)
```bash
for m in medgemma qwen; do
  python scripts/predict_vlm.py --model $m --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --splits val test test_nih test_jsrt --data-root $D --config C0-$m --run 0 --temperature 0 --out $O/preds/C0-$m.jsonl
  for r in 1 2; do
    python scripts/predict_vlm.py --model $m --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --splits val test test_nih test_jsrt --data-root $D --config C0-$m --run $r --temperature 0.7 --out $O/preds/C0-$m.jsonl
  done
done
```
Script tự bỏ qua ảnh đã chạy, nên nếu Colab ngắt kết nối thì chỉ cần chạy lại cùng lệnh.

## E3. LoRA MedGemma C1–C3 (3 seed)
Pilot đo thời gian và loss đầu tiên (kiểm tra mục 3 ở trên): thêm `--max-train 64 --epochs 1` vào một lệnh `train_lora.py` rồi đọc `timing.json` và log.
```bash
for s in 0 1 2; do
  python scripts/train_lora.py --model medgemma --mode cls --manifest $M/manifest_vinbig.csv --data-root $D --seed $s --out $O/lora/C1-s$s
  python scripts/train_lora.py --model medgemma --mode loc --manifest $M/manifest_vinbig.csv --data-root $D --seed $s --out $O/lora/C2-s$s
  python scripts/train_lora.py --model medgemma --mode loc --balance --pos-weight 2.0 --manifest $M/manifest_vinbig.csv --data-root $D --seed $s --out $O/lora/C3-s$s
done
```
Dự đoán:
- C1 dùng `--prompt cls --generate none`, vì chỉ cần điểm.
- C2 và C3 dùng `--prompt loc --generate all`.
- Mỗi lệnh chạy trên `val test test_nih test_jsrt`, với `--adapter $O/lora/<cfg>-s$s/final --config <cfg> --seed $s`, ghi vào `$O/preds/<cfg>.jsonl`.

Ví dụ cho C3:
```bash
for s in 0 1 2; do
  python scripts/predict_vlm.py --model medgemma --adapter $O/lora/C3-s$s/final --prompt loc --generate all --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --splits val test test_nih test_jsrt --data-root $D --config C3 --seed $s --out $O/preds/C3.jsonl
done
```

Đường học (sensitivity theo epoch): chạy `predict_vlm.py --splits val --generate none` cho từng `checkpoint-*` của seed 0, với `--config <cfg>-ep<k>`.

## E4. C4: cấu hình tốt nhất trên Qwen2.5-VL-3B (1–3 seed)
Chọn cấu hình tốt nhất trong C1–C3 theo **sens ở spec 0,90 trên val** (cột `sens` của split `val` không có trong bảng test; xem `thresholds.json` và `summary_per_seed.csv`). Chạy lại `train_lora.py --model qwen` với đúng các cờ của cấu hình đó, rồi dự đoán với `--config C4`.

## Báo cáo
```bash
python scripts/make_report.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --preds "$O/preds/*.jsonl" --edges-json $O/e0_data_edges.json --out $O/report
```

## E5. Kiểm toán che vùng (C0-medgemma, cấu hình LoRA tốt nhất, B2), trên `test` và `test_jsrt`
Lấy ngưỡng `t90` từ `$O/report/thresholds.json` (khóa dạng `config|seed|run`), rồi chạy (`--manifest` theo split: `manifest_vinbig.csv` cho `test`, `manifest_jsrt.csv` cho `test_jsrt`):
```bash
# LoRA tốt nhất (ví dụ C3, seed 0)
python scripts/run_audit.py --scorer vlm --model medgemma --adapter $O/lora/C3-s0/final --manifest $M/manifest_jsrt.csv --split test_jsrt --data-root $D --threshold <t90> --edges-json $O/e0_data_edges.json --out $O/audit/C3_jsrt.json
# C0 zero-shot (không có --adapter)
python scripts/run_audit.py --scorer vlm --model medgemma --manifest $M/manifest_jsrt.csv --split test_jsrt --data-root $D --threshold <t90 của C0-medgemma|0|0> --edges-json $O/e0_data_edges.json --out $O/audit/C0_jsrt.json
# B2 YOLO
python scripts/run_audit.py --scorer yolo --yolo-weights $O/yolo/seed0/weights/best.pt --manifest $M/manifest_jsrt.csv --split test_jsrt --data-root $D --threshold <t90 của B2|0|0> --edges-json $O/e0_data_edges.json --out $O/audit/B2_jsrt.json
```
Thêm `--lung-mask heuristic` nếu mặt nạ PSPNet trên CPU quá chậm (kiểm tra mục 6).

Tóm tắt trong file `.json` có `flip_t1`, `flip_t2`, `delta` tính trên tập con ghép cặp (những ảnh T2 tạo được), cùng `flip_t1_all` (toàn bộ ảnh) và `n_t2_ok` (cỡ tập con ghép cặp). **Số cần báo cáo là `delta` (= `flip_t1` − `flip_t2`)**, vì cả hai vế dùng cùng một tập ảnh; báo kèm `n_t2_ok`. `flip_t1_all` chỉ để tham khảo.

## Ngân sách (spec §7)
Đo thời gian ở Task 13 Bước 2. Nếu một lượt vượt 3 giờ, cắt theo thứ tự:
1. `--neg-per-pos` 4 → 2.
2. Giảm độ phân giải.
3. `--epochs 2`.
4. C1 chỉ 1 seed.

Ghi mọi quyết định vào `$O/budget.md`.
