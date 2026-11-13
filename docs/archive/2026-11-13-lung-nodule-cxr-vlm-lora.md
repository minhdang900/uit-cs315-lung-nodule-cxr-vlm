> **LƯU TRỮ — chỉ để tham khảo lịch sử.** Kế hoạch này đã được thực hiện xong ngày 2026-11-13. Code trong `code/` là chuẩn; khi khác nhau, tin vào code, spec (`docs/design/`, §8.5) và sổ tay bàn giao (`docs/handoff/`). Đừng chép code từ file này.

# Kế hoạch triển khai: Trợ lý AI phát hiện và định vị nốt phổi trên X-quang ngực (tinh chỉnh LoRA VLM cỡ nhỏ)

> **Trạng thái (2026-11-13):** đã triển khai và merge vào `main` của repo `code/`. Một số chỗ code trong kế hoạch này đã được sửa khi triển khai (xem spec §8.5 và ledger `code/.superpowers/sdd/2026-11-13-lung-nodule-cxr-vlm-lora/progress.md`); code trong repo là bản chuẩn.

> **Dành cho agent thực thi:** BẮT BUỘC dùng sub-skill `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để làm từng task. Các bước dùng checkbox `- [ ]` để theo dõi.

**Mục tiêu.** Xây dựng một pipeline tái lập được, gồm:
- chuẩn bị ba bộ dữ liệu CXR: VinBigData, NIH ChestX-ray14, JSRT;
- chạy zero-shot hai VLM: MedGemma-4B-it, Qwen2.5-VL-3B;
- tinh chỉnh LoRA (C1–C4);
- huấn luyện hai baseline CNN: DenseNet-121, YOLOv8;
- đánh giá phát hiện và định vị, kiểm toán che vùng, rồi tổng hợp bảng benchmark có khoảng tin cậy.

**Kiến trúc.** Một gói Python nhỏ tên `nodulecxr`, gồm các module tách biệt:
- dữ liệu → manifest CSV chung;
- prompt và parser;
- VLM wrapper;
- huấn luyện LoRA;
- baseline;
- metric;
- kiểm toán;
- báo cáo.

Mọi mô hình ghi dự đoán ra cùng một định dạng JSONL, và module báo cáo chỉ đọc manifest cùng JSONL.

Phần logic thuần (nhãn, chia tập, parser, metric, thống kê, che vùng, collator, loss) có unit test chạy trên CPU. Phần cần GPU (MedGemma, Qwen, YOLO, DenseNet) là script mỏng gọi các hàm đã test, chạy trên Colab.

**Tech stack.**
- Python ≥ 3.9, numpy, pandas, scikit-learn, scipy, statsmodels, Pillow, pydicom, ensemble-boxes, pytest.
- GPU: torch ≥ 2.3, transformers ≥ 4.50, peft ≥ 0.14, accelerate, torchxrayvision ≥ 1.2, ultralytics ≥ 8.3.

**Spec:** `docs/design/2026-11-12-lung-nodule-cxr-vlm-lora-design.md` (v2, đã được duyệt ngày 2026-11-12). Agent thực thi phải đọc cả spec lẫn kế hoạch này.

## Ràng buộc chung (chép từ spec, áp cho mọi task)

**Mô hình và cấu hình**
- VLM chính là `google/medgemma-4b-it`. VLM phụ là `Qwen/Qwen2.5-VL-3B-Instruct`.
- LoRA: r = 16, α = 32, dropout 0,05. Gắn vào các lớp attention và MLP của LLM; **bộ mã hóa ảnh đông cứng**.
- Ca dương gộp **Nodule/Mass**.
- Cấu hình: C0 (zero-shot), C1 (LoRA, chỉ phân loại), C2 (LoRA, phân loại + box), C3 (C2 + cân bằng lớp: oversampling ca dương 1:1 và trọng số loss cho token trả lời), C4 (cấu hình tốt nhất trong C1–C3, áp lên Qwen2.5-VL-3B), B1 (DenseNet-121 từ `torchxrayvision`), B2 (YOLOv8).
- Số lượt chạy:
  - C1–C3: mỗi cấu hình 3 seed.
  - C4: 1–3 seed.
  - B1, B2: mỗi loại 3 seed.
  - E2 (zero-shot): mỗi VLM chạy 3 lần, gồm nhiệt độ 0 và 0,7.

**Prompt và nhãn**
- Câu hỏi cố định: "Is there a lung nodule or mass on this chest X-ray? If yes, give the bounding box(es) in pixel coordinates. Answer in JSON."
- Đáp án đích sinh tự động từ nhãn gốc (box hợp nhất bằng weighted boxes fusion). Ca âm có đáp án `{"nodule":"no","boxes":[]}`.
- **Không dùng câu hỏi–đáp do LLM sinh.**
- **Không đưa ngữ cảnh lâm sàng vào prompt** trong điều kiện chính.

**Dữ liệu và đánh giá**
- VinBigData chia theo `image_id`, tỷ lệ 70/10/20, phân tầng theo nhãn Nodule/Mass.
- Ngưỡng và siêu tham số chọn trên validation. **Mỗi tập test chỉ dùng một lần.**
- Tiêu chí thành công: sensitivity ≥ 0,60 ở specificity 0,90, với ngưỡng chọn trên validation.
- Báo cáo **không** viết "ngang bác sĩ".
- Chỉ số mức ảnh:
  - Sens, spec, PPV, NPV, F1.
  - AUC-ROC và **AUC-PR**.
  - Sens ở spec 0,90 và 0,95.
  - **ECE, Brier, balanced Brier.**
  - PPV quy về tỷ lệ bệnh thấp (1% và 3%).
  - Chỉ số chế độ báo bình thường.
- PPV trên JSRT không có ý nghĩa lâm sàng (tỷ lệ nốt 62%), chỉ báo kèm chú thích.
- Định vị: IoU, tỷ lệ "đúng và IoU ≥ 0,3" (báo thêm ở 0,5), FROC. Với JSRT: "trúng" khi tâm box nằm trong bán kính nốt cộng biên, kèm khoảng cách tâm trung vị (mm).
- Kiểm toán che vùng:
  - T1: che nốt, box nới 10%, tô bằng giá trị trung bình.
  - T2: che vùng ngẫu nhiên cùng diện tích, không giao nốt, nằm trong trường phổi.
  - T3: tráo sang ảnh âm của bệnh nhân khác.
  - Chỉ số chính: tỷ lệ đổi đáp án ở T1 trừ ở T2.
  - Phân tầng theo tertile diện tích box (mm với JSRT) và theo độ mờ JSRT 1–5.
- Thống kê:
  - Trung bình ± độ lệch chuẩn qua các seed.
  - Khoảng tin cậy 95% bằng bootstrap theo ảnh.
  - McNemar cho sens/spec, DeLong cho AUC, hiệu chỉnh Holm.

**Vận hành**
- Ngân sách: nếu một lượt MedGemma-4B vượt 3 giờ, cắt theo thứ tự: giảm số ảnh âm (giữ toàn bộ ảnh dương) → giảm độ phân giải → còn 2 epoch → C1 còn 1 seed.
- Không phân phối lại ảnh. Chỉ bàn giao nhãn và dự đoán.

## Quy ước dùng chung (khóa từ đầu)

**Thư mục.** Gốc dự án là `P = .../CS315.G12.LT.TTNT - MachineLearningAdvance/`.
- Mã nguồn ở `P/code/` (repo git riêng).
- Dữ liệu ở `P/data/` gồm `raw/` và `processed/`; không commit.
- Kết quả ở `P/outputs/`; không commit.

**Ảnh đã xử lý.** PNG xám, kích thước **896×896** (`IMG_SIZE = 896`). Đây là kích thước đầu vào của MedGemma. Qwen2.5-VL được ép `min_pixels = max_pixels = 896*896`. Mọi box trong manifest và trong đáp án dùng **tọa độ pixel trên ảnh 896×896**, dạng `[x1, y1, x2, y2]`.

**Manifest CSV.** Các cột: `image_id, dataset, path, orig_w, orig_h, label, boxes, split, subtlety, size_mm, patient_id, n_rads`.
- `path` là đường dẫn tương đối so với `--data-root`.
- `boxes` là chuỗi JSON.
- `split` nhận một trong các giá trị `train | val | test | test_nih | test_jsrt`.

**Dự đoán JSONL.** Mỗi dòng một ảnh, gồm các khóa: `image_id, dataset, split, config, seed, run, score, label_pred, boxes_pred, parse_ok, raw_text`.
- `score ∈ [0, 1]`.
- `label_pred` là `0 | 1 | null`.
- `boxes_pred` là `[[x1, y1, x2, y2, s], ...]`.

**Cấu trúc file**

```
code/
  pyproject.toml, .gitignore, RUNBOOK.md
  src/nodulecxr/
    __init__.py
    records.py        # manifest, box, JSONL I/O
    images.py         # đọc DICOM/JSRT, chuẩn hóa, resize, load ảnh
    strata.py         # diện tích box, tertile
    data/vinbig.py    # nhãn VinBigData + WBF
    data/jsrt.py      # parse CLNDAT/NNDAT, manifest JSRT
    data/nih.py       # manifest NIH
    data/split.py     # chia tập, giảm ảnh âm, oversampling
    data/stats.py     # E0: mô tả dữ liệu
    prompts.py        # câu hỏi, đáp án đích, parser
    metrics/detection.py, metrics/localization.py, metrics/stats.py
    audit/occlusion.py
    vlm/chat.py       # dựng chuỗi chat template
    vlm/scoring.py    # xác suất yes/no từ log-prob
    vlm/hf.py         # nạp MedGemma/Qwen, generate, yes_probability
    vlm/predict.py    # vòng dự đoán chung (dùng được với VLM giả)
    train/sft_data.py # build_example, collate, SFTDataset, regex LoRA
    train/lora.py     # LoraConfig, weighted loss, Trainer
    baselines/densenet.py, baselines/yolo.py
    eval/report.py    # bảng tổng hợp, phân tầng, so sánh
  scripts/
    prepare_vinbig.py, prepare_jsrt.py, prepare_nih.py, draw_boxes.py, data_stats.py,
    predict_vlm.py, train_lora.py, train_densenet.py, train_yolo.py, run_audit.py, make_report.py
  tests/  (một file test cho mỗi module)
```

---

### Task 0: Việc thủ công và khung dự án

**Files**
- Create: `code/pyproject.toml`, `code/.gitignore`, `code/src/nodulecxr/__init__.py`, `code/src/nodulecxr/{data,metrics,audit,vlm,train,baselines,eval}/__init__.py`, `code/tests/test_smoke.py`

**Interfaces**
- Produces: gói `nodulecxr` cài được bằng `pip install -e ".[dev]"`.

- [ ] **Bước 1: Việc thủ công (người dùng tự làm, agent không tự đăng nhập hay chấp nhận điều khoản)**
  1. Đọc điều khoản dữ liệu và xác nhận được phép dùng cho học thuật:
     - VinBigData: Kaggle competition `vinbigdata-chest-xray-abnormalities-detection`.
     - NIH ChestX-ray14.
     - JSRT: đăng ký tải tại `http://db.jsrt.or.jp/eng.php`.
  2. Chấp nhận điều khoản Health AI Developer Foundations của `google/medgemma-4b-it` trên Hugging Face. Tạo HF token có quyền đọc.
  3. Tạo Kaggle API token (`kaggle.json`).
  4. Ghi lại link và ngày chấp nhận điều khoản vào `P/data/LICENSES.md`.

- [ ] **Bước 2: Khởi tạo repo**

```bash
cd "P/code"   # thay P bằng đường dẫn thật; tạo thư mục nếu chưa có
git init
mkdir -p src/nodulecxr/{data,metrics,audit,vlm,train,baselines,eval} scripts tests
touch src/nodulecxr/__init__.py src/nodulecxr/{data,metrics,audit,vlm,train,baselines,eval}/__init__.py
```

- [ ] **Bước 3: Viết `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "nodulecxr"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
  "numpy>=1.24", "pandas>=2.0", "pillow>=10", "scikit-learn>=1.3", "scipy>=1.10",
  "statsmodels>=0.14", "ensemble-boxes>=1.0.9", "pydicom>=2.4", "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
torchcpu = ["torch>=2.3"]
gpu = ["torch>=2.3", "transformers>=4.50", "peft>=0.14", "accelerate>=0.34",
       "torchxrayvision>=1.2", "ultralytics>=8.3"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Bước 4: Viết `.gitignore` và test khói**

`.gitignore`:
```
__pycache__/
*.egg-info/
.venv/
outputs/
data/
*.png
*.pt
```

`tests/test_smoke.py`:
```python
def test_package_imports():
    import nodulecxr  # noqa: F401
```

- [ ] **Bước 5: Cài đặt và chạy test**

Chạy:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torchcpu]"
pytest -q
```
Kỳ vọng: `1 passed`.

- [ ] **Bước 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold nodulecxr package"
```

---

### Task 1: Manifest, box và JSONL I/O

**Files**
- Create: `code/src/nodulecxr/records.py`
- Test: `code/tests/test_records.py`

**Interfaces**
- Produces:
  - `IMG_SIZE: int = 896`
  - `MANIFEST_COLUMNS: list[str]`
  - `boxes_to_json(boxes) -> str`
  - `boxes_from_json(s) -> list[list[float]]`
  - `scale_boxes(boxes, orig_w, orig_h, size=IMG_SIZE) -> list[list[float]]`
  - `write_manifest(df, path) -> None`
  - `read_manifest(path) -> pd.DataFrame` (cột `boxes` đã là list)
  - `append_jsonl(row: dict, path) -> None`
  - `read_jsonl(path) -> pd.DataFrame`
  - `done_ids(path, **match) -> set[str]`

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import pandas as pd
from nodulecxr.records import (MANIFEST_COLUMNS, append_jsonl, done_ids, read_jsonl,
                               read_manifest, scale_boxes, write_manifest)


def test_scale_boxes_maps_original_to_896():
    assert scale_boxes([[0, 0, 1792, 896]], 1792, 896) == [[0.0, 0.0, 896.0, 896.0]]


def test_manifest_roundtrip(tmp_path):
    df = pd.DataFrame([{"image_id": "a", "dataset": "vinbig", "path": "vinbig/a.png",
                        "orig_w": 100, "orig_h": 100, "label": 1, "boxes": [[1, 2, 3, 4]],
                        "split": "train"},
                       {"image_id": "b", "dataset": "vinbig", "path": "vinbig/b.png",
                        "orig_w": 100, "orig_h": 100, "label": 0, "boxes": [], "split": "val"}])
    p = tmp_path / "m.csv"
    write_manifest(df, p)
    back = read_manifest(p)
    assert list(back.columns) == MANIFEST_COLUMNS
    assert back.loc[0, "boxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert back.loc[1, "boxes"] == []
    assert back["label"].tolist() == [1, 0]


def test_jsonl_append_and_resume(tmp_path):
    p = tmp_path / "p.jsonl"
    append_jsonl({"image_id": "a", "config": "C0", "run": 0, "score": 0.2}, p)
    append_jsonl({"image_id": "b", "config": "C0", "run": 1, "score": 0.9}, p)
    df = read_jsonl(p)
    assert df["image_id"].tolist() == ["a", "b"]
    assert done_ids(p, config="C0", run=0) == {"a"}
    assert done_ids(tmp_path / "missing.jsonl") == set()
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_records.py -q`
Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'nodulecxr.records'`.

- [ ] **Bước 3: Viết code**

```python
"""Manifest format and prediction I/O shared by every dataset, model and metric."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

IMG_SIZE = 896
MANIFEST_COLUMNS = ["image_id", "dataset", "path", "orig_w", "orig_h", "label",
                    "boxes", "split", "subtlety", "size_mm", "patient_id", "n_rads"]


def boxes_to_json(boxes) -> str:
    if not isinstance(boxes, (list, tuple)):
        return "[]"
    return json.dumps([[round(float(v), 1) for v in b[:4]] for b in boxes])


def boxes_from_json(s) -> list:
    if not isinstance(s, str) or not s.strip():
        return []
    return [[float(v) for v in b[:4]] for b in json.loads(s)]


def scale_boxes(boxes, orig_w, orig_h, size: int = IMG_SIZE) -> list:
    sx, sy = size / float(orig_w), size / float(orig_h)
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]


def write_manifest(df: pd.DataFrame, path) -> None:
    out = df.copy()
    for c in MANIFEST_COLUMNS:
        if c not in out.columns:
            out[c] = None
    out["boxes"] = out["boxes"].apply(lambda b: b if isinstance(b, str) else boxes_to_json(b))
    out[MANIFEST_COLUMNS].to_csv(path, index=False)


def read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str, "patient_id": str})
    df["boxes"] = df["boxes"].fillna("[]").apply(boxes_from_json)
    df["label"] = df["label"].astype(int)
    return df


def append_jsonl(row: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str)
    return df


def done_ids(path, **match) -> set:
    if not Path(path).exists():
        return set()
    df = read_jsonl(path)
    if df.empty:
        return set()
    for k, v in match.items():
        df = df[df[k] == v]
    return set(df["image_id"])
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_records.py -q`
Kỳ vọng: `3 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/records.py tests/test_records.py
git commit -m "feat: manifest and jsonl io"
```

---

### Task 2: Nhãn VinBigData (WBF) và chia tập

**Files**
- Create: `code/src/nodulecxr/data/vinbig.py`, `code/src/nodulecxr/data/split.py`
- Test: `code/tests/test_vinbig.py`, `code/tests/test_split.py`

**Interfaces**
- Consumes: không có.
- Produces:
  - `NODULE_CLASS = "Nodule/Mass"`
  - `build_vinbig_labels(ann: DataFrame, dims: DataFrame, iou_thr=0.4) -> DataFrame`
    - Cột `ann`: `image_id, class_name, rad_id, x_min, y_min, x_max, y_max`. Cột `rad_ID` được tự đổi tên thành `rad_id`.
    - Cột `dims`: `image_id, orig_w, orig_h`.
    - Trả về cột: `image_id, orig_w, orig_h, label, boxes` (pixel gốc), `n_rads`.
  - `stratified_split(labels: Series, ratios=(0.7, 0.1, 0.2), seed=0) -> Series` với giá trị `"train" | "val" | "test"`. Nếu `ratios[2] == 0` thì chỉ có train/val.
  - `subsample_negatives(df, neg_per_pos: float, seed=0) -> DataFrame`
  - `oversample_indices(labels, seed=0) -> list[int]` (vị trí trong `labels`; số ca dương được lặp lên bằng số ca âm)

**Quy tắc nhãn.** Một ảnh là dương nếu **ít nhất một** bác sĩ đánh dấu Nodule/Mass. Box của các bác sĩ được hợp nhất bằng WBF với `iou_thr = 0,4`. `n_rads` là số bác sĩ đồng ý; con số này được báo cáo ở E0.

- [ ] **Bước 1: Viết test (sẽ fail)**

`tests/test_vinbig.py`:
```python
import numpy as np
import pandas as pd
from nodulecxr.data.vinbig import build_vinbig_labels


def test_wbf_fuses_two_radiologists_and_keeps_negatives():
    nan = np.nan
    ann = pd.DataFrame([
        {"image_id": "p1", "class_name": "Nodule/Mass", "rad_id": "R1", "x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
        {"image_id": "p1", "class_name": "Nodule/Mass", "rad_id": "R2", "x_min": 120, "y_min": 120, "x_max": 220, "y_max": 220},
        {"image_id": "p1", "class_name": "Aortic enlargement", "rad_id": "R3", "x_min": 0, "y_min": 0, "x_max": 50, "y_max": 50},
        {"image_id": "n1", "class_name": "No finding", "rad_id": "R1", "x_min": nan, "y_min": nan, "x_max": nan, "y_max": nan},
    ])
    dims = pd.DataFrame({"image_id": ["p1", "n1"], "orig_w": [1000, 1000], "orig_h": [1000, 1000]})
    out = build_vinbig_labels(ann, dims).set_index("image_id")
    assert out.loc["p1", "label"] == 1 and out.loc["p1", "n_rads"] == 2
    boxes = out.loc["p1", "boxes"]
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [110, 110, 210, 210], atol=1.0)
    assert out.loc["n1", "label"] == 0 and out.loc["n1", "boxes"] == []


def test_accepts_physionet_rad_ID_column():
    ann = pd.DataFrame([{"image_id": "p1", "class_name": "Nodule/Mass", "rad_ID": "R8",
                         "x_min": 10, "y_min": 10, "x_max": 30, "y_max": 30}])
    dims = pd.DataFrame({"image_id": ["p1"], "orig_w": [100], "orig_h": [100]})
    out = build_vinbig_labels(ann, dims)
    assert out.loc[0, "label"] == 1
```

`tests/test_split.py`:
```python
import pandas as pd
from nodulecxr.data.split import oversample_indices, stratified_split, subsample_negatives


def _labels():
    return pd.Series([1] * 30 + [0] * 970, index=[f"i{k}" for k in range(1000)])


def test_stratified_split_sizes_and_positives():
    s = stratified_split(_labels(), seed=0)
    counts = s.value_counts()
    assert abs(counts["train"] - 700) <= 1 and abs(counts["val"] - 100) <= 1 and abs(counts["test"] - 200) <= 1
    pos = _labels()[_labels() == 1].index
    assert abs((s.loc[pos] == "test").sum() - 6) <= 1
    assert s.equals(stratified_split(_labels(), seed=0))


def test_split_without_test():
    s = stratified_split(_labels(), ratios=(0.85, 0.15, 0.0), seed=0)
    assert set(s.unique()) == {"train", "val"}


def test_subsample_negatives_keeps_all_positives():
    df = pd.DataFrame({"label": [1] * 10 + [0] * 100})
    out = subsample_negatives(df, neg_per_pos=4, seed=0)
    assert (out.label == 1).sum() == 10 and (out.label == 0).sum() == 40


def test_oversample_balances_classes():
    labels = [1] * 10 + [0] * 90
    idx = oversample_indices(labels, seed=0)
    picked = [labels[i] for i in idx]
    assert len(idx) == 180 and sum(picked) == 90
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_vinbig.py tests/test_split.py -q`
Kỳ vọng: FAIL, `ModuleNotFoundError`.

- [ ] **Bước 3: Viết `data/vinbig.py`**

```python
"""VinBigData / VinDr-CXR annotations -> one row per image, Nodule/Mass positives."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

NODULE_CLASS = "Nodule/Mass"


def build_vinbig_labels(ann: pd.DataFrame, dims: pd.DataFrame, iou_thr: float = 0.4) -> pd.DataFrame:
    ann = ann.rename(columns={"rad_ID": "rad_id"})
    dims = dims.set_index("image_id")
    rows = []
    for image_id, g in ann.groupby("image_id"):
        w = float(dims.loc[image_id, "orig_w"])
        h = float(dims.loc[image_id, "orig_h"])
        nod = g[g["class_name"] == NODULE_CLASS]
        if nod.empty:
            rows.append({"image_id": image_id, "orig_w": w, "orig_h": h, "label": 0, "boxes": [], "n_rads": 0})
            continue
        boxes_list, scores_list, labels_list = [], [], []
        for _, r in nod.groupby("rad_id"):
            b = np.stack([r.x_min / w, r.y_min / h, r.x_max / w, r.y_max / h], axis=1).clip(0, 1)
            boxes_list.append(b.tolist())
            scores_list.append([1.0] * len(b))
            labels_list.append([0] * len(b))
        fused, _, _ = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                            iou_thr=iou_thr, skip_box_thr=0.0)
        boxes = [[x1 * w, y1 * h, x2 * w, y2 * h] for x1, y1, x2, y2 in fused]
        rows.append({"image_id": image_id, "orig_w": w, "orig_h": h, "label": 1, "boxes": boxes,
                     "n_rads": int(nod["rad_id"].nunique())})
    return pd.DataFrame(rows)
```

- [ ] **Bước 4: Viết `data/split.py`**

```python
"""Train/val/test splitting, negative subsampling and positive oversampling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(labels: pd.Series, ratios=(0.7, 0.1, 0.2), seed: int = 0) -> pd.Series:
    idx = labels.index.to_numpy()
    tr, rest = train_test_split(idx, train_size=ratios[0], stratify=labels.to_numpy(), random_state=seed)
    out = pd.Series("train", index=labels.index, dtype=object)
    if ratios[2] == 0:
        out.loc[rest] = "val"
        return out
    val_frac = ratios[1] / (ratios[1] + ratios[2])
    va, te = train_test_split(rest, train_size=val_frac, stratify=labels.loc[rest].to_numpy(),
                              random_state=seed)
    out.loc[va] = "val"
    out.loc[te] = "test"
    return out


def subsample_negatives(df: pd.DataFrame, neg_per_pos: float, seed: int = 0) -> pd.DataFrame:
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    k = min(len(neg), int(round(neg_per_pos * len(pos))))
    return pd.concat([pos, neg.sample(n=k, random_state=seed)]).sort_index()


def oversample_indices(labels, seed: int = 0) -> list:
    y = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    reps = np.concatenate([np.tile(pos, len(neg) // len(pos)),
                           rng.choice(pos, len(neg) % len(pos), replace=False)])
    idx = np.concatenate([neg, reps])
    rng.shuffle(idx)
    return idx.tolist()
```

- [ ] **Bước 5: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_vinbig.py tests/test_split.py -q`
Kỳ vọng: `6 passed`.

- [ ] **Bước 6: Commit**

```bash
git add src/nodulecxr/data/vinbig.py src/nodulecxr/data/split.py tests/test_vinbig.py tests/test_split.py
git commit -m "feat: vinbig labels with WBF and stratified split"
```

---

### Task 3: Đọc và chuẩn hóa ảnh

**Files**
- Create: `code/src/nodulecxr/images.py`
- Test: `code/tests/test_images.py`

**Interfaces**
- Produces:
  - `percentile_to_uint8(arr, lo=2.0, hi=98.0) -> np.ndarray[uint8]`
  - `read_jsrt_img(path, shape=(2048, 2048), invert=True) -> np.ndarray[uint8]`
  - `dicom_to_uint8(path) -> np.ndarray[uint8]`
  - `resize_square(img, size=896) -> PIL.Image` (mode L)
  - `load_image(root, rel) -> PIL.Image` (mode RGB, dùng khi đưa vào mô hình)

Cách xử lý JSRT theo [L2]: đọc file 16-bit big-endian, kéo tương phản theo phân vị 2–98, chuyển 8-bit, đảo xám. Kiểm tra bằng mắt ở Task 5 Bước 4: phổi phải có màu tối.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import numpy as np
from PIL import Image
from nodulecxr.images import load_image, percentile_to_uint8, read_jsrt_img, resize_square


def test_percentile_stretch_hits_extremes():
    out = percentile_to_uint8(np.arange(100, dtype=np.float32).reshape(10, 10), 0, 100)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_constant_image_does_not_divide_by_zero():
    assert percentile_to_uint8(np.ones((4, 4))).max() == 0


def test_read_jsrt_big_endian_and_invert(tmp_path):
    raw = np.arange(16, dtype=">u2").reshape(4, 4)
    p = tmp_path / "x.img"
    raw.tofile(p)
    img = read_jsrt_img(p, shape=(4, 4), invert=True)
    assert img[0, 0] == 255 and img[-1, -1] == 0


def test_resize_and_load(tmp_path):
    im = resize_square(np.zeros((50, 80), dtype=np.uint8))
    assert im.size == (896, 896) and im.mode == "L"
    im.save(tmp_path / "a.png")
    assert load_image(tmp_path, "a.png").mode == "RGB"
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_images.py -q`
Kỳ vọng: FAIL, `ModuleNotFoundError`.

- [ ] **Bước 3: Viết code**

```python
"""Image reading and normalisation to 896x896 grayscale PNG."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from nodulecxr.records import IMG_SIZE


def percentile_to_uint8(arr, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    p_lo, p_hi = np.percentile(a, [lo, hi])
    if p_hi <= p_lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - p_lo) / (p_hi - p_lo), 0, 1) * 255).round().astype(np.uint8)


def read_jsrt_img(path, shape=(2048, 2048), invert: bool = True) -> np.ndarray:
    raw = np.fromfile(path, dtype=">u2")
    if raw.size != shape[0] * shape[1]:
        raise ValueError(f"{path}: expected {shape[0] * shape[1]} pixels, got {raw.size}")
    img = percentile_to_uint8(raw.reshape(shape))
    return 255 - img if invert else img


def dicom_to_uint8(path) -> np.ndarray:
    import pydicom
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    ds = pydicom.dcmread(path)
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return percentile_to_uint8(arr, 0.0, 100.0)


def resize_square(img, size: int = IMG_SIZE) -> Image.Image:
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    return im.convert("L").resize((size, size), Image.BILINEAR)


def load_image(root, rel) -> Image.Image:
    return Image.open(Path(root) / rel).convert("RGB")
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_images.py -q`
Kỳ vọng: `4 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/images.py tests/test_images.py
git commit -m "feat: image io and normalisation"
```

---

### Task 4: Manifest JSRT và NIH, phân tầng kích thước

**Files**
- Create: `code/src/nodulecxr/data/jsrt.py`, `code/src/nodulecxr/data/nih.py`, `code/src/nodulecxr/strata.py`
- Test: `code/tests/test_jsrt.py`, `code/tests/test_nih.py`, `code/tests/test_strata.py`

**Interfaces**
- Consumes: `scale_boxes`, `IMG_SIZE` (Task 1).
- Produces:
  - `JSRT_PIXEL_MM = 0.175`
  - `JSRT_MM_PER_PX_896 = 0.175 * 2048 / 896` (= 0,4)
  - `parse_clndat(text) -> DataFrame[filename, subtlety, size_mm, x, y, malignancy]`
  - `parse_nndat(text) -> DataFrame[filename]`
  - `build_jsrt_manifest(cln, nn) -> DataFrame` (theo cột manifest; split `test_jsrt`; `path = "jsrt/<stem>.png"`)
  - `build_nih_manifest(data_entry, bbox, neg_ratio=3, seed=0) -> DataFrame` (split `test_nih`; `path = "nih/<Image Index>"`)
  - `box_rel_area(box, size=896) -> float`
  - `tertile_edges(areas) -> tuple[float, float]`
  - `size_bin(area, edges) -> str` (một trong `"small" | "medium" | "large"`)

**Định nghĩa tập test NIH**
- Ca dương là mọi ảnh có box Nodule hoặc Mass trong `BBox_List_2017.csv`.
- Ca âm là `3 ×` số ca dương, lấy ngẫu nhiên (seed 0) từ các ảnh "No Finding" của bệnh nhân **không** có ảnh dương.

**Định dạng `CLNDAT_EN.txt` của JSRT.** Mỗi dòng gồm: `tên file, độ mờ (1–5), kích thước (mm), tuổi, giới, x, y, lành/ác, ...`. Tọa độ x, y tính trên ảnh 2048 px.

- [ ] **Bước 1: Viết test (sẽ fail)**

`tests/test_jsrt.py`:
```python
import pytest
from nodulecxr.data.jsrt import build_jsrt_manifest, parse_clndat, parse_nndat

CLN = "JPCLN001.IMG\t3\t15\t59\tfemale\t1634\t692\tmalignant\tl.upper lobe\tlung cancer\n"
NN = "JPCNN001.IMG\t60\tmale\tnon-nodule\n"


def test_parse_and_build():
    cln, nn = parse_clndat(CLN), parse_nndat(NN)
    assert cln.loc[0, "subtlety"] == 3 and cln.loc[0, "size_mm"] == 15.0
    m = build_jsrt_manifest(cln, nn).set_index("image_id")
    pos = m.loc["JPCLN001"]
    x1, y1, x2, y2 = pos["boxes"][0]
    assert (x1 + x2) / 2 == pytest.approx(1634 * 896 / 2048)
    assert (x2 - x1) / 2 == pytest.approx(15 / 0.175 / 2 * 896 / 2048)
    assert pos["label"] == 1 and pos["split"] == "test_jsrt" and pos["path"] == "jsrt/JPCLN001.png"
    assert m.loc["JPCNN001", "label"] == 0
```

`tests/test_nih.py`:
```python
import pandas as pd
from nodulecxr.data.nih import build_nih_manifest


def test_nih_positives_and_patient_disjoint_negatives():
    bbox = pd.DataFrame({"Image Index": ["a.png", "a.png", "b.png", "c.png"],
                         "Finding Label": ["Nodule", "Mass", "Nodule", "Atelectasis"],
                         "Bbox [x": [10, 100, 20, 0], "y": [10, 100, 20, 0],
                         "w": [30, 50, 40, 10], "h]": [30, 50, 40, 10]})
    entries = pd.DataFrame({"Image Index": ["a.png", "b.png", "c.png"] + [f"n{i}.png" for i in range(10)],
                            "Finding Labels": ["Nodule|Mass", "Nodule", "Atelectasis"] + ["No Finding"] * 10,
                            "Patient ID": [1, 2, 3] + [1] + list(range(100, 109))})
    m = build_nih_manifest(entries, bbox, neg_ratio=3, seed=0)
    pos = m[m.label == 1].set_index("image_id")
    assert set(pos.index) == {"a.png", "b.png"} and len(pos.loc["a.png", "boxes"]) == 2
    assert pos.loc["b.png", "boxes"][0] == [20 * 0.875, 20 * 0.875, 60 * 0.875, 60 * 0.875]
    neg = m[m.label == 0]
    assert len(neg) == 6 and "n0.png" not in set(neg.image_id)
```

`tests/test_strata.py`:
```python
from nodulecxr.strata import box_rel_area, size_bin, tertile_edges


def test_tertiles():
    edges = tertile_edges([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    assert size_bin(0.0015, edges) == "small" and size_bin(0.0055, edges) == "large"
    assert box_rel_area([0, 0, 448, 448]) == 0.25
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_jsrt.py tests/test_nih.py tests/test_strata.py -q`
Kỳ vọng: FAIL, `ModuleNotFoundError`.

- [ ] **Bước 3: Viết `data/jsrt.py`**

```python
"""JSRT nodule database: metadata parsing and manifest."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nodulecxr.records import IMG_SIZE, scale_boxes

JSRT_SIZE_PX = 2048
JSRT_PIXEL_MM = 0.175
JSRT_MM_PER_PX_896 = JSRT_PIXEL_MM * JSRT_SIZE_PX / IMG_SIZE


def parse_clndat(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 8 or not p[0].upper().startswith("JPCLN"):
            continue
        rows.append({"filename": p[0], "subtlety": int(p[1]), "size_mm": float(p[2]),
                     "x": float(p[5]), "y": float(p[6]), "malignancy": p[7].lower()})
    return pd.DataFrame(rows)


def parse_nndat(text: str) -> pd.DataFrame:
    names = [line.split()[0] for line in text.splitlines() if line.strip().upper().startswith("JPCNN")]
    return pd.DataFrame({"filename": names})


def build_jsrt_manifest(cln: pd.DataFrame, nn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in cln.itertuples():
        half = r.size_mm / JSRT_PIXEL_MM / 2
        box = [r.x - half, r.y - half, r.x + half, r.y + half]
        stem = Path(r.filename).stem.upper()
        rows.append({"image_id": stem, "dataset": "jsrt", "path": f"jsrt/{stem}.png",
                     "orig_w": JSRT_SIZE_PX, "orig_h": JSRT_SIZE_PX, "label": 1,
                     "boxes": scale_boxes([box], JSRT_SIZE_PX, JSRT_SIZE_PX), "split": "test_jsrt",
                     "subtlety": r.subtlety, "size_mm": r.size_mm})
    for f in nn["filename"]:
        stem = Path(f).stem.upper()
        rows.append({"image_id": stem, "dataset": "jsrt", "path": f"jsrt/{stem}.png",
                     "orig_w": JSRT_SIZE_PX, "orig_h": JSRT_SIZE_PX, "label": 0, "boxes": [],
                     "split": "test_jsrt"})
    return pd.DataFrame(rows)
```

- [ ] **Bước 4: Viết `data/nih.py`**

```python
"""NIH ChestX-ray14 external test set: Nodule/Mass boxes vs No Finding."""
from __future__ import annotations

import pandas as pd

from nodulecxr.records import scale_boxes

NIH_SIZE = 1024


def build_nih_manifest(data_entry: pd.DataFrame, bbox: pd.DataFrame, neg_ratio: int = 3,
                       seed: int = 0) -> pd.DataFrame:
    b = bbox.rename(columns={"Bbox [x": "x", "h]": "h"})
    b = b[b["Finding Label"].isin(["Nodule", "Mass"])]
    pid = data_entry.set_index("Image Index")["Patient ID"].astype(str)
    rows = []
    for img, g in b.groupby("Image Index"):
        boxes = [[r.x, r.y, r.x + r.w, r.y + r.h] for r in g.itertuples()]
        rows.append({"image_id": img, "dataset": "nih", "path": f"nih/{img}",
                     "orig_w": NIH_SIZE, "orig_h": NIH_SIZE, "label": 1,
                     "boxes": scale_boxes(boxes, NIH_SIZE, NIH_SIZE), "split": "test_nih",
                     "patient_id": pid.get(img, "")})
    pos_patients = {r["patient_id"] for r in rows}
    pool = data_entry[(data_entry["Finding Labels"] == "No Finding")
                      & (~data_entry["Patient ID"].astype(str).isin(pos_patients))]
    neg = pool.sample(n=min(len(pool), neg_ratio * len(rows)), random_state=seed)
    for img, p in zip(neg["Image Index"], neg["Patient ID"].astype(str)):
        rows.append({"image_id": img, "dataset": "nih", "path": f"nih/{img}",
                     "orig_w": NIH_SIZE, "orig_h": NIH_SIZE, "label": 0, "boxes": [],
                     "split": "test_nih", "patient_id": p})
    return pd.DataFrame(rows)
```

Ca âm được duyệt bằng `zip` theo tên cột, nên không phụ thuộc thứ tự cột của `data_entry`.

- [ ] **Bước 5: Viết `strata.py`**

```python
"""Nodule size strata."""
from __future__ import annotations

import numpy as np

from nodulecxr.records import IMG_SIZE


def box_rel_area(box, size: int = IMG_SIZE) -> float:
    return (box[2] - box[0]) * (box[3] - box[1]) / float(size * size)


def tertile_edges(areas) -> tuple:
    q1, q2 = np.quantile(np.asarray(areas, dtype=float), [1 / 3, 2 / 3])
    return float(q1), float(q2)


def size_bin(area: float, edges) -> str:
    if area <= edges[0]:
        return "small"
    if area <= edges[1]:
        return "medium"
    return "large"
```

- [ ] **Bước 6: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_jsrt.py tests/test_nih.py tests/test_strata.py -q`
Kỳ vọng: `3 passed`.

- [ ] **Bước 7: Commit**

```bash
git add src/nodulecxr/data/jsrt.py src/nodulecxr/data/nih.py src/nodulecxr/strata.py tests/test_jsrt.py tests/test_nih.py tests/test_strata.py
git commit -m "feat: jsrt and nih manifests, size strata"
```

---

### Task 5: Script chuẩn bị dữ liệu và E0 (mô tả dữ liệu)

**Files**
- Create: `code/src/nodulecxr/data/stats.py`, `code/scripts/prepare_vinbig.py`, `code/scripts/prepare_jsrt.py`, `code/scripts/prepare_nih.py`, `code/scripts/draw_boxes.py`, `code/scripts/data_stats.py`
- Test: `code/tests/test_stats.py`

**Interfaces**
- Consumes: Task 1–4.
- Produces:
  - `P/data/processed/manifest_{vinbig,jsrt,nih}.csv`
  - Ảnh PNG 896 tại `P/data/processed/{vinbig,jsrt,nih}/`
  - `describe_manifest(df, edges) -> dict` gồm khóa `per_split` (list dict có `split, n, n_pos, prevalence`), `n_rads`, `size_bins`
  - `P/outputs/e0_data.md`

**Nguồn VinBigData.** Mặc định dùng bản PNG đã resize trên Kaggle (`xhlulu/vinbigdata-chest-xray-resized-png-1024x1024`), vì bản DICOM gốc khoảng 200 GB. Bản PNG kèm `train_meta.csv` với cột `image_id, dim0, dim1`, trong đó dim0 là chiều cao và dim1 là chiều rộng ảnh gốc. Bước 1 phải kiểm tra tên cột này. Nếu dùng tập test chính thức của VinDr-CXR (S8), dùng `--official-test-csv/--official-test-dicom` với ảnh DICOM.

- [ ] **Bước 1: Tải dữ liệu thô (Colab hoặc máy có ổ lớn)**

```bash
mkdir -p data/raw/vinbig data/raw/nih data/raw/jsrt
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection -f train.csv -p data/raw/vinbig
kaggle datasets download -d xhlulu/vinbigdata-chest-xray-resized-png-1024x1024 -p data/raw/vinbig --unzip
kaggle datasets download -d nih-chest-xrays/data -p data/raw/nih --unzip
python -c "import pandas as pd; print(pd.read_csv('data/raw/vinbig/train_meta.csv').head())"
```

Kỳ vọng: in ra các cột `image_id, dim0, dim1`. Nếu tên cột khác, ghi lại và dùng tham số `--h-col/--w-col` ở Bước 3.

Với JSRT: giải nén các file `.IMG`, `CLNDAT_EN.txt` và `NNDAT_EN.txt` vào `data/raw/jsrt/`.

- [ ] **Bước 2: Viết test cho `describe_manifest` (sẽ fail)**

```python
import pandas as pd
from nodulecxr.data.stats import describe_manifest


def test_describe_manifest_counts():
    df = pd.DataFrame({"split": ["train", "train", "val", "test"], "label": [1, 0, 0, 1],
                       "boxes": [[[0, 0, 10, 10]], [], [], [[0, 0, 300, 300]]],
                       "n_rads": [2, 0, 0, 1]})
    d = describe_manifest(df, edges=(0.001, 0.01))
    tr = [r for r in d["per_split"] if r["split"] == "train"][0]
    assert tr["n"] == 2 and tr["n_pos"] == 1 and tr["prevalence"] == 0.5
    assert d["n_rads"] == {1: 1, 2: 1}
    assert d["size_bins"] == {"small": 1, "large": 1}
```

Chạy: `pytest tests/test_stats.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 3: Viết `data/stats.py` và các script**

`src/nodulecxr/data/stats.py`:
```python
"""E0 descriptive statistics of a manifest."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from nodulecxr.strata import box_rel_area, size_bin


def describe_manifest(df: pd.DataFrame, edges) -> dict:
    per_split = []
    for split, g in df.groupby("split"):
        n, n_pos = len(g), int(g["label"].sum())
        per_split.append({"split": split, "n": n, "n_pos": n_pos, "prevalence": n_pos / n})
    pos = df[df["label"] == 1]
    n_rads = dict(Counter(int(v) for v in pos["n_rads"].dropna())) if "n_rads" in df else {}
    bins = Counter(size_bin(max(box_rel_area(b) for b in boxes), edges)
                   for boxes in pos["boxes"] if boxes)
    return {"per_split": per_split, "n_rads": n_rads, "size_bins": dict(bins)}


def to_markdown(name: str, d: dict) -> str:
    lines = [f"## {name}", "", "| split | n | n_pos | prevalence |", "|---|---|---|---|"]
    lines += [f"| {r['split']} | {r['n']} | {r['n_pos']} | {r['prevalence']:.4f} |" for r in d["per_split"]]
    lines += ["", f"- Số bác sĩ đồng ý (ca dương): {d['n_rads']}", f"- Tầng kích thước: {d['size_bins']}", ""]
    return "\n".join(lines)
```

`scripts/prepare_vinbig.py`:
```python
"""Build manifest_vinbig.csv and 896px PNGs from VinBigData."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from nodulecxr.data.split import stratified_split
from nodulecxr.data.vinbig import build_vinbig_labels
from nodulecxr.images import dicom_to_uint8, resize_square
from nodulecxr.records import scale_boxes, write_manifest


def dims_from_dicom(dicom_dir, ids):
    import pydicom
    rows = []
    for i in ids:
        ds = pydicom.dcmread(Path(dicom_dir) / f"{i}.dicom", stop_before_pixels=True)
        rows.append({"image_id": i, "orig_w": int(ds.Columns), "orig_h": int(ds.Rows)})
    return pd.DataFrame(rows)


def convert(labels, src_png_dir, src_dicom_dir, out_root, split_name=None):
    out_dir = Path(out_root) / "vinbig"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in tqdm(labels.itertuples(), total=len(labels)):
        dst = out_dir / f"{r.image_id}.png"
        if not dst.exists():
            if src_png_dir:
                img = Image.open(Path(src_png_dir) / f"{r.image_id}.png")
            else:
                img = dicom_to_uint8(Path(src_dicom_dir) / f"{r.image_id}.dicom")
            resize_square(img).save(dst)
        recs.append({"image_id": r.image_id, "dataset": "vinbig", "path": f"vinbig/{r.image_id}.png",
                     "orig_w": r.orig_w, "orig_h": r.orig_h, "label": r.label,
                     "boxes": scale_boxes(r.boxes, r.orig_w, r.orig_h), "n_rads": r.n_rads,
                     "split": split_name})
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--png-dir")
    ap.add_argument("--meta-csv")
    ap.add_argument("--h-col", default="dim0")
    ap.add_argument("--w-col", default="dim1")
    ap.add_argument("--dicom-dir")
    ap.add_argument("--official-test-csv")
    ap.add_argument("--official-test-dicom")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ann = pd.read_csv(a.train_csv)
    if a.meta_csv:
        dims = pd.read_csv(a.meta_csv).rename(columns={a.h_col: "orig_h", a.w_col: "orig_w"})
    else:
        dims = dims_from_dicom(a.dicom_dir, ann["image_id"].unique())
    labels = build_vinbig_labels(ann, dims[["image_id", "orig_w", "orig_h"]])
    man = convert(labels, a.png_dir, a.dicom_dir, a.out_root)
    ratios = (0.85, 0.15, 0.0) if a.official_test_csv else (0.7, 0.1, 0.2)
    man["split"] = stratified_split(man.set_index("image_id")["label"], ratios, a.seed).to_numpy()
    if a.official_test_csv:
        t_ann = pd.read_csv(a.official_test_csv)
        t_dims = dims_from_dicom(a.official_test_dicom, t_ann["image_id"].unique())
        t_labels = build_vinbig_labels(t_ann, t_dims)
        man = pd.concat([man, convert(t_labels, None, a.official_test_dicom, a.out_root, "test")])
    write_manifest(man, Path(a.out_root) / "manifest_vinbig.csv")
    print(man.groupby("split")["label"].agg(["count", "sum"]))


if __name__ == "__main__":
    main()
```

`scripts/prepare_jsrt.py`:
```python
"""Build manifest_jsrt.csv and 896px PNGs from JSRT .IMG files."""
import argparse
from pathlib import Path

from nodulecxr.data.jsrt import build_jsrt_manifest, parse_clndat, parse_nndat
from nodulecxr.images import read_jsrt_img, resize_square
from nodulecxr.records import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    cln = parse_clndat((raw / "CLNDAT_EN.txt").read_text(encoding="latin-1"))
    nn = parse_nndat((raw / "NNDAT_EN.txt").read_text(encoding="latin-1"))
    man = build_jsrt_manifest(cln, nn)
    out = Path(a.out_root) / "jsrt"
    out.mkdir(parents=True, exist_ok=True)
    for stem in man["image_id"]:
        src = next(raw.rglob(f"{stem}.IMG"), None) or next(raw.rglob(f"{stem.lower()}.img"))
        resize_square(read_jsrt_img(src)).save(out / f"{stem}.png")
    write_manifest(man, Path(a.out_root) / "manifest_jsrt.csv")
    print(man["label"].value_counts())


if __name__ == "__main__":
    main()
```

`scripts/prepare_nih.py`:
```python
"""Build manifest_nih.csv and 896px PNGs from NIH ChestX-ray14."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from nodulecxr.data.nih import build_nih_manifest
from nodulecxr.images import resize_square
from nodulecxr.records import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--neg-ratio", type=int, default=3)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    man = build_nih_manifest(pd.read_csv(raw / "Data_Entry_2017.csv"),
                             pd.read_csv(raw / "BBox_List_2017.csv"), a.neg_ratio, seed=0)
    index = {p.name: p for p in raw.rglob("*.png")}
    out = Path(a.out_root) / "nih"
    out.mkdir(parents=True, exist_ok=True)
    for img in man["image_id"]:
        resize_square(Image.open(index[img])).save(out / img)
    write_manifest(man, Path(a.out_root) / "manifest_nih.csv")
    print(man["label"].value_counts())


if __name__ == "__main__":
    main()
```

`scripts/draw_boxes.py`:
```python
"""QA: draw ground-truth boxes on n positive images per manifest."""
import argparse
from pathlib import Path

from PIL import ImageDraw

from nodulecxr.images import load_image
from nodulecxr.records import read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    df = read_manifest(a.manifest)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for r in df[df.label == 1].head(a.n).itertuples():
        im = load_image(a.data_root, r.path)
        d = ImageDraw.Draw(im)
        for b in r.boxes:
            d.rectangle(b, outline=(255, 0, 0), width=3)
        im.save(Path(a.out) / f"{r.dataset}_{r.image_id}.png")


if __name__ == "__main__":
    main()
```

`scripts/data_stats.py`:
```python
"""E0: descriptive table for all manifests -> outputs/e0_data.md."""
import argparse

import pandas as pd

from nodulecxr.data.stats import describe_manifest, to_markdown
from nodulecxr.records import read_manifest
from nodulecxr.strata import box_rel_area, tertile_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dfs = {m: read_manifest(m) for m in a.manifests}
    vin = next(d for m, d in dfs.items() if "vinbig" in m)
    train_pos = vin[(vin.split == "train") & (vin.label == 1)]
    edges = tertile_edges([max(box_rel_area(b) for b in bs) for bs in train_pos.boxes])
    text = [f"# E0: mô tả dữ liệu\n\nTertile diện tích box (tương đối, từ VinBigData train): {edges}\n"]
    for m, d in dfs.items():
        text.append(to_markdown(m, describe_manifest(d, edges)))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    pd.Series(edges).to_json(a.out.replace(".md", "_edges.json"))


if __name__ == "__main__":
    main()
```

Chạy: `pytest tests/test_stats.py -q`. Kỳ vọng: `1 passed`.

- [ ] **Bước 4: Chạy trên dữ liệu thật và kiểm tra bằng mắt**

```bash
python scripts/prepare_vinbig.py --train-csv data/raw/vinbig/train.csv --png-dir data/raw/vinbig/train --meta-csv data/raw/vinbig/train_meta.csv --out-root data/processed
python scripts/prepare_jsrt.py --raw-dir data/raw/jsrt --out-root data/processed
python scripts/prepare_nih.py --raw-dir data/raw/nih --out-root data/processed
for m in vinbig jsrt nih; do python scripts/draw_boxes.py --manifest data/processed/manifest_$m.csv --data-root data/processed --out outputs/qa; done
python scripts/data_stats.py --manifests data/processed/manifest_vinbig.csv data/processed/manifest_jsrt.csv data/processed/manifest_nih.csv --out outputs/e0_data.md
```

Kỳ vọng:
- VinBigData có khoảng 15.000 ảnh, khoảng 400 ca dương (con số của VinDr-CXR theo [L37] là 410).
- JSRT có 154 ca dương và 93 ca âm.
- NIH có khoảng 150 ca dương và 3 lần số đó ca âm.
- Mở `outputs/qa/*.png` và kiểm tra:
  - box nằm đúng trên nốt;
  - phổi JSRT có màu tối (nếu sáng, chạy lại với `invert=False`).

Nếu box VinBigData bị lệch, nghĩa là bản PNG giữ tỷ lệ khung hình kèm phần đệm. Khi đó phải chuyển sang `--dicom-dir`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/data/stats.py scripts/prepare_*.py scripts/draw_boxes.py scripts/data_stats.py tests/test_stats.py
git commit -m "feat: data preparation scripts and E0 stats"
```

---

### Task 6: Prompt, đáp án đích và parser

**Files**
- Create: `code/src/nodulecxr/prompts.py`
- Test: `code/tests/test_prompts.py`

**Interfaces**
- Produces:
  - `QUESTION_LOC: str` (câu hỏi cố định của spec, dùng cho C0, C2, C3, C4)
  - `QUESTION_CLS: str` (dùng cho C1)
  - `ANSWER_PREFIX = '{"nodule": "'`
  - `build_target(label: int, boxes, with_boxes=True) -> str`
  - `ParsedAnswer(label: int | None, boxes: list, ok: bool)`
  - `parse_answer(text, size=896) -> ParsedAnswer`

**Quy tắc parser.**
- JSON hợp lệ có khóa `nodule` mang giá trị yes/no thì `ok = True`.
- Chỉ bắt được nhãn bằng regex thì `ok = False`, vẫn có nhãn.
- Không đọc được gì thì `label = None`, `ok = False`. Ở bước tính metric, `None` được tính là "no" và báo riêng tỷ lệ parse lỗi.
- Box được cắt về khoảng [0, 896]; box suy biến (cạnh < 1 px) bị bỏ. Chấp nhận cả dạng `{"bbox_2d": [...]}` của Qwen.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
from nodulecxr.prompts import ANSWER_PREFIX, QUESTION_LOC, build_target, parse_answer


def test_question_matches_spec():
    assert QUESTION_LOC == ("Is there a lung nodule or mass on this chest X-ray? If yes, give the "
                            "bounding box(es) in pixel coordinates. Answer in JSON.")


def test_target_format_and_prefix():
    t = build_target(1, [[10.4, 20.6, 30.0, 40.0]])
    assert t == '{"nodule": "yes", "boxes": [[10, 21, 30, 40]]}' and t.startswith(ANSWER_PREFIX)
    assert build_target(0, []) == '{"nodule": "no", "boxes": []}'
    assert build_target(1, [[1, 2, 3, 4]], with_boxes=False) == '{"nodule": "yes"}'


def test_roundtrip_and_fences():
    p = parse_answer("```json\n" + build_target(1, [[10, 20, 30, 40]]) + "\n```")
    assert p.ok and p.label == 1 and p.boxes == [[10.0, 20.0, 30.0, 40.0]]


def test_clip_drop_and_qwen_format():
    p = parse_answer('{"nodule": "yes", "boxes": [[-5, 10, 2000, 50], [5, 5, 5.5, 9]]}')
    assert p.boxes == [[0.0, 10.0, 896.0, 50.0]]
    q = parse_answer('{"nodule": "yes", "boxes": [{"bbox_2d": [1, 2, 30, 40], "label": "nodule"}]}')
    assert q.boxes == [[1.0, 2.0, 30.0, 40.0]]


def test_no_drops_boxes_and_fallbacks():
    assert parse_answer('{"nodule": "no", "boxes": [[1,2,3,4]]}').boxes == []
    f = parse_answer('Answer: nodule: yes, right upper lobe')
    assert f.label == 1 and not f.ok
    g = parse_answer("No acute cardiopulmonary abnormality.")
    assert g.label is None and not g.ok
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_prompts.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 3: Viết code**

```python
"""Fixed prompts, target strings and a tolerant JSON answer parser."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from nodulecxr.records import IMG_SIZE

QUESTION_LOC = ("Is there a lung nodule or mass on this chest X-ray? If yes, give the bounding "
                "box(es) in pixel coordinates. Answer in JSON.")
QUESTION_CLS = "Is there a lung nodule or mass on this chest X-ray? Answer in JSON."
ANSWER_PREFIX = '{"nodule": "'

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
_JSON_RE = re.compile(r"\{.*\}", re.S)
_LABEL_RE = re.compile(r'"?nodule"?\s*[:=]\s*"?(yes|no)\b', re.I)


def build_target(label: int, boxes, with_boxes: bool = True) -> str:
    d = {"nodule": "yes" if label else "no"}
    if with_boxes:
        d["boxes"] = [[int(round(v)) for v in b[:4]] for b in boxes] if label else []
    return json.dumps(d)


@dataclass
class ParsedAnswer:
    label: Optional[int]
    boxes: list = field(default_factory=list)
    ok: bool = False


def _clean_boxes(raw, size: int) -> list:
    out = []
    for b in raw if isinstance(raw, list) else []:
        if isinstance(b, dict):
            b = b.get("bbox_2d") or b.get("bbox") or b.get("box")
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b[:4])
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((min(max(x1, 0.0), size), min(max(x2, 0.0), size)))
        y1, y2 = sorted((min(max(y1, 0.0), size), min(max(y2, 0.0), size)))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            out.append([x1, y1, x2, y2])
    return out


def parse_answer(text: str, size: int = IMG_SIZE) -> ParsedAnswer:
    t = _FENCE_RE.sub("", text or "").strip()
    m = _JSON_RE.search(t)
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = None
        if isinstance(d, dict) and str(d.get("nodule", "")).lower() in ("yes", "no"):
            label = 1 if str(d["nodule"]).lower() == "yes" else 0
            return ParsedAnswer(label, _clean_boxes(d.get("boxes", []), size) if label else [], True)
    m = _LABEL_RE.search(t)
    if m:
        return ParsedAnswer(1 if m.group(1).lower() == "yes" else 0, [], False)
    return ParsedAnswer(None, [], False)
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_prompts.py -q`. Kỳ vọng: `5 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/prompts.py tests/test_prompts.py
git commit -m "feat: prompts, targets and answer parser"
```

---

### Task 7: Chỉ số phát hiện mức ảnh

**Files**
- Create: `code/src/nodulecxr/metrics/detection.py`
- Test: `code/tests/test_detection.py`

**Interfaces**
- Produces:
  - `binary_metrics(y, yhat) -> dict` với các khóa `sens, spec, ppv, npv, f1, tp, fp, tn, fn`
  - `threshold_at_specificity(y, s, target=0.90) -> float` (ngưỡng nhỏ nhất đạt spec ≥ target; dương khi `s >= t`)
  - `predict_at(s, t) -> np.ndarray[int]`
  - `score_metrics(y, s) -> dict` với các khóa `auc_roc, auc_pr, brier, balanced_brier, ece`
  - `ppv_at_prevalence(sens, spec, prev) -> float`

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import math

import numpy as np
import pytest
from nodulecxr.metrics.detection import (binary_metrics, ppv_at_prevalence, predict_at,
                                         score_metrics, threshold_at_specificity)


def test_binary_metrics():
    m = binary_metrics([1, 1, 0, 0, 0], [1, 0, 0, 0, 1])
    assert m["sens"] == 0.5 and m["spec"] == pytest.approx(2 / 3) and m["ppv"] == 0.5
    assert math.isnan(binary_metrics([0, 0], [0, 0])["sens"])


def test_threshold_from_validation():
    y = np.array([0] * 10 + [1, 1])
    s = np.array([i / 10 for i in range(10)] + [0.95, 0.5])
    t = threshold_at_specificity(y, s, 0.90)
    assert t == pytest.approx(0.9)
    m = binary_metrics(y, predict_at(s, t))
    assert m["spec"] == pytest.approx(0.9) and m["sens"] == 0.5


def test_score_metrics_and_ppv():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    m = score_metrics(y, s)
    assert m["auc_roc"] == 1.0 and m["auc_pr"] == 1.0 and m["ece"] == 0.0 and m["brier"] == 0.0
    assert ppv_at_prevalence(0.6, 0.9, 0.02) == pytest.approx(0.012 / (0.012 + 0.098))
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_detection.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 3: Viết code**

```python
"""Image-level detection metrics, threshold selection and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _div(a, b) -> float:
    return float(a) / float(b) if b else float("nan")


def binary_metrics(y, yhat) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return {"sens": _div(tp, tp + fn), "spec": _div(tn, tn + fp), "ppv": _div(tp, tp + fp),
            "npv": _div(tn, tn + fn), "f1": _div(2 * tp, 2 * tp + fp + fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def threshold_at_specificity(y, s, target: float = 0.90) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    neg = s[y == 0]
    for t in np.unique(np.concatenate([s, [np.inf]])):
        if np.mean(neg < t) >= target:
            return float(t)
    return float("inf")


def predict_at(s, t: float) -> np.ndarray:
    return (np.asarray(s, dtype=float) >= t).astype(int)


def ece(y, s, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    bins = np.minimum((s * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += abs(y[m].mean() - s[m].mean()) * m.sum()
    return float(total / len(y))


def score_metrics(y, s) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    both = len(np.unique(y)) == 2
    brier_pos = float(np.mean((s[y == 1] - 1) ** 2)) if (y == 1).any() else float("nan")
    brier_neg = float(np.mean(s[y == 0] ** 2)) if (y == 0).any() else float("nan")
    return {"auc_roc": float(roc_auc_score(y, s)) if both else float("nan"),
            "auc_pr": float(average_precision_score(y, s)) if both else float("nan"),
            "brier": float(np.mean((s - y) ** 2)),
            "balanced_brier": float(np.nanmean([brier_pos, brier_neg])),
            "ece": ece(y, s)}


def ppv_at_prevalence(sens: float, spec: float, prev: float) -> float:
    return _div(sens * prev, sens * prev + (1 - spec) * (1 - prev))
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_detection.py -q`. Kỳ vọng: `3 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/metrics/detection.py tests/test_detection.py
git commit -m "feat: detection and calibration metrics"
```

---

### Task 8: Thống kê (bootstrap, McNemar, DeLong, Holm)

**Files**
- Create: `code/src/nodulecxr/metrics/stats.py`
- Test: `code/tests/test_stats_tests.py`

**Interfaces**
- Produces:
  - `bootstrap_ci(metric_fn, y, *arrays, n_boot=2000, alpha=0.05, seed=0) -> (point, lo, hi)`
    - Bootstrap phân tầng theo nhãn, để giữ tỷ lệ bệnh.
    - `metric_fn(y, *arrays) -> float`.
  - `mcnemar_p(correct_a, correct_b) -> float` (kiểm định chính xác)
  - `delong_test(y, s1, s2) -> (auc1, auc2, p)`
  - `holm(pvals) -> list[float]`

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from nodulecxr.metrics.stats import bootstrap_ci, delong_test, holm, mcnemar_p


def test_bootstrap_contains_point():
    rng = np.random.default_rng(0)
    y = np.array([1] * 50 + [0] * 150)
    s = np.where(y == 1, rng.normal(1, 1, 200), rng.normal(0, 1, 200))
    point, lo, hi = bootstrap_ci(roc_auc_score, y, s, n_boot=300)
    assert lo < point < hi and 0.6 < point < 0.9


def test_mcnemar_identical_is_one():
    a = np.array([True, False, True, True])
    assert mcnemar_p(a, a) == 1.0


def test_delong_matches_sklearn_and_detects_difference():
    rng = np.random.default_rng(1)
    y = np.array([1] * 200 + [0] * 200)
    good = np.where(y == 1, rng.normal(2, 1, 400), rng.normal(0, 1, 400))
    bad = rng.normal(0, 1, 400)
    a1, a2, p = delong_test(y, good, bad)
    assert a1 == pytest.approx(roc_auc_score(y, good)) and a2 == pytest.approx(roc_auc_score(y, bad))
    assert p < 1e-6
    assert delong_test(y, good, good)[2] == 1.0


def test_holm_monotone():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_stats_tests.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 3: Viết code**

```python
"""Confidence intervals and paired tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


def bootstrap_ci(metric_fn, y, *arrays, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    arrays = [np.asarray(a) for a in arrays]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v = metric_fn(y[idx], *[a[idx] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    point = float(metric_fn(y, *arrays))
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mcnemar_p(correct_a, correct_b) -> float:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    table = [[int(np.sum(a & b)), int(np.sum(a & ~b))], [int(np.sum(~a & b)), int(np.sum(~a & ~b))]]
    return float(mcnemar(table, exact=True).pvalue)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_test(y, s1, s2):
    y = np.asarray(y).astype(bool)
    scores = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.concatenate([p, q])) for p, q in zip(pos, neg)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))


def holm(pvals) -> list:
    return multipletests(list(pvals), method="holm")[1].tolist()
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_stats_tests.py -q`. Kỳ vọng: `4 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/metrics/stats.py tests/test_stats_tests.py
git commit -m "feat: bootstrap CI, McNemar, DeLong, Holm"
```

---

### Task 9: Chỉ số định vị (IoU, hit, JSRT, FROC)

**Files**
- Create: `code/src/nodulecxr/metrics/localization.py`
- Test: `code/tests/test_localization.py`

**Interfaces**
- Produces:
  - `iou(a, b) -> float`
  - `best_iou(preds, gts) -> float` (0 nếu một trong hai rỗng)
  - `jsrt_hit(pred_box, gt_box, margin_px) -> bool` (tâm box dự đoán cách tâm nốt ≤ bán kính + biên)
  - `center_distance(pred_box, gt_box) -> float`
  - `froc_points(cases, iou_thr=0.3, fp_rates=(0.125, 0.25, 0.5, 1, 2, 4, 8)) -> dict[float, float]`
    - `cases` là list các cặp `(preds [[x1, y1, x2, y2, s], ...], gts [[x1, y1, x2, y2], ...])`.

**Quy ước JSRT.** Biên là 5 mm = `5 / JSRT_MM_PER_PX_896` = 12,5 px. Khoảng cách được báo bằng mm, bằng cách nhân với 0,4.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import pytest
from nodulecxr.metrics.localization import best_iou, center_distance, froc_points, iou, jsrt_hit


def test_iou_and_best():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert best_iou([], [[0, 0, 1, 1]]) == 0.0
    assert best_iou([[0, 0, 10, 10, 0.9]], [[0, 0, 10, 10]]) == 1.0


def test_jsrt_hit_and_distance():
    gt = [90, 90, 110, 110]               # radius 10
    assert jsrt_hit([100, 100, 120, 120], gt, margin_px=0) is False   # centre 110,110 -> dist 14.1
    assert jsrt_hit([100, 100, 120, 120], gt, margin_px=12.5) is True
    assert center_distance([0, 0, 2, 2], [3, 4, 5, 6]) == pytest.approx(5.0)


def test_froc_tp_then_fp():
    cases = [([[0, 0, 10, 10, 0.9], [50, 50, 60, 60, 0.8]], [[0, 0, 10, 10]])]
    pts = froc_points(cases, fp_rates=(0.5, 1))
    assert pts == {0.5: 1.0, 1: 1.0}
    pts2 = froc_points([([[50, 50, 60, 60, 0.9], [0, 0, 10, 10, 0.8]], [[0, 0, 10, 10]])], fp_rates=(0.5, 1))
    assert pts2 == {0.5: 0.0, 1: 1.0}
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_localization.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 3: Viết code**

```python
"""Localisation metrics: IoU, centre hits and FROC."""
from __future__ import annotations

import math


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_iou(preds, gts) -> float:
    if not preds or not gts:
        return 0.0
    return max(iou(p[:4], g) for p in preds for g in gts)


def _centre(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def center_distance(pred_box, gt_box) -> float:
    (px, py), (gx, gy) = _centre(pred_box), _centre(gt_box)
    return math.hypot(px - gx, py - gy)


def jsrt_hit(pred_box, gt_box, margin_px: float) -> bool:
    radius = (gt_box[2] - gt_box[0]) / 2.0
    return center_distance(pred_box, gt_box) <= radius + margin_px


def _match_image(preds, gts, thr):
    used, out = set(), []
    for p in sorted(preds, key=lambda q: -q[4]):
        best, bj = -1.0, None
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p[:4], g)
            if v >= thr and v > best:
                best, bj = v, j
        if bj is None:
            out.append((p[4], False))
        else:
            used.add(bj)
            out.append((p[4], True))
    return out


def froc_points(cases, iou_thr: float = 0.3, fp_rates=(0.125, 0.25, 0.5, 1, 2, 4, 8)) -> dict:
    n_gt = sum(len(g) for _, g in cases)
    if n_gt == 0:
        raise ValueError("FROC needs at least one ground-truth box")
    n_img = len(cases)
    scored = []
    for preds, gts in cases:
        scored += _match_image(preds, gts, iou_thr)
    scored.sort(key=lambda x: -x[0])
    tp = fp = 0
    curve = [(0.0, 0.0)]
    for i, (s, is_tp) in enumerate(scored):
        tp += int(is_tp)
        fp += int(not is_tp)
        if i + 1 == len(scored) or scored[i + 1][0] != s:
            curve.append((fp / n_img, tp / n_gt))
    return {r: max(sens for f, sens in curve if f <= r) for r in fp_rates}
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_localization.py -q`. Kỳ vọng: `3 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/metrics/localization.py tests/test_localization.py
git commit -m "feat: localisation metrics and FROC"
```

---

### Task 10: Công cụ kiểm toán che vùng

**Files**
- Create: `code/src/nodulecxr/audit/occlusion.py`
- Test: `code/tests/test_occlusion.py`

**Interfaces**
- Produces:
  - `expand_box(box, frac=0.10, size=896) -> list[float]`
  - `occlude(img: PIL.Image, boxes, fill=None) -> PIL.Image` (mặc định tô bằng trung bình độ xám của ảnh)
  - `boxes_intersect(a, b) -> bool`
  - `sample_random_box(w, h, lung_mask: np.ndarray[bool], forbidden, rng, min_inside=0.9, max_tries=2000) -> list | None`
  - `heuristic_lung_mask(size=896) -> np.ndarray[bool]` (dự phòng khi không có mô hình phân đoạn)
  - `flip_rate(orig_pos: array[int], new_pos: array[int]) -> float` (tỷ lệ đổi từ 1 sang 0 trong các ca gốc là 1)

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import math

import numpy as np
from PIL import Image
from nodulecxr.audit.occlusion import (boxes_intersect, expand_box, flip_rate, heuristic_lung_mask,
                                       occlude, sample_random_box)


def test_expand_and_occlude_with_mean():
    assert expand_box([100, 100, 200, 200]) == [95.0, 95.0, 205.0, 205.0]
    arr = np.zeros((896, 896), dtype=np.uint8)
    arr[:, 448:] = 200
    out = np.array(occlude(Image.fromarray(arr).convert("RGB"), [[0, 0, 10, 10]]).convert("L"))
    assert out[5, 5] == 100 and out[500, 500] == 200


def test_random_box_inside_mask_and_disjoint():
    mask = heuristic_lung_mask()
    rng = np.random.default_rng(0)
    nod = [200, 300, 260, 360]
    b = sample_random_box(60, 60, mask, [nod], rng)
    x1, y1, x2, y2 = map(int, b)
    assert mask[y1:y2, x1:x2].mean() >= 0.9 and not boxes_intersect(b, nod)
    assert sample_random_box(900, 900, mask, [], rng) is None


def test_flip_rate():
    assert flip_rate([1, 1, 0, 1], [0, 1, 1, 0]) == 2 / 3
    assert math.isnan(flip_rate([0, 0], [1, 1]))
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_occlusion.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 3: Viết code**

```python
"""Occlusion audit utilities (T1 nodule, T2 random same-area, T3 swap)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from nodulecxr.records import IMG_SIZE


def expand_box(box, frac: float = 0.10, size: int = IMG_SIZE) -> list:
    w, h = box[2] - box[0], box[3] - box[1]
    dx, dy = w * frac / 2, h * frac / 2
    return [max(0.0, box[0] - dx), max(0.0, box[1] - dy), min(float(size), box[2] + dx),
            min(float(size), box[3] + dy)]


def occlude(img: Image.Image, boxes, fill=None) -> Image.Image:
    a = np.array(img).copy()
    value = int(round(a.mean())) if fill is None else fill
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b[:4])
        a[y1:y2, x1:x2] = value
    return Image.fromarray(a)


def boxes_intersect(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def sample_random_box(w, h, lung_mask, forbidden, rng, min_inside: float = 0.9, max_tries: int = 2000):
    H, W = lung_mask.shape
    w, h = int(round(w)), int(round(h))
    if w >= W or h >= H or w < 1 or h < 1:
        return None
    for _ in range(max_tries):
        x1 = int(rng.integers(0, W - w + 1))
        y1 = int(rng.integers(0, H - h + 1))
        box = [float(x1), float(y1), float(x1 + w), float(y1 + h)]
        if lung_mask[y1:y1 + h, x1:x1 + w].mean() >= min_inside and \
                not any(boxes_intersect(box, f) for f in forbidden):
            return box
    return None


def heuristic_lung_mask(size: int = IMG_SIZE) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    y1, y2 = int(0.15 * size), int(0.85 * size)
    m[y1:y2, int(0.12 * size):int(0.47 * size)] = True
    m[y1:y2, int(0.53 * size):int(0.88 * size)] = True
    return m


def flip_rate(orig_pos, new_pos) -> float:
    o = np.asarray(orig_pos).astype(int)
    n = np.asarray(new_pos).astype(int)
    sel = o == 1
    return float(np.mean(n[sel] == 0)) if sel.any() else float("nan")
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_occlusion.py -q`. Kỳ vọng: `3 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/audit/occlusion.py tests/test_occlusion.py
git commit -m "feat: occlusion audit utilities"
```

---

### Task 11: VLM wrapper, chấm điểm yes/no, vòng dự đoán và `predict_vlm.py`

**Files**
- Create: `code/src/nodulecxr/vlm/chat.py`, `code/src/nodulecxr/vlm/scoring.py`, `code/src/nodulecxr/vlm/hf.py`, `code/src/nodulecxr/vlm/predict.py`, `code/scripts/predict_vlm.py`
- Test: `code/tests/test_vlm.py`

**Interfaces**
- Consumes: `ANSWER_PREFIX`, `QUESTION_LOC`, `QUESTION_CLS`, `parse_answer` (Task 6); `append_jsonl`, `done_ids` (Task 1); `load_image` (Task 3).
- Produces:
  - `build_chat_text(processor, prompt, answer=None) -> str`
  - `first_divergence(a, b) -> int`
  - `two_way_probability(logp_yes, logp_no) -> float`
  - `token_logprob_sum(logits, ids, start) -> float` (torch)
  - `MODEL_IDS = {"medgemma": "google/medgemma-4b-it", "qwen": "Qwen/Qwen2.5-VL-3B-Instruct"}`
  - `load_base(key, dtype=None) -> (model, processor)`
  - `class HFVLM(key, adapter_path=None)` với các phương thức:
    - `generate(image, prompt, temperature=0.0, max_new_tokens=96, seed=None) -> str`
    - `yes_probability(image, prompt) -> float`
  - `iter_predictions(vlm, records, prompt, load_image, temperature=0.0, seed=None, generate=True, max_new_tokens=96, meta=None, skip=())`, sinh ra từng dict theo định dạng JSONL

**Cách lấy điểm liên tục (spec §8.1).** Đưa vào mô hình hai chuỗi `ANSWER_PREFIX + 'yes"'` và `ANSWER_PREFIX + 'no"'` (teacher-forcing). Tính tổng log-prob của các token sau điểm hai chuỗi tách nhau, rồi lấy `p_yes = softmax(lp_yes, lp_no)[0]`. Cách này không phụ thuộc việc tokenizer có gộp `"yes` thành một token hay không. Box của VLM dùng `score` của ảnh làm điểm cho FROC.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import math

import pandas as pd
import pytest
from PIL import Image
from nodulecxr.vlm.chat import build_chat_text
from nodulecxr.vlm.predict import iter_predictions
from nodulecxr.vlm.scoring import first_divergence, two_way_probability


class FakeProcessor:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "|".join(f"{m['role']}:" + "".join(c.get("text", "<img>") for c in m["content"]) for m in msgs)
        return s + ("|assistant:" if add_generation_prompt else "")


def test_chat_text_prefix_property():
    p = FakeProcessor()
    head = build_chat_text(p, "Q")
    full = build_chat_text(p, "Q", "A")
    assert head == "user:<img>Q|assistant:" and full == "user:<img>Q|assistant:A"
    assert full.startswith(head)


def test_scoring_helpers():
    assert first_divergence([1, 2, 3, 9], [1, 2, 4]) == 2
    assert two_way_probability(math.log(0.3), math.log(0.1)) == pytest.approx(0.75)


class FakeVLM:
    def yes_probability(self, image, prompt):
        return 0.8

    def generate(self, image, prompt, temperature=0.0, max_new_tokens=96, seed=None):
        return '{"nodule": "yes", "boxes": [[10, 10, 50, 50]]}'


def test_iter_predictions_rows_and_skip():
    recs = pd.DataFrame({"image_id": ["a", "b"], "dataset": "vinbig", "split": "test", "path": ["a", "b"]})
    rows = list(iter_predictions(FakeVLM(), recs, "Q", lambda p: Image.new("RGB", (896, 896)),
                                 meta={"config": "C0", "seed": 0, "run": 0}, skip={"b"}))
    assert len(rows) == 1
    r = rows[0]
    assert r["image_id"] == "a" and r["score"] == 0.8 and r["label_pred"] == 1 and r["parse_ok"]
    assert r["boxes_pred"] == [[10.0, 10.0, 50.0, 50.0, 0.8]] and r["config"] == "C0"
```

Thêm test cho `token_logprob_sum`, chạy được khi có torch:

```python
def test_token_logprob_sum():
    torch = pytest.importorskip("torch")
    from nodulecxr.vlm.scoring import token_logprob_sum
    logits = torch.log(torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]]))
    ids = torch.tensor([0, 1, 0])
    # tokens from position 1: P(ids[1]=1 | pos0)=0.5, P(ids[2]=0 | pos1)=0.25
    assert token_logprob_sum(logits, ids, 1) == pytest.approx(math.log(0.5) + math.log(0.25))
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_vlm.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 3: Viết `vlm/chat.py` và `vlm/scoring.py`**

`vlm/chat.py`:
```python
"""Chat-template text shared by inference and training (no heavy imports)."""
from __future__ import annotations


def build_chat_text(processor, prompt: str, answer=None) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=answer is None)
```

`vlm/scoring.py`:
```python
"""Turn two teacher-forced continuations into P(yes)."""
from __future__ import annotations

import math


def first_divergence(a, b) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def two_way_probability(logp_yes: float, logp_no: float) -> float:
    m = max(logp_yes, logp_no)
    ey, en = math.exp(logp_yes - m), math.exp(logp_no - m)
    return ey / (ey + en)


def token_logprob_sum(logits, ids, start: int) -> float:
    import torch
    logp = torch.log_softmax(logits[start - 1:-1].float(), dim=-1)
    return float(logp.gather(1, ids[start:].unsqueeze(1)).sum())
```

- [ ] **Bước 4: Viết `vlm/predict.py`**

```python
"""Model-agnostic prediction loop producing JSONL rows."""
from __future__ import annotations

from nodulecxr.prompts import ParsedAnswer, parse_answer


def iter_predictions(vlm, records, prompt, load_image, temperature=0.0, seed=None, generate=True,
                     max_new_tokens=96, meta=None, skip=()):
    for r in records.itertuples():
        if r.image_id in skip:
            continue
        img = load_image(r.path)
        score = float(vlm.yes_probability(img, prompt))
        if generate:
            text = vlm.generate(img, prompt, temperature=temperature,
                                max_new_tokens=max_new_tokens, seed=seed)
            pa = parse_answer(text)
        else:
            text, pa = "", ParsedAnswer(None, [], False)
        yield {"image_id": r.image_id, "dataset": r.dataset, "split": r.split, "score": score,
               "label_pred": pa.label, "boxes_pred": [b + [score] for b in pa.boxes],
               "parse_ok": pa.ok, "raw_text": text, **(meta or {})}
```

- [ ] **Bước 5: Viết `vlm/hf.py`** (cần GPU; không có unit test, được kiểm ở Bước 8)

```python
"""Hugging Face loaders and inference for MedGemma-4B-it and Qwen2.5-VL-3B."""
from __future__ import annotations

from nodulecxr.prompts import ANSWER_PREFIX
from nodulecxr.records import IMG_SIZE
from nodulecxr.vlm.chat import build_chat_text
from nodulecxr.vlm.scoring import first_divergence, token_logprob_sum, two_way_probability

MODEL_IDS = {"medgemma": "google/medgemma-4b-it", "qwen": "Qwen/Qwen2.5-VL-3B-Instruct"}


def load_base(key: str, dtype=None):
    import torch
    from transformers import AutoProcessor
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    mid = MODEL_IDS[key]
    if key == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
        processor = AutoProcessor.from_pretrained(mid, min_pixels=IMG_SIZE * IMG_SIZE,
                                                  max_pixels=IMG_SIZE * IMG_SIZE)
    else:
        from transformers import AutoModelForImageTextToText as M
        processor = AutoProcessor.from_pretrained(mid)
    model = M.from_pretrained(mid, torch_dtype=dtype, device_map="auto")
    return model, processor


class HFVLM:
    def __init__(self, key: str, adapter_path=None):
        import torch
        self.torch = torch
        self.model, self.processor = load_base(key)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _encode(self, text, image):
        enc = self.processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
        return enc.to(self.model.device, dtype=self.model.dtype)

    def generate(self, image, prompt, temperature=0.0, max_new_tokens=96, seed=None) -> str:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        enc = self._encode(build_chat_text(self.processor, prompt), image)
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        n = enc["input_ids"].shape[1]
        return self.processor.tokenizer.decode(out[0, n:], skip_special_tokens=True)

    def _forced(self, image, prompt, stub):
        enc = self._encode(build_chat_text(self.processor, prompt) + stub, image)
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0]
        return logits, enc["input_ids"][0]

    def yes_probability(self, image, prompt) -> float:
        ly, iy = self._forced(image, prompt, ANSWER_PREFIX + 'yes"')
        ln, in_ = self._forced(image, prompt, ANSWER_PREFIX + 'no"')
        k = first_divergence(iy.tolist(), in_.tolist())
        return two_way_probability(token_logprob_sum(ly, iy, k), token_logprob_sum(ln, in_, k))
```

- [ ] **Bước 6: Viết `scripts/predict_vlm.py`**

```python
"""Run a VLM (zero-shot or with a LoRA adapter) over manifest splits -> JSONL (resumable)."""
import argparse

import pandas as pd
from tqdm import tqdm

from nodulecxr.images import load_image
from nodulecxr.prompts import QUESTION_CLS, QUESTION_LOC
from nodulecxr.records import append_jsonl, done_ids, read_manifest
from nodulecxr.vlm.hf import HFVLM
from nodulecxr.vlm.predict import iter_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["medgemma", "qwen"], required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--prompt", choices=["loc", "cls"], default="loc")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--generate", choices=["all", "none"], default="all")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([read_manifest(m) for m in a.manifests])
    df = df[df.split.isin(a.splits)]
    if a.limit:
        df = df.groupby("split", group_keys=False).apply(lambda g: g.head(a.limit))
    vlm = HFVLM(a.model, a.adapter)
    prompt = QUESTION_CLS if a.prompt == "cls" else QUESTION_LOC
    meta = {"config": a.config, "seed": a.seed, "run": a.run, "model": a.model}
    skip = done_ids(a.out, config=a.config, seed=a.seed, run=a.run)
    rows = iter_predictions(vlm, df, prompt, lambda p: load_image(a.data_root, p), a.temperature,
                            a.seed + 1000 * a.run, a.generate == "all", a.max_new_tokens, meta, skip)
    for row in tqdm(rows, total=len(df) - len(skip)):
        append_jsonl(row, a.out)


if __name__ == "__main__":
    main()
```

- [ ] **Bước 7: Chạy unit test, xác nhận pass**

Chạy: `pytest tests/test_vlm.py -q`. Kỳ vọng: `4 passed` (hoặc `3 passed, 1 skipped` nếu chưa cài torch).

- [ ] **Bước 8: Chạy thử trên Colab với 50 ảnh** ("Việc phải làm đầu tiên", mục 3)

```bash
python scripts/predict_vlm.py --model medgemma --manifests data/processed/manifest_vinbig.csv --splits val --limit 50 --data-root data/processed --config pilot --out outputs/pilot_medgemma.jsonl
python -c "import pandas as pd; d=pd.read_json('outputs/pilot_medgemma.jsonl', lines=True); print(d.parse_ok.mean(), d.score.describe()); print(d.raw_text.head(5).tolist())"
```

Kỳ vọng:
- Tệp có 50 dòng.
- `score` nằm trong [0, 1] và không phải tất cả bằng nhau.
- In ra tỷ lệ parse thành công và 5 câu trả lời mẫu.

Ghi lại:
- MedGemma có xuất được JSON và box không (rủi ro nêu ở spec §5.1);
- thời gian trung bình mỗi ảnh.

Lặp lại với `--model qwen`.

- [ ] **Bước 9: Commit**

```bash
git add src/nodulecxr/vlm scripts/predict_vlm.py tests/test_vlm.py
git commit -m "feat: VLM wrappers, yes-probability scoring and prediction script"
```

---

### Task 12: Dữ liệu SFT, collator, loss có trọng số, regex LoRA

**Files**
- Create: `code/src/nodulecxr/train/sft_data.py`, `code/src/nodulecxr/train/losses.py`, `code/src/nodulecxr/train/lora.py`
- Test: `code/tests/test_train.py`

**Interfaces**
- Consumes: `build_chat_text` (Task 11); `build_target`, `QUESTION_LOC`, `QUESTION_CLS` (Task 6); `oversample_indices` (Task 2).
- Produces:
  - `LORA_TARGET_REGEX: str`. Khớp `q/k/v/o/gate/up/down_proj` của LLM; **loại trừ** mọi module thuộc `visual | vision_tower | vision_model | multi_modal_projector`, nên bộ mã hóa ảnh đông cứng.
  - `build_example(processor, image, prompt, answer) -> dict[str, Tensor]`. Nhãn của phần prompt bằng −100.
  - `collate(batch, pad_id) -> dict[str, Tensor]`
  - `SFTDataset(df, processor, mode: "cls" | "loc", load_image, pos_weight=1.0, indices=None)`
  - `make_lora_config() -> peft.LoraConfig` (r = 16, α = 32, dropout 0,05)
  - `weighted_token_ce(logits, labels, weights) -> Tensor` (trong `train/losses.py`, không import transformers)
  - `WeightedSFTTrainer(Trainer)`: lấy `sample_weight` ra khỏi batch và dùng `weighted_token_ce`.

**Trọng số loss.** Ca dương có `sample_weight = pos_weight`, ca âm bằng 1. C1 và C2 dùng `pos_weight = 1`. C3 dùng oversampling 1:1 kèm `pos_weight = 2.0`; giá trị này được chọn trong {1, 2} trên validation ở Task 13 Bước 3.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import re

import pytest

torch = pytest.importorskip("torch")
from nodulecxr.train.losses import weighted_token_ce  # noqa: E402
from nodulecxr.train.sft_data import LORA_TARGET_REGEX, build_example, collate  # noqa: E402


def test_lora_regex_targets_llm_only():
    r = re.compile(LORA_TARGET_REGEX)
    assert r.fullmatch("model.language_model.layers.0.self_attn.q_proj")
    assert r.fullmatch("model.layers.3.mlp.down_proj")
    assert not r.fullmatch("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")
    assert not r.fullmatch("visual.blocks.0.attn.qkv")
    assert not r.fullmatch("visual.blocks.0.mlp.down_proj")
    assert not r.fullmatch("model.multi_modal_projector.mm_input_projection")


class TinyProcessor:
    """Character-level fake: each char is a token; '<img>' expands to 4 image tokens."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        s = "".join(f"[{m['role']}]" + "".join(c.get("text", "<img>") for c in m["content"]) for m in msgs)
        return s + ("[assistant]" if add_generation_prompt else "") + ("" if add_generation_prompt else "<e>")

    def __call__(self, text, images, return_tensors="pt"):
        t = text[0].replace("<img>", "@@@@")
        ids = torch.tensor([[ord(c) % 97 + 3 for c in t]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros(1, 3, 2, 2)}


def test_build_example_masks_prompt():
    from PIL import Image
    ex = build_example(TinyProcessor(), Image.new("RGB", (4, 4)), "Q?", "A!")
    n_answer = len("A!") + len("<e>")
    assert (ex["labels"] != -100).sum().item() == n_answer
    assert ex["pixel_values"].shape == (1, 3, 2, 2)


def test_collate_pads_and_concats():
    a = {"input_ids": torch.tensor([5, 6, 7]), "attention_mask": torch.ones(3, dtype=torch.long),
         "labels": torch.tensor([-100, 6, 7]), "pixel_values": torch.zeros(1, 3, 2, 2), "sample_weight": 2.0}
    b = {"input_ids": torch.tensor([5]), "attention_mask": torch.ones(1, dtype=torch.long),
         "labels": torch.tensor([5]), "pixel_values": torch.zeros(1, 3, 2, 2), "sample_weight": 1.0}
    out = collate([a, b], pad_id=0)
    assert out["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert out["labels"].tolist()[1] == [5, -100, -100]
    assert out["attention_mask"].tolist()[1] == [1, 0, 0]
    assert out["pixel_values"].shape == (2, 3, 2, 2) and out["sample_weight"].tolist() == [2.0, 1.0]


def test_weighted_ce_reduces_to_mean_and_weights_samples():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, 0]])
    plain = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1),
                                              ignore_index=-100)
    assert weighted_token_ce(logits, labels, torch.ones(2)) == pytest.approx(plain.item(), rel=1e-5)
    heavy = weighted_token_ce(logits, labels, torch.tensor([10.0, 1.0]))
    only_first = weighted_token_ce(logits, labels, torch.tensor([1.0, 0.0]))
    assert abs(heavy.item() - only_first.item()) < abs(plain.item() - only_first.item())
```

- [ ] **Bước 2: Chạy test, xác nhận fail**

Chạy: `pytest tests/test_train.py -q`. Kỳ vọng: FAIL, `ModuleNotFoundError`.

- [ ] **Bước 3: Viết `train/sft_data.py`**

```python
"""Supervised fine-tuning examples: prompt-masked labels, padding collator, dataset."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nodulecxr.prompts import QUESTION_CLS, QUESTION_LOC, build_target
from nodulecxr.vlm.chat import build_chat_text

LORA_TARGET_REGEX = (r"^(?!.*(?:visual|vision_tower|vision_model|multi_modal_projector)).*\."
                     r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
SEQ_KEYS = ("input_ids", "attention_mask", "labels", "token_type_ids")
IMAGE_KEYS = ("pixel_values", "image_grid_thw")


def build_example(processor, image, prompt: str, answer: str) -> dict:
    img = image.convert("RGB")
    full = processor(text=[build_chat_text(processor, prompt, answer)], images=[img], return_tensors="pt")
    head = processor(text=[build_chat_text(processor, prompt)], images=[img], return_tensors="pt")
    n_head = head["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][0, :n_head], head["input_ids"][0]):
        raise ValueError("chat template: prompt is not a prefix of prompt+answer")
    ex = {k: v[0] for k, v in full.items() if k in SEQ_KEYS}
    for k in IMAGE_KEYS:
        if k in full:
            ex[k] = full[k]
    labels = ex["input_ids"].clone()
    labels[:n_head] = -100
    ex["labels"] = labels
    return ex


def collate(batch, pad_id: int) -> dict:
    out = {}
    maxlen = max(len(b["input_ids"]) for b in batch)
    for k in SEQ_KEYS:
        if k not in batch[0]:
            continue
        fill = pad_id if k == "input_ids" else (-100 if k == "labels" else 0)
        out[k] = torch.stack([F.pad(b[k], (0, maxlen - len(b[k])), value=fill) for b in batch])
    for k in IMAGE_KEYS:
        if k in batch[0]:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    if "sample_weight" in batch[0]:
        out["sample_weight"] = torch.tensor([float(b["sample_weight"]) for b in batch])
    return out


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, mode: str, load_image, pos_weight: float = 1.0, indices=None):
        assert mode in ("cls", "loc")
        self.df = df.reset_index(drop=True)
        self.processor, self.mode, self.load_image = processor, mode, load_image
        self.pos_weight = pos_weight
        self.indices = list(indices) if indices is not None else list(range(len(self.df)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        r = self.df.iloc[self.indices[i]]
        prompt = QUESTION_CLS if self.mode == "cls" else QUESTION_LOC
        answer = build_target(int(r.label), r.boxes, with_boxes=self.mode == "loc")
        ex = build_example(self.processor, self.load_image(r.path), prompt, answer)
        ex["sample_weight"] = self.pos_weight if int(r.label) == 1 else 1.0
        return ex
```

- [ ] **Bước 4: Viết `train/losses.py` và `train/lora.py`**

`train/losses.py` (chỉ cần torch, nên test chạy được mà không cần transformers):
```python
"""Sample-weighted token cross-entropy."""
from __future__ import annotations

import torch.nn.functional as F


def weighted_token_ce(logits, labels, weights):
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    b, t, v = shift_logits.shape
    ce = F.cross_entropy(shift_logits.reshape(-1, v), shift_labels.reshape(-1),
                         ignore_index=-100, reduction="none").view(b, t)
    w = weights.view(-1, 1).to(ce.dtype) * (shift_labels != -100).to(ce.dtype)
    return (ce * w).sum() / w.sum().clamp_min(1e-8)
```

`train/lora.py`:
```python
"""LoRA configuration and Trainer subclass using the sample-weighted loss."""
from __future__ import annotations

from nodulecxr.train.losses import weighted_token_ce
from nodulecxr.train.sft_data import LORA_TARGET_REGEX


def make_lora_config():
    from peft import LoraConfig
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGET_REGEX,
                      bias="none", task_type="CAUSAL_LM")


def _trainer_base():
    from transformers import Trainer
    return Trainer


class WeightedSFTTrainer(_trainer_base()):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("sample_weight")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = weighted_token_ce(outputs.logits, labels, weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss
```

- [ ] **Bước 5: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_train.py -q`. Kỳ vọng: `4 passed`.

- [ ] **Bước 6: Commit**

```bash
git add src/nodulecxr/train tests/test_train.py
git commit -m "feat: SFT examples, collator, weighted loss and LoRA config"
```

---

### Task 13: Script huấn luyện LoRA (C1–C4) và lượt đo thời gian

**Files**
- Create: `code/scripts/train_lora.py`

**Interfaces**
- Consumes:
  - `load_base` (Task 11);
  - `SFTDataset`, `collate`, `make_lora_config`, `WeightedSFTTrainer` (Task 12);
  - `subsample_negatives`, `oversample_indices` (Task 2).
- Produces:
  - adapter của từng epoch tại `<out>/checkpoint-*`;
  - adapter cuối tại `<out>/final`;
  - `<out>/timing.json` với các khóa `seconds, n_train, epochs`.

**Bảng cấu hình (khóa cứng ở script runbook)**

| Config | `--model` | `--mode` | `--balance` | `--pos-weight` |
|---|---|---|---|---|
| C1 | medgemma | cls | không | 1.0 |
| C2 | medgemma | loc | không | 1.0 |
| C3 | medgemma | loc | có | 2.0 |
| C4 | qwen | theo cấu hình tốt nhất trong C1–C3 | như cấu hình đó | như cấu hình đó |

- [ ] **Bước 1: Viết script**

```python
"""LoRA fine-tuning of MedGemma-4B-it / Qwen2.5-VL-3B on VinBigData nodule targets."""
import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from transformers import TrainingArguments

from nodulecxr.data.split import oversample_indices, subsample_negatives
from nodulecxr.images import load_image
from nodulecxr.records import read_manifest
from nodulecxr.train.lora import WeightedSFTTrainer, make_lora_config
from nodulecxr.train.sft_data import SFTDataset, collate
from nodulecxr.vlm.hf import load_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["medgemma", "qwen"], required=True)
    ap.add_argument("--mode", choices=["cls", "loc"], required=True)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--neg-per-pos", type=float, default=4.0)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, help="timing pilot: cap training samples")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = read_manifest(a.manifest)
    train = subsample_negatives(df[df.split == "train"], a.neg_per_pos, a.seed).reset_index(drop=True)
    if a.max_train:
        train = train.sample(n=min(a.max_train, len(train)), random_state=a.seed).reset_index(drop=True)
    indices = oversample_indices(train.label, a.seed) if a.balance else None

    model, processor = load_base(a.model)
    model.enable_input_require_grads()
    from peft import get_peft_model
    model = get_peft_model(model, make_lora_config())
    model.print_trainable_parameters()

    ds = SFTDataset(train, processor, a.mode, lambda p: load_image(a.data_root, p), a.pos_weight, indices)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch", bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none", seed=a.seed, dataloader_num_workers=2)
    trainer = WeightedSFTTrainer(model=model, args=targs, train_dataset=ds,
                                 data_collator=partial(collate, pad_id=processor.tokenizer.pad_token_id))
    t0 = time.time()
    trainer.train()
    model.save_pretrained(Path(a.out) / "final")
    Path(a.out, "timing.json").write_text(json.dumps(
        {"seconds": time.time() - t0, "n_train": len(ds), "epochs": a.epochs}))


if __name__ == "__main__":
    main()
```

- [ ] **Bước 2: Lượt đo thời gian trên Colab** ("Việc phải làm đầu tiên", mục 4)

```bash
python scripts/train_lora.py --model medgemma --mode loc --manifest data/processed/manifest_vinbig.csv --data-root /content/data --max-train 64 --epochs 1 --out outputs/lora/pilot
cat outputs/lora/pilot/timing.json
```

Kỳ vọng:
- Không lỗi.
- Log có `trainable params`, chiếm khoảng 0,3–1% tổng tham số.
- Loss giảm.
- Có `timing.json`.

Ước lượng thời gian một lượt đầy đủ = `seconds / 64 × n_train × epochs`. Nếu vượt 3 giờ, áp quy tắc cắt giảm của spec theo thứ tự:
1. Giảm `--neg-per-pos` (4 → 2).
2. Giảm độ phân giải (chỉ khi bước 1 không đủ; ghi vào báo cáo).
3. `--epochs 2`.
4. C1 chỉ chạy 1 seed.

Ghi quyết định vào `outputs/budget.md`.

- [ ] **Bước 3: Chọn `pos-weight` cho C3 trên validation (một seed)**

Huấn luyện C3 với `--pos-weight 1.0` và `--pos-weight 2.0` (seed 0). Chạy `predict_vlm.py --splits val --generate none` cho cả hai, rồi chọn giá trị có sensitivity cao hơn ở specificity 0,90 trên val (dùng `make_report.py` ở Task 17). Ghi lại lựa chọn trong `outputs/budget.md`.

- [ ] **Bước 4: Commit**

```bash
git add scripts/train_lora.py
git commit -m "feat: LoRA training script with timing"
```

---

### Task 14: Baseline B1, DenseNet-121 (`torchxrayvision`)

**Files**
- Create: `code/src/nodulecxr/baselines/densenet.py`, `code/scripts/train_densenet.py`
- Test: `code/tests/test_densenet.py`

**Interfaces**
- Consumes: manifest (Task 1), `append_jsonl` (Task 1), `threshold`/metrics (Task 7).
- Produces:
  - `xrv_normalize(arr_uint8) -> np.ndarray[float32]` (về khoảng [−1024, 1024], quy ước của torchxrayvision)
  - `to_xrv_array(pil, size=224) -> np.ndarray` với shape `(1, size, size)`
  - `build_model() -> nn.Module` (lớp `NoduleDenseNet` bên trong), với `forward(x: (B, 1, 224, 224)) -> (B,) logits`
  - JSONL với `config = "B1"`, `label_pred = null`, `boxes_pred = []`

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import numpy as np
from PIL import Image
from nodulecxr.baselines.densenet import to_xrv_array, xrv_normalize


def test_xrv_normalize_range_and_shape():
    out = xrv_normalize(np.array([0, 255], dtype=np.uint8))
    assert out.tolist() == [-1024.0, 1024.0]
    arr = to_xrv_array(Image.new("RGB", (896, 896), (128, 128, 128)))
    assert arr.shape == (1, 224, 224) and arr.dtype == np.float32
```

Chạy: `pytest tests/test_densenet.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 2: Viết `baselines/densenet.py`**

```python
"""B1: torchxrayvision DenseNet-121 backbone with a binary nodule/mass head."""
from __future__ import annotations

import numpy as np
from PIL import Image


def xrv_normalize(arr) -> np.ndarray:
    return (np.asarray(arr, dtype=np.float32) / 255.0) * 2048.0 - 1024.0


def to_xrv_array(pil: Image.Image, size: int = 224) -> np.ndarray:
    a = np.array(pil.convert("L").resize((size, size), Image.BILINEAR))
    return xrv_normalize(a)[None]


def build_model():
    import torch.nn as nn
    import torchxrayvision as xrv

    class NoduleDenseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.head = nn.Linear(1024, 1)

        def forward(self, x):
            return self.head(self.backbone.features2(x)).squeeze(1)

    return NoduleDenseNet()
```


- [ ] **Bước 3: Viết `scripts/train_densenet.py`**

```python
"""Train/evaluate B1 DenseNet-121 on VinBigData; predict all splits -> JSONL."""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from nodulecxr.baselines.densenet import build_model, to_xrv_array
from nodulecxr.images import load_image
from nodulecxr.records import append_jsonl, read_manifest


class DS(Dataset):
    def __init__(self, df, root):
        self.df, self.root = df.reset_index(drop=True), root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return torch.from_numpy(to_xrv_array(load_image(self.root, r.path))), float(r.label)


def scores(model, df, root, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(DS(df, root), batch_size=64, num_workers=2):
            out.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    df = pd.concat([read_manifest(m) for m in a.manifests])
    tr, va = df[df.split == "train"], df[df.split == "val"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    pos_w = torch.tensor((tr.label == 0).sum() / max(1, (tr.label == 1).sum()), device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_auc, best_state = -1.0, None
    for epoch in range(a.epochs):
        model.train()
        for x, y in DataLoader(DS(tr, a.data_root), batch_size=32, shuffle=True, num_workers=2):
            opt.zero_grad()
            loss = loss_fn(model(x.to(dev)), y.float().to(dev))
            loss.backward()
            opt.step()
        auc = roc_auc_score(va.label, scores(model, va, a.data_root, dev))
        print(f"epoch {epoch} val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ev = df[df.split != "train"]
    for r, s in zip(ev.itertuples(), scores(model, ev, a.data_root, dev)):
        append_jsonl({"image_id": r.image_id, "dataset": r.dataset, "split": r.split, "score": float(s),
                      "label_pred": None, "boxes_pred": [], "parse_ok": True, "raw_text": "",
                      "config": "B1", "seed": a.seed, "run": 0}, a.out)


if __name__ == "__main__":
    main()
```

- [ ] **Bước 4: Chạy test, rồi chạy thử trên Colab**

Chạy:
- local: `pytest tests/test_densenet.py -q`. Kỳ vọng: `1 passed`.
- Colab: `python scripts/train_densenet.py --manifests data/processed/manifest_vinbig.csv data/processed/manifest_nih.csv data/processed/manifest_jsrt.csv --data-root /content/data --epochs 1 --seed 0 --out outputs/preds/B1_smoke.jsonl`. Kỳ vọng: in ra val AUC và tạo file JSONL có các split `val, test, test_nih, test_jsrt`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/baselines/densenet.py scripts/train_densenet.py tests/test_densenet.py
git commit -m "feat: B1 DenseNet-121 baseline"
```

---

### Task 15: Baseline B2, YOLOv8

**Files**
- Create: `code/src/nodulecxr/baselines/yolo.py`, `code/scripts/train_yolo.py`
- Test: `code/tests/test_yolo.py`

**Interfaces**
- Consumes: manifest, `subsample_negatives`, `append_jsonl`.
- Produces:
  - `yolo_label_lines(boxes, size=896) -> list[str]` (`"0 cx cy w h"`, đã chuẩn hóa)
  - `export_yolo(df_train, df_val, data_root, out_dir) -> Path` (đường dẫn `data.yaml`)
  - `yolo_scorer(model) -> callable(PIL) -> (score, boxes)`
  - JSONL với `config = "B2"`, `score` là confidence lớn nhất (0 nếu không có box), `boxes_pred` kèm confidence

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import pandas as pd
from PIL import Image
from nodulecxr.baselines.yolo import export_yolo, yolo_label_lines


def test_label_lines():
    assert yolo_label_lines([[0, 0, 448, 224]]) == ["0 0.250000 0.125000 0.500000 0.250000"]


def test_export(tmp_path):
    root = tmp_path / "data"
    (root / "vinbig").mkdir(parents=True)
    for i in ("a", "b"):
        Image.new("L", (896, 896)).save(root / "vinbig" / f"{i}.png")
    tr = pd.DataFrame({"image_id": ["a"], "path": ["vinbig/a.png"], "label": [1], "boxes": [[[0, 0, 448, 224]]]})
    va = pd.DataFrame({"image_id": ["b"], "path": ["vinbig/b.png"], "label": [0], "boxes": [[]]})
    yaml_path = export_yolo(tr, va, root, tmp_path / "yolo")
    assert (tmp_path / "yolo/labels/train/a.txt").read_text().strip() == "0 0.250000 0.125000 0.500000 0.250000"
    assert (tmp_path / "yolo/labels/val/b.txt").read_text() == ""
    assert "names" in yaml_path.read_text()
```

Chạy: `pytest tests/test_yolo.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 2: Viết `baselines/yolo.py`**

```python
"""B2: YOLOv8 export, scoring helper."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from nodulecxr.records import IMG_SIZE


def yolo_label_lines(boxes, size: int = IMG_SIZE) -> list:
    lines = []
    for x1, y1, x2, y2 in (b[:4] for b in boxes):
        lines.append(f"0 {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} "
                     f"{(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}")
    return lines


def _link(src: Path, dst: Path):
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy(src, dst)


def export_yolo(df_train, df_val, data_root, out_dir) -> Path:
    out = Path(out_dir)
    for split, df in (("train", df_train), ("val", df_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in df.itertuples():
            dst = out / "images" / split / f"{r.image_id}.png"
            if not dst.exists():
                _link(Path(data_root) / r.path, dst)
            lines = yolo_label_lines(r.boxes) if int(r.label) == 1 else []
            (out / "labels" / split / f"{r.image_id}.txt").write_text("\n".join(lines))
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: nodule\n")
    return yaml_path


def yolo_scorer(model, imgsz: int = IMG_SIZE):
    def score(pil):
        res = model.predict(pil, imgsz=imgsz, conf=0.001, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy().tolist()
        conf = res.boxes.conf.cpu().numpy().tolist()
        boxes = [b + [c] for b, c in zip(xyxy, conf)]
        return (max(conf) if conf else 0.0), boxes
    return score
```

- [ ] **Bước 3: Viết `scripts/train_yolo.py`**

```python
"""Train YOLOv8 on VinBigData Nodule/Mass boxes and predict all evaluation splits -> JSONL."""
import argparse

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from nodulecxr.baselines.yolo import export_yolo, yolo_scorer
from nodulecxr.data.split import subsample_negatives
from nodulecxr.images import load_image
from nodulecxr.records import append_jsonl, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--neg-per-pos", type=float, default=4.0)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([read_manifest(m) for m in a.manifests])
    tr = subsample_negatives(df[df.split == "train"], a.neg_per_pos, a.seed)
    yaml_path = export_yolo(tr, df[df.split == "val"], a.data_root, f"{a.work}/yolo_data")
    model = YOLO(a.weights)
    model.train(data=str(yaml_path), imgsz=896, epochs=a.epochs, batch=16, seed=a.seed,
                project=a.work, name=f"seed{a.seed}", exist_ok=True)
    model = YOLO(f"{a.work}/seed{a.seed}/weights/best.pt")  # best epoch, not last
    score = yolo_scorer(model)
    ev = df[df.split != "train"]
    for r in tqdm(ev.itertuples(), total=len(ev)):
        s, boxes = score(load_image(a.data_root, r.path))
        append_jsonl({"image_id": r.image_id, "dataset": r.dataset, "split": r.split, "score": float(s),
                      "label_pred": None, "boxes_pred": boxes, "parse_ok": True, "raw_text": "",
                      "config": "B2", "seed": a.seed, "run": 0}, a.out)


if __name__ == "__main__":
    main()
```

- [ ] **Bước 4: Chạy test, rồi chạy thử trên Colab**

- Local: `pytest tests/test_yolo.py -q`. Kỳ vọng: `2 passed`.
- Colab: `python scripts/train_yolo.py --manifests data/processed/manifest_vinbig.csv data/processed/manifest_nih.csv data/processed/manifest_jsrt.csv --data-root /content/data --epochs 1 --seed 0 --work outputs/yolo --out outputs/preds/B2_smoke.jsonl`. Kỳ vọng: huấn luyện xong 1 epoch, nạp được `best.pt` và tạo JSONL.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/baselines/yolo.py scripts/train_yolo.py tests/test_yolo.py
git commit -m "feat: B2 YOLOv8 baseline"
```

---

### Task 16: Kiểm toán "nhìn đúng chỗ" (E5)

**Files**
- Create: `code/src/nodulecxr/audit/run.py`, `code/scripts/run_audit.py`
- Test: `code/tests/test_audit_run.py`

**Interfaces**
- Consumes:
  - Task 10: `occlude`, `expand_box`, `sample_random_box`, `flip_rate`, `heuristic_lung_mask`.
  - Task 4: `box_rel_area`, `size_bin`.
  - Task 11: `HFVLM`.
  - Task 15: `yolo_scorer`.
- Produces:
  - `audit_images(score_fn, positives, negatives, load_image, threshold, lung_mask_fn, edges, seed=0) -> DataFrame`. Mỗi dòng là một ca dương, gồm các cột `image_id, size_bin, subtlety, s_orig, s_t1, s_t2, s_t3, t2_ok`.
  - `summarize_audit(df, threshold) -> dict`, gồm các khóa `n, flip_t1, flip_t2, delta, yes_rate_t3`, cùng các khóa `by_size` và `by_subtlety`.
  - `score_fn(PIL) -> float`

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import numpy as np
import pandas as pd
from PIL import Image
from nodulecxr.audit.occlusion import heuristic_lung_mask
from nodulecxr.audit.run import audit_images, summarize_audit


def _img_with_blob(box):
    a = np.full((896, 896), 60, dtype=np.uint8)
    x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = 250
    return Image.fromarray(a).convert("RGB")


def test_scorer_that_looks_at_nodule_flips_only_on_t1():
    box = [200, 300, 260, 360]
    images = {"p": _img_with_blob(box), "n": _img_with_blob([0, 0, 1, 1])}
    score = lambda im: float(np.array(im.convert("L")).max() > 200)  # noqa: E731
    pos = pd.DataFrame({"image_id": ["p"], "path": ["p"], "boxes": [[box]], "subtlety": [3]})
    neg = pd.DataFrame({"image_id": ["n"], "path": ["n"]})
    df = audit_images(score, pos, neg, images.__getitem__, 0.5, lambda im: heuristic_lung_mask(),
                      edges=(0.001, 0.01))
    s = summarize_audit(df, 0.5)
    assert s["flip_t1"] == 1.0 and s["flip_t2"] == 0.0 and s["delta"] == 1.0 and s["yes_rate_t3"] == 0.0
```

Chạy: `pytest tests/test_audit_run.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 2: Viết `audit/run.py`**

```python
"""E5 occlusion audit over positive images."""
from __future__ import annotations

import numpy as np
import pandas as pd

from nodulecxr.audit.occlusion import expand_box, flip_rate, occlude, sample_random_box
from nodulecxr.strata import box_rel_area, size_bin


def audit_images(score_fn, positives, negatives, load_image, threshold, lung_mask_fn, edges, seed=0):
    rng = np.random.default_rng(seed)
    neg_list = list(negatives.itertuples())
    rows = []
    for r in positives.itertuples():
        img = load_image(r.path)
        boxes = [list(b) for b in r.boxes]
        s_orig = score_fn(img)
        s_t1 = score_fn(occlude(img, [expand_box(b) for b in boxes]))
        mask = lung_mask_fn(img)
        rand = [sample_random_box(b[2] - b[0], b[3] - b[1], mask, boxes, rng) for b in boxes]
        t2_ok = all(b is not None for b in rand)
        s_t2 = score_fn(occlude(img, [b for b in rand if b is not None])) if t2_ok else float("nan")
        neg = neg_list[int(rng.integers(len(neg_list)))]
        s_t3 = score_fn(load_image(neg.path))
        area = max(box_rel_area(b) for b in boxes)
        rows.append({"image_id": r.image_id, "size_bin": size_bin(area, edges),
                     "subtlety": getattr(r, "subtlety", None), "s_orig": s_orig, "s_t1": s_t1,
                     "s_t2": s_t2, "s_t3": s_t3, "t2_ok": t2_ok})
    return pd.DataFrame(rows)


def _summary(df, thr):
    o = (df.s_orig >= thr).astype(int)
    ok = df.t2_ok.astype(bool)
    f1 = flip_rate(o, (df.s_t1 >= thr).astype(int))
    f2 = flip_rate(o[ok], (df.s_t2[ok] >= thr).astype(int))
    return {"n": int(len(df)), "n_orig_yes": int(o.sum()), "flip_t1": f1, "flip_t2": f2,
            "delta": f1 - f2, "yes_rate_t3": float((df.s_t3 >= thr).mean())}


def summarize_audit(df, threshold) -> dict:
    out = _summary(df, threshold)
    out["by_size"] = {k: _summary(g, threshold) for k, g in df.groupby("size_bin")}
    if df["subtlety"].notna().any():
        out["by_subtlety"] = {int(k): _summary(g, threshold) for k, g in df.dropna(subset=["subtlety"]).groupby("subtlety")}
    return out
```

- [ ] **Bước 3: Viết `scripts/run_audit.py`**

```python
"""E5: occlusion audit for a VLM config (C0 or best LoRA) or B2 on one test split."""
import argparse
import json

import numpy as np
import pandas as pd

from nodulecxr.audit.occlusion import heuristic_lung_mask
from nodulecxr.audit.run import audit_images, summarize_audit
from nodulecxr.images import load_image
from nodulecxr.prompts import QUESTION_CLS, QUESTION_LOC
from nodulecxr.records import read_manifest

_SEG = {}


def lung_mask_xrv(img):
    import torch
    import torchxrayvision as xrv
    from PIL import Image
    from nodulecxr.baselines.densenet import xrv_normalize
    if "m" not in _SEG:
        _SEG["m"] = xrv.baseline_models.chestx_det.PSPNet().eval()
    seg = _SEG["m"]
    x = xrv_normalize(np.array(img.convert("L").resize((512, 512))))[None, None]
    with torch.no_grad():
        out = torch.sigmoid(seg(torch.from_numpy(x)))[0]
    idx = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]
    m = (out[idx] > 0.5).any(0).numpy().astype(np.uint8) * 255
    mask = np.array(Image.fromarray(m).resize((896, 896), Image.NEAREST)) > 0
    return mask if mask.mean() > 0.05 else heuristic_lung_mask()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["vlm", "yolo"], required=True)
    ap.add_argument("--model", choices=["medgemma", "qwen"])
    ap.add_argument("--adapter")
    ap.add_argument("--prompt", choices=["loc", "cls"], default="loc")
    ap.add_argument("--yolo-weights")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="from outputs/report/thresholds.json")
    ap.add_argument("--edges-json", required=True, help="outputs/e0_data_edges.json")
    ap.add_argument("--lung-mask", choices=["xrv", "heuristic"], default="xrv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = read_manifest(a.manifest)
    df = df[df.split == a.split]
    edges = tuple(pd.read_json(a.edges_json, typ="series").tolist())
    if a.scorer == "vlm":
        from nodulecxr.vlm.hf import HFVLM
        vlm = HFVLM(a.model, a.adapter)
        prompt = QUESTION_CLS if a.prompt == "cls" else QUESTION_LOC
        score_fn = lambda im: vlm.yes_probability(im, prompt)  # noqa: E731
    else:
        from ultralytics import YOLO
        from nodulecxr.baselines.yolo import yolo_scorer
        ys = yolo_scorer(YOLO(a.yolo_weights))
        score_fn = lambda im: ys(im)[0]  # noqa: E731
    mask_fn = lung_mask_xrv if a.lung_mask == "xrv" else (lambda im: heuristic_lung_mask())
    res = audit_images(score_fn, df[df.label == 1], df[df.label == 0],
                       lambda p: load_image(a.data_root, p), a.threshold, mask_fn, edges)
    res.to_csv(a.out.replace(".json", ".csv"), index=False)
    with open(a.out, "w") as f:
        json.dump(summarize_audit(res, a.threshold), f, indent=2, default=float)


if __name__ == "__main__":
    main()
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_audit_run.py -q`. Kỳ vọng: `1 passed`.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/audit/run.py scripts/run_audit.py tests/test_audit_run.py
git commit -m "feat: E5 occlusion audit"
```

---

### Task 17: Báo cáo tổng hợp (bảng 7 cấu hình × 3 tập test)

**Files**
- Create: `code/src/nodulecxr/eval/report.py`, `code/scripts/make_report.py`
- Test: `code/tests/test_report.py`

**Interfaces**
- Consumes: Task 7, 8, 9; `JSRT_MM_PER_PX_896`; `box_rel_area`, `size_bin`.
- Produces:
  - `TEST_SPLITS = ("test", "test_nih", "test_jsrt")`
  - `compute_thresholds(labelled) -> dict[(config, seed, run), {"t90": float, "t95": float}]` (đầu vào là `_with_labels(preds, manifest)`; chỉ dùng split `val`)
  - `evaluate_group(g, manifest, thr, edges) -> list[dict]` (mỗi tập test một dòng, kèm metric và khoảng tin cậy)
  - `evaluate_all(preds, manifest, edges) -> (summary DataFrame, strata DataFrame)`
  - `aggregate_seeds(summary) -> DataFrame` (mean ± sd theo `config, split`)
  - `compare_configs(preds, manifest, thresholds, split, seed=0, run=0) -> DataFrame` (McNemar sens/spec, DeLong AUC, Holm)
  - `run_agreement(preds) -> DataFrame` (độ ổn định giữa các lần chạy E2)

**Quy ước tính**
- Mọi metric dùng ngưỡng `t90` (hoặc `t95`) lấy từ val **của cùng** `(config, seed, run)`.
- `label_pred` rỗng được tính là 0. `raw_*` là metric dựa trên câu trả lời văn bản.
- "Đúng vì đúng lý do": dự đoán dương ở `t90` **và** `best_iou ≥ 0,3` (báo thêm ở 0,5).
- JSRT: dùng `jsrt_hit` với biên 12,5 px; khoảng cách được báo bằng mm.

- [ ] **Bước 1: Viết test (sẽ fail)**

```python
import numpy as np
import pandas as pd
from nodulecxr.eval.report import (_with_labels, aggregate_seeds, compare_configs, compute_thresholds,
                                   evaluate_all, run_agreement)


def _data():
    rng = np.random.default_rng(0)
    rows, man = [], []
    for split, n in (("val", 200), ("test", 200)):
        for i in range(n):
            y = int(i < n // 5)
            iid = f"{split}{i}"
            man.append({"image_id": iid, "dataset": "vinbig", "split": split, "label": y,
                        "boxes": [[100, 100, 150, 150]] if y else [], "subtlety": None, "size_mm": None})
            for cfg, sep in (("C0", 0.2), ("C3", 2.5)):
                s = float(1 / (1 + np.exp(-(rng.normal(sep * y, 1)))))
                rows.append({"image_id": iid, "dataset": "vinbig", "split": split, "config": cfg, "seed": 0,
                             "run": 0, "score": s, "label_pred": int(s > 0.5),
                             "boxes_pred": [[100, 100, 150, 150, s]] if s > 0.5 else [], "parse_ok": True})
    return pd.DataFrame(rows), pd.DataFrame(man)


def test_thresholds_from_val_only_and_summary():
    preds, man = _data()
    thr = compute_thresholds(_with_labels(preds, man))
    assert set(thr) == {("C0", 0, 0), ("C3", 0, 0)}
    summary, strata = evaluate_all(preds, man, edges=(0.001, 0.01))
    row = summary[(summary.config == "C3") & (summary.split == "test")].iloc[0]
    assert 0.8 <= row["spec"] <= 1.0 and row["sens"] > 0.6 and row["auc_roc"] > 0.9
    for col in ("sens_lo", "sens_hi", "auc_pr", "ece", "balanced_brier", "ppv_prev_1pct",
                "raw_sens", "parse_fail", "hit_iou30", "froc_1fp"):
        assert col in summary.columns
    assert summary[summary.config == "C3"]["sens"].iloc[0] > summary[summary.config == "C0"]["sens"].iloc[0]
    agg = aggregate_seeds(summary)
    assert {"sens_mean", "sens_sd"} <= set(agg.columns)
    cmp = compare_configs(preds, man, thr, "test")
    assert {"a", "b", "p_sens", "p_spec", "p_auc", "p_auc_holm"} <= set(cmp.columns)
    assert len(strata) > 0


def test_run_agreement():
    p = pd.DataFrame({"image_id": ["a", "a", "b", "b"], "config": "C0", "seed": 0, "run": [0, 1, 0, 1],
                      "label_pred": [1, 1, 0, 1]})
    assert run_agreement(p).loc[0, "agreement"] == 0.5
```

Chạy: `pytest tests/test_report.py -q`. Kỳ vọng: FAIL.

- [ ] **Bước 2: Viết `eval/report.py`**

```python
"""Benchmark tables: thresholds from val, metrics with CIs on each test split, strata, comparisons."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from nodulecxr.data.jsrt import JSRT_MM_PER_PX_896
from nodulecxr.metrics.detection import (binary_metrics, ppv_at_prevalence, predict_at, score_metrics,
                                         threshold_at_specificity)
from nodulecxr.metrics.localization import best_iou, center_distance, froc_points, jsrt_hit
from nodulecxr.metrics.stats import bootstrap_ci, delong_test, holm, mcnemar_p
from nodulecxr.strata import box_rel_area, size_bin

TEST_SPLITS = ("test", "test_nih", "test_jsrt")
KEY = ["config", "seed", "run"]
JSRT_MARGIN_PX = 5.0 / JSRT_MM_PER_PX_896


def _sens(y, p):
    return binary_metrics(y, p)["sens"]


def _spec(y, p):
    return binary_metrics(y, p)["spec"]


def _auc(y, s):
    return score_metrics(y, s)["auc_roc"]


def compute_thresholds(labelled) -> dict:
    """labelled = _with_labels(preds, manifest); thresholds come from the val split only."""
    out = {}
    for k, g in labelled[labelled.split == "val"].groupby(KEY):
        out[k] = {"t90": threshold_at_specificity(g.y, g.score, 0.90),
                  "t95": threshold_at_specificity(g.y, g.score, 0.95)}
    return out


def _with_labels(preds, manifest):
    m = manifest[["image_id", "split", "label", "boxes", "subtlety", "size_mm"]].rename(
        columns={"label": "y", "boxes": "gt"})
    return preds.merge(m, on=["image_id", "split"], how="inner")


def evaluate_group(g, thr, edges) -> tuple:
    rows, strata = [], []
    for split in TEST_SPLITS:
        d = g[g.split == split]
        if d.empty:
            continue
        y, s = d.y.to_numpy(), d.score.to_numpy()
        p90, p95 = predict_at(s, thr["t90"]), predict_at(s, thr["t95"])
        m = binary_metrics(y, p90)
        row = {"split": split, "n": len(d), "n_pos": int(y.sum()), "threshold": thr["t90"], **m}
        row["sens_at_spec95"] = binary_metrics(y, p95)["sens"]
        row.update(score_metrics(y, s))
        for name, fn, arr in (("sens", _sens, p90), ("spec", _spec, p90), ("auc_roc", _auc, s)):
            _, lo, hi = bootstrap_ci(fn, y, arr, n_boot=1000)
            row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
        row["ppv_prev_1pct"] = ppv_at_prevalence(m["sens"], m["spec"], 0.01)
        row["ppv_prev_3pct"] = ppv_at_prevalence(m["sens"], m["spec"], 0.03)
        raw = d.label_pred.fillna(0).astype(int).to_numpy() if "label_pred" in d else np.zeros(len(d))
        rm = binary_metrics(y, raw)
        row.update({"raw_sens": rm["sens"], "raw_spec": rm["spec"],
                    "parse_fail": float(1 - d.parse_ok.astype(bool).mean()) if "parse_ok" in d else 0.0,
                    "no_on_pos": 1 - rm["sens"], "yes_on_neg": 1 - rm["spec"]})
        pos = d[d.y == 1]
        ious = [best_iou(pb, gt) for pb, gt in zip(pos.boxes_pred, pos.gt)]
        pos_pred = predict_at(pos.score, thr["t90"])
        row["mean_best_iou"] = float(np.mean(ious)) if ious else float("nan")
        row["hit_iou30"] = float(np.mean([pp and i >= 0.3 for pp, i in zip(pos_pred, ious)])) if ious else float("nan")
        row["hit_iou50"] = float(np.mean([pp and i >= 0.5 for pp, i in zip(pos_pred, ious)])) if ious else float("nan")
        has_boxes = d.boxes_pred.apply(len).sum() > 0
        if has_boxes and len(pos):
            fr = froc_points(list(zip(d.boxes_pred, d.gt)), iou_thr=0.3)
            row["froc_0.5fp"], row["froc_1fp"] = fr[0.5], fr[1]
        else:
            row["froc_0.5fp"] = row["froc_1fp"] = float("nan")
        if split == "test_jsrt" and len(pos):
            hits, dists = [], []
            for pb, gt, pp in zip(pos.boxes_pred, pos.gt, pos_pred):
                if pp and pb:
                    top = max(pb, key=lambda b: b[4])
                    hits.append(jsrt_hit(top, gt[0], JSRT_MARGIN_PX))
                    dists.append(center_distance(top, gt[0]) * JSRT_MM_PER_PX_896)
                else:
                    hits.append(False)
            row["jsrt_hit"] = float(np.mean(hits))
            row["jsrt_median_dist_mm"] = float(np.median(dists)) if dists else float("nan")
        rows.append(row)
        for r, pp in zip(pos.itertuples(), pos_pred):
            strata.append({"split": split, "size_bin": size_bin(max(box_rel_area(b) for b in r.gt), edges) if r.gt else None,
                           "subtlety": r.subtlety, "detected": int(pp)})
    return rows, strata


def evaluate_all(preds, manifest, edges):
    data = _with_labels(preds, manifest)
    thresholds = compute_thresholds(data)
    summary, strata = [], []
    for k, g in data.groupby(KEY):
        if k not in thresholds:
            continue
        rows, st = evaluate_group(g, thresholds[k], edges)
        meta = dict(zip(KEY, k))
        summary += [{**meta, **r} for r in rows]
        strata += [{**meta, **r} for r in st]
    st = pd.DataFrame(strata)
    if not st.empty:
        parts = [st.groupby(KEY + ["split", "size_bin"]).detected.agg(["mean", "count"]).reset_index()
                   .rename(columns={"size_bin": "stratum"}).assign(kind="size")]
        js = st.dropna(subset=["subtlety"])
        if not js.empty:
            parts.append(js.groupby(KEY + ["split", "subtlety"]).detected.agg(["mean", "count"]).reset_index()
                         .rename(columns={"subtlety": "stratum"}).assign(kind="subtlety"))
        st = pd.concat(parts, ignore_index=True).rename(columns={"mean": "sensitivity"})
    return pd.DataFrame(summary), st


def aggregate_seeds(summary) -> pd.DataFrame:
    num = summary.select_dtypes("number").columns.difference(["seed", "run"])
    g = summary.groupby(["config", "split"])[list(num)]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()
    out["n_seeds"] = summary.groupby(["config", "split"]).seed.nunique().to_numpy()
    return out


def compare_configs(preds, manifest, thresholds, split, seed=0, run=0) -> pd.DataFrame:
    data = _with_labels(preds, manifest)
    d = data[(data.split == split) & (data.run == run)]
    configs = sorted(c for c in d.config.unique() if (c, seed, run) in thresholds)
    piv = {c: d[(d.config == c) & (d.seed == seed)].set_index("image_id") for c in configs}
    rows = []
    for a, b in itertools.combinations(configs, 2):
        ids = piv[a].index.intersection(piv[b].index)
        A, B = piv[a].loc[ids], piv[b].loc[ids]
        y = A.y.to_numpy()
        pa = predict_at(A.score, thresholds[(a, seed, run)]["t90"])
        pb = predict_at(B.score, thresholds[(b, seed, run)]["t90"])
        pos, neg = y == 1, y == 0
        _, _, p_auc = delong_test(y, A.score.to_numpy(), B.score.to_numpy())
        rows.append({"split": split, "a": a, "b": b,
                     "p_sens": mcnemar_p(pa[pos] == 1, pb[pos] == 1),
                     "p_spec": mcnemar_p(pa[neg] == 0, pb[neg] == 0), "p_auc": p_auc})
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ("p_sens", "p_spec", "p_auc"):
            out[f"{col}_holm"] = holm(out[col])
    return out


def run_agreement(preds) -> pd.DataFrame:
    p = preds.copy()
    p["label_pred"] = p["label_pred"].fillna(0).astype(int)
    rows = []
    for (cfg, seed), g in p.groupby(["config", "seed"]):
        if g.run.nunique() < 2:
            continue
        same = g.groupby("image_id").label_pred.nunique() == 1
        rows.append({"config": cfg, "seed": seed, "n_runs": g.run.nunique(), "agreement": float(same.mean())})
    return pd.DataFrame(rows)
```

- [ ] **Bước 3: Viết `scripts/make_report.py`**

```python
"""Build benchmark tables from manifests + prediction JSONL files."""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from nodulecxr.eval.report import (TEST_SPLITS, _with_labels, aggregate_seeds, compare_configs,
                                   compute_thresholds, evaluate_all, run_agreement)
from nodulecxr.records import read_jsonl, read_manifest

COLS = ["sens", "spec", "sens_at_spec95", "auc_roc", "auc_pr", "ece", "balanced_brier", "ppv_prev_1pct",
        "raw_sens", "parse_fail", "hit_iou30", "froc_1fp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--preds", required=True, help="glob, e.g. 'outputs/preds/*.jsonl'")
    ap.add_argument("--edges-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.concat([read_manifest(m) for m in a.manifests])
    preds = pd.concat([read_jsonl(p) for p in glob.glob(a.preds)], ignore_index=True)
    preds = preds[~preds.config.isin(["pilot"])]
    edges = tuple(pd.read_json(a.edges_json, typ="series").tolist())
    thr = compute_thresholds(_with_labels(preds, man))
    (out / "thresholds.json").write_text(json.dumps({f"{c}|{s}|{r}": v for (c, s, r), v in thr.items()}, indent=2))
    summary, strata = evaluate_all(preds, man, edges)
    summary.to_csv(out / "summary_per_seed.csv", index=False)
    strata.to_csv(out / "strata.csv", index=False)
    agg = aggregate_seeds(summary)
    agg.to_csv(out / "summary.csv", index=False)
    pd.concat([compare_configs(preds, man, thr, s) for s in TEST_SPLITS]).to_csv(out / "comparisons.csv", index=False)
    run_agreement(preds).to_csv(out / "run_agreement.csv", index=False)
    lines = ["# Bảng benchmark (mean ± sd qua seed; ngưỡng chọn trên val ở spec 0,90)", ""]
    for split in TEST_SPLITS:
        t = agg[agg.split == split]
        if t.empty:
            continue
        lines += [f"## {split}", "", "| config | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
        for r in t.itertuples():
            cells = [f"{getattr(r, c + '_mean'):.3f} ± {getattr(r, c + '_sd'):.3f}"
                     if pd.notna(getattr(r, c + "_sd")) else f"{getattr(r, c + '_mean'):.3f}" for c in COLS]
            lines.append(f"| {r.config} | " + " | ".join(cells) + " |")
        if split == "test_jsrt":
            lines.append("\n*PPV trên JSRT không có ý nghĩa lâm sàng (tỷ lệ nốt 62%).*")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Bước 4: Chạy test, xác nhận pass**

Chạy: `pytest tests/test_report.py -q`. Kỳ vọng: `2 passed`.

Chạy toàn bộ: `pytest -q`. Kỳ vọng: tất cả pass.

- [ ] **Bước 5: Commit**

```bash
git add src/nodulecxr/eval/report.py scripts/make_report.py tests/test_report.py
git commit -m "feat: benchmark report with CIs, strata and paired tests"
```

---

### Task 18: RUNBOOK chạy E0–E5 trên Colab

**Files**
- Create: `code/RUNBOOK.md`

**Interfaces**
- Consumes: mọi script ở Task 5–17.
- Produces: trình tự chạy có thể làm theo từng bước. Mỗi thí nghiệm ghi kết quả vào `P/outputs/`.

- [ ] **Bước 1: Viết `RUNBOOK.md` với nội dung sau**

````markdown
# RUNBOOK (Colab, GPU L4 hoặc A100; không dùng T4 vì MedGemma cần bf16)

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
Chạy 3 script `prepare_*`, rồi `draw_boxes.py` và `data_stats.py`. Kiểm tra `outputs/qa/` bằng mắt.
Kết quả: `$O/e0_data.md`, `$O/e0_data_edges.json`.

## E1. Baseline CNN (3 seed mỗi loại)
```bash
for s in 0 1 2; do
  python scripts/train_densenet.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --data-root $D --seed $s --out $O/preds/B1.jsonl
  python scripts/train_yolo.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --data-root $D --seed $s --work $O/yolo --out $O/preds/B2.jsonl
done
```

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

Đường học (sensitivity theo epoch): chạy `predict_vlm.py --splits val --generate none` cho từng `checkpoint-*` của seed 0, với `--config <cfg>-ep<k>`.

## E4. C4: cấu hình tốt nhất trên Qwen2.5-VL-3B (1–3 seed)
Chọn cấu hình tốt nhất trong C1–C3 theo **sens ở spec 0,90 trên val** (cột `sens` của split `val` không có trong bảng test; xem `thresholds.json` và `summary_per_seed.csv`). Chạy lại `train_lora.py --model qwen` với đúng các cờ của cấu hình đó, rồi dự đoán với `--config C4`.

## Báo cáo
```bash
python scripts/make_report.py --manifests $M/manifest_vinbig.csv $M/manifest_nih.csv $M/manifest_jsrt.csv --preds "$O/preds/*.jsonl" --edges-json $O/e0_data_edges.json --out $O/report
```

## E5. Kiểm toán che vùng (C0-medgemma, cấu hình LoRA tốt nhất, B2), trên `test` và `test_jsrt`
Lấy ngưỡng `t90` từ `$O/report/thresholds.json`, rồi chạy:
```bash
python scripts/run_audit.py --scorer vlm --model medgemma --adapter $O/lora/C3-s0/final --manifest $M/manifest_jsrt.csv --split test_jsrt --data-root $D --threshold <t90> --edges-json $O/e0_data_edges.json --out $O/audit/C3_jsrt.json
```

## Ngân sách (spec §7)
Đo thời gian ở Task 13 Bước 2. Nếu một lượt vượt 3 giờ, cắt theo thứ tự:
1. `--neg-per-pos` 4 → 2.
2. Giảm độ phân giải.
3. `--epochs 2`.
4. C1 chỉ 1 seed.

Ghi mọi quyết định vào `$O/budget.md`.
````

- [ ] **Bước 2: Kiểm tra mọi lệnh trong RUNBOOK khớp tham số của script**

Chạy: `for f in scripts/*.py; do python $f --help > /dev/null || echo "FAIL $f"; done`. Script GPU sẽ báo `ImportError` nếu máy local chưa cài torch hay transformers; khi đó chạy lệnh này trên Colab.

Kỳ vọng: không có dòng `FAIL`. Sau đó đối chiếu từng cờ trong RUNBOOK với `--help`.

- [ ] **Bước 3: Commit**

```bash
git add RUNBOOK.md
git commit -m "docs: Colab runbook for E0-E5"
```

---

## Ánh xạ spec → task (tự rà soát)

| Mục spec | Task |
|---|---|
| §2 mục tiêu 1, E2: zero-shot chuẩn hóa, chạy lặp, ngưỡng từ val | 6, 11, 17, RUNBOOK E2 |
| §2 mục tiêu 2: LoRA phân loại / có box (C1, C2) | 12, 13 |
| §2 mục tiêu 3: cân bằng lớp (C3) | 2 (`oversample_indices`), 12 (`pos_weight`), 13 |
| §2 mục tiêu 4, §8.3, E5: định vị và che vùng | 9, 10, 16 |
| §5.1 mô hình, LoRA r16/α32/dropout 0,05, encoder đông cứng | 11, 12 (regex có test) |
| §5.2 C4 trên Qwen | 13, RUNBOOK E4 |
| §5.2 C5, C6 (mở rộng, không cam kết) | Không có task. Chỉ làm khi còn thời gian; cần kế hoạch riêng |
| §5.3 định dạng đáp án, không dùng QA do LLM sinh, không có ngữ cảnh lâm sàng | 6 (prompt cố định), 12 |
| §6 dữ liệu, chia 70/10/20, tập test chính thức của VinDr (S8) | 2, 4, 5 |
| §7 E0–E5, số seed, ngân sách | 5, 13, 14, 15, 18 |
| §8.1 chỉ số mức ảnh, AUC-PR, calibration, PPV theo tỷ lệ bệnh | 7, 17 |
| §8.2 IoU, FROC, JSRT theo bán kính và mm | 9, 17 |
| §8.4 bootstrap, McNemar, DeLong, Holm | 8, 17 |
| "Việc phải làm đầu tiên" 1–4 | 0, 5, 11 (Bước 8), 13 (Bước 2) |
| "Việc phải làm đầu tiên" 5: đọc toàn văn L45, L50 | Việc tài liệu, không phải code; làm song song |

**Lưu ý cho agent thực thi**
- Mọi bước chạy trên GPU (Task 11 Bước 8, Task 13 Bước 2–3, Task 14–15 Bước 4, RUNBOOK) phải ghi lại thời gian thực tế vào `outputs/budget.md`.
