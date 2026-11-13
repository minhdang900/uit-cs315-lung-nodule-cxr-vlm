# data/ — không commit lên git

Thư mục này chứa dữ liệu ảnh X-quang. Dữ liệu y tế có giấy phép riêng, chỉ dùng cho nghiên cứu và **không được phân phối lại**, nên `.gitignore` chặn mọi thứ trong đây trừ file này.

Cấu trúc mong đợi sau bước E0 (xem `code/RUNBOOK.md` và `docs/handoff/so_tay_ban_giao.pdf`, chương Runbook):

```
data/
  raw/          # dữ liệu tải về: VinBigData, NIH ChestX-ray14, JSRT
  processed/    # ảnh 896x896 và manifest_*.csv do scripts/prepare_*.py tạo ra
```

Các lệnh trong RUNBOOK chạy từ `code/`, nên dùng đường dẫn `../data/processed`.
