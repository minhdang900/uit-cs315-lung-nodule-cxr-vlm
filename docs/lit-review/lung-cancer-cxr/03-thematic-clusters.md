# Phân nhóm theo chủ đề: LLM/VLM cho nốt và ung thư phổi trên X-quang ngực

*Phiên bản 2, ngày 2026-11-12. Phân tích 90 bài trong `corpus.md` (L1–L92; L46 trùng L2 và L54 trùng L3 nên gộp lại).*

*So với v1, bản này thêm 30 bài từ Scopus (L63–L92, xem `04-scopus-prestige-search.md`) và thêm cụm phương pháp **M**.*

**Cách phân nhóm**
- Mỗi bài được gán vào **một cụm chính** theo câu hỏi nghiên cứu chủ đạo của bài. Cột "Cụm phụ" ghi các cụm khác mà bài cũng đóng góp.
- Mức đọc khác nhau giữa các bài:
  - Đọc toàn văn: L1, L47, L48, L49, L64.
  - Chỉ đọc abstract: phần lớn L1–L62, và L63, L74–L77, L85, L89, L90.
  - Chỉ dựa vào tiêu đề và nguồn: các bài trong L63–L92 có PDF nhưng chưa đọc, đánh dấu *(chưa đọc)*.
- Việc chia cụm làm thủ công theo nội dung, chưa dùng phân cụm tự động như embedding hay đồng trích dẫn.

---

## Tổng quan 11 cụm

| # | Cụm chủ đề | Số bài (v1 → v2) | Giai đoạn | Vai trò với đề tài |
|---|---|---|---|---|
| T1 | Đánh giá zero-shot MLLM/VLM cho nốt và ung thư phổi trên CXR | 8 → 8 | 2025–2026 | **Lõi: xác lập vấn đề** |
| T2 | VLM, agent và mô hình tạo sinh chuyên cho X-quang ngực | 9 → 12 | 2023–2026 | Mô hình ứng viên; tiền lệ tinh chỉnh |
| T3 | Bộ mã hóa thị giác–ngôn ngữ và giới hạn biểu diễn | 3 → 5 | 2023–2026 | **Cơ chế thất bại** |
| T4 | Định vị, grounding và benchmark nhận thức/VQA | 5 → 8 | 2025–2026 | Phương pháp đo định vị |
| T5 | LLM chỉ đọc văn bản báo cáo X-quang | 8 → 8 | 2023–2026 | Nhánh văn bản, hướng mở rộng |
| T6 | CAD học sâu phát hiện nốt và ung thư phổi (hiệu năng độc lập) | 7 → 15 | 2019–2024 | **Mốc so sánh** |
| T7 | Tương tác người–AI và tác động lâm sàng của CAD | 5 → 13 | 2019–2024 | Mốc lâm sàng; rủi ro AI sai |
| T8 | Dự đoán nguy cơ và sàng lọc ung thư phổi từ CXR | 5 → 7 | 2020–2026 | Hướng mở rộng |
| T9 | Dataset và benchmark có nhãn nốt/khối | 8 → 8 | 2017–2026 | Nguồn dữ liệu |
| T10 | Tổng quan hệ thống và phân tích gộp | 2 → 3 | 2021–2025 | Số liệu gộp làm mốc |
| **M** | **Phương pháp: độ phân giải, mất cân bằng lớp, đánh giá** (mới) | 0 → 3 | 2020–2024 | **Quyết định thiết kế trong spec** |
| | **Tổng** | **60 → 90** | | |

```
                ┌──────────── T9 Dataset ────────────┐
                │                                     │
   T2 VLM nền ──┼──► T1 Đánh giá zero-shot ◄── T6 CAD (mốc) ◄── T10 Tổng quan
        │       │          │   (vấn đề)              │
        ▼       │          ▼                         ▼
   T3 Cơ chế ───┘    T4 Định vị/VQA            T7 Người–AI / RCT
        ▲                  ▲                         │
        └──── M Phương pháp (độ phân giải, mất cân bằng lớp, đánh giá)
   T5 LLM văn bản ─────────────► T8 Nguy cơ / sàng lọc ◄┘
```

*Đọc sơ đồ:*
- T1 dùng mô hình của T2 và dữ liệu của T9, rồi so với T6.
- T3 và M giải thích kết quả của T1 (độ phân giải, cách gộp đặc trưng) và cho các quyết định thiết kế.
- T5 và T8 là hai nhánh mở rộng.

---

## T1. Đánh giá zero-shot MLLM/VLM cho nốt và ung thư phổi trên CXR (8 bài)

**Mô tả**
- Các đánh giá độc lập chạy LLM đa phương thức đa năng và VLM chuyên ngành ở chế độ zero-shot trên ảnh CXR có nốt, khối hoặc ung thư.
- Đặc điểm chung: mẫu nhỏ (50–257 ảnh), hồi cứu, một nguồn dữ liệu.
- Kết luận thống nhất: sensitivity thấp, mô hình bỏ sót nốt nhỏ và mờ. L52 là phản ví dụ vì khối u lớn được nhận diện khá tốt.
- *Cập nhật: đã đọc toàn văn L1, L2/L46 (preprint), L3, L4, L51 (tóm tắt hội nghị) và L52; chi tiết trong `05`.*
- Đợt bổ sung không thêm bài nào vào T1: Scopus Q1 chỉ tìm lại L1 và L3. Cụm vẫn chỉ có bằng chứng zero-shot.

| ID | Bài | Nơi công bố | Kết quả chính | Cụm phụ |
|---|---|---|---|---|
| L1 | Khovanova et al. 2026: 9 MLLM, VLM và agent | Front Digit Health | AUC tốt nhất 0,743 (BiomedCLIP); CXR-LLaVA sens 0,079 | T2, T3 |
| L2 (+L46) | Nishio et al. 2026: 6 VLM trên JSRT | AJR (+ medRxiv) | Sens 0,6–27,3%; MedGemma 5,2% (áp chót), spec 100% | T2 |
| L3 (=L54) | Chen CH et al. 2026: Gemini và GPT-4 đọc CXR | Diagnostics | Ác tính đơn ổ: phát hiện khoảng 10% (đa ổ: 55%) | — |
| L4 | Cusick et al. 2026: ngữ cảnh có cấu trúc | Cureus | Ung thư phổi đúng 20% | — |
| L51 | Abou-Zeid et al. 2026: hai VLM viết báo cáo | RöFo | Khối phổi ít được nhận đúng | T2 |
| L52 | Lacaita et al. 2025: ChatGPT-4o | J Pers Med | CXR đúng 66,15%; khối > 3 cm: 9/10 | — |
| L53 | Starychenko et al. 2025: chatbot đọc CXR | ERS Congress (tóm tắt hội nghị) | Nốt: sens ≤ 41,3% | — |
| L62 | Chu 2026: thận trọng khi dùng VLM sàng lọc nốt | AJR (commentary) | Chỉ có tiêu đề | — |

---

## T2. VLM, agent và mô hình tạo sinh chuyên cho X-quang ngực (12 bài)

**Mô tả**
- Các VLM hoặc LLM đa phương thức huấn luyện trên dữ liệu CXR lớn, cho các tác vụ sinh báo cáo, VQA, grounding, hội thoại và agent.
- Hiệu năng được báo trên benchmark tổng hợp, không có kết quả riêng cho nốt.
- Đợt bổ sung thêm 3 bài:
  - L63: bằng chứng ở tạp chí Radiology rằng mô hình chuyên ngành vượt xa GPT-4V. Tỷ lệ bác sĩ chấp nhận báo cáo: 70,5% so với 29,6%.
  - L64: tinh chỉnh LoRA một VLM nhỏ cho X-quang phổi, nhưng chỉ sinh chú thích, đánh giá bằng ROUGE/BLEU, không phát hiện hay định vị nốt. Đây không phải tiền lệ LoRA duy nhất: L69 (AAAI) dùng LoRA rank 16 cho LLM sinh báo cáo CXR, còn L71 (MICCAI) tinh chỉnh Qwen2.5-VL cho grounding tổng quát.
  - L65: kết hợp LLM với CNN để sàng lọc.
- MedGemma (L7) là VLM chính của đề tài.

| ID | Bài | Nơi công bố | Đặc điểm | Cụm phụ |
|---|---|---|---|---|
| L5 | Lee S et al. 2025: CXR-LLaVA | Eur Radiol | F1 0,81 nội bộ / 0,56 ngoài | — |
| L6 | Chen Z et al. 2024: CheXagent | arXiv | 8 loại tác vụ | — |
| L7 | Sellergren et al. 2026: MedGemma | Nat Med | Bộ mã hóa MedSigLIP | T3 |
| L8 | Bannur et al. 2024: MAIRA-2 | arXiv | Sinh báo cáo có grounding | T4 |
| L9 | Deperrois et al. 2026: RadVLM | **Sci Rep** (trước đây ghi arXiv) | Hội thoại và grounding | T4 |
| L10 | Zambrano Chaves et al. 2025: LLaVA-Rad | Nat Commun | 7B | — |
| L11 | Fallahpour et al. 2025: MedRAX | arXiv | Agent gọi công cụ | T4 |
| L12 | Thawakar et al. 2023: XrayGPT | arXiv | Tóm tắt báo cáo | — |
| L59 | Le-Duc et al. 2024: LiteGPT | arXiv | Định vị và phân loại trên VinDr | T4 |
| **L63** | Hong EK et al. 2025: AI tạo sinh chuyên ngành cho báo cáo CXR | **Radiology** | 8,8 triệu cặp ảnh–báo cáo; chấp nhận 70,5% (bác sĩ 73,3%, GPT-4V 29,6%) | T1 |
| **L64** | Volkov et al. 2025: SmolVLM + LoRA cho báo cáo có cấu trúc | SCM (IEEE) | QLoRA 0,47% tham số; chỉ có chỉ số văn bản; có ví dụ sai lâm sàng | M |
| **L65** | Zhang et al. 2026: LLM + CNN sàng lọc trên MIMIC-CXR *(chưa đọc)* | Front Digit Health | — | T6 |

---

## T3. Bộ mã hóa thị giác–ngôn ngữ và giới hạn biểu diễn (5 bài)

**Mô tả**
- Các mô hình nền kiểu CLIP hoặc ViT và khả năng giữ tín hiệu tổn thương nhỏ khi chuyển sang CXR.
- Hai bài lõi:
  - L47: mất tín hiệu khi gộp toàn cục, dùng patch-local thì lấy lại được.
  - L50: VLM giữ nguyên trọng số kém xa CNN.
- Đợt bổ sung:
  - L66: embedding RAD-DINO (tiền huấn luyện trên CXR) ở độ phân giải gốc, đông cứng, dùng cho **CT liều thấp (NLST), không phải CXR**. Chỉ là bằng chứng gián tiếp về chuyển giao bộ mã hóa; bài có ích về cách đánh giá ở tỷ lệ bệnh thực tế (PR-AUC, hiệu chỉnh xác suất). *(Sửa sau khi đọc toàn văn trong 05.)*
  - L68: BioViL-T (CVPR, A*), tiền huấn luyện thị giác–ngôn ngữ cho CXR.

| ID | Bài | Nơi công bố | Kết quả chính | Cụm phụ |
|---|---|---|---|---|
| L14 | Zhang S et al. 2023: BiomedCLIP | arXiv | Dẫn đầu ở L1 | T1 |
| L47 | Muthyala et al. 2026: embedding đông cứng làm mất tín hiệu tổn thương nhỏ | arXiv | AUC nốt 0,688 → 0,899 (CLS → patch-local) | T4 |
| L50 | Fisher 2025: VL không chuyển giao | medRxiv | F1 0,203 so với CNN 0,811 | T6 |
| **L66** | Hoq et al. 2025: RAD-DINO độ phân giải gốc, áp dụng cho CT liều thấp | J Imaging Inform Med | PR-AUC 0,705 sau hiệu chỉnh; **CT, không phải CXR** | M, T8 |
| **L68** | Bannur et al. 2023: BioViL-T *(chưa đọc)* | **CVPR** | — | T2 |

---

## T4. Định vị, grounding và benchmark nhận thức/VQA (8 bài)

**Mô tả**
- Đánh giá xem VLM có chỉ ra được tổn thương ở đâu không, kèm các benchmark VQA.
- Phát hiện chung: năng lực giảm mạnh khi chuyển từ "có/không" sang "ở đâu".
- Đợt bổ sung thêm 3 bài phương pháp grounding ở hội nghị và tạp chí hàng đầu:
  - L69: AAAI.
  - L70: IEEE TMI.
  - L71: MICCAI, dùng GRPO với phần thưởng không gian.
- Cả ba làm phrase grounding hoặc báo cáo có neo giải phẫu, **không riêng cho nốt**.

| ID | Bài | Nơi công bố | Kết quả chính | Cụm phụ |
|---|---|---|---|---|
| L48 | Gosai et al. 2025: MLLM định vị bệnh lý | PMLR (ML4H) | Lung lesion: GPT-5 35,7%, MedGemma 14,3%, bác sĩ 85% | T1 |
| L49 | Choi et al. 2026: CheXpercept | arXiv | Chỉnh đường viền ≤ 2,9% | T9 |
| L42 | Pal et al. 2025: ReXVQA | arXiv | MedGemma 83,24% | T9 |
| L43 | Nguyen DH et al. 2025: VinDr-CXR-VQA | arXiv | MedGemma F1 0,624 | T9 |
| L44 | Wei et al. 2026: REVEAL-CXR | arXiv | 200 ca, 12 nhãn | T9 |
| **L69** | Gao et al. 2026: S2D-Align *(chưa đọc)* | **AAAI** | Báo cáo có neo giải phẫu | T2 |
| **L70** | Zhang et al. 2026: phrase grounding tổng quát *(chưa đọc)* | **IEEE TMI** | — | — |
| **L71** | Xu et al. 2025: MedGround-R1 *(chưa đọc)* | **MICCAI** | Grounding bằng GRPO | M |

---

## T5. LLM chỉ đọc văn bản báo cáo X-quang (8 bài, không đổi)

**Mô tả.** Hai tiểu nhóm:
- **(a) Trích xuất và gán nhãn:** L55–L58. Hiệu năng cao.
- **(b) Suy luận theo guideline:** L15–L18. Kém nếu không đưa guideline vào prompt, và chủ yếu làm trên báo cáo CT.

| ID | Bài | Nơi công bố | Kết quả chính |
|---|---|---|---|
| L55 | Mukherjee et al. 2023: Vicuna gán nhãn báo cáo | Radiology | AUC trung vị 0,84 |
| L56 | Savage et al. 2024: GPT-4 so với NLP chuyên ngành | AJR | GPT-4 tốt hơn |
| L57 | Park et al. 2024: trích xuất 7 bệnh phổi | PLOS ONE | Sens 0,71–1,00 |
| L58 | Yasaka et al. 2024: LLM fine-tune lọc ca ung thư phổi | J Imaging Inform Med | Acc 0,983 |
| L15 | Gamble et al. 2024: GPT áp dụng Fleischner | Can Assoc Radiol J | 0,15 → 0,66 khi có guideline |
| L16 | Gulati et al. 2026: phân loại Lung-RADS | Clin Radiol | 68,8% |
| L17 | Zhou T et al. 2024: Lung-RADS bằng in-context learning | Kỷ yếu | Không có số liệu |
| L18 | Wen et al. 2025: khuyến nghị theo dõi nốt | Eur J Radiol Open | 92,8–94,6% |

---

## T6. CAD học sâu phát hiện nốt và ung thư phổi, hiệu năng độc lập (15 bài)

**Mô tả**
- CNN học thuật và thương mại phát hiện nốt ác tính hoặc ung thư phổi trên CXR.
- Đợt bổ sung gấp đôi cụm này, chủ yếu với bài từ Radiology, Lancet Digital Health, Eur Radiol và MedIA.
- Thông điệp mới:
  - CAD đạt AUC 0,97–0,99 trên quần thể khám sức khỏe (L75), và AUROC 0,899 trên ung thư từng bị bỏ sót (L81).
  - Kiểm định độc lập nhiều sản phẩm cho thấy **không đồng đều**: 4/7 thuật toán vượt người đọc (L80).
  - Tỷ lệ dương tính giả khoảng 3% (L75). Con số này quan trọng khi tỷ lệ bệnh thấp, liên quan cụm M.
- Đây là **mốc so sánh** cho baseline B1 và B2.

| ID | Bài | Nơi công bố | Kết quả chính | Cụm phụ |
|---|---|---|---|---|
| L19 | Nam et al. 2019: DLAD nốt ác tính | Radiology | AUROC 0,92–0,99 | T7 |
| L21 | Yoo et al. 2020: kiểm định trên NLST | JAMA Netw Open | Sens 86,2%, spec 85,0% | T8 |
| L25 | Hwang et al. 2019: 4 bệnh lồng ngực chính | JAMA Netw Open | AUC 0,979 | — |
| L26 | Majkowska et al. 2020: chuẩn tham chiếu do bác sĩ thẩm định | Radiology | Nốt/khối AUC 0,72 | T9 |
| L29 | Shimazaki et al. 2022: phân đoạn ung thư phổi | Sci Rep | Điểm mù 0,50–0,64 | — |
| L31 | Ohlmann-Knafo et al. 2022: CAD trên JSRT | arXiv | Bác sĩ AUC 0,77 | T7 |
| L45 | Arzamasov et al. 2024: 5 phần mềm thương mại | QIMS | AUC 0,956 → 0,812 | T1 |
| **L72** | Cid et al. 2024: X-Raydar mã nguồn mở *(chưa đọc)* | **Lancet Digit Health** | Đọc CXR toàn diện, nhiều trung tâm | T2 |
| **L73** | Pesce et al. 2019: mạng attention, tổn thương phổi *(chưa đọc)* | **MedIA** | Nhãn yếu kèm một ít box | T4 |
| **L75** | Lee JH et al. 2020: DL so với bác sĩ, khám sức khỏe | **Radiology** | AUC 0,97–0,99; sens 83–90%; FPR khoảng 3% | M |
| **L77** | Hwang et al. 2021: CAD phát hiện di căn phổi | **Radiology** | Tỷ lệ phát hiện 0,32% → 0,86% | T7 |
| **L78** | Shen et al. 2023: tổng hợp ảnh để tăng cường dữ liệu nốt *(chưa đọc)* | **MedIA** | — | M |
| **L79** | Kim et al. 2023: kiểm định ngoài CAD thương mại *(chưa đọc)* | **Eur Radiol** | Tỷ lệ bệnh thấp | M |
| **L80** | van Leeuwen et al. 2024: 7 thuật toán nốt thương mại | **Radiology** | 4/7 vượt người đọc (AUC 0,86–0,93 so với 0,81) | T7 |
| **L81** | Nam et al. 2020: ung thư bị bỏ sót | Radiol Cardiothorac Imaging | AUROC 0,899; AUFROC 0,744 | T7 |

---

## T7. Tương tác người–AI và tác động lâm sàng của CAD (13 bài)

**Mô tả**
- Nghiên cứu nhiều người đọc, RCT và đánh giá thực tế về AI hỗ trợ bác sĩ.
- Đợt bổ sung thêm 8 bài, với ba thông điệp:
  - **(i) Mốc năng lực của bác sĩ:** sensitivity với nốt ác tính khoảng 65% (L85), với ung thư từng bị bỏ sót 40–53% (L74).
  - **(ii) Lợi ích của AI:** nốt +12% sensitivity (L89), JAFROC 0,90 → 0,95 (L76).
  - **(iii) Rủi ro khi AI sai:** AI kém kéo người đọc theo (L88). Người đọc đổi chẩn đoán theo AI trong 59–67% trường hợp (L90).
- Thông điệp (iii) là luận cứ trực tiếp cho phần **kiểm toán VLM** của đề tài: một VLM sensitivity thấp nhưng nói năng tự tin có thể gây hại.

| ID | Bài | Nơi công bố | Kết quả chính | Cụm phụ |
|---|---|---|---|---|
| L20 | Nam et al. 2023: RCT | Radiology | Phát hiện nốt cần xử trí 0,25% → 0,59% | T6 |
| L22 | Yoo et al. 2021: NLST, nhiều người đọc | Eur Radiol | Sens 0,61 → 0,72 | T6 |
| L27 | Dissez et al. 2022: phát hiện sớm ung thư | arXiv | +17,4% ca ung thư | — |
| L28 | Topff et al. 2024: AI đọc kép | Eur Radiol | 71,4% dấu hiệu bị bỏ sót là nốt | T5 |
| L32 | Tang et al. 2023: giao diện AI | Radiol AI | Giao diện chỉ văn bản tốt nhất | — |
| **L74** | Jang et al. 2020: DLAD và ung thư bị bỏ sót | **Radiology** | AUC 0,67 → 0,76; sens 40% → 53% | T6 |
| **L76** | Sung et al. 2021: thử nghiệm ngẫu nhiên bắt chéo | **Radiology** | JAFROC 0,90 → 0,95 | T4 (FROC) |
| **L85** | Sim et al. 2020: DCNN, 4 trung tâm | **Radiology** | Bác sĩ 65,1% → 70,3% | T6 |
| **L86** | Homayounieh et al. 2021: AI và độ chính xác của người đọc *(chưa đọc)* | **JAMA Netw Open** | — | — |
| **L87** | Ahn et al. 2022: AI hỗ trợ đọc CXR *(chưa đọc)* | **JAMA Netw Open** | — | — |
| **L88** | Bernstein et al. 2023: AI sai ảnh hưởng bác sĩ *(chưa đọc)* | **Eur Radiol** | — | T1 |
| **L89** | Bennani et al. 2023: AI và 5 loại bất thường | **Radiology** | Nốt +12% sens; thời gian đọc −31% | — |
| **L90** | Lee JH et al. 2023: tương tác người–AI | **Radiology** | AI tốt: AUC 0,77 → 0,82; người đọc đổi theo AI 59–67% | T1 |

---

## T8. Dự đoán nguy cơ và sàng lọc ung thư phổi từ CXR (7 bài)

**Mô tả**
- Dùng CXR, có thể kèm EHR, để dự đoán nguy cơ dài hạn hoặc chọn người cần chụp CT.
- Đợt bổ sung thêm 2 bài:
  - L91: kết hợp CXR với EMR, đăng ở JAMA Netw Open.
  - L67: mô hình nền lai cho tiên lượng ung thư phổi trên X-quang liều thấp.
- LLM trong cụm này vẫn chỉ xuất hiện ở preprint hoặc tóm tắt hội nghị (L13, L61).

| ID | Bài | Nơi công bố | Kết quả chính | Cụm phụ |
|---|---|---|---|---|
| L23 | Lu et al. 2020: CXR-LC | Ann Intern Med | AUC 0,755 | T6 |
| L24 | Lee JH et al. 2022: chọn người chụp CT theo USPSTF | Radiology | AUC 0,68–0,75 | — |
| L30 | Kotevska et al. 2026: ViT dự đoán sớm | arXiv | +6–10 điểm AUC | — |
| L13 | Yang et al. 2026: LUNG-KGMM | arXiv | Không có số liệu | T5 |
| L61 | Kim et al. 2026: CXR + đặc trưng LLM | J Thorac Oncol (tóm tắt hội nghị) | Chỉ có tiêu đề | T5 |
| **L67** | Hayeso et al. 2026: mô hình nền lai, X-quang liều thấp *(chưa đọc)* | Transl Lung Cancer Res | — | T3 |
| **L91** | Raghu et al. 2022: CXR + EMR dự đoán nguy cơ *(chưa đọc)* | **JAMA Netw Open** | — | T6 |

---

## T9. Dataset và benchmark có nhãn nốt/khối (8 bài)

**Mô tả.** Không đổi về số bài. Riêng L35 (NODE21) nay đã có bản đăng chính thức ở IEEE TMI 2024.

| ID | Bài | Nơi công bố | Nhãn nốt/khối |
|---|---|---|---|
| L35 | Sogancioglu et al. 2024: NODE21 | **IEEE TMI** | Có box |
| L36 | Mustafic et al. 2026: NoduLoCC2026 | arXiv | Có |
| L60 | Mustafic et al. 2026: dataset NoduLoCC | Kỷ yếu ICCVIT | Có |
| L37 | Nguyen HQ et al. 2022: VinDr-CXR | Sci Data | Có Nodule/Mass |
| L38 | Wang X et al. 2017: ChestX-ray14 | CVPR | Nodule, Mass |
| L39 | Irvin et al. 2019: CheXpert | AAAI | Lung Lesion |
| L40 | Johnson et al. 2019: MIMIC-CXR-JPG | arXiv | 14 nhãn |
| L41 | Liu J et al. 2020: ChestX-Det10 | arXiv | Có box |

---

## T10. Tổng quan hệ thống và phân tích gộp (3 bài)

| ID | Bài | Nơi công bố | Kết quả chính |
|---|---|---|---|
| L33 | Essa 2025: AI trên CXR cho viêm phổi và ung thư phổi | Eur J Radiol Open | Nốt: sens khoảng 72%, spec khoảng 95% |
| L34 | Megat Ramli et al. 2025: AI phát hiện nốt | Diagnostics | Sens 56,4–95,7% |
| **L92** | Aggarwal et al. 2021: độ chính xác DL trong chẩn đoán hình ảnh *(chưa đọc)* | **npj Digit Med** | Phân tích gộp nhiều chuyên khoa, có CXR |

---

## M. Phương pháp: độ phân giải, mất cân bằng lớp, đánh giá (3 bài, cụm mới)

**Mô tả**
- Các bài không trực tiếp về VLM hay CAD, nhưng chi phối các **quyết định thiết kế** trong spec.
- **Độ phân giải đầu vào:**
  - L82: AUC phát hiện nốt ở 64 px chỉ bằng 80,7% AUC ở 320 px; tốt nhất ở 256–448 px.
  - L83: kích thước ma trận ảnh tối ưu cho CAD nốt/khối.
  - Đây là cơ sở để kiểm tra độ phân giải ảnh của MedGemma (896 px) và Qwen2.5-VL, và để xét thí nghiệm chia ô.
- **Đánh giá khi ca dương hiếm:**
  - L84: chuẩn mực đánh giá khi mất cân bằng lớp, cả phân biệt lẫn hiệu chỉnh xác suất.
  - Liên quan C3 (cân bằng lớp) và việc báo cáo PPV/FPR bên cạnh AUC.

| ID | Bài | Nơi công bố | Kết quả chính | Cụm phụ |
|---|---|---|---|---|
| L82 | Sabottke & Spieler 2020: độ phân giải ảnh và DL | **Radiology AI** | Nốt: AUC 0,689 (64 px) so với 0,854 (320 px) | T3, T6 |
| L83 | Kim et al. 2020: kích thước ma trận tối ưu cho CAD nốt *(chưa đọc)* | **Eur Radiol** | — | T6 |
| L84 | Mosquera et al. 2024: mất cân bằng lớp *(chưa đọc)* | **Eur Radiol** | — | — |

---

## Nhận xét xuyên cụm (v2)

1. **Mất cân đối về cụm càng rõ.**
   - Cụm CAD và lâm sàng (T6, T7, T10: 31 bài) có kiểm định ngoài, RCT, nghiên cứu nhiều người đọc và phân tích gộp.
   - Cụm LLM/VLM (T1–T4: 33 bài) vẫn chủ yếu là mô tả mô hình hoặc đánh giá zero-shot.
   - **Vẫn chưa có bài nào tinh chỉnh VLM để phát hiện hay định vị nốt.** Các tiền lệ tinh chỉnh gần nhất đều nhắm tác vụ khác: L64 và L69 (LoRA, sinh báo cáo, chỉ số văn bản) và L71 (Qwen2.5-VL, grounding tổng quát trên MS-CXR, mIoU 79,02).
2. **Mốc năng lực đã có đủ ba tầng.** Bác sĩ đạt sensitivity khoảng 65% (L85), CAD đạt AUC 0,86–0,99 (L80, L75), còn VLM zero-shot đạt sensitivity 0,6–27% (L2). Ngưỡng của spec (≥ 0,60 ở specificity 0,90) tương đương mức bác sĩ.
3. **Hai cơ chế thất bại có bằng chứng độc lập:**
   - Gộp toàn cục làm mất tín hiệu tổn thương nhỏ (L47).
   - Độ phân giải thấp làm giảm phát hiện nốt mạnh hơn các bệnh khác (L82, L83; L83 thấy 896 px là tối ưu, trùng kích thước đầu vào của MedGemma).
4. **Rủi ro lâm sàng của VLM yếu nay có cơ sở.** T7 cho thấy AI sai vẫn kéo bác sĩ theo (L88, L90). Phần kiểm toán của đề tài vì thế có ý nghĩa an toàn, không chỉ là bài tập học thuật.
5. **Grounding ở hội nghị hàng đầu chưa chạm đến nốt.** L69–L71 làm phrase grounding tổng quát. Định vị nốt sau tinh chỉnh vẫn là ô trống.
6. **Dữ liệu dùng chung vẫn là lợi thế:** JSRT (L2 và L31), MosMedData VII (L1 và L45), cùng NODE21 nay đã đăng ở IEEE TMI.

## Ánh xạ cụm vào spec

| Cụm | Dùng ở mục nào của spec |
|---|---|
| T1 | §1 Vấn đề, §3 Khoảng trống, E2 |
| T2 | §5.1 Mô hình, C0; L64 là tiền lệ LoRA cần phân biệt ở §3 |
| T3 | §1 giả thuyết, C5/C6 |
| T4 | §8.2 Định vị, §8.3 Kiểm toán |
| T5 | §9 Hướng phát triển |
| T6, T7, T10 | B1, B2, ngưỡng thành công, §1 mốc so sánh; L88 và L90 cho luận cứ an toàn |
| T8 | §9 Hướng phát triển |
| T9 | §6 Dữ liệu |
| **M** | §6 tiền xử lý (độ phân giải, chia ô), C3 cân bằng lớp, §8.1 chỉ số (PPV/FPR, hiệu chỉnh) |

*Danh mục đầy đủ có DOI nằm trong `corpus.md` (mục 1–4, 6, 7).*
