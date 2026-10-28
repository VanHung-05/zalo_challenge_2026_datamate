# PLAN: DataMate – Agent hỏi đáp tiếng Việt trên dữ liệu bảng

> Mục tiêu: xây agent nhận (data package + câu hỏi tiếng Việt + as-of date) và trả về đáp án đúng, hoặc `unanswerable` / `abstain` khi phù hợp. Đóng gói Docker chạy offline, tái lập được, xong trong 6 giờ trên 2×RTX 5090.

---

## 0. Nguyên tắc chỉ đạo

1. **LLM lập kế hoạch, code tính toán.** Mọi con số phải đến từ thực thi code (SQL/pandas), không từ "nhẩm" của LLM.
2. **Phần xác định (deterministic) làm bằng code:** giải quyết thời gian, tra lịch âm, chuẩn hóa số/tiền/ngày, định dạng đầu ra.
3. **Tổng quát, không hard-code** theo business ID hoặc câu hỏi (Rule 4).
4. **Luôn có đáp án sơ bộ + ghi từng dòng + flush** để an toàn khi hết giờ (Rule 7, 8).
5. **Tái lập:** temperature = 0, seed cố định, tránh logic phụ thuộc đồng hồ (Rule 8: chênh PA ≤ 0.03).
6. **Đo trước khi tối ưu:** mọi thay đổi phải được đánh giá trên tập dev nội bộ và phân loại lỗi.

---

## 1. Ràng buộc từ rules.txt (checklist tuân thủ)

| # | Ràng buộc | Hành động của team |
|---|-----------|--------------------|
| R2 | Model công khai trước 01/10/2026, license cho phép; **tổng tham số tất cả model ≤ 15B** (MoE tính toàn bộ) | Lập bảng "model budget"; khai báo trong README |
| R3 | Offline hoàn toàn, không API ngoài, không download | Bake weights + thư viện vào image; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; test với `--network none` |
| R4 | Không human-in-the-loop; cấm hard-code đáp án / logic theo question hoặc business ID | Code review checklist; grep cấm ID cụ thể |
| R5 | Được sinh và chạy code; `/data` read-only | DB in-memory hoặc ghi vào `/tmp`, `/work` |
| R6 | Không dùng test public/private để train, không thu thập đáp án test | Chỉ dùng train data BTC cấp + dữ liệu tự tổng hợp |
| R7 | 6 giờ wall-clock toàn bộ private test, gồm load model; CUDA ≥ 12.8 | Time budget + watchdog; test trên GPU Blackwell nếu có thể |
| R8 | Tái lập: \|PA(upload) − PA(re-run)\| ≤ 0.03; log đầy đủ | Chạy lặp 2–3 lần đo độ lệch; log có cấu trúc |

**Việc cần xác nhận với BTC (ghi lại câu trả lời):**
- [ ] Cách chấm abstain vs sai vs unanswerable (có phạt không, hệ số bao nhiêu?).
- [ ] Ranh giới "logic keyed on business identifiers" (ví dụ: heuristic theo tên cột phổ biến có được không?).
- [ ] Rulebook và bảng lịch âm được mount ở đâu trong container (`/data/...`?).
- [ ] Định dạng chi tiết cho từng loại đáp án (số, %, ngày, list, bảng, boolean).
- [ ] Docker guideline: base image, entrypoint, định dạng log.

---

<!-- ## 2. Cấu trúc repo đề xuất

```
datamate/
├── README.md                  # khai báo model, cách chạy, kiến trúc (bắt buộc)
├── plan.md
├── Dockerfile
├── entrypoint.sh
├── configs/
│   ├── default.yaml           # model path, seed, timeouts, thresholds
│   └── fast.yaml              # chế độ nhanh khi gần hết giờ
├── src/
│   ├── main.py                # vòng lặp chính, ghi submission từng dòng
│   ├── ingest/
│   │   ├── detect.py          # encoding, delimiter, loại file
│   │   ├── csv_loader.py
│   │   ├── excel_loader.py    # tìm header, merged cells, nhiều bảng/sheet
│   │   ├── sqlite_loader.py
│   │   └── clean.py           # ngày, tiền, Unicode NFC, kiểu dữ liệu
│   ├── profile/
│   │   ├── profiler.py        # thống kê cột, top-k giá trị, phạm vi ngày
│   │   └── linking.py         # schema linking + value linking
│   ├── knowledge/
│   │   ├── rulebook.py        # parse + retrieval rulebook
│   │   └── lunar.py           # tra bảng solar–lunar
│   ├── nlu/
│   │   ├── time_resolver.py   # "tháng này", "tuần rồi", "cùng kỳ năm ngoái"...
│   │   ├── vn_normalize.py    # bỏ dấu, số tiếng Việt, "1tr5", "15k"
│   │   └── question_parser.py # phân loại câu hỏi, trích xuất thực thể
│   ├── agent/
│   │   ├── planner.py
│   │   ├── codegen.py         # sinh SQL/pandas
│   │   ├── executor.py        # sandbox, timeout, self-debug
│   │   ├── verifier.py        # sanity check, unanswerable, abstain
│   │   └── consistency.py     # self-consistency + bỏ phiếu
│   ├── output/
│   │   └── formatter.py       # ép kiểu đầu ra đúng spec
│   └── utils/
│       ├── timing.py          # budget theo câu + đồng hồ tổng
│       └── logging.py
├── data_dev/                  # dữ liệu dev tự tổng hợp (KHÔNG chứa test)
├── eval/
│   ├── run_eval.py
│   ├── error_taxonomy.md
│   └── reports/
└── tests/                     # unit test cho time_resolver, clean, formatter
``` -->

---

## 3. Lộ trình theo giai đoạn

### Giai đoạn 0 – Khởi động & nắm đề (Hạn: ____)

- [ ] Đọc kỹ đề, rules, Docker guideline, định dạng `submission.jsonl`.
- [ ] Xem toàn bộ dữ liệu train/public test BTC cấp: số business, loại file, loại câu hỏi, phân bố kiểu đáp án.
- [ ] Đọc rulebook và bảng solar–lunar; ghi lại các định nghĩa nghiệp vụ (doanh thu, tồn kho, công nợ, kỳ báo cáo, đầu tuần...).
- [ ] Hoàn thành danh sách câu hỏi gửi BTC (mục 1).
- [ ] Chốt phân công, quy ước code, kênh trao đổi, quy trình ghi lại thí nghiệm.
- [ ] Dựng repo, CI đơn giản (lint + unit test).

**Deliverable:** tài liệu "Hiểu đề" 1–2 trang; repo khung; danh sách loại câu hỏi.

### Giai đoạn 1 – Baseline end-to-end (Hạn: ____)

- [ ] Loader cho CSV/XLSX/SQLite → nạp vào DuckDB (hoặc SQLite) in-memory.
- [ ] Profiler cơ bản (schema, kiểu, mẫu, min/max ngày).
- [ ] Prompt: README + rulebook + schema profile + câu hỏi + as-of date → LLM sinh SQL.
- [ ] Executor có timeout + self-debug (2–3 vòng).
- [ ] Formatter tối thiểu; ghi `submission.jsonl` từng dòng.
- [ ] Chạy trên public test, nộp lần đầu để có mốc điểm.
- [ ] **Chạy thử Docker sớm** (dù chỉ baseline) để phát hiện lỗi môi trường.

**Deliverable:** pipeline chạy end-to-end, điểm baseline, image Docker chạy được offline.

### Giai đoạn 2 – Nền tảng xác định (deterministic) (Hạn: ____)

**2.1 Time resolver**
- [ ] Xây `resolve_time(expr, as_of) → [start, end)` cho: hôm nay/qua/kia, tuần này/rồi/trước, tháng này/trước, quý, năm nay/ngoái, "N ngày/tuần/tháng gần đây", "cùng kỳ năm ngoái", "từ đầu tháng", "cuối tuần".
- [ ] Quy ước: thứ Hai đầu tuần, kiểm tra rulebook về "tuần rồi" (tuần lịch vs 7 ngày gần nhất).
- [ ] Unit test phủ edge case: đầu/cuối tháng, năm nhuận, qua năm, Tết.
- [ ] Tích hợp lịch âm: tra bảng cho Tết, mùng 1, rằm, giỗ, các dịp lễ âm.

**2.2 Chuẩn hóa dữ liệu**
- [ ] Ngày: `dd/mm/yyyy`, ISO, text, Excel serial, năm 2 chữ số; phát hiện thứ tự ngày/tháng theo phân bố giá trị.
- [ ] Tiền/số: `1.500.000`, `1,5tr`, `15k`, "VNĐ", "đ", dấu thập phân kiểu VN/Mỹ.
- [ ] Text: Unicode NFC, trim, chuẩn hóa khoảng trắng, cột `_norm` bỏ dấu cho tìm kiếm.
- [ ] Excel: tìm header row, merged cells, header nhiều tầng, dòng tổng cộng, nhiều bảng trên một sheet, sheet/cột ẩn.
- [ ] Ghi lại (trong log) mọi phép biến đổi để dễ truy vết.

**2.3 Schema & value linking**
- [ ] Index giá trị distinct của cột dạng danh mục (bỏ dấu, fuzzy).
- [ ] Tìm thực thể được nhắc trong câu hỏi ("trà sữa", "chị Lan") → cột/bảng tương ứng.
- [ ] Chọn top bảng/cột liên quan để đưa vào prompt (giảm nhiễu).

**Deliverable:** các module có unit test; điểm tăng so với baseline; báo cáo lỗi.

### Giai đoạn 3 – Tri thức nghiệp vụ & độ chính xác (Hạn: ____)

- [ ] Parse rulebook thành các định nghĩa có cấu trúc (doanh thu, lợi nhuận, công nợ, tồn kho, hết hạn, bán chạy...).
- [ ] Viết sẵn macro/view SQL tái sử dụng (ví dụ `valid_orders`, `revenue`, `debt`, `stock_on_hand`).
- [ ] Phân loại câu hỏi (aggregate, top-k, so sánh kỳ, tỷ lệ, tồn đọng, join, lịch âm, boolean, list/table) → chọn template prompt phù hợp.
- [ ] Few-shot theo **loại** câu hỏi (viết tay/tự tổng hợp, không chép từ test).
- [ ] Bước "khám phá" trước khi viết truy vấn chính (`SELECT DISTINCT`, `LIMIT`).
- [ ] Xử lý tie, thứ tự, làm tròn, đơn vị theo spec.

**Deliverable:** tăng điểm rõ rệt ở nhóm câu cần định nghĩa nghiệp vụ.

### Giai đoạn 4 – Unanswerable / Abstain / Tin cậy (Hạn: ____)

- [ ] **Kiểm tra trước khi sinh code:** khái niệm trong câu hỏi có ánh xạ được vào cột/bảng không? (ví dụ hỏi lợi nhuận mà không có giá vốn).
- [ ] **Kiểm tra phạm vi:** khoảng thời gian hỏi có nằm trong phạm vi dữ liệu? thực thể có tồn tại?
- [ ] **Kiểm tra sau khi chạy:** kết quả rỗng/null/vô lý (âm, vượt ngưỡng) → xử lý.
- [ ] Phân biệt câu hỏi không phải dữ liệu (dự đoán, "vì sao", ý kiến).
- [ ] Self-consistency (N=3–5, bỏ phiếu theo kết quả thực thi) làm tín hiệu tin cậy.
- [ ] Hiệu chỉnh ngưỡng abstain trên dev theo công thức chấm của BTC.
- [ ] Tạo tập dev riêng cho nhóm unanswerable (cần đa dạng, không chỉ vài mẫu).

**Deliverable:** bảng precision/recall của unanswerable; ngưỡng abstain đã hiệu chỉnh.

### Giai đoạn 5 – Mô hình & hiệu năng (Hạn: ____)

- [ ] Lập **model budget** (tổng ≤ 15B, gồm LLM + embedding + reranker + classifier). Ví dụ hướng cần cân nhắc: một LLM mạnh về code/SQL + tiếng Việt cỡ 7–14B, một embedding nhỏ. *Cần tự kiểm chứng ngày phát hành weights, license và số tham số thực tế.*
- [ ] So sánh vài LLM trên dev (độ đúng, tốc độ, tiếng Việt, khả năng sinh SQL).
- [ ] Cân nhắc quantization (AWQ/GPTQ/FP8) để tăng tốc; đo ảnh hưởng đến độ đúng.
- [ ] Cân nhắc fine-tune (LoRA) trên dữ liệu **tự tổng hợp** / train được cấp (text-to-SQL tiếng Việt); không dùng test.
- [ ] Phân bổ 2 GPU: chạy song song 2 worker (mỗi GPU một instance) hoặc tensor parallel; chọn phương án nhanh và ổn định hơn.
- [ ] Router theo độ khó: câu dễ → 1 lần sinh; câu khó → self-consistency/nhiều vòng.
- [ ] Cache profile/schema theo business để không tính lại cho mỗi câu hỏi.
- [ ] Ước lượng thời gian: `tổng thời gian ≈ load model + Σ(thời gian/câu)`; dự phòng ≥ 30% so với giới hạn 6 giờ.

**Deliverable:** cấu hình model cuối, bảng đo thời gian/câu, kế hoạch chế độ nhanh.

### Giai đoạn 6 – Đóng gói Docker & kiểm thử tái lập (Hạn: ____)

- [ ] Dockerfile theo đúng base image trong Docker guideline (CUDA ≥ 12.8, PyTorch/vLLM tương thích Blackwell).
- [ ] Bake sẵn: weights, tokenizer, embedding, thư viện (duckdb, pandas, openpyxl, chardet...), rulebook/lịch âm nếu cần.
- [ ] Biến môi trường offline; **không** có lệnh nào tải từ mạng.
- [ ] Entry point: đọc `/data`, ghi `/result/submission.jsonl` từng dòng + flush, ghi log vào `/result/` (theo guideline).
- [ ] Watchdog thời gian tổng; chuyển chế độ nhanh khi còn < X phút; tuyệt đối không vượt 6 giờ.
- [ ] Xử lý lỗi từng câu: lỗi/timeout → ghi `abstain` rồi tiếp tục, không để crash cả run.
- [ ] Test `docker run --network none --gpus all -v <data>:/data:ro -v <out>:/result ...`.
- [ ] **Chạy lặp 2–3 lần** trên cùng test set; đo |ΔPA|, mục tiêu ≪ 0.03. Nếu lệch: kiểm tra seed, batching vLLM, thứ tự xử lý, logic phụ thuộc thời gian.
- [ ] Kiểm tra dung lượng image, thời gian khởi động, chạy được trên máy khác.
- [ ] Viết README: kiến trúc, model đã dùng (tên, nguồn, số tham số, license), cách chạy, lưu ý tái lập.

**Deliverable:** Docker image cuối + README + báo cáo tái lập.

### Giai đoạn 7 – Private test & nộp cuối (Hạn: ____)

- [ ] **Đóng băng** (freeze) code & image; ghi lại hash/tag.
- [ ] Chạy image trên private test; kiểm tra file `submission.jsonl` hợp lệ (đủ dòng, đúng schema).
- [ ] Upload; lưu bản sao log để giải trình khi BTC hỏi.
- [ ] Chuẩn bị sẵn thông tin trả lời BTC nếu PA lệch (log, cấu hình, seed).

---

## 4. Chiến lược đánh giá

### 4.1 Tập dev
- Dùng dữ liệu train/dev BTC cấp (nếu có đáp án).
- Tự tổng hợp thêm **50–200 câu** (có đáp án tính tay/bằng code độc lập) bao phủ:
  - Tổng hợp (sum/count/avg), top-k, so sánh kỳ, tỷ lệ/%, tăng trưởng.
  - Thời gian tương đối, cùng kỳ năm ngoái, lịch âm.
  - Join nhiều bảng, công nợ, tồn kho, hạn sử dụng.
  - Boolean, list, table.
  - **Unanswerable** các kiểu: thiếu cột, ngoài phạm vi thời gian, thực thể không tồn tại, câu hỏi không phải dữ liệu.
- Tạo business mẫu **bẩn**: header lệch, ngày lẫn lộn định dạng, tiền dạng chuỗi, Unicode NFD, dòng tổng, sheet ẩn.
- **Tuyệt đối không** đưa dữ liệu/câu hỏi test vào dev để tinh chỉnh (Rule 6).

### 4.2 Chỉ số
- PA (độ đúng) tổng và theo từng loại câu hỏi / loại đáp án.
- Tỷ lệ unanswerable đúng/sai, tỷ lệ abstain.
- Điểm theo công thức chính thức của BTC (khi rõ).
- Thời gian/câu (p50, p95), tổng thời gian run.
- Độ lệch giữa các lần chạy (tái lập).

### 4.3 Error taxonomy (cập nhật liên tục)
| Nhóm lỗi | Ví dụ | Hướng sửa |
|----------|-------|-----------|
| Sai thời gian | "tuần rồi" lệch khoảng | Sửa `time_resolver`, thêm test |
| Sai cột / bảng | Chọn nhầm cột ngày | Cải thiện linking, mô tả cột |
| Sai thực thể | Không khớp "trà sữa" | Value linking, fuzzy |
| Sai định nghĩa nghiệp vụ | Doanh thu tính cả đơn hủy | Macro từ rulebook |
| Dữ liệu bẩn | Tiền dạng chuỗi | Cải thiện `clean.py` |
| Sai định dạng | % vs hệ số | Formatter |
| Sai unanswerable | Trả bừa / từ chối thừa | Verifier, ngưỡng |
| Lỗi hệ thống | Timeout, crash | Xử lý lỗi, fallback |

Quy trình mỗi vòng: chạy dev → gom lỗi → xếp theo số lượng → sửa nhóm lớn nhất → chạy lại → ghi nhật ký.

---

## 5. Thiết kế kỹ thuật chi tiết (tóm tắt)

**Luồng xử lý một câu hỏi**
```
Câu hỏi + as_of + business package
   ↓
[Cache] profile + DB đã nạp theo business
   ↓
Chuẩn hóa câu hỏi (NFC, bỏ dấu bản phụ, số/tiền)
   ↓
Time resolver (code) → khoảng thời gian cụ thể + lịch âm nếu có
   ↓
Schema/value linking → bảng/cột/thực thể liên quan
   ↓
Pre-check: khái niệm có ánh xạ được không? (unanswerable sớm)
   ↓
Prompt (README + rulebook liên quan + schema + thời gian đã giải quyết)
   ↓
Sinh SQL/pandas (N phương án nếu cần) → thực thi sandbox → self-debug
   ↓
Post-check: rỗng/null/vô lý/ngoài phạm vi dữ liệu
   ↓
Bỏ phiếu + ước lượng tin cậy → answer / unanswerable / abstain
   ↓
Formatter → ghi JSONL + flush + log
```

**Sandbox thực thi:** timeout mỗi truy vấn, giới hạn bộ nhớ, chỉ đọc dữ liệu gốc, không mạng, chặn thao tác ghi/xóa ngoài thư mục tạm, chặn `import` nguy hiểm.

**Log mỗi câu (JSON):** id, câu hỏi, as_of, khoảng thời gian đã giải quyết, bảng/cột được chọn, các truy vấn sinh ra, kết quả/lỗi, phiếu bầu, quyết định cuối, thời gian, seed.

**Kiểm soát thời gian:**
- Ngân sách mỗi câu (ví dụ soft 60s, hard 180s — điều chỉnh theo đo đạc).
- Đồng hồ tổng: khi còn ít hơn ngưỡng an toàn thì chuyển `fast.yaml`.
- Để đảm bảo tái lập, chọn chế độ **theo tiến độ xử lý đã định sẵn** (ví dụ số câu đã xong) thay vì theo giây thực khi có thể, và đặt dư thời gian để hầu như không bao giờ kích hoạt chế độ nhanh.

---

## 6. Phân công gợi ý (điều chỉnh theo số thành viên)

| Vai trò | Phạm vi |
|---------|---------|
| **A – Data/Ingestion** | Loader, làm sạch, profiler, Excel bẩn, dữ liệu dev bẩn |
| **B – NLU/Thời gian/Lịch âm** | `time_resolver`, `lunar`, chuẩn hóa tiếng Việt, unit test |
| **C – Agent/LLM** | Prompt, codegen, self-debug, self-consistency, router |
| **D – Eval/Verifier** | Tập dev, error taxonomy, unanswerable/abstain, hiệu chỉnh ngưỡng |
| **E – Infra/Docker** | Dockerfile, offline, hiệu năng GPU, watchdog, kiểm thử tái lập |

Nếu team nhỏ, gộp A+B và C+D, E làm xuyên suốt nhưng bắt đầu sớm (từ Giai đoạn 1).

---

## 7. Rủi ro & phương án giảm thiểu

| Rủi ro | Mức | Giảm thiểu |
|--------|-----|-----------|
| Vượt 6 giờ khi chạy lại | Cao | Ghi từng dòng; đo thời gian trên máy tương đương; dư ≥ 30%; chế độ nhanh |
| Lệch PA > 0.03 giữa hai lần chạy | Cao | temp = 0, seed, tránh phụ thuộc thời gian, đo lặp nhiều lần |
| Docker lỗi trên Blackwell | Cao | Dùng đúng base image BTC; test sớm |
| Vi phạm offline (âm thầm tải model/tokenizer) | Trung bình | Biến môi trường offline; test `--network none` |
| Vượt 15B tham số | Trung bình | Bảng model budget; kiểm tra bằng code đếm tham số |
| Overfit vào public test | Trung bình | Không tinh chỉnh theo từng câu; dùng dev riêng |
| Hard-code ngầm (vi phạm R4) | Trung bình | Code review; grep ID; chỉ logic tổng quát |
| Trả lời bừa cho unanswerable | Trung bình | Pre/post-check, self-consistency, ngưỡng hiệu chỉnh |
| Dữ liệu bẩn kiểu chưa gặp | Trung bình | Đa dạng dữ liệu dev; fallback an toàn khi parse lỗi |
| Crash giữa chừng | Trung bình | try/except từng câu, ghi `abstain`, tiếp tục |

---

## 8. Checklist trước khi nộp Docker

- [ ] Chạy `--network none` thành công, không lỗi tải.
- [ ] Tổng tham số model ≤ 15B (đã tính MoE, embedding, reranker...).
- [ ] Tất cả weights công khai trước 01/10/2026 và license phù hợp; đã khai báo trong README.
- [ ] `/data` mount read-only vẫn chạy bình thường.
- [ ] `/result/submission.jsonl` ghi từng dòng; đủ số dòng; đúng schema.
- [ ] Log đầy đủ trong thư mục theo guideline.
- [ ] Thời gian toàn bộ ≪ 6 giờ (kèm load model).
- [ ] Chạy lặp ≥ 2 lần: |ΔPA| nhỏ.
- [ ] Không có hard-code đáp án / logic theo ID.
- [ ] Không dùng dữ liệu test để train.
- [ ] README đầy đủ; tag/hash image đã ghi lại.

---
