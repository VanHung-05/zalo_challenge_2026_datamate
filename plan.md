# PLAN GIẢI BÀI TOÁN DATAMATE — Zalo AI Challenge 2026 (v2, dựa trên tài liệu chính thức)

> Nguồn: `description.md`, `data_overview.md`, `evaluation.md`, `rules.md` do BTC cung cấp.
> Ngày lập: 03/11/2026. Mốc nộp bài public theo báo chí: 23/11/2026.

## 0. Thay đổi so với bản plan trước

| Điểm | Giả định cũ | Sự thật theo tài liệu chính thức | Hệ quả |
|---|---|---|---|
| `unanswerable` | Coi như "lối thoát an toàn" | **Đây là một câu trả lời**: đúng khi và chỉ khi đáp án chuẩn là unanswerable (+1), ngược lại **−1**. Chỉ `abstain` mới luôn là 0 | Phải coi 3 hành động (trả giá trị / unanswerable / abstain) là 3 lựa chọn có kỳ vọng điểm riêng |
| Kiểu đáp án | Cần xác minh | 9 kiểu đã chốt; **sai kiểu = sai** | Phải dự đoán đúng *kiểu đáp án* (vd. "tháng nào" → `date` dạng `yyyy-mm`) |
| Phần cứng/thời gian | Chưa biết | 2×RTX 5090 (32 GB), 512 GB RAM, **6 giờ** cho toàn bộ private test (~500 câu) gồm cả load model | Ngân sách ≈ 43 giây/câu nếu chạy tuần tự; có dư địa cho code agent + bỏ phiếu |
| Model | ≤15B | Đúng; **cộng tổng mọi model** (LLM, embedding, reranker, classifier), MoE tính toàn bộ tham số, trọng số phải công bố **trước 01/10/2026** | Chọn bộ model theo "ngân sách 15B" |
| Tái lập | Chưa biết | `|PA_upload − PA_rerun| ≤ 0.03`, nếu không có thể bị loại | Cần tính tất định và quyết định abstain ổn định |
| Rulebook | Có | **Mỗi split có rulebook riêng, có thể khác nhau** — agent phải đọc rulebook của split đang chạy | Không hardcode luật; nạp rulebook làm đầu vào |
| Private test | Shop lạ | Có **loại hình kinh doanh mới** chưa xuất hiện ở train/public | Ưu tiên khả năng tổng quát hơn luật cho từng ngành |
| Đóng gói | Docker | Docker phải chứa **mã train + inference, trọng số, `predict.sh`, `README.md`**, ghi log từng câu vào `/result/logs/` | Thêm hạng mục log & README vào kế hoạch |

---

## 1. Tóm tắt bài toán

- **Đầu vào mỗi câu:** `{id, business_id, question, as_of_date}` + gói dữ liệu của business (`.csv/.xlsx/.sqlite` + `README.md`, chất lượng README từ đầy đủ → không có gì) + `rulebook.md` và `calendar.csv` dùng chung cho cả split.
- **Đầu ra:** một dòng `submission.jsonl` với `type` ∈ {`number`, `percent`, `string`, `date`, `boolean`, `list`, `table`, `unanswerable`, `abstain`}. Câu thiếu = abstain.
- **Dữ liệu:** 30 business train (~3.000 câu, có `answers.jsonl`), public test 10–20 business (~1.000 câu, ẩn đáp án), private test 10–20 business (~500 câu, có ngành mới). Mỗi business 12–30 tháng, từ vài trăm đến ~300.000 dòng, có tính mùa vụ (Tết, Trung thu, lễ, ngày lương, cuối tuần). Ngành: tạp hóa, cà phê/trà sữa, nhà hàng, nhà thuốc, vật liệu xây dựng, sửa xe máy, mẹ & bé…
- **Điểm:** PA = trung bình của {+1 đúng, 0 abstain, −1 sai}. Hòa hạng → so tổng thời gian inference khi chạy lại, rồi tới thời điểm nộp sớm hơn.

### 1.1 Luật so khớp (quyết định cách format đầu ra)

| Kiểu | Luật khớp | Việc phải làm |
|---|---|---|
| `number` | Khớp chính xác; đáp án chuẩn là số nguyên (VND, đếm, số lượng) hoặc làm tròn 2 chữ số thập phân | Số nguyên thì xuất nguyên; còn lại làm tròn đúng 2 số lẻ. Tránh sai số float (dùng `Decimal`/làm tròn cuối cùng một lần) |
| `percent` | Đơn vị là **phần trăm** (vd. `-12.45`), làm tròn 2 số lẻ, **đúng dấu** | Không xuất `0.1245`; chú ý mẫu số "so với kỳ trước" |
| `string` | NFC → lowercase → ký tự không phải chữ/số/khoảng trắng thành khoảng trắng → gộp khoảng trắng; **giữ dấu**. Đáp án chuẩn có `alias` gồm *mọi cách viết trong dữ liệu của business đó* + mã (vd. `SP0042`) | **Luôn trả giá trị lấy nguyên văn từ dữ liệu** (tên hoặc mã), không tự "làm đẹp"/dịch/đổi hoa thường theo ý mình |
| `date` | `yyyy-mm-dd`; nếu hỏi về tháng thì `yyyy-mm` | Suy luận độ phân giải thời gian từ câu hỏi ("ngày nào" vs "tháng nào") |
| `boolean` | `true/false` | — |
| `list` | **Có thứ tự**, khớp từng vị trí; thứ tự theo câu hỏi hoặc mặc định rulebook; phần tử khớp theo luật string/number | Sắp xếp đúng; ít hơn *k* ứng viên thì trả hết |
| `table` | Cột khớp theo tên đã chuẩn hóa (không khớp thì theo vị trí); **hàng so như tập hợp**; ô tên theo luật string, ô số theo luật number | Đặt tên cột theo chữ trong câu hỏi/rulebook |
| `unanswerable` | Đúng ⟺ đáp án chuẩn là unanswerable | Mọi câu trả lời cho câu unanswerable đều là sai |
| `abstain` | Luôn 0 điểm | Lựa chọn an toàn khi không chắc |

**Xếp hạng:** theo tiêu chí trong câu hỏi, nếu không nói thì theo mặc định rulebook; giảm dần; nếu các phần tử bằng nhau thì xếp theo **mã tăng dần**. Câu có hòa ở biên (hoặc rank 1 của "… nhất") bị loại khỏi tập test → không cần mất công cho các tình huống hòa nhập nhằng ở top-1.

> Lưu ý: **sai kiểu cũng là sai**. Ví dụ hỏi "tuần này bán được bao nhiêu" mà trả `string` là −1 dù số đúng.

### 1.2 Toán học của quyết định (3 hành động)

Với mỗi câu, gọi `p_v` = xác suất đáp án ứng viên tốt nhất đúng, `p_u` = xác suất đáp án chuẩn là unanswerable.

| Hành động | Kỳ vọng điểm |
|---|---|
| Trả giá trị | `2·p_v − 1` |
| Trả `unanswerable` | `2·p_u − 1` |
| `abstain` | `0` |

Chọn hành động lớn nhất; trên thực tế dùng **ngưỡng an toàn** `τ ≈ 0.65–0.75` (thay vì 0.5) vì xác suất do mô hình ước lượng thường quá lạc quan, và vì yêu cầu tái lập (mục 6.4). `τ` được tối ưu bằng cross-validation theo business.

---

## 2. Nguyên tắc thiết kế

1. **LLM không tự tính số.** LLM chỉ "hiểu và viết chương trình"; số liệu do code (pandas/SQL) tính.
2. **Tổng quát hóa là ưu tiên số 1.** Private test có ngành mới và rulebook có thể khác → không xây bộ luật riêng cho từng ngành; thay vào đó dựa vào *rulebook được nạp động* + *thư viện hàm ngữ nghĩa (semantic layer)* + LLM viết code.
3. **Mặc định của rulebook được thực thi bởi code, không phải bởi trí nhớ của LLM.** Ví dụ: "doanh thu" (trừ giảm giá cấp dòng, không trừ voucher cấp đơn, loại VAT, phí giao, đơn hủy, đơn có hàng trả) nằm trong hàm `revenue()`; LLM chỉ gọi hàm.
4. **Câu hỏi có thể ghi đè mặc định** ("tính cả đơn hủy", "theo doanh thu thay vì số lượng") → hệ thống phải nhận ra override và truyền tham số vào hàm.
5. **Abstain là công dân hạng nhất.** Mỗi tầng có thể từ chối; quyết định cuối cùng dựa trên xác suất đã hiệu chỉnh.
6. **Đo bằng group-CV theo business**, không theo câu.
7. **Tất định, có log, có watchdog thời gian**, ghi kết quả streaming.

---

## 3. Kiến trúc tổng thể

```
/data/<biz>/*.{csv,xlsx,sqlite}, README.md ──► [1] Ingestion & Business Card (cache /tmp)
/data/rulebook.md, calendar.csv ─────────────► [2] Rulebook & Calendar config
questions.jsonl (id, biz, question, as_of) ──► [3] Question Understanding
                                                  (chuẩn hóa, thời gian, thực thể, kiểu đáp án, override)
                                                         │
                         ┌───────────────────────────────┼───────────────────────────┐
                         ▼                               ▼                           ▼
                [4a] Answerability Guard      [4b] Program Synthesis          [4c] Fast paths
                (unanswerable / abstain       LLM + semantic-layer helpers    (mẫu câu phổ biến,
                 classifier)                  + sandbox + tự sửa + N mẫu      deterministic, tùy chọn)
                         └───────────────────────────────┼───────────────────────────┘
                                                         ▼
                                        [5] Aggregation & Calibrated Decision
                                           (bỏ phiếu → chọn: value / unanswerable / abstain)
                                                         ▼
                                        [6] Formatter (9 kiểu) → submission.jsonl (streaming) + /result/logs/
```

### Vì sao chọn "LLM viết code trên semantic layer" làm lõi?
- Private test có **ngành lạ** (vd. phòng khám, trường học, hợp tác xã…) → luật cứng theo ngành không phủ được; LLM thì tổng quát hơn.
- Nhưng LLM 14B tự viết toàn bộ logic nghiệp vụ dễ sai ngầm (quên loại đơn hủy, quên trừ giảm giá) → đưa các định nghĩa chuẩn của rulebook vào **thư viện hàm** để giảm bề mặt lỗi.
- Fast path deterministic cho các dạng câu phổ biến giúp **nhanh** (hòa hạng dựa trên thời gian) và **ổn định** (đáp ứng yêu cầu tái lập).

---

## 4. Chi tiết từng module

### 4.1 Ingestion & Business Card

**Loader** (có test với dữ liệu ~300k dòng):
- CSV: thử `utf-8`, `utf-8-sig`, `cp1258`; tự dò dấu phân cách.
- Excel: nhiều sheet; header lệch dòng, merged cells, dòng tổng/dòng trống; dùng engine nhanh (vd. `python-calamine`/`openpyxl read_only`) vì file lớn.
- SQLite: đọc mọi bảng/view, lấy cả schema và khóa ngoại.
- Chuẩn hóa NFC, strip khoảng trắng & ký tự vô hình; ép kiểu số/ngày với **định dạng Việt**: `1.250.000`, `1,25tr`, `250k`, ngày `dd/mm/yyyy` vs `yyyy-mm-dd` vs serial Excel. Phát hiện `dd/mm` hay `mm/dd` bằng thống kê cả cột.
- Gộp bảng phân mảnh theo tháng (cùng schema) → union; bảng đơn hàng–chi tiết dòng hàng → join theo khóa.
- Flag chất lượng dữ liệu: trùng dòng, giá trị âm, ngày tương lai so với `as_of_date`, thiếu khóa, đơn vị lẫn lộn.

**Business Card** (chạy 1 lần/business, cache `/tmp`):
1. **Profiling tất định:** tên bảng/cột, kiểu, tỉ lệ null, cardinality, min/max, 5 dòng mẫu, khoảng thời gian bao phủ của dữ liệu (cần cho Guard: câu hỏi ngoài phạm vi thời gian).
2. **Vai trò ngữ nghĩa của cột** (`date`, `qty`, `unit_price`, `line_amount`, `discount`, `voucher`, `vat`, `delivery_fee`, `cost`, `status`, `customer`, `product`, `branch`, `employee`, `due_date`, `paid`, `lot`, `expiry`, `stock`…) kết hợp: tên cột + từ điển đồng nghĩa VI, mô tả trong README (nếu có), kiểm chứng bằng giá trị (vd. `qty×price−discount ≈ amount`).
3. **LLM sinh "data dictionary"** khi README nghèo/không có; kết quả luôn được **kiểm chứng bằng ràng buộc số học** hoặc đánh dấu `low_confidence` (dùng ở Guard).
4. **Bảng thực thể chuẩn:** gom các cách viết của cùng một sản phẩm/khách hàng theo mã (khớp với cơ chế alias của BTC: mọi cách viết trong dữ liệu + mã).
5. **Các khái niệm có/không có trong dữ liệu:** ví dụ có cột giá vốn/lô nhập không, có cột hạn dùng không, có công nợ không… → **bảng "khả năng"** (capability map) phục vụ Guard.

### 4.2 Rulebook & Calendar

- Nạp `rulebook.md` **của split đang chạy** (toàn văn đưa vào prompt vì thường ngắn) và đồng thời **parse các tham số chính** thành `rules.json` để hàm semantic layer dùng: định nghĩa doanh thu, "bán chạy" (theo số lượng), tuần Thứ Hai–Chủ Nhật, "tháng này" = tháng chứa `as_of_date`, "tồn kho" = số lượng đang có, giá vốn mặc định = bình quân gia quyền theo lô, cửa sổ Tết = 23 tháng Chạp → mùng 10 (âm lịch), tie-break, cách xếp hạng…
- Nếu parse không chắc chắn một tham số → đánh dấu rủi ro, câu nào phụ thuộc tham số đó bị trừ điểm tự tin (không tự đoán).
- `calendar.csv`: ánh xạ âm–dương để giải quyết Tết/rằm/mùng… Kiểm tra bằng test với các mốc đã biết.
- Test cấu hình: chạy thử với **cả rulebook train lẫn rulebook public test** để chắc chắn không phụ thuộc bản train.

### 4.3 Question Understanding

1. **Chuẩn hóa văn bản:** NFC, teencode/viết tắt (`ko`, `dc`, `sp`, `kh`, `hnay`, `tuần r`), gõ không dấu, Telex lỗi.
2. **Bộ phân giải thời gian tất định (quan trọng nhất cho độ chính xác):** từ `as_of_date` + rulebook + calendar sinh ra khoảng `[start, end]` rõ ràng cho: hôm nay/qua/kia, tuần này/rồi, tháng này/trước, quý, năm nay/ngoái, **cùng kỳ năm ngoái**, X ngày/tuần/tháng qua, đầu/cuối tháng, ngày lương, dịp Tết (23 Chạp–mùng 10), Trung thu, lễ… Truyền kết quả đã tính sẵn vào prompt, **không để LLM tự tính ngày**.
3. **Dự đoán kiểu đáp án** (`answer_type`) và độ phân giải ngày (`yyyy-mm` vs `yyyy-mm-dd`), số lượng phần tử k của list ("top 3", "5 mặt hàng"), thứ tự yêu cầu.
4. **Entity linking:** map tên trong câu hỏi → thực thể chuẩn (exact → token-set/`rapidfuzz` → n-gram TF-IDF → embedding nhỏ tùy chọn). Nếu top-1 và top-2 sát nhau → hạ độ tin cậy/abstain. Nếu thực thể **không tồn tại** trong dữ liệu → tín hiệu cho Guard.
5. **Phát hiện override** của mặc định rulebook ("kể cả đơn hủy", "tính theo doanh thu", "chưa trừ giảm giá").
6. **Phát hiện câu mơ hồ** (thiếu mốc thời gian, thuật ngữ không định nghĩa).

### 4.4 Semantic Layer (thư viện hàm mà LLM được phép gọi)

Hàm thuần, có unit test, đọc tham số từ `rules.json` + Business Card. Danh sách khởi đầu (mở rộng theo lỗi trên train):

```python
valid_lines(df, overrides)         # lọc đơn hợp lệ theo rulebook (bỏ hủy, đơn có hàng trả…)
revenue(lines, period, by=None)    # sau giảm giá dòng, trước voucher đơn, bỏ VAT & phí giao
quantity(lines, period, by=None)
orders_count(lines, period)
avg_order_value(lines, period)
gross_profit(lines, period, cost_method="wavg_lot")
period_compare(metric, p1, p2)     # → percent đúng dấu & đúng mẫu số
rank(metric, by, k, order, tie="code")   # tie-break đúng luật BTC
share_of_total(...)
stock_on_hand(asof, by=None); slow_movers(...); expiring(asof, days)
receivables(asof, overdue_only=False)    # công nợ / quá hạn
resolve_entity(name, kind)         # trả giá trị nguyên văn trong dữ liệu
```

- **Quy tắc vàng:** hàm trả về giá trị nguyên văn của dữ liệu (tên hoặc mã) cho kiểu `string`/`list`.
- Hàm nào không áp dụng được (thiếu cột) phải `raise CapabilityMissing("giá vốn")` → tín hiệu **unanswerable có lý do**, thay vì LLM chế dữ liệu.
- LLM vẫn có thể viết pandas thuần cho các nghiệp vụ chưa có hàm (đây là cơ chế tổng quát hóa sang ngành lạ).

### 4.5 Program Synthesis Agent (LLM)

**Prompt gồm:** Business Card (schema + vai trò + mẫu dòng), phần README liên quan, **toàn văn rulebook**, danh sách hàm semantic layer + ví dụ, câu hỏi chuẩn hóa, `as_of_date`, khoảng thời gian đã giải sẵn, thực thể đã link, `answer_type` dự đoán.

**Vòng lặp (tối đa ~4–6 lượt):**
1. LLM sinh code → chạy trong **sandbox** (tiến trình con, timeout, giới hạn RAM, whitelist thư viện, chặn mạng/`os`/`subprocess`; thư mục data read-only; file tạm ở `/tmp`).
2. Nếu lỗi/kết quả rỗng/vô lý → trả traceback + quan sát cho LLM sửa.
3. Cho phép bước **khám phá** (xem giá trị distinct của cột trạng thái, kiểm tra tên thực thể) trước khi chốt truy vấn.
4. Kết quả cuối phải ở dạng có cấu trúc `{type, value, evidence}` (evidence: bảng/cột/bộ lọc đã dùng; hữu ích cho log và Guard).

**Self-consistency:** sinh `N` lời giải độc lập (4–8) với các "góc nhìn" khác nhau (pandas/SQL/viết lại theo hàm semantic layer/tính ngược để đối chiếu). Đưa mọi đáp án qua **cùng bộ chuẩn hóa của BTC** để so sánh đồng nhất.

**Kiểm tra hợp lý tự động:** kiểu đúng `answer_type`; không NaN/inf; phần trăm trong biên hợp lý; số lượng ≥ 0; tổng thành phần ≈ tổng; list đúng độ dài k; đáp án `string` có nằm trong tập giá trị của dữ liệu không.

### 4.6 Answerability Guard (nguồn điểm lớn và cũng là nguồn mất điểm lớn)

Hai loại tín hiệu:
- **Capability map (tất định):** câu hỏi cần khái niệm X (giá vốn, lợi nhuận ròng, hạn dùng, tồn kho, công nợ, chi phí lương, đánh giá hài lòng, …) mà Business Card xác nhận **không có cột/bảng tương ứng** → ứng viên `unanswerable`. Ví dụ chính thức: *"Dữ liệu không có giá vốn nên không tính được lợi nhuận"*.
- **Phạm vi dữ liệu:** khoảng thời gian hỏi nằm ngoài vùng dữ liệu bao phủ; thực thể hỏi không tồn tại; hỏi dự báo/ý kiến chủ quan.

Phân biệt rõ:
| Tình huống | Hành động |
|---|---|
| Chắc chắn dữ liệu thiếu (nhiều tín hiệu cùng chỉ) | `unanswerable` |
| Không chắc đây là thiếu dữ liệu hay hệ thống chưa hiểu | **`abstain`** (an toàn hơn) |
| Câu hỏi mơ hồ nhưng có thể trả lời theo mặc định rulebook | Trả lời theo mặc định rulebook |

Classifier Guard huấn luyện trên 3.000 câu train (nhãn `answer: null` là unanswerable), đặc trưng gồm capability map, tín hiệu thực thể/thời gian, và đầu ra của LLM; **đánh giá riêng precision của nhãn unanswerable**, vì sai là −1.

### 4.7 Aggregation & Calibrated Decision

- Với mỗi câu, gom candidate `{answer, type, source, exec_ok, checks}`; tính **tỉ lệ đồng thuận** (theo luật so khớp BTC) giữa các lời giải + tín hiệu từ Guard + cờ rủi ro (entity mơ hồ, tham số rulebook không chắc, dữ liệu kém chất lượng, README nghèo…).
- **Calibrator** (vd. gradient boosting/logistic) học `P(đúng)` từ các đặc trưng trên bằng **group-CV theo business**; xuất `p_v`, `p_u`.
- Quyết định theo mục 1.2 với `τ` đã tối ưu theo PA.
- **Veto cứng → abstain** khi: các lời giải bất đồng; thực thể link sát nhau; câu chạm tới luật chưa xác nhận được; code đụng cột có vai trò độ tin cậy thấp; sandbox timeout.

### 4.8 Formatter & ghi kết quả

- Xuất đúng schema từng kiểu (mục 1.1); `percent` là số phần trăm; `date` đúng độ phân giải; `string` lấy nguyên văn từ dữ liệu; `table` có `columns` + `rows`.
- **Ghi streaming** từng dòng vào `/result/submission.jsonl` ngay khi xong (Rule 8: nếu hết giờ/crash chỉ phần đã ghi được chấm). Hỗ trợ **resume**.
- **Log** `/result/logs/<id>.json` tối thiểu: `id`, đáp án cuối, thời gian; cộng thêm intent, các candidate, code đã chạy, output, lý do abstain/unanswerable (để BTC đối chiếu khi chênh lệch PA).
- Hỗ trợ `reason` tùy chọn cho `unanswerable`.

---

## 6. Ràng buộc kỹ thuật & vận hành

### 6.1 Ngân sách model (≤ 15B tổng, công bố trong README, trọng số phát hành trước 01/10/2026, giấy phép cho phép)
- **Tổng tham số của mọi model dùng khi inference** tính cộng dồn, kể cả không nạp đồng thời. Ví dụ phương án: 1 LLM ~14B + 1 embedding nhỏ (<0.5B), hoặc 1 LLM ~8B + 1 LLM code ~4–7B.
- Ứng viên cần kiểm tra thực tế (giấy phép, ngày phát hành, khả năng tiếng Việt, khả năng viết code): các dòng Qwen (instruct/coder) cỡ 7–14B, và các model mở khác ≤15B phát hành trước 01/10/2026. **Cần benchmark nội bộ trên train** trước khi chọn; quantization (AWQ/GPTQ/FP8) được phép.
- Fine-tune bằng LoRA trên 2×5090 (hoặc máy tương đương) rồi merge.

### 6.2 Phần cứng chấm: 2×RTX 5090 (32 GB), 512 GB RAM, 6 giờ
- Blackwell cần **CUDA ≥ 12.8** và bản PyTorch/vLLM tương thích; dùng base image theo **Docker guideline riêng của BTC**.
- Phương án: **2 replica vLLM** (mỗi GPU một bản, model 8–14B lượng tử hóa) để tăng thông lượng, hoặc tensor-parallel 2 GPU nếu model lớn hơn. Cần đo.
- **Ngân sách thời gian:** private ~500 câu. Mục tiêu tổng ≤ **2–2.5 giờ** (chừa dư ≥ 60% so với 6 giờ). Cụ thể: load model ≤ 10 phút; mỗi business build catalog 1 lần; xử lý các câu theo nhóm business để tái dùng cache; chạy song song nhiều câu bằng asyncio.
- **Watchdog:** timeout mỗi câu (vd. 120–180 s) → abstain; timeout toàn cục (vd. 5h) → ngừng và các câu còn lại là abstain. Tối ưu thời gian cũng là tie-breaker xếp hạng.

### 6.3 Docker & offline
- Image chứa: mã nguồn **train + inference**, trọng số model, `predict.sh`, `README.md` (khai báo model), thư viện cài sẵn (không `pip install` lúc chạy).
- `predict.sh` đọc `/data` (read-only), ghi `/result/submission.jsonl` và `/result/logs/`.
- Chạy kiểm thử với `--network none`; đặt `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; cache/tạm ra `/tmp`, **không ghi vào `/data`**.
- Không có logic gắn với `question_id`/`business_id`, không có đáp án hardcode (Rule 4) → kiểm tra bằng grep + review code trước khi nộp.

### 6.4 Tái lập (|ΔPA| ≤ 0.03) — dễ bị bỏ quên
Với ~500 câu, ΔPA 0.03 tương đương lệch ~15 điểm; sự bất định của vLLM (batch, GPU khác) có thể làm vài câu đổi quyết định.
- Seed cố định, temperature thấp/0 cho lời giải chính; với lời giải lấy mẫu dùng seed theo từng request.
- Ưu tiên **đường tất định** (fast path + semantic layer); LLM chỉ chạm vào phần cần thiết.
- Quyết định abstain dùng **biên an toàn** (không đặt ngưỡng sát vùng dày đặc điểm) để thay đổi nhỏ không lật quyết định hàng loạt.
- **Kiểm tra tái lập nội bộ:** chạy image 2 lần trên public test (có thể trên cấu hình GPU khác nhau); yêu cầu |ΔPA| ≪ 0.03 (mục tiêu ≤ 0.01) và đo số câu đổi kết quả.
- Log đủ để giải trình mọi chênh lệch.

---

## 7. Đánh giá & vòng lặp cải tiến

- **Scorer cục bộ** cài đúng luật BTC (chuẩn hóa string, alias, làm tròn, so list theo vị trí, so table theo tập hàng, quy tắc unanswerable/abstain) + PA; breakdown theo `answer_type`, intent, business, ngành, nguồn lời giải.
- **Quy trình mỗi vòng:** chạy dev → đọc `errors.jsonl` (kèm plan/code/log) → phân loại lỗi (hiểu câu hỏi / thời gian / entity / luật nghiệp vụ / format / Guard / abstain-threshold) → sửa nguyên nhân gốc → thêm test hồi quy → chạy holdout *một con số*.
- Nếu lỡ nhìn lỗi của business holdout → chuyển business đó sang dev.
- **Public leaderboard** chỉ để tham chiếu; theo dõi chênh lệch giữa PA holdout cục bộ và PA public để biết mức tổng quát hóa.

**Chỉ số theo dõi:** PA (holdout theo business, theo ngành); wrong-rate trên câu đã trả lời; coverage; precision của `unanswerable`; thời gian/câu; số câu đổi kết quả giữa 2 lần chạy.

---

## 8. Lộ trình (03/11 → 23/11 và giai đoạn private)

> Mốc nộp public theo báo chí là 23/11; **ngày đóng băng Docker và lịch private test cần xác nhận** trên trang cuộc thi/Docker guideline (`[CẦN XÁC MINH]`). Plan dưới đây giả định phải có image chạy được trước 20/11.

| Giai đoạn | Ngày | Việc chính | Đầu ra |
|---|---|---|---|
| **P0 – Khảo sát dữ liệu** | 03–05/11 | Tải train/public; thống kê: số business/ngành, phân bố `type` đáp án, tỉ lệ `answer=null`, độ dài rulebook, mức độ README, kích thước file; đọc 15–20 cặp (câu hỏi, đáp án) theo từng nhóm; so sánh rulebook train vs public | `data_notes.md`, danh sách intent thực tế |
| **P1 – Nền tảng** | 04–09/11 | Loader + profiler + Business Card; scorer cục bộ; split dev/holdout (theo business & ngành); abstain-all baseline nộp thử; khung Docker offline chạy được | Pipeline end-to-end + PA baseline |
| **P2 – Semantic layer & thời gian** | 07–13/11 | Rulebook parser, calendar, time resolver, entity linking, các hàm metric hàng đầu theo phân bố intent | PA dev tăng, wrong-rate thấp trên intent đã phủ |
| **P3 – LLM agent** | 10–17/11 | vLLM (2 replica), prompt, sandbox, vòng sửa lỗi, self-consistency; chọn model ≤15B bằng benchmark | Coverage tăng không kéo wrong-rate |
| **P4 – Self-training & Guard** | 14–19/11 | Rejection sampling → LoRA SFT; classifier Guard; calibrator; quét ngưỡng `τ` | PA holdout tăng; precision unanswerable cao |
| **P5 – Đóng gói & chống rủi ro** | 18–22/11 | Docker chuẩn BTC, test `--network none`, đo thời gian/VRAM trên cấu hình tương đương, kiểm tra tái lập 2 lần, README khai báo model, rà soát Rule 4 | Image sẵn sàng nộp |
| **Buffer** | 22–23/11 | Nộp public cuối, dự phòng sự cố | — |
| **Private phase** | theo BTC | Chạy bản đã đóng băng trên private set, upload `submission.jsonl` | — |

**Nguyên tắc ưu tiên thời gian:** ưu tiên các intent chiếm tỉ trọng lớn nhất trên train; đạt độ chính xác rất cao rồi mới mở rộng sang đuôi dài.

---

## 9. Bẫy thường gặp & cách phòng

| Bẫy | Hệ quả | Phòng tránh |
|---|---|---|
| Sai kiểu đáp án (`number` vs `string`, `date` ngày vs tháng) | −1 dù giá trị đúng | Dự đoán `answer_type` riêng, kiểm thử theo kiểu |
| `percent` xuất dạng 0.12 thay vì 12.00; sai dấu; sai mẫu số | Sai | Hàm `period_compare` chuẩn; test dấu |
| Làm tròn/float | Lệch 0.01 | Làm tròn một lần cuối bằng `Decimal`; số nguyên giữ nguyên |
| `string` không nguyên văn dữ liệu | Sai | Chỉ trả giá trị có trong dữ liệu (tên hoặc mã) |
| Thứ tự list, tie-break theo mã tăng dần | Sai vị trí | `rank()` tuân thủ luật BTC |
| "Tuần rồi / cùng kỳ năm ngoái / Tết" lệch quy ước | Sai hàng loạt câu thời gian | Resolver tất định + rulebook + calendar; test dày |
| Quên loại đơn hủy / đơn có hàng trả / VAT / phí giao / dùng voucher cấp đơn | Sai doanh thu | Hàm `valid_lines`/`revenue` theo rulebook; test từng cấu phần |
| Câu hỏi override mặc định | Sai | Detector override + tham số hàm |
| Trả `unanswerable` khi chỉ là hệ thống chưa hiểu | −1 | Dùng `abstain` khi không chắc; tối ưu precision |
| Bỏ sót câu unanswerable (trả đại một giá trị) | −1 | Capability map + classifier Guard |
| Rulebook split khác train | Sai nặng ở public/private | Nạp động rulebook từng split; test bằng nhiều rulebook |
| Ngành lạ ở private | Hụt điểm | Hold-out theo ngành; semantic layer + LLM tổng quát |
| README nghèo / không có | Hiểu sai schema | Data dictionary tự sinh + kiểm chứng số học; hạ confidence |
| File 300k dòng / Excel nặng | Quá thời gian | Engine đọc nhanh, cache, lọc sớm, giới hạn sandbox |
| Vượt 6 giờ | Phần còn lại tính abstain | Watchdog + ghi streaming + mục tiêu ≤2.5h |
| Chênh PA upload vs re-run > 0.03 | Bị loại | Mục 6.4 |
| Hardcode theo id/biz hoặc dùng dữ liệu test | Bị loại | Review code; tách rạch ròi train/test |
| Dùng mạng / ghi vào `/data` | Vi phạm | Chạy `--network none`, cache ở `/tmp` |

---

## 10. Cấu trúc repo đề xuất

```
datamate/
├── ingest/        # loader, profiler, business_card, entity_index
├── knowledge/     # rulebook_parser, calendar, capability_map
├── understand/    # normalize, time_resolver, entity_link, answer_type, override_detector
├── semantic/      # metrics (revenue, quantity, rank, compare, stock, receivables…)
├── agent/         # prompts, llm_client (vLLM), synthesis loop, sandbox, self_consistency
├── decision/      # guard, calibrator, voter, thresholds
├── output/        # formatter (9 kiểu), writer (streaming, resume), logs
├── offline/       # scorer (luật BTC), eval (PA+breakdown), cv splits, error analysis
├── train/         # rejection_sampling, sft_lora, guard_train, calibrator_train
├── runtime/       # orchestrator (asyncio), watchdog, CLI
configs/  docker/ (Dockerfile, predict.sh)  tests/  README.md
```

---

## 11. Checklist 48 giờ đầu

- [ ] Tải đầy đủ train + public + rulebook + calendar + **Docker guideline** (lấy lịch private và base image).
- [ ] Thống kê train: phân bố `type`, tỉ lệ `answer=null`, intent chính, số câu/business, ngành, kích thước file lớn nhất.
- [ ] So sánh `rulebook.md` train vs public: những điểm khác nhau.
- [ ] Cài scorer cục bộ đúng luật BTC và kiểm tra bằng chính `answers.jsonl` (PA = 1.0 khi nộp đáp án chuẩn).
- [ ] Chia dev/holdout theo business và theo ngành.
- [ ] Nộp thử file abstain-all (kiểm tra quy trình nộp) và dựng khung Docker offline.
- [ ] Benchmark nhanh 2–3 model ≤15B (tiếng Việt + sinh code) trên một lát dev; đo tốc độ vLLM trên máy có GPU tương đương.
- [ ] Chốt phân công (data/ingest, understand+semantic, LLM+sandbox, eval+Docker+train).

---

## 12. Các điểm còn cần xác nhận

1. Lịch chính xác: hạn Docker, thời điểm phát private test, số lượt nộp/ngày (xem "general regulations" & Docker guideline).
2. Định dạng chuỗi cho một số đáp án đặc thù (vd. tên thứ trong tuần: "Thứ Hai"/"Monday"?) → xem trong `answers.jsonl` train.
3. Quy tắc làm tròn chính xác (half-up hay banker's) → kiểm bằng train.
4. Có được dùng model/dịch vụ thương mại để **sinh dữ liệu huấn luyện** hay không (Rule 3 chỉ cấm ở inference; nên hỏi BTC nếu muốn dùng).
5. Cách BTC đếm "tổng tham số" khi dùng nhiều model/embedding (README phải khai báo rõ).