# Báo cáo tiến độ DataMate — 28/10/2026

> Phạm vi: review source code (commit `87eae99` + file chưa commit) đối chiếu với `plan.md` và `docs/` (description, data_overview, evaluation, rules).
> Phương pháp: đọc toàn bộ `src/`, `scripts/`, `tests/`, `configs/`, `predict.sh`; chạy unit test; chạy script chẩn đoán **không cần LLM** trên `train/` (3.000 câu) và nạp dữ liệu toàn bộ 30 business train + 14 business public_test.

## 0. Tóm tắt

| Hạng mục | Trạng thái |
|---|---|
| Giai đoạn 0 – Nắm đề | ~70%: docs đã tóm tắt; câu hỏi gửi BTC chưa có câu trả lời; chưa có phân công/CI |
| Giai đoạn 1 – Baseline end-to-end | ~75% code, **chưa có điểm baseline**; chưa chạy Docker |
| Giai đoạn 2 – Deterministic | ~45%: time resolver, loader, entity linker có bản đầu, còn nhiều lỗ hổng |
| Giai đoạn 3 – Tri thức nghiệp vụ | ~15%: rulebook **không được đọc** (hard-code), chưa phân loại câu hỏi/few-shot |
| Giai đoạn 4 – Unanswerable/Abstain | ~5% |
| Giai đoạn 5 – Mô hình & hiệu năng | ~10%: mới chọn model trên giấy |
| Giai đoạn 6 – Docker & tái lập | 0% |
| Giai đoạn 7 – Nộp cuối | 0% |

**Chưa có bằng chứng về điểm số:** trong repo không có file submission/kết quả đánh giá nào, và lúc review máy không chạy Ollama, nên **PA hiện tại chưa được đo**. Việc cần làm đầu tiên là có một con số PA baseline trên train.

**Ba lỗi nghiêm trọng nhất (đã kiểm chứng bằng cách chạy code):**
1. **Rulebook của split không được dùng:** `get_rulebook()` tìm file ở thư mục business nên đọc ra nội dung rỗng (độ dài 0). Prompt lấy một bản luật **hard-code chép từ rulebook public_test**, đi ngược yêu cầu "đọc và áp dụng rulebook của split đang trả lời" (data_overview.md).
2. **Kết quả dạng DataFrame/Series/ndarray luôn bị abstain:** `agent.py:129` so sánh `final_result == "UNANSWERABLE"`, phép so sánh này ném `ValueError` với các kiểu đó; runner bắt lỗi và ghi abstain. Các câu list (193) và table (19) trên train gần như mất trắng nếu LLM trả về kiểu pandas.
3. **Sandbox không thực sự giới hạn thời gian và không cô lập dữ liệu:** đặt timeout 1s nhưng 3,8s sau mới trả về (khối `with ThreadPoolExecutor` chờ thread chạy xong). Code sinh ra có thể sửa `data` **dùng chung giữa các câu**: `inplace=True` làm mất dòng ở các câu sau, gây sai số và rủi ro không tái lập (Rule 8).

---

## 1. Hiện trạng theo module

| Module | Đã làm | Vấn đề đã kiểm chứng |
|---|---|---|
| `loader.py` | Đọc CSV (dò encoding/sep), XLSX nhiều sheet, dò header 10 dòng đầu, bỏ dòng "Tổng", SQLite; chuẩn hoá ngày VN, số VN; gộp file bán hàng theo tháng thành `BanHang` | Nạp đủ 44/44 business, không lỗi. **Chậm:** train 642s, public 204s (business chậm nhất ~60s). `date_range` bị nhiễu (`0001-01-01`, `1965-…` do lẫn cột ngày sinh/hạn dùng; `biz_107` = None). Nhận diện cột ngày/số dựa vào từ khoá tên cột (`"no"`, `"ky"`, `"tra"`… dễ khớp nhầm). Gộp `BanHang` chỉ theo tiền tố tên file (`BanHang`, `So_chi_tiet_ban_hang`…), trùng với tên file trong public_test. Chưa xử lý: merged cell, header nhiều tầng, ô trống "kế thừa dòng đầu nhóm" (rulebook §8), dòng trùng trong cùng đơn |
| `profiler.py` | `SchemaCard`, `ValueIndex` (RapidFuzz) | `SchemaCard`/`profile_business` **không được dùng** trong pipeline (prompt tự dựng schema riêng) |
| `time_resolver.py` | Hôm nay/qua/kia, N ngày gần nhất, tuần này/trước, tháng này/trước, tháng M/Y, quý, năm, Tết Y (tra `calendar.csv`) | Chỉ trả **một** khoảng, nên câu so sánh kỳ (cùng kỳ, so tháng trước, Tết năm nay so năm ngoái) bị thiếu kỳ so. Có **gợi ý sai được đưa vào prompt kèm lệnh "Hãy lọc đúng khoảng này"**, ví dụ: "Tết năm nay so với Tết năm ngoái" → `2026-01-01..as_of` ("Năm nay"); "từ đầu tháng này đến hết hôm nay" → chỉ ngày hôm nay. Chưa có: tháng/quý không ghi năm (rulebook §2), "trước Tết", ca sáng/chiều, "năm ngoái", luật "tháng này so tháng trước cùng số ngày", từ viết tắt/sai chính tả ("thangs 5/2026" → None). Vì vậy độ phủ đo được 91% chỉ là tỉ lệ *có trả về khoảng*, **không phải độ đúng** |
| `entity_linker.py` | Trích n-gram, cụm viết hoa, fuzzy `token_sort_ratio` ≥ 75 | Chưa đo precision/recall; n-gram 2–4 từ đem so khớp với toàn bộ index có thể chậm với business lớn |
| `rules.py` | Parse rulebook theo `##` | Đường dẫn sai (xem mục 0); `get_core_rules_prompt()` trả chuỗi hard-code; cache global không reset khi đổi split |
| `code_generator.py` | Prompt có rulebook + schema + README (cắt 800 ký tự) + thời gian + entity + vòng self-repair | Schema chỉ hiển thị **tối đa 6 bảng**, trong khi nhiều business có 17–28 bảng (biz_119: 28, dev_004: 24), nên LLM không thấy phần lớn dữ liệu. Prompt khai báo `start_date`/`end_date` như biến nhưng sandbox **không inject** (thử thì gặp `NameError: start_date`) |
| `sandbox.py` | Kiểm tra AST cấm import/hàm nguy hiểm, timeout bằng thread | Timeout không hiệu lực; dữ liệu bị mutate (xem mục 0). Lách được: `pd.io.common.os.getcwd()` chạy thành công. Không giới hạn bộ nhớ |
| `llm.py` | Ollama (dev) / vLLM (prod), temperature 0, seed | vLLM gọi `generate()` với **prompt thô, không áp chat template** của bản Instruct. `vllm` chưa có trong `requirements.txt`. Stop token `"```\n\n"` có thể cắt code giữa chừng |
| `agent.py` | Vòng generate → execute → repair (≤ `max_retries`) → format | `infer_expected_type` dùng từ khoá, đúng **87,4%** (2561/2930 câu có đáp án): boolean 2/23, string 19/88, list 54/193, table 0/19, date 92/145. Sai type là bị chấm sai. Mọi giá trị trả về đều được nộp, **không có verifier** |
| `formatter.py` | Đủ 9 type theo §9 | `format_percent` tự nhân 100 khi \|x\| < 1, nên `0.5` (0,5%) thành `50.0`. Tiền không được ép thành số nguyên (`format_number(3.3333)` cho `3.33`) |
| `runner.py` | Ghi JSONL từng dòng + flush, cache business, lỗi thì abstain | **Không ghi log từng câu vào `/result/logs/`** (bắt buộc theo data_overview.md, phục vụ giải trình Rule 8). Không có watchdog 6 giờ, không chạy song song. `parallel_workers`, `n_samples`, `confidence_threshold` trong config **không được code dùng** |
| `scripts/evaluate.py` | Chấm PA, thống kê theo type/business | `match_percent` có dung sai 0,015, lỏng hơn luật "khớp sau làm tròn 2 chữ số". Bảng so cột theo vị trí, chưa so theo tên |
| `tests/` | 19 test (time, formatter, entity) | **19/19 pass** (trong venv có đủ `requirements.txt`). Chưa có test cho loader, sandbox, agent |
| `src/main.py` | — | File rỗng, chưa commit |
| Docker | — | Chưa có `Dockerfile`, chưa test `--network none` |

## 2. Đối chiếu tuân thủ luật

| Luật | Hiện trạng | Đánh giá |
|---|---|---|
| R2 – ≤ 15B tham số | Dự kiến Qwen2.5-14B (~14,7B) | Đạt nếu chỉ dùng một model; **không còn chỗ** cho embedding/reranker. Cần ghi bảng model budget vào README |
| R3 – Offline | Chưa có Docker, chưa đặt `HF_HUB_OFFLINE` | Chưa kiểm chứng |
| R4 – Không hard-code theo ID | grep `src/` không thấy business ID | Đạt ở mức grep. Tuy nhiên luật nghiệp vụ và heuristic tên file đang chép từ **public_test**; nên chuyển sang đọc rulebook động |
| R5 – `/data` read-only | Code không ghi vào thư mục dữ liệu | Đạt về thiết kế, chưa test mount `:ro` |
| R6 – Không dùng test để tinh chỉnh | Luật trong `rules.py` lấy từ rulebook public_test (rulebook là tài liệu đề cung cấp, không phải đáp án) | Rủi ro thấp, nhưng nên lấy động từ split để vừa tránh tranh cãi vừa đúng khi rulebook private khác |
| R7 – 6 giờ | Chưa đo thời gian/câu. Riêng nạp dữ liệu đã mất 3–10 phút mỗi split | Chưa kiểm chứng |
| R8 – Tái lập ≤ 0,03 | temperature 0 + seed; nhưng dữ liệu dùng chung bị mutate và thread timeout có thể chạy song song với câu sau | **Rủi ro**, cần sửa sandbox và chạy lặp để đo |

## 3. Dữ liệu tham chiếu (train)

Phân bố type đáp án trên 3.000 câu: number 1884 · percent 578 · list 193 · date 145 · string 88 · unanswerable 70 · boolean 23 · table 19.
Number + percent chiếm 82%, nên độ đúng của luật doanh thu/thời gian quyết định phần lớn điểm. Unanswerable chỉ 2,3%; vì sai bị trừ 1 điểm, abstain khi không chắc thường lợi hơn đoán.

## 4. Đề xuất việc tiếp theo (theo thứ tự ưu tiên)

1. **Đo baseline:** chạy `runner.py` trên một mẫu train cố định (ví dụ 300 câu trải đều các business), chấm bằng `evaluate.py`, lưu kết quả vào `reports/`. Chưa có con số này thì chưa thể ưu tiên việc khác một cách có căn cứ.
2. **Sửa 3 lỗi nghiêm trọng ở mục 0:**
   - Đọc `rulebook.md` từ thư mục split (truyền `data_dir` của split vào), đưa các mục liên quan vào prompt thay cho bản hard-code.
   - Kiểm tra `isinstance(result, str)` trước khi so với `"UNANSWERABLE"`.
   - Chạy sandbox trong **process riêng** có thể kill (multiprocessing/subprocess), truyền bản sao dữ liệu hoặc khoá ghi.
3. **Sửa prompt/sandbox:** inject `start_date`/`end_date`; hiển thị đủ bảng (hoặc chọn bảng theo mức liên quan thay vì cắt ở 6); áp chat template cho vLLM.
4. **Time resolver:** trả về nhiều khoảng (kỳ hỏi + kỳ so); thêm tháng/quý không ghi năm, năm ngoái, trước Tết, ca sáng/chiều, các cách viết tắt. Khi không chắc thì **không đưa gợi ý** vào prompt, vì gợi ý sai còn hại hơn không có gợi ý. Đo **độ đúng** trên một tập nhỏ có nhãn tay.
5. **Type inference:** để LLM xuất type cùng kết quả, hoặc dùng bộ phân loại tốt hơn; sửa nhóm boolean/string/list/table trước.
6. **Unanswerable/abstain (Giai đoạn 4):** dùng `date_range` (sau khi lọc cột ngày giao dịch) để bắt kỳ nằm ngoài dữ liệu; thêm self-consistency (`n_samples`) làm tín hiệu tin cậy để abstain.
7. **Log `/result/logs/` từng câu** (id, đáp án, thời gian, code, output) và **Dockerfile** chạy `--network none`. Nên làm sớm theo đúng plan Giai đoạn 1.
8. **Hiệu năng loader:** cache kết quả nạp (parquet trong `/tmp`) và tối ưu các `apply` theo dòng.

## 5. Hạn chế của báo cáo này

- Chưa chạy được LLM nên **không có PA thực**. Mọi kết luận về ảnh hưởng tới điểm đều là suy luận từ code và từ chẩn đoán không cần LLM.
- Độ chính xác của loader (header đúng hay chưa, số/ngày parse đúng hay chưa) chưa được kiểm tra theo từng file; mới xác nhận là nạp không lỗi và đếm số dòng.
- Các câu hỏi gửi BTC trong `plan.md` mục 1 vẫn chưa có câu trả lời ghi lại.
