# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Thạch Minh Quân · **MSSV:** 2A202601585
**Khóa học:** AICB-K34 · Track 3: GraphRAG

> Đây là bản tổng hợp. Chi tiết đầy đủ ở ba file chuyên đề:
> [`technical_defense.md`](technical_defense.md) · [`failure_analysis.md`](failure_analysis.md) · [`reflection_ThachMinhQuan.md`](reflection_ThachMinhQuan.md)

---

## 📊 Pipeline đã chạy

| Bước | Kết quả |
|---|---|
| Stream dataset HackerNoon | 137 577 dòng / 80 MB |
| Lọc thân bài ≥ 80 ký tự | → 63 717 (73 860 dòng hỏng sẵn trong dataset gốc) |
| Exact dedup SHA-1 | → 57 085 |
| Curation giàu quan hệ | 1 936 bài (3.4%) → lấy 1 500 |
| Near-dedup MinHash/LSH @0.75 | → 1 495 bài |
| Chunking 220 từ / overlap 40 | **1 501 chunk** |
| Coreference (200 chunk) | 98 chunk được áp phép thế |
| NER + RE (200 chunk) | **306 triple**, 0 vi phạm schema |
| Entity Resolution | 288 mention → 281 canonical, audit 12 dòng |
| Neo4j (UNWIND batch 1000) | **286 node / 305 cạnh**, `invalid_provenance_edges = 0` ✅ |

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution

**Thay đổi kiến trúc:** thay vì bắt LLM in lại toàn văn `resolved_text` (~1 500 output token/request chỉ để chép lại), tôi cho LLM trả về danh sách `{mention → antecedent}` rồi Python tự áp. Giảm ~10× output token và **an toàn ngữ nghĩa hơn** — LLM không còn cơ hội sửa lén số liệu/ngày tháng ở phần văn bản không liên quan.

**Ca sai — `chunk_id = cf095df30bfa3f72399f::c0000`** (similarity 0.461, thấp nhất trong 98 chunk bị sửa):

> *"Samsung unveils OLED display with embedded heart rate sensor. **It** is a separate module attached under the display panel."*
> → *"**OLED display with embedded heart rate sensor** is a separate module attached under the display panel."*

`It` chỉ **riêng cảm biến**, nhưng model chọn cụm danh từ ngoài cùng. Câu trở nên vô nghĩa ("màn hình OLED là module gắn dưới tấm nền màn hình") và sinh ra cạnh sai.

**Điểm nguy hiểm nhất:** cạnh sai đó vẫn có đủ `source_chunk_id`, `published_date`, `evidence` nên **vượt qua 100% kiểm tra provenance**. Kiểm tra cấu trúc không bao giờ bắt được lỗi ngữ nghĩa.

### 2. Entity Resolution Threshold & Lexical Guard

**Ngưỡng:** cosine gộp **0.90**, ngưỡng ghi audit **0.80** (tách riêng — nếu chỉ log từ 0.90 thì dải 0.80–0.90 biến mất và không thể trả lời chính câu hỏi này).

**Cặp cosine > 0.85 bị Guard chặn:**

| Left | Right | Cosine | Decision |
|---|---|---|---|
| `Azure` | `Microsoft Azure` | **0.961** | `REJECT_GUARD` / `GUARD_TOKEN_SUBSET` |

`{azure}` là tập con thật sự của `{microsoft, azure}` → Guard chặn. Guard này sinh ra để chặn `Apple` ⊂ `Apple Watch`.

**Nhưng ở ca này Guard đã SAI** — hai cái đó cùng một thực thể. Guard thuần cú pháp không phân biệt được "tên đầy đủ của X" với "sản phẩm con của X". Tổng kết ở ngưỡng 0.90: **0 false merge, 2 false reject** (`Azure`, `Amazon.com`) — đánh đổi đúng hướng cho KG, vì false merge làm hỏng vĩnh viễn mọi truy vấn qua node đó.

### 3. Super-node Analysis

| Hạng | Tên | Type | Degree |
|---|---|---|---|
| 1 | Microsoft | Company | **38** |
| 2 | **IoT** | Technology | **32** |
| 3 | SpaceX | Company | **31** |

`IoT` hạng 2 dù **không phải thực thể mà là chủ đề** — super-node "rác" làm loãng ngữ cảnh.

Bậc lớn nhất 38 < ngưỡng 100 nên nhánh cắt tỉa không tự kích hoạt. Test 2 tầng: **[A]** chứng minh `LIMIT` chạy phía DB + thứ tự `published_date DESC`; **[B]** hạ tạm ngưỡng để kích hoạt đúng code path, khôi phục trong `finally`.

**Ưu điểm** "50 cạnh mới nhất": trần cứng cho token/latency, hợp với tin thời sự. **Rủi ro**: recency bias — câu hỏi lịch sử bị cắt sạch cạnh cũ mà **không báo lỗi**.

### 4. Bảng so sánh Benchmark

| Tiêu chí | Flat RAG | GraphRAG | Δ |
|---|---|---|---|
| Comprehensiveness | 3.20 | **4.40** | +1.20 |
| Faithfulness | 3.40 | **4.80** | +1.40 |
| Multi-hop reasoning | 3.20 | **4.60** | +1.40 |
| Token / câu | **1 035** | 3 194 | 3.09× |
| Latency (s) | 43.6 | 2.7 | *không dùng được* |

> ⚠️ **Latency không kết luận được**: Flat RAG luôn chạy trước GraphRAG nên hứng trọn thời gian chờ rate limiter. Lỗi thiết kế phép đo, không phải đặc tính kiến trúc.

| Nhóm | n | Flat | Graph | Bên thắng |
|---|---|---|---|---|
| `factoid` | 2 | **5.00** | **5.00** | Hoà — Graph tốn 1.74× token vô ích |
| `multi-hop` | 1 | 4.33 | **5.00** | GraphRAG |
| `cross-doc` | 2 | **1.00** | **4.00** | GraphRAG |

**Ca 1 — Flat RAG thua (G05, cross-doc, Δ +4.00):** câu hỏi cần **loại quan hệ + tập ngày xuất bản** tổng hợp qua 13 chunk. Thông tin này không tồn tại trong bất kỳ đoạn văn xuôi nào — chỉ có ở dạng cấu trúc trên thuộc tính cạnh. Tăng `k` bao nhiêu cũng vô ích.

**Ca 2 — GraphRAG suy giảm (G04, cross-doc, Comprehensiveness 3/5):** thu thập 82 cạnh nhưng **chỉ 47 cạnh vào được context** — mất 43% do `MAX_GRAPH_CONTEXT_CHARS = 8000`, chính là giá trị tôi hạ từ 14 000 để vừa trần TPM 8 000 token/phút. Đối chiếu G05 (47 cạnh, không bị cắt → 5.00) cho quan hệ nhân quả rõ ràng. Việc cắt xảy ra **âm thầm**, không có cảnh báo nào trong log.

### 5. Trade-offs, Agent Control & Scale 350 MB

**Đề xuất của AI Agent tôi từ chối:** (1) pairwise cosine O(N²) cho ER; (2) giữ ngưỡng `merge_guard` chung 0.72 — `SequenceMatcher("sam altman","steve altman") = 0.727` sẽ gộp nhầm hai người; (3) thêm `ai`/`labs` vào `CORP_SUFFIXES` — sẽ vô hiệu hoá chính guard vừa viết; (4) coref in lại toàn văn; (5) coi mọi 429 như nhau — lỗi TPD chỉ hồi sau nhiều giờ, retry là vô ích.

**Scale 350 MB — bottleneck đầu tiên là tầng LLM extraction**, không phải Neo4j hay FAISS. Đo được 200 chunk = ~19 000 token; ngoại suy 100 000 bài ≈ **9.5 triệu token**, gấp 47× hạn mức 200k/ngày. Giải pháp: pre-filter trước (đã chứng minh 3.4% corpus cho 6.6× triple/chunk), định tuyến 2 tầng model, cache theo hash chunk, blocking + HNSW cho ER, community partitioning cho retrieval.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN

### 1. Mapping bài giảng vào code

| Khái niệm | Module | Hàm | Quan sát |
|---|---|---|---|
| Conservative Coreference | M1 | `resolve_coref_batch()`, `apply_substitutions()` | 98/200 chunk bị sửa |
| Schema & Allowlist Guard | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | 0 vi phạm / 306 triple |
| Bulk Cypher Ingestion | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND` batch 1000 |
| Entity Resolution & Union-Find | M3 | `build_resolution_map()`, `UF` | 288 → 281 canonical |
| Super-node Degree Cap | M4 | `retrieve_graph_context()` | Degree max 38 |
| LLM-as-a-Judge | M5 | `judge_answer()` | Judge khác họ với generator |

*(Bảng đầy đủ 17 dòng ở `reflection_ThachMinhQuan.md`.)*

### 2. Debugging & bài học

**Lỗi khó nhất: pipeline "chạy đúng" nhưng cho đồ thị vô dụng.** Toàn bộ assert pass, `invalid_provenance_edges = 0`, nhưng đồ thị chỉ có 69 cạnh, degree max 4, ER gộp 0 cặp. Truy vấn Cypher đếm cấu trúc GraphRAG *cần* mới lộ ra: **1 đường 2-hop xuyên chunk, 0 cặp entity cross-doc**.

Nguyên nhân không phải prompt mà là **lấy mẫu dữ liệu**: 1 500 bài ngẫu nhiên từ 168 công ty, mỗi bài là snippet ~44 từ về một sản phẩm riêng → thực thể không bao giờ lặp lại. Sau khi curation: **306 triple / 200 chunk (6.6× tốt hơn), degree max 38, ER gộp 7**.

**Bài học:** assert kiểm tra *tính toàn vẹn cấu trúc* không thay được kiểm tra *tính hữu dụng*. Phải đếm số 2-hop path và số entity đa tài liệu **trước** khi chạy benchmark.

**Ba bài học vận hành:** (1) tách "gọi API" khỏi "parse" — một `AttributeError` đã xoá sạch 50 request đã trả tiền; (2) ràng buộc nguy hiểm nhất không nằm trong header — TPD 200k/ngày chỉ hiện trong nội dung lỗi 429; (3) ước lượng token phải tự hiệu chuẩn — ước lượng thô hụt ~2×.

### 3. Kế hoạch đồ án

Quyết định dựa trên **phân bố loại câu hỏi**, không dựa vào việc "GraphRAG hiện đại hơn":

- `factoid` → **Flat RAG** (đo được: chất lượng bằng nhau, Graph tốn 1.74× token)
- `multi-hop` / `cross-doc` → **GraphRAG** (Flat RAG thất bại có hệ thống ở cross-doc: 1.00/5)
- Thực tế nên dùng **query router** phân loại trước, ước tính tiết kiệm ~27% token mà không mất điểm chất lượng.

*(Thiết kế Node/Relation, chiến lược ER và Super-node cho đồ án cụ thể: xem `reflection_ThachMinhQuan.md` mục 3.)*

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|---|---|---|
| Mức độ hiểu bài giảng GraphRAG | | |
| Khả năng kiểm soát AI Coding Agent | | |
| Chất lượng đồ thị tri thức xây dựng | | 286 node / 305 cạnh sau khi sửa lỗi lấy mẫu |
| Khả năng phân tích và debug hệ thống | | |
