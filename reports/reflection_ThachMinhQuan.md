# Suy Ngẫm Cá Nhân & Kế Hoạch Đồ Án — Lab 19

**Học viên:** Thạch Minh Quân · **MSSV:** 2A202601585
**Khóa học:** AICB-K34 · Track 3: GraphRAG

---

## 1. Mapping bài giảng vào code

| Khái niệm trong bài giảng | Module | Hàm / khối code | Cell | Quan sát thực tế |
|---|---|---|---|---|
| Exact dedup (SHA-1) | M1 | `standardize_news()` | 1.5 | 63 717 → 57 085 bài (loại 6 632) |
| **Curation giàu quan hệ** | M1 | `select_relation_dense()` | 1.5 | 1 936/57 085 bài (3.4%) — quyết định quan trọng nhất toàn lab, xem mục 2 |
| Near-dedup (MinHash/LSH) | M1 (bonus) | `near_dedup()`, `near_dup_sweep()` | 1.5b | @0.75 loại 5 bài; sweep 0.9→0.5 cho 0/2/12/20/40 cặp |
| Rolling-window chunking | M1 | `chunk_text()`, `build_chunks()` | 1.5b | 1 495 bài → 1 501 chunk (bài chỉ ~44 từ nên hầu như 1 chunk/bài) |
| Conservative Coreference | M1 | `resolve_coref_batch()`, `apply_substitutions()` | 1.7 | 98/200 chunk được áp phép thế |
| Schema & Allowlist Guard | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, `_parse_extraction()` | 2.1 | **0 vi phạm** schema trên 306 triple |
| Provenance bắt buộc | M2 | `run_extraction()`, `bulk_insert_edges()` | 2.1 / 2.3 | `invalid_provenance_edges = 0` |
| Entity Resolution + Union-Find | M3 | `build_resolution_map()`, `UF`, `merge_guard()` | 2.2 | 288 mention → 281 canonical (gộp 7) |
| Lexical Guard | M3 | `merge_guard()` | 2.2 | 1 `REJECT_GUARD` (`Azure`/`Microsoft Azure`, cosine 0.961) |
| Bulk Cypher Ingestion | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | 2.3 | `UNWIND $rows` batch 1000 → 286 node / 305 cạnh |
| Flat RAG (FAISS IndexFlatIP) | M4 | `build_flat_index()`, `retrieve_flat_context()` | 3.1 | 1 501 vector, embedding CPU ~30s |
| Seed extraction + fuzzy fallback | M4 | `extract_seeds()`, `match_seeds()` | 3.2 | **100% seed khớp `EXACT`**, không câu nào phải dùng fuzzy |
| BFS + Super-node cap | M4 | `retrieve_graph_context()`, `recent_edges()` | 3.3 / 5.1 | Degree max 38 (Microsoft); cap chứng minh bằng test 2 tầng |
| Hybrid context | M4 | `answer_graph_rag()` | 3.4 | `=== GRAPH ===` + `=== VECTOR ===` |
| LLM-as-a-Judge | M5 | `judge_answer()` | 4.2 | Judge `qwen/qwen3.6-27b` khác họ với generator `gpt-oss-120b` |
| Community detection | Bonus A | `build_communities()`, `summarize_communities()` | Bonus A | NetworkX greedy modularity, 5 community report |
| Self-correction | Bonus B | `self_correcting_context()` | Bonus B | Đã cài đặt; không chạy được vì cạn quota 200k token/ngày |

---

## 2. Quá trình debugging & bài học

### Lỗi khó nhất: pipeline "chạy đúng" nhưng cho ra đồ thị vô dụng

**Hiện tượng:** toàn bộ Phần 1–2 chạy sạch, không exception, `invalid_provenance_edges = 0`, mọi assert đều pass. Nhưng đồ thị thu được chỉ có 69 cạnh / 120 node, degree cao nhất = 4, Entity Resolution gộp được **0** cặp.

**Chẩn đoán sai ban đầu:** tôi tưởng do prompt extraction quá thiên về precision nên recall thấp, và định sửa prompt.

**Nguyên nhân thật:** tôi viết truy vấn Cypher đếm cấu trúc mà GraphRAG *cần* để hoạt động, và con số nói thẳng ra vấn đề:

```
2-hop path xuyên chunk : 1
cặp entity ≥ 2 chunk   : 0
```

Không phải lỗi prompt. Là lỗi **lấy mẫu dữ liệu**: 1 500 bài ngẫu nhiên từ corpus trải trên 168 công ty, mỗi bài là snippet ~44 từ về một sản phẩm riêng biệt (`SumoFlo® CPFM-8103 Single-Use Coriolis Mass Flow Meter`). Thực thể gần như không bao giờ lặp lại, nên **không có gì để nối**.

**Cách xử lý:** giữ nguyên `LAB_MAX_ARTICLES = 1500` nhưng chọn có chủ đích bài vừa có tín hiệu quan hệ vừa nhắc công ty lớn. Kết quả:

| | Ngẫu nhiên | Curated |
|---|---|---|
| Triple | 69 / 300 chunk | **306 / 200 chunk** (6.6×/chunk) |
| Degree max | 4 | **38** |
| ER gộp được | 0 | 7 |

**Bài học:** *một pipeline pass hết mọi assert vẫn có thể vô dụng.* Các assert của tôi kiểm tra **tính toàn vẹn cấu trúc** (provenance đủ, schema hợp lệ) chứ không kiểm tra **tính hữu dụng** (đồ thị có đủ kết nối để truy vấn không). Bài học là phải viết thêm loại kiểm tra thứ hai — đếm số 2-hop path, số entity xuất hiện đa tài liệu — và coi đó là điều kiện cần trước khi chạy benchmark.

### Ba lỗi vận hành khác và cách khắc phục

| # | Lỗi | Thiệt hại | Khắc phục |
|---|---|---|---|
| 1 | `AttributeError` trong parser xảy ra **sau** khi 50 request đã trả tiền | Mất ~15 phút + ~80k token, không cứu được gì | Ghi raw response xuống JSONL **ngay khi nhận**, parse là bước riêng → sửa parser sau tốn 0 request |
| 2 | Coi mọi lỗi 429 như nhau rồi backoff-retry | 50 request bò mất 53 phút | Nhận diện riêng **TPD** (quota ngày), đánh dấu model chết trong ngày, chuyển model dự phòng thay vì retry vô vọng |
| 3 | Ước lượng token `ký tự/3.2` hụt ~2× thực tế | Liên tục vượt TPM → 429 dây chuyền | Hệ số EMA tự hiệu chuẩn từ `usage` thật của mỗi response |

### Bài học về kiểm soát AI Coding Agent

- **Code chạy được ≠ code đúng.** `merge_guard` ngưỡng 0.72 chạy trơn tru và không báo lỗi gì, nhưng `SequenceMatcher("sam altman", "steve altman") = 0.727` → gộp nhầm hai người khác nhau. Chỉ phát hiện được khi tự tay tính giá trị cho đúng cặp mà đề bài cảnh báo.
- **Lỗi ngữ nghĩa không bị kiểm tra cấu trúc bắt.** Cạnh sai do coreference sai vẫn có đủ `source_chunk_id` + `published_date` nên vượt qua 100% bài kiểm tra provenance.
- **Một test không bao giờ fail thì không phải test.** `test_supernode_policy()` bản gốc rơi vào nhánh `else` khi không có node degree > 100 và **không assert gì cả** — trông như pass nhưng không chứng minh điều gì.
- **Lỗi âm thầm nguy hiểm hơn lỗi ồn ào.** `textualize()` cắt mất 43% số cạnh của G04 mà không phát cảnh báo nào; chỉ khi đối chiếu `graph_collected_edges` với số dòng thật trong context mới lộ ra (xem `failure_analysis.md`).

---

## 3. Kế hoạch áp dụng vào đồ án thực tế

### 3.1. Bài toán

- **Tên đồ án:** _[điền tên đồ án của bạn]_
- **Mô tả:** _[mô tả ngắn]_
- **Phân bố câu hỏi người dùng:** _[ước lượng % factoid / multi-hop / tổng hợp]_

### 3.2. Có cần GraphRAG không?

Từ số liệu lab này, quyết định nên dựa vào **phân bố loại câu hỏi**, không phải vào việc "GraphRAG hiện đại hơn":

| Loại câu hỏi | Kết quả đo được | Khuyến nghị |
|---|---|---|
| `factoid` (1-hop) | Flat 5.00 = Graph 5.00, nhưng Graph tốn **1.74× token** | **Flat RAG** — GraphRAG lãng phí thuần tuý |
| `multi-hop` | Flat 4.33 → Graph 5.00 | GraphRAG |
| `cross-doc` | Flat **1.00** → Graph 4.00 | **GraphRAG** — Flat RAG thất bại có hệ thống |

**Quy tắc rút ra:** GraphRAG chỉ hoàn vốn khi câu hỏi cần **thông tin dạng cấu trúc** (loại quan hệ, dòng thời gian, đường nối giữa các thực thể) — thứ không tồn tại trong bất kỳ đoạn văn xuôi nào và chỉ xuất hiện sau khi trích xuất. Nếu bài toán chỉ tra cứu sự thật đơn lẻ, chi phí index của GraphRAG là lãng phí.

**Kết luận cho đồ án của tôi:** `[ ] Flat RAG` · `[ ] Hybrid + query router` · `[ ] GraphRAG đầy đủ`
**Lý do:** _[điền]_

### 3.3. Thiết kế Node / Relation dự kiến

**Nodes:** _[điền — mỗi node cần `id`, `name`, `name_norm`, `aliases_norm`]_

**Relations (allowlist đóng):** _[điền]_

> Giữ nguyên nguyên tắc từ lab: **mọi cạnh phải có `source_chunk_id` + `published_date` + `evidence` + `confidence`**, và relation type phải nằm trong allowlist — không cho LLM tự đặt tên quan hệ.

### 3.4. Chiến lược Entity Resolution

1. **Alias map thủ công** cho định danh không suy ra được bằng lexical/vector (mã sản phẩm, ticker, viết tắt nội bộ).
2. **Blocking** theo type + token đầu để tránh O(N²) khi corpus lớn.
3. **Vector ANN** (HNSW) trong từng block, ngưỡng gộp ~0.90.
4. **Lexical Guard riêng theo từng loại thực thể** — bài học rõ nhất từ lab: một ngưỡng chung cho mọi loại chắc chắn sai ở loại có không gian tên dày đặc (tên người).
5. **Guard ngữ nghĩa thay vì thuần cú pháp** — ca `Azure`/`Microsoft Azure` cho thấy quan hệ tập con token không phân biệt được "tên đầy đủ" với "sản phẩm con". Nếu token thừa là tên một `Company` đã có trong đồ thị thì nên gộp.
6. **Audit table bắt buộc** với lý do merge/reject, review định kỳ.

### 3.5. Chiến lược Super-node

- Ngưỡng: `degree > N` → cap M cạnh. **Không nên xếp hạng thuần theo recency** như lab — sẽ mất sạch bằng chứng lịch sử mà không báo lỗi.
- Đề xuất: chấm điểm hỗn hợp `w1·recency + w2·confidence + w3·cosine(query, evidence)`, và nếu câu hỏi có mốc thời gian thì lọc `published_date` **trước** khi `LIMIT`.
- **Cảnh báo khi cắt**: ghi `truncated_edges` vào diagnostics. Lab này mất 43% cạnh ở một câu mà log không hề báo gì.
- Cảnh giác với **super-node "rác"**: trong lab, `IoT` đứng hạng 2 với degree 32 dù nó là *chủ đề* chứ không phải thực thể. Nên có danh sách chặn cho các khái niệm quá chung.

---

## 🎯 Tự đánh giá

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|---|---|---|
| Mức độ hiểu bài giảng GraphRAG | | |
| Khả năng kiểm soát AI Coding Agent | | |
| Chất lượng đồ thị tri thức xây dựng | | Đồ thị 286 node/305 cạnh sau khi sửa lỗi lấy mẫu |
| Khả năng phân tích và debug hệ thống | | |
