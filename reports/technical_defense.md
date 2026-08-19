# Thuyết Minh Kỹ Thuật — Lab 19: Production-Grade GraphRAG vs Flat RAG

**Học viên:** Thạch Minh Quân · **MSSV:** 2A202601585
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Môi trường:** Local (Python 3.11) · Neo4j AuraDB · Groq
**Model:** Bulk (coref/NER-RE/seed) `openai/gpt-oss-120b` · Generator `openai/gpt-oss-120b` · Judge `qwen/qwen3.6-27b`

---

## 0. Pipeline thực tế đã chạy

| Bước | Kết quả |
|---|---|
| Stream dataset | 137 577 dòng / 80 MB |
| Lọc thân bài ≥ 80 ký tự | 137 577 → **63 717** (73 860 dòng hỏng/rỗng trong dataset gốc) |
| Exact dedup (SHA-1 `title+text`) | 63 717 → **57 085** |
| Curation giàu quan hệ | 1 936 / 57 085 bài (3.4%) → lấy **1 500** |
| Near-dedup MinHash/LSH @0.75 | 1 500 → **1 495** |
| Chunking (220 từ / overlap 40) | **1 501 chunk** |
| Coreference (200 chunk) | **98 chunk** được áp phép thế |
| NER + RE (200 chunk) | **306 triple**, 0 vi phạm schema allowlist |
| Entity Resolution | 288 mention → **281 canonical** (gộp 7), audit **12 dòng** |
| Neo4j | **286 node / 305 cạnh**, `invalid_provenance_edges = 0` ✅ |

---

## 1. Coreference Resolution — tình huống phân giải sai và hậu quả

### Quyết định kiến trúc: trả về PHÉP THẾ, không trả về toàn văn

Bản khung bắt LLM in lại toàn bộ `resolved_text`. Với 5 chunk/request, mỗi lời gọi phải sinh ~1 500 output token chỉ để **chép lại** phần văn bản không đổi. Tôi đổi sang cho LLM trả về danh sách `{mention → antecedent}` rồi Python tự áp phép thế. Hai lợi ích:

1. Output giảm ~10 lần → thoát khỏi trần TPM của Groq free tier.
2. **An toàn ngữ nghĩa hơn**: LLM không còn cơ hội sửa lén số liệu, ngày tháng, tên sản phẩm ở phần văn bản không liên quan — đúng tinh thần "conservative" mà đề bài yêu cầu. Mọi thay đổi đều truy vết được qua cột `applied_subs`.

### Ca phân giải SAI (dẫn chứng thật)

`chunk_id = cf095df30bfa3f72399f::c0000` (similarity gốc/sau = 0.461 — thấp nhất toàn bộ 98 chunk bị sửa)

> **Nguyên văn:** *"Samsung unveils OLED display with embedded heart rate sensor. **It** is a separate module attached under the display panel."*
>
> **Sau coref:** *"...**OLED display with embedded heart rate sensor** is a separate module attached under the display panel."*

- **Đại từ:** `It`
- **Phân giải thành:** `OLED display with embedded heart rate sensor`
- **Đáng lẽ phải là:** `heart rate sensor` (chỉ riêng cảm biến)

**Vì sao model sai:** tiền ngữ đúng (`heart rate sensor`) nằm *lồng bên trong* một cụm danh từ dài hơn (`OLED display with embedded heart rate sensor`). LLM chọn cụm danh từ ngoài cùng vì nó là chủ ngữ nổi bật nhất của câu trước.

**Hậu quả lên Knowledge Graph:** câu sau khi phân giải trở thành vô nghĩa — *"màn hình OLED là một module gắn dưới tấm nền màn hình"*. Nếu tầng NER+RE trích từ câu này, ta được cạnh sai kiểu `OLED display -USES-> display panel` thay vì quan hệ đúng về cảm biến.

**Điểm nguy hiểm nhất:** cạnh sai đó **vẫn có đầy đủ `source_chunk_id`, `published_date`, `evidence`, `confidence`**, nên nó vượt qua 100% bài kiểm tra provenance ở cell 2.4. *Kiểm tra cấu trúc không bao giờ bắt được lỗi ngữ nghĩa* — đây là bài học lớn nhất của module này.

### Ca thứ hai — ngữ cảnh listicle

`chunk_id = 07b11f3305d87962b831::c0000`

> *"20 Big Companies That Hire Remote Workers. Learn more about **3M**. What **the company** is: **Adobe**..."*  → `"What Adobe is: Adobe..."`

Ở đây `the company` nằm giữa hai công ty khác nhau (3M phía trước, Adobe phía sau). Model chọn Adobe (may mắn là đúng), nhưng đây là loại ngữ cảnh mà xác suất sai rất cao — văn bản dạng danh sách không có cấu trúc diễn ngôn tuyến tính để suy ra tiền ngữ.

### Guard đã thêm

Sau khi quan sát rằng thay **mọi** occurrence của một đại từ là rủi ro (occurrence sau thường trôi sang thực thể khác), tôi giới hạn đại từ trần (`it/its/they/them/he/his/she/her`) **chỉ thay occurrence đầu tiên** — vị trí gần tiền ngữ nhất, đáng tin nhất. Ví dụ kiểm chứng:

```
"Microsoft ... The company said it would invest $10B in OpenAI. It is the largest round yet."
```
`It` thứ hai chỉ *vòng gọi vốn*, không phải Microsoft. Guard giữ nguyên câu cuối.

---

## 2. Entity Resolution — ngưỡng similarity & Lexical Guard

### Cấu hình

| Tham số | Giá trị | Lý do |
|---|---|---|
| `threshold` (ngưỡng GỘP) | **0.90** | Dưới mức này, cặp sản phẩm cùng dòng (`GPT 3`/`GPT-4`, `Copilot`/`Copilot X`) bắt đầu lọt lưới |
| `audit_floor` (ngưỡng GHI LOG) | **0.80** | Tách khỏi ngưỡng gộp để dải 0.80–0.90 vẫn truy vết được |
| `DEFAULT_MIN_RATIO` | 0.72 | SequenceMatcher cho Company/Technology |
| `PERSON_MIN_RATIO` | **0.90** | Xem phân tích bên dưới |

**Tại sao phải tách `audit_floor` khỏi `threshold`:** nếu chỉ ghi log từ 0.90 trở lên thì toàn bộ dải 0.80–0.90 biến mất khỏi audit và câu hỏi này (yêu cầu dẫn chứng cặp > 0.85 bị chặn) **không thể trả lời được**. Đây là thay đổi bắt buộc so với bản khung.

### Cặp similarity > 0.85 bị Lexical Guard CHẶN

| Left | Right | Cosine | Decision | Reason |
|---|---|---|---|---|
| `Azure` | `Microsoft Azure` | **0.961** | `REJECT_GUARD` | `GUARD_TOKEN_SUBSET` |

**Cơ chế:** `strip_suffix("azure") = {azure}` là **tập con thật sự** của `{microsoft, azure}` → Guard 1 chặn. Guard này tồn tại để chặn `Apple` ⊂ `Apple Watch`, `Google` ⊂ `Google Cloud` — sản phẩm mang tên công ty mẹ.

**Nhưng ở ca này Guard đã SAI.** `Azure` và `Microsoft Azure` là *cùng một thực thể*. Đây là **false reject** — cái giá phải trả cho một guard thuần cú pháp: nó không phân biệt được "tên đầy đủ của X" với "sản phẩm con của X", vì cả hai đều có quan hệ tập con về mặt token.

**Hệ quả đo được:** đồ thị có 2 node riêng cho cùng một dịch vụ, làm đứt các đường đi qua Azure và hạ degree của Microsoft.

**Cách khắc phục đúng đắn:** thay quan hệ tập con thuần cú pháp bằng kiểm tra ngữ nghĩa — nếu token thừa chính là tên một `Company` đã có trong đồ thị (`Microsoft`) thì đó là *tên đầy đủ*, nên gộp; nếu token thừa là danh từ sản phẩm (`Watch`, `Cloud`, `Music`) thì đó là *sản phẩm con*, không gộp.

### Các quyết định khác trong audit (12 dòng)

| Cặp | Cosine | Quyết định | Đánh giá |
|---|---|---|---|
| `GPT-3.5 turbo` / `GPT3.5-turbo` | 0.976 | `MERGE_VECTOR` (ratio 0.870) | ✅ Đúng — chỉ khác dấu cách |
| `Copilot` / `Copilot X` | 0.873 | `REJECT_LOW_SIM` | ✅ Đúng — hai sản phẩm khác nhau |
| `GPT 3` / `GPT-4` | 0.850 | `REJECT_LOW_SIM` | ✅ Đúng — hai thế hệ model |
| `Uber` / `Uber Direct` | 0.803 | `REJECT_LOW_SIM` | ✅ Đúng |
| `ChatGPT` / `ChatGPT API` | 0.836 | `REJECT_LOW_SIM` | ✅ Đúng |
| `Amazon.com` / `Amazon` | 0.848 | `REJECT_LOW_SIM` | ❌ **Sai** — cùng thực thể, ngưỡng 0.90 quá cao |
| `AI technology` / `AI` | 0.810 | `REJECT_LOW_SIM` | ⚠️ Tranh cãi |
| 4 × `MERGE_MANUAL` | 1.000 | Alias map | ✅ `Apple Inc→Apple`, `Meta`, `IBM`, `AMD` |

**Tổng kết trung thực:** ở ngưỡng 0.90, hệ thống thiên nặng về **precision** — 0 false merge, nhưng **2 false reject** (`Azure`/`Microsoft Azure`, `Amazon.com`/`Amazon`). Với knowledge graph thì đánh đổi này đúng hướng: một false merge làm hỏng vĩnh viễn mọi truy vấn đi qua node bị gộp nhầm, còn một false reject chỉ làm đồ thị thưa hơn.

### Cải tiến Guard so với bản khung (Challenge B)

| Guard | Chặn trường hợp | Lý do bắt buộc |
|---|---|---|
| `GUARD_TOKEN_SUBSET` | `Apple` ⊂ `Apple Watch` | Sản phẩm mang tên công ty mẹ |
| `GUARD_PERSON_DIFFERENT_GIVEN_NAME` | `Sam Altman` vs `Steve Altman` | `SequenceMatcher = 0.727` — **vượt ngưỡng chung 0.72, bản khung sẽ gộp nhầm hai người khác nhau** |
| `PERSON_MIN_RATIO = 0.90` | Họ/tên gần giống | Không gian tên người dày đặc hơn tên công ty |
| Alias map chỉ áp cho `Company` | Ticker áp nhầm sang `Person`/`Technology` | Ticker là khái niệm của pháp nhân |
| `CORP_SUFFIXES` **không** chứa `ai`/`labs`/`technologies` | `Meta AI` → `meta` → gộp vào `Meta` | `strip_suffix` chạy **trước** Guard 1 nên sẽ vô hiệu hoá chính guard đó |

**Quyết định có chủ đích:** không map `Alphabet` → `Google`. Alphabet là holding, Google là subsidiary; gộp lại làm sai các quan hệ `INVESTED_IN`/`ACQUIRED` ở cấp tập đoàn.

---

## 3. Super-node Analysis

### Top thực thể bậc cao nhất

| Hạng | Tên | Type | Degree |
|---|---|---|---|
| 1 | **Microsoft** | Company | **38** |
| 2 | **IoT** | Technology | **32** |
| 3 | **SpaceX** | Company | **31** |
| 4 | Starlink | Technology | 24 |
| 5 | Elon Musk | Person | 22 |
| 6 | Google | Company | 14 |
| 7 | OpenAI | Company | 12 |
| 8 | Amazon | Company | 12 |

Phân bố bậc lệch mạnh (power law): 3 node đầu chiếm ~33% tổng số đầu mút cạnh, trong khi trung vị degree ≈ 2.

Đáng chú ý: **`IoT` hạng 2** — đó không phải thực thể mà là một *chủ đề*. Tầng NER coi nó là `Technology` và mọi bài về thiết bị kết nối đều gắn cạnh vào nó. Đây là một dạng super-node "rác" làm loãng ngữ cảnh mà không mang thông tin phân biệt.

### Chính sách và bằng chứng đã chạy

Cấu hình: `SUPER_NODE_DEGREE=100` → cap `50` cạnh mới nhất (`ORDER BY published_date DESC LIMIT 50`), `GLOBAL_EDGE_CAP=250`, `MAX_GRAPH_CONTEXT_CHARS=8000`.

Bậc lớn nhất thực tế là **38 < 100**, nên nhánh cắt tỉa **không tự kích hoạt**. Bản test gốc khi đó rơi vào `else` và **không assert gì cả** — tức là không có bằng chứng nào cho rubric. Tôi thay bằng test hai tầng:

- **[A] Tầng Cypher:** so `recent_edges(hub, 100000)` với `recent_edges(hub, 50)`, khẳng định `len ≤ 50` **và** dãy `published_date` giảm dần. Chứng minh `LIMIT` chạy **phía database** — super-node không bao giờ truyền hàng nghìn cạnh về client rồi mới cắt.
- **[B] Tầng BFS:** hạ **tạm thời** `SUPER_NODE_DEGREE` xuống dưới bậc lớn nhất quan sát được và `cap=5`, chạy `retrieve_graph_context(seed_ids=[hub])` để kích hoạt **đúng nhánh code** cắt tỉa, kiểm tra `supernode_events` khác rỗng và mọi `fetched ≤ cap`, rồi khôi phục hằng số trong `finally`. Cùng một code path, chỉ khác tham số.

Tham số `seed_ids` được thêm vào `retrieve_graph_context()` để test **tất định và không tốn quota LLM**.

### Ưu điểm và rủi ro của "50 cạnh mới nhất"

**Ưu điểm:** chặn context explosion (node bậc 5 000 vẫn chỉ đóng góp 50 dòng → token/latency có trần cứng); tin công nghệ nặng tính thời sự nên cạnh mới thường là cạnh người dùng đang hỏi; cắt ở tầng Cypher tiết kiệm cả băng thông lẫn RAM client.

**Rủi ro:**
- **Recency bias** — câu hỏi lịch sử ("Microsoft mua lại những công ty nào 2016–2018") bị cắt sạch cạnh cũ và GraphRAG trả lời thiếu mà **không hề báo lỗi**.
- Xếp hạng thuần thời gian bỏ qua `confidence` và độ liên quan tới truy vấn.
- Cạnh thiếu `published_date` bị `coalesce(...,'')` đẩy xuống cuối → luôn bị cắt trước. Trong đồ thị này rủi ro đó bằng 0 vì 100% cạnh có ngày.

**Đề xuất:** cắt tỉa nhận biết truy vấn — nếu câu hỏi chứa mốc thời gian thì lọc `published_date` theo khoảng đó **trước** khi `LIMIT`; hoặc chấm điểm hỗn hợp `w1·recency + w2·confidence + w3·cosine(query, evidence)`.

---

## 4. So sánh thực nghiệm Flat RAG vs GraphRAG

> Nguồn: `outputs/graphrag_vs_flatrag_summary.csv` (dòng `ALL`) và `outputs/graphrag_eval_results.csv`.
> Golden Dataset: 5 câu (`factoid` 2 · `multi-hop` 1 · `cross-doc` 2), sinh từ đường đi CÓ THẬT trong đồ thị.

### Bảng tổng hợp (trung bình toàn bộ Golden Dataset)

| Tiêu chí | Flat RAG | GraphRAG | Δ | Nhận xét |
|---|---|---|---|---|
| Comprehensiveness (1–5) | 3.20 | **4.40** | **+1.20** | GraphRAG bao phủ đủ quan hệ + ngày tháng |
| Faithfulness (1–5) | 3.40 | **4.80** | **+1.40** | Cạnh có `evidence` + `chunk_id` nên dễ trích dẫn đúng |
| Multi-hop reasoning (1–5) | 3.20 | **4.60** | **+1.40** | Quan hệ bắc cầu đã vật chất hoá thành cạnh |
| Token usage / câu | **1 035** | 3 194 | **3.09× đắt hơn** | Chi phí thật của subgraph context |
| Latency (s) | 43.6 | 2.7 | *(không dùng được — xem cảnh báo)* | |

> ⚠️ **Số liệu latency KHÔNG dùng được để kết luận.** Trong `run_evaluation()`, Flat RAG luôn
> được gọi **trước** GraphRAG cho mỗi câu, nên nó hứng trọn thời gian chờ của token-bucket
> rate limiter (TPM 8 000), còn GraphRAG chạy ngay sau khi ngân sách vừa giải phóng. Đây là
> **lỗi thiết kế phép đo**, không phải đặc tính kiến trúc. Muốn đo đúng phải tách riêng
> retrieval latency khỏi thời gian chờ rate limit, hoặc hoán đổi thứ tự gọi giữa các câu.
> Chỉ số **token/câu (3.09×) mới phản ánh đúng chi phí**, và nó khớp với lý thuyết.

### Theo từng nhóm câu hỏi

| Nhóm | n | Flat (TB 3 tiêu chí) | Graph (TB 3 tiêu chí) | Bên thắng |
|---|---|---|---|---|
| `factoid` | 2 | **5.00** | **5.00** | Hoà tuyệt đối |
| `multi-hop` | 1 | 4.33 | **5.00** | GraphRAG |
| `cross-doc` | 2 | **1.00** | **4.00** | GraphRAG (cách biệt lớn nhất) |

**Flat RAG thắng ở nhóm nào?** Không nhóm nào — nhưng nó **hoà tuyệt đối 5.0/5.0 ở `factoid`**
trong khi tốn ít hơn 2.1 lần token. Với câu 1-hop, subgraph không thêm thông tin gì mà chỉ
thêm chi phí. Kết luận thực tiễn: **nên có query router**, không nên luôn chạy hybrid.

**GraphRAG thắng ở nhóm nào?** `cross-doc` (+3.00) và `multi-hop` (+0.67). Nguyên nhân chung:
cả hai đòi **tổng hợp bằng chứng nằm rải ở nhiều chunk**. Flat RAG top-k chỉ trả về những chunk
giống câu hỏi nhất về mặt ngữ nghĩa, không có cơ chế nào bảo đảm gom đủ *tất cả* chunk liên quan
đến một cặp thực thể. GraphRAG duyệt cạnh nên gom đủ theo cấu trúc.

**Vì sao Flat RAG bị 1.0/1.0/1.0 ở `cross-doc`:** câu hỏi yêu cầu liệt kê **loại quan hệ và ngày
xuất bản** tổng hợp qua nhiều bài. Không chunk văn xuôi nào chứa sẵn danh sách đó — thông tin
này chỉ tồn tại ở **dạng cấu trúc**, tức là ở các thuộc tính cạnh của đồ thị. Judge nhận xét
đúng: *"provides a narrative summary of business events"* thay vì trả lời đúng ràng buộc.

Chi tiết truy vết 2 ca lỗi: xem `reports/failure_analysis.md`.

---

## 5. Trade-offs, kiểm soát AI Coding Agent & scale 350 MB

### 5a. Đề xuất của AI Coding Agent mà tôi TỪ CHỐI

| # | Đề xuất | Từ chối vì | Thay bằng |
|---|---|---|---|
| 1 | Pairwise cosine O(N²) cho Entity Resolution | 288 mention còn chạy được, nhưng scale lên 350 MB thì ma trận nổ RAM | FAISS `IndexFlatIP` + top-k ANN; near-dedup dùng MinHash/LSH O(N) |
| 2 | Giữ `merge_guard` ngưỡng chung 0.72 cho mọi loại thực thể | `SequenceMatcher("sam altman","steve altman") = 0.727` → **gộp nhầm hai người** | `PERSON_MIN_RATIO = 0.90` + `GUARD_PERSON_DIFFERENT_GIVEN_NAME` |
| 3 | Thêm `ai`, `labs`, `technologies` vào `CORP_SUFFIXES` để "gộp tốt hơn" | `strip_suffix` chạy trước Guard 1 → `Meta AI` rút thành `meta` rồi gộp thẳng vào `Meta`, vô hiệu hoá chính guard vừa viết | Chỉ giữ hậu tố **pháp lý** |
| 4 | Coref in lại toàn văn `resolved_text` | Đốt ~1 500 output token/request để chép lại văn bản không đổi, và cho model cơ hội sửa lén số liệu | Trả về danh sách phép thế, Python tự áp |
| 5 | Coi mọi lỗi 429 như nhau rồi backoff + retry | Lỗi TPD (quota **ngày**) chỉ hồi sau nhiều giờ; retry là vô ích và đốt sạch quota còn lại | Nhận diện riêng TPD, đánh dấu model chết trong ngày, chuyển sang model dự phòng |

### 5b. Ba bài học vận hành (từ lỗi thật đã gặp)

1. **Tách "gọi API" khỏi "parse".** Một `AttributeError` trong parser đã xoá sạch kết quả của 50 request (~15 phút và ~80k token) vì exception ném ra trước khi kịp lưu gì. Sau khi sửa: raw response ghi xuống JSONL **ngay khi nhận**, parse là bước riêng — sửa parser sau đó tốn 0 request.
2. **Ràng buộc nguy hiểm nhất không nằm trong header.** TPM = 8 000 token/phút có trong `x-ratelimit-limit-tokens`, nhưng **TPD = 200 000 token/ngày dùng chung cho cả tổ chức** chỉ hiện ra trong nội dung lỗi 429 khi đã quá muộn.
3. **Ước lượng token phải tự hiệu chuẩn.** Ước lượng thô `ký tự/3.2` hụt ~2 lần so với thực tế (chunk_id dạng hex + JSON escape tokenize ở mức ~1.4 ký tự/token), khiến hệ thống liên tục vượt TPM. Đã thay bằng hệ số EMA cập nhật từ `usage` thật của mỗi response.

### 5c. Quyết định curation dữ liệu (quan trọng nhất)

Lần chạy đầu lấy **mẫu ngẫu nhiên** 1 500/57 085 bài. Kết quả đo được:

| | Ngẫu nhiên | Sau curation |
|---|---|---|
| Triple trích được | 69 / 300 chunk | **306 / 200 chunk** |
| Node / cạnh | 120 / 69 | **286 / 305** |
| Entity Resolution gộp được | **0** | 7 |
| Degree cao nhất | **4** | **38** |
| **Đường 2-hop xuyên chunk** | **1** | *(xem mục 4)* |
| Cặp entity ≥ 2 chunk | **0** | *(xem mục 4)* |

Corpus ngẫu nhiên **không thể benchmark được**: đồ thị gần như rời rạc nên GraphRAG không có gì để duyệt, và cả 3 nhóm câu hỏi đều sẽ cho GraphRAG ≈ Flat RAG vì lý do sai (không phải vì kiến trúc, mà vì dữ liệu).

Nguyên nhân: dataset trải trên 168 công ty, mỗi bài là snippet ~44 từ về một sản phẩm riêng biệt (`SumoFlo® CPFM-8103 Single-Use Coriolis Mass Flow Meter`) — thực thể gần như không bao giờ lặp lại.

Xử lý: giữ nguyên `LAB_MAX_ARTICLES = 1500` theo Scale Guard, nhưng **chọn có chủ đích** những bài vừa có tín hiệu quan hệ (`acquire|invest|partner|launch|found|...`) vừa nhắc tên công ty lớn — 1 936 bài (3.4%). Đây chính là bước **pre-filter rẻ tiền trước khi gọi LLM đắt tiền** mà mọi pipeline production đều làm.

### 5d. Scale lên 350 MB (~100 000 bài)

**Bottleneck đầu tiên là tầng LLM extraction, không phải Neo4j hay FAISS.** Đo được: 200 chunk tốn ~19 000 token. Ngoại suy tuyến tính cho 100 000 bài (~100 000 chunk) là ~9.5 triệu token — gấp **47 lần** hạn mức 200k/ngày của free tier.

| Tầng | Vấn đề khi scale | Giải pháp |
|---|---|---|
| Extraction | Chi phí token là ràng buộc thống trị | **Pre-filter trước** (đã chứng minh: 3.4% corpus cho 6.6× triple/chunk); định tuyến 2 tầng — model nhỏ lọc chunk có khả năng chứa quan hệ, model lớn chỉ chạy trên phần đã lọc; cache theo hash chunk |
| Rate limit | TPD dùng chung toàn tổ chức | Worker queue có token-budget tập trung; nâng tier trả phí; phân bổ theo model |
| Entity Resolution | O(N²) không khả thi | Blocking (type + token đầu) rồi HNSW ANN trong từng block; Union-Find phân tán |
| Dedup | Exact hash bỏ sót bản repost | MinHash/LSH (đã triển khai, O(N)) |
| Neo4j | Aura Free 200k node/400k rel sẽ vỡ | Tier trả phí hoặc self-host; `apoc.periodic.iterate`; index composite |
| Retrieval | Super-node bùng nổ theo N | Precompute community + tóm tắt phân tầng; query router chọn tầng local/global |
| Đánh giá | Judge tốn tiền theo số câu | Judge mẫu phân tầng + metric tự động (recall trên gold path) cho phần còn lại |

---

## 📎 Phụ lục — Reproducibility

| Thông số | Giá trị |
|---|---|
| `LAB_MAX_ARTICLES` / `LAB_MAX_CHUNKS` / `EXTRACTION_MAX_CHUNKS` | 1500 / 3000 / 200 |
| `CHUNK_WORDS` / `CHUNK_OVERLAP_WORDS` | 220 / 40 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| `SEED` | 42 |
| Rate limit đo được | TPM 8 000/model · TPD 200 000 dùng chung tổ chức · RPM 1 000 |
| `reasoning_effort` | `gpt-oss` → `low`; `qwen` → `none` (bắt buộc, nếu không model phát `<think>` và Groq trả 400 khi bật JSON mode) |

> 🔐 Không có API key hay password nào hard-code trong notebook — toàn bộ đọc qua `get_secret()` từ `.env` / Colab Secrets.
