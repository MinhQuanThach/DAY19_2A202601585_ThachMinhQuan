# Phân Tích Ca Lỗi (Root-Cause Analysis) — Flat RAG vs GraphRAG

**Học viên:** Thạch Minh Quân · **MSSV:** 2A202601585
**Nguồn:** `outputs/graphrag_eval_results.csv`, `outputs/failure_analysis_table.csv`

---

## Bảng xếp hạng chênh lệch toàn bộ Golden Dataset

| id | group | Flat (TB) | Graph (TB) | Δ | seeds khớp | cạnh thu thập |
|---|---|---|---|---|---|---|
| G05 | cross-doc | **1.00** | **5.00** | **+4.00** | `Elon Musk[EXACT]`, `SpaceX[EXACT]` | 47 |
| G04 | cross-doc | **1.00** | 3.00 | +2.00 | `SpaceX[EXACT]`, `Starlink[EXACT]` | 82 |
| G03 | multi-hop | 4.33 | 5.00 | +0.67 | `Elon Musk[EXACT]`, `Starlink[EXACT]` | 84 |
| G01 | factoid | 5.00 | 5.00 | 0.00 | `audio video calling[EXACT]` | 2 |
| G02 | factoid | 5.00 | 5.00 | 0.00 | `X[EXACT]` | 23 |

Mẫu hình rõ: **chênh lệch tỉ lệ thuận với số chunk cần tổng hợp**. Câu factoid (1 cạnh) hoà tuyệt đối; câu cross-doc (13–16 chunk) chênh 2–4 điểm.

---

## Quy trình truy vết áp dụng

```
Câu trả lời sai
   ├─ 1. Bằng chứng có trong context không?  Có → lỗi generation. Không → đi tiếp
   ├─ 2. [Flat] chunk chứa bằng chứng có trong top-k không?   (flat_retrieved_chunks)
   ├─ 3. [Graph] seed có khớp entity không?                   (graph_matched_seeds)
   ├─ 4. [Graph] cạnh cần thiết có trong triples_df không?    (lỗi NER+RE / coref)
   ├─ 5. [Graph] có bị cắt bởi cap nào không?                 (supernode / global / char)
   └─ 6. Entity Resolution có làm phân mảnh hay gộp nhầm không?
```

---

## CA 1 — Flat RAG thất bại hoàn toàn, GraphRAG thành công (G05)

| Trường | Giá trị |
|---|---|
| Question ID | **G05** · nhóm `cross-doc` |
| Câu hỏi | *"Multiple articles mention both Elon Musk and SpaceX. Summarize what the corpus reports about their relationship across those articles, including the relationship types and publication dates."* |
| Reference | *"Across 13 distinct source chunks the corpus links Elon Musk to SpaceX via: FOUNDED, LEADS. Reported dates: 2023-01-12, 2023-07-27, …"* |
| Flat RAG | **1.00 / 1.00 / 1.00** |
| GraphRAG | **5.00 / 5.00 / 5.00** |

**Flat RAG trả lời:**
> *"The corpus reports that Elon Musk is the CEO and leader of SpaceX [chunk_id=136d721e…, 6f87cb5e…, def53852…]. The relationship is characterized by Musk's leadership…"*

**Judge chấm Flat RAG:**
> *"The candidate answer fails to address the specific constraints of the question, which asks for a summary of relationship types and publication dates across the corpus. Instead, it provides a narrative summary of business events."*

**GraphRAG trả lời:**
> *"…characterized by Musk founding, leading, and owning the company. The specific relationship types and publication dates are as follows: **FOUNDED** …, **LEADS** …"*

### Truy vết nguyên nhân gốc rễ

- **Bước 1 — retrieval hay generation?** Retrieval. Flat RAG lấy đúng chunk *nói về* Musk và SpaceX, nhưng không chunk nào chứa thứ câu hỏi cần.
- **Bước 2 — vì sao top-k không đủ?** Đây **không phải** lỗi ranking. Câu hỏi yêu cầu **loại quan hệ** (`FOUNDED`, `LEADS`) và **tập hợp ngày xuất bản** *tổng hợp qua 13 chunk*. Thông tin này **không tồn tại trong bất kỳ đoạn văn xuôi nào** — nó chỉ xuất hiện sau khi đã trích xuất và tổng hợp thành cấu trúc. Dù có tăng `k` lên bao nhiêu, Flat RAG cũng không thể có được nó, vì thứ nó cần đọc không nằm trong văn bản.
- **GraphRAG giải quyết ra sao:** cả hai seed khớp `EXACT`, BFS gom 47 cạnh, mỗi cạnh mang sẵn `type(r)` (nhãn quan hệ), `published_date` và `source_chunk_id`. Bước `textualize()` biến chúng thành dòng có cấu trúc — model chỉ cần đọc và nhóm lại, **không cần suy luận**.

> **Kết luận:** đây là ca cho thấy đúng bản chất khác biệt kiến trúc. Flat RAG truy hồi **văn bản**; GraphRAG truy hồi **quan hệ đã được vật chất hoá lúc index**. Câu hỏi nào cần chính cái cấu trúc đó thì Flat RAG thua bằng 0, không phải vì tìm kém mà vì thứ cần tìm chưa từng được tạo ra.

---

## CA 2 — GraphRAG bị suy giảm nghiêm trọng (G04)

| Trường | Giá trị |
|---|---|
| Question ID | **G04** · nhóm `cross-doc` |
| Câu hỏi | *"Multiple articles mention both SpaceX and Starlink. Summarize … relationship types and publication dates."* |
| Reference | *"Across **16** distinct source chunks … via: DEVELOPED. Reported dates: 2023-01-12, 2023-07-27, 2023-10-07, 2023-10-11, …"* |
| GraphRAG | Comprehensiveness **3** · Faithfulness 5 · Multi-hop 4 → TB **3.00** |
| Seeds | `SpaceX[EXACT]`, `Starlink[EXACT]` — khớp tốt |
| `graph_collected_edges` | **82** |
| `supernode_events` / `global_cap_hit` | 0 / `False` |

**Judge chấm GraphRAG:**
> *"The candidate answer correctly identifies the primary relationship type (DEVELOPED) and lists several publication dates present in the context. **However, it is significantly less comprehensive than the reference answer, listing only** [một phần] **dates."*

### Truy vết nguyên nhân gốc rễ

Loại trừ lần lượt:

- [ ] Seed extraction fail → **không**, cả hai seed khớp `EXACT`.
- [ ] Missing edge do NER+RE → **không**, đồ thị có đủ 16 chunk cho cặp này (chính reference sinh ra từ đó).
- [ ] Super-node cap → **không**, `supernode_events = 0` (degree SpaceX 31 < ngưỡng 100).
- [ ] `GLOBAL_EDGE_CAP` → **không**, `global_cap_hit = False` (82 < 250).
- [x] **`MAX_GRAPH_CONTEXT_CHARS` cắt mất cạnh** → **ĐÚNG**.

**Bằng chứng định lượng** (đo trực tiếp trên cột `graph_context`):

| id | Cạnh thu thập | Cạnh vào được context | Mất | Độ dài phần GRAPH |
|---|---|---|---|---|
| G05 | 47 | 47 | 0 | 7 781 |
| **G04** | **82** | **47** | **35 (43%)** | **7 911** |
| **G03** | **84** | **47** | **37 (44%)** | **7 843** |
| G02 | 23 | 23 | 0 | 3 515 |
| G01 | 2 | 2 | 0 | 373 |

Hàm `textualize()` dừng khi chạm `MAX_GRAPH_CONTEXT_CHARS = 8000`. G04 và G03 đều bị cắt xuống đúng 47 dòng vì độ dài trung bình mỗi dòng cạnh ~168 ký tự.

**Nguyên nhân gốc thật sự là một quyết định của chính tôi:** tôi hạ `MAX_GRAPH_CONTEXT_CHARS` từ 14 000 xuống **8 000** để một request nằm gọn dưới trần TPM 8 000 token/phút của Groq free tier. Ràng buộc hạ tầng đã trực tiếp gây mất 43% bằng chứng, và biểu hiện ra ngoài thành điểm Comprehensiveness thấp.

Đối chiếu G05 (47 cạnh, không bị cắt → 5.00) với G04 (82 cạnh, bị cắt 43% → 3.00) cho thấy quan hệ nhân quả rất rõ: **cùng loại câu hỏi, cùng chất lượng seed, khác nhau ở chỗ có bị cắt hay không**.

**Điểm nguy hiểm:** việc cắt xảy ra **âm thầm**. `textualize()` chỉ `break` khi hết chỗ, không phát cảnh báo nào và cũng không ghi vào `diagnostics`. Nhìn từ log thì câu này trông "thành công" — chỉ khi đối chiếu `graph_collected_edges` với số dòng thật trong context mới lộ ra.

### Yếu tố thứ hai: reference answer quá khắt khe

Reference của nhóm `cross-doc` được sinh tự động và liệt kê **toàn bộ** ngày của cả 16 chunk. Một câu trả lời tốt của con người cũng khó liệt kê đủ. Đây là hạn chế của cách sinh golden tự động, và nó khiến điểm Comprehensiveness của nhóm cross-doc bị hạ một cách hệ thống.

### Đề xuất khắc phục (theo thứ tự ưu tiên)

| Đề xuất | Chi phí | Rủi ro |
|---|---|---|
| **Ghi cảnh báo khi `textualize()` cắt cạnh** — thêm `truncated_edges` vào `diagnostics` | ~0 | Không có. Nên làm đầu tiên: lỗi âm thầm nguy hiểm hơn lỗi ồn ào |
| **Gộp cạnh trùng trước khi textualize** — 16 chunk cùng nói `SpaceX -DEVELOPED-> Starlink` có thể nén thành 1 dòng kèm danh sách ngày/chunk | ~0 | Mất chi tiết `evidence` riêng của từng cạnh |
| Nâng `MAX_GRAPH_CONTEXT_CHARS` về 14 000 | Cần TPM cao hơn (tier trả phí) | Vượt trần → 429 vô hạn |
| Xếp hạng cạnh theo độ liên quan tới truy vấn thay vì thuần recency | Vừa | Phức tạp hoá retrieval |

**Đề xuất số 2 là đáng giá nhất**: với đồ thị này, 82 cạnh của G04 chỉ tương ứng vài quan hệ *riêng biệt* lặp lại qua nhiều chunk. Nén lại thì vừa đủ chỗ trong 8 000 ký tự vừa trả lời trọn vẹn câu hỏi.

---

## CA 3 (phụ) — Cả hai hoà nhau nhưng GraphRAG lãng phí (G01, G02)

Ở hai câu `factoid`, cả hai đạt **5.00/5.00** nhưng GraphRAG tốn **1 505 token** so với **864 token** của Flat RAG (1.74×). Với G02 (`"Who leads X?"`), GraphRAG kéo về 23 cạnh trong khi chỉ cần đúng 1 cạnh `Elon Musk -LEADS-> X`.

**Kết luận vận hành:** luôn chạy hybrid là lãng phí. Nên có **query router** phân loại câu hỏi trước — factoid thì đi Flat RAG, multi-hop/cross-doc mới kích hoạt graph traversal. Ước tính trên phân bố Golden Dataset này (2/5 câu là factoid), router sẽ tiết kiệm ~27% token tổng mà không mất điểm chất lượng nào.
