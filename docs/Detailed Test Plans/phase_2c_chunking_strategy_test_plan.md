# Kế Hoạch Chi Tiết - Giai Đoạn 2C: Ablation Chiến Lược Chunking

## 1. Mục tiêu và phạm vi

Phase 2C đánh giá ảnh hưởng của **chiến lược chia đoạn và cách dựng context**
đến retrieval lẫn câu trả lời cuối của RAG. Đây là controlled ablation: toàn bộ
retriever, reranker, prompt, generator và judge đã khóa từ Phase 1/2B; chỉ thay
đổi cách biến một bài báo thành các đơn vị được tìm kiếm và đưa vào generator.

Câu hỏi nghiên cứu chính:

1. Giữ trọn câu hoặc đoạn văn có tăng khả năng chứa đầy đủ evidence không?
2. Retrieve một đoạn nhỏ rồi mở rộng sang parent có cân bằng được retrieval
   precision và context completeness không?
3. Chiến lược nào tăng Answer Correctness mà không làm giảm Faithfulness,
   Citation F1 và Citation Validity?
4. Mức cải thiện có đủ lớn so với chi phí index, token và latency phát sinh?

Phase 2C **không thay thế hoặc xóa** ablation kích thước chunk của Phase 1.
Kết quả `256/32`, `512/64`, `1024/128` được giữ như bằng chứng hỗ trợ cho việc
chọn target size `512/64`; Phase 2C kiểm tra một trục khác là cấu trúc chunk.

Ngoài phạm vi:

- thay embedding/retrieval model hoặc reranker;
- sửa prompt P2, context depth, temperature hoặc generator model;
- query rewriting và Agentic RAG;
- supervised fine-tuning model;
- dùng câu hỏi abstention để chọn chunker;
- chọn lại cấu hình dựa trên 284 câu Phase 2 held-out đã mở.

## 2. Bối cảnh và cấu hình đã khóa

### 2.1. Kết quả đầu vào

Phase 1 đã khóa recursive `512/64`, BGE-M3 learned sparse và BGE-large
reranker. Trên 871 câu resolved thuộc 150 bài final-test, cấu hình này đạt:

| Hit@1 | Hit@5 | MRR@5 | NDCG@5 | Recall@5 | P50 |
|---:|---:|---:|---:|---:|---:|
| 0,7543 | 0,8978 | 0,8112 | 0,8313 | 0,8955 | 550,0 ms |

Phase 2B đã khóa generator configuration:

| Thành phần | Giá trị |
|---|---|
| Prompt | `P2` - trả lời trực tiếp, ngắn, đúng loại thông tin được hỏi |
| Context depth | `5` |
| Generator | `gemini-3.1-flash-lite` |
| Generator reasoning | `minimal` |
| Temperature / max output | `0` / `512` tokens |
| Judge | `accounts/fireworks/models/glm-5p3-flash` |
| Judge reasoning | `low` |

P2-D5 đạt trên 284 câu Phase 2 held-out: Answer Correctness `0,7157`,
Faithfulness `0,9249`, Citation F1 `0,7289` và Citation Validity `0,9437`.
Đây là mốc đối chiếu tuyệt đối, không phải dữ liệu để tiếp tục chọn chiến lược.

### 2.2. Hợp đồng dữ liệu

| Thuộc tính | Giá trị khóa |
|---|---|
| Raw dataset | `MatchaMacchiato/newsqa_200_11064_v2.0.0` |
| Raw revision | `b81c8db6847a23272665946c0c43c72e9a212fd9` |
| Corpus | 11.064 bài báo |
| Primary set | 1.152 semantic-deduplicated resolved questions |
| Development | 50 bài, 281 câu, seed `42` |
| Phase 2 held-out đã mở | 50 bài, 284 câu, seed `46` - không dùng trong 2C |
| Generation reserve | 100 bài, 587 câu |

Không sửa article text, resolved question, accepted answers hoặc source evidence
spans. Mỗi strategy phải sinh lại chunk IDs và ánh xạ evidence từ cùng character
span `[start, end)` trong bài báo.

Tập 587 câu chưa được dùng cho generation hoặc chọn P2-D5, nhưng đã nằm trong
held-out retrieval aggregate của Phase 1. Vì vậy nó là **generation/context
reserve**, không được mô tả là một retrieval test hoàn toàn chưa từng truy cập.
Trong Phase 2C, mọi quyết định vẫn phải được khóa trước khi chạy tập này.

## 3. Ma trận chiến lược

Target size giữ ở `512` tokens để tránh trộn lại ablation kích thước chunk.

| ID | Chiến lược | Đơn vị retrieve/rerank | Context đưa cho generator |
|---|---|---|---|
| `C0` | Recursive `512/64` | Chunk 512 | Chính chunk đã retrieve |
| `C1` | Sentence-aware | Nhóm câu, tối đa 512 | Chính nhóm câu đã retrieve |
| `C2` | Paragraph-aware | Nhóm đoạn, tối đa 512 | Chính nhóm đoạn đã retrieve |
| `C3` | Hierarchical | Child `256/32` | Parent `512/64` chứa child |

### 3.1. Recursive control (`C0`)

Đây là cấu hình hiện hành và là control. Tái sử dụng artifact/index và output
P2-D5 khi hash của question, context order và prompt khớp; không gọi API lại chỉ
để tạo một bản sao baseline.

### 3.2. Sentence-aware (`C1`)

- Tách câu bằng sentence segmenter có phiên bản được pin.
- Ghép các câu liên tiếp đến tối đa 512 tokens.
- Overlap theo câu hoàn chỉnh, hướng tới khoảng 64 tokens nhưng không cắt câu.
- Một câu dài hơn 512 tokens được recursive-split và phải được đánh dấu fallback.

### 3.3. Paragraph-aware (`C2`)

- Bảo toàn ranh giới paragraph từ article text.
- Ghép paragraph ngắn liên tiếp đến tối đa 512 tokens.
- Paragraph dài hơn giới hạn được recursive-split.
- Overlap bằng paragraph cuối khi không vượt budget; lưu số token overlap thực.

### 3.4. Hierarchical child-to-parent (`C3`)

- Parent: recursive `512/64`; child: recursive `256/32` bên trong parent.
- Chỉ child được index, retrieve và rerank.
- Sau rerank, mở rộng child sang exact parent text.
- Nếu nhiều child cùng parent, chỉ giữ parent có child score cao nhất.
- Giữ thứ tự theo best-child score và tiếp tục duyệt ranked children cho đến khi
  đủ context hợp lệ hoặc hết candidate pool.
- Citation `[i]` trỏ tới thứ tự parent thực sự được đưa cho generator, không trỏ
  tới vị trí child ban đầu.

Semantic chunking không nằm trong ma trận chính vì nó cần embedding model và
threshold riêng, tạo thêm biến gây nhiễu. Chỉ mở `C4` exploratory sau khi C0-C3
hoàn tất; không dùng kết quả exploratory để thay winner đã xác nhận.

## 4. Điều kiện kiểm soát công bằng

Mọi arm dùng chung:

| Thành phần | Giá trị |
|---|---|
| Question variant | Semantic-deduplicated `resolved` |
| Retriever | BGE-M3 learned sparse |
| Candidate pool | `top_k=20` |
| Reranker | `BAAI/bge-reranker-large` |
| Max contexts | `5` |
| Prompt/generator/judge | Cấu hình P2-D5 đã khóa |
| Seed | `42` |

`context_depth=5` chưa đủ để kiểm soát chi phí vì năm chunk của các strategy có
thể dài khác nhau. Phase 2C phải khóa thêm `max_context_tokens` theo context
budget của C0. Quy trình:

1. Đo phân phối token của năm context C0 trên 281 câu development.
2. Đăng ký budget trước khi chạy generation của arm mới.
3. Thêm context theo ranked order; không cắt giữa câu nếu strategy bảo toàn câu.
4. Ghi cả số context và token thực tế cho từng câu.

Nếu context kế tiếp vượt budget, bỏ context đó và thử context tiếp theo chỉ khi
việc này đã được đăng ký trước. Không điều chỉnh budget riêng cho từng arm.

## 5. Gold evidence và hai tầng đánh giá

Vì C3 retrieve child nhưng generator đọc parent, cần hai relevance mapping:

1. **Retrieval relevance:** chunk/child có character range giao với gold evidence.
2. **Delivered-context relevance:** context/parent thực sự đưa cho generator có
   character range giao với gold evidence.

Mỗi chunk lưu tối thiểu:

```json
{
  "id": "...",
  "text": "...",
  "metadata": {
    "article_id": "...",
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 512,
    "strategy": "hierarchical",
    "parent_id": "..."
  }
}
```

Quy tắc mapping:

- range dùng `[start, end)` và exact text phải khớp source article;
- báo cáo cả `overlap_any` và `fully_contains_evidence`;
- không dùng tìm kiếm fuzzy để che lỗi offset;
- mọi `relevant_chunk_id` phải tồn tại trong corpus tương ứng;
- citation score dùng context IDs sau expansion, không dùng child IDs ẩn;
- một parent xuất hiện tối đa một lần trong prompt.

## 6. Quy trình thực nghiệm

### 6.1. Phase 2C.0 - Implementation và preparation

1. Bổ sung `SentenceChunker` và `ParagraphChunker` vào chunker factory.
2. Hoàn thiện hierarchical artifact: lưu parent text/range và child-parent map.
3. Bổ sung context-expansion stage dùng chung cho benchmark và app.
4. Viết unit/integration tests cho boundary, offsets, IDs, dedup parent và budget.
5. Materialize C0-C3 từ cùng raw revision và human-reviewed dedup decisions.
6. Build một BGE-M3 index riêng cho mỗi strategy.
7. Lưu manifest, config, hash, environment, chunk statistics và build time.

Điều kiện qua preparation:

- 100% chunks là substring chính xác của article;
- 100% gold spans ánh xạ được tới ít nhất một chunk/context;
- không trùng chunk ID trong một variant;
- child-parent references không bị dangling;
- cùng đúng 1.152 question IDs ở cả bốn arm;
- smoke retrieval trả về ID thuộc đúng index/strategy.

### 6.2. Phase 2C.1 - Retrieval screening

Chạy C0-C3 trên toàn bộ 281 câu development, chưa gọi generator/judge. Đánh giá
trước và sau rerank. C0 được phép tái sử dụng trace chỉ khi fingerprint khớp.

Một arm bị loại nếu:

- có lỗi mapping/integrity;
- coverage nhỏ hơn 100%;
- Hit@5 hoặc Recall@5 thấp hơn C0 quá `0,02` và paired article-cluster bootstrap
  CI không chứa `0`;
- latency/index cost bất thường mà không có cải thiện quality tương ứng.

Ngưỡng `0,02` ở screening chỉ dùng loại arm yếu rõ ràng; với 281 câu, nó tương
đương khoảng 6 câu. Không dùng chênh `0,01` để loại sớm vì chỉ tương đương khoảng
3 câu và dễ bị sampling noise.

### 6.3. Phase 2C.2 - End-to-end screening

Chạy các arm còn lại trên cùng 80 câu screening đã khóa của Phase 2B:

- generation/deterministic metrics trên 80 câu;
- GLM judge trên cùng 20 câu judge-calibration;
- tái sử dụng C0 P2-D5 output khi fingerprint khớp;
- không thay câu hỏi hoặc subset sau khi xem output.

Vòng này dùng để phát hiện effect lớn và lỗi vận hành. Không tuyên bố winner chỉ
từ 20 judge samples và không áp dụng máy móc margin `0,01` trên mẫu nhỏ.

### 6.4. Phase 2C.3 - Development finalists

Chọn tối đa hai strategy mới cùng C0 và chạy đầy đủ 281 câu development. Mỗi arm
có retrieval, generation, deterministic scoring và full RAGAS/GLM scoring.

So sánh paired theo `question_id`, đồng thời phân tầng:

- gold nằm/không nằm trong retrieved top 5;
- gold nằm/không nằm trong delivered contexts;
- evidence chứa trọn trong một context hay bị chia;
- question type;
- độ dài article và evidence.

### 6.5. Phase 2C.4 - Final reserve confirmation

Sau khi ký `phase2c_winner_decision.json`, chạy winner đúng một lần trên 587 câu
thuộc 100 bài generation reserve. Không chạy các loser trên reserve và không sửa
strategy/budget sau khi xem kết quả.

Nếu final reserve kém development, vẫn báo cáo nguyên trạng và phân tích
generalization gap. Không quay lại chọn arm khác bằng reserve.

## 7. Metrics

### 7.1. Corpus và chunking

| Nhóm | Metrics |
|---|---|
| Quy mô | Tổng chunks, chunks/article, index bytes, build time |
| Độ dài | Mean/P50/P95/max tokens per chunk/context |
| Boundary | Tỷ lệ kết thúc ở sentence/paragraph boundary |
| Evidence | Any-overlap rate, full-containment rate, chunks per evidence |
| Redundancy | Overlap-token ratio, duplicate-context ratio |
| Budget | Context count và input tokens thực tế mỗi câu |

### 7.2. Retrieval

- Hit Rate, MRR, NDCG và Recall tại `k=1,3,5`;
- initial retrieval và post-rerank;
- gold-in-retrieved-child và gold-in-delivered-context;
- retrieve, rerank, expansion và total P50/P95 latency;
- reranker delta.

### 7.3. Generation và grounding

| Vai trò | Metrics |
|---|---|
| Primary | RAGAS Answer Correctness |
| Grounding | Faithfulness |
| Citation | Citation F1, Validity, Precision, Recall, Coverage |
| Supporting | QA EM/F1, Answer Relevancy, Context Precision/Recall |
| Vận hành | Coverage, retries, input/output tokens, cost, P50/P95 latency |

Không loại retrieval miss khỏi điểm tổng. Chỉ dùng subgroup để chẩn đoán vì
retrieval miss là lỗi end-to-end hợp lệ.

## 8. Phân tích thống kê

- Primary unit là question nhưng bootstrap theo **article cluster**, vì các câu
  trong cùng bài không độc lập.
- Báo cáo question-micro và article-macro.
- Với mỗi metric, báo cáo paired mean difference, effect size và CI95 từ ít
  nhất 2.000 bootstrap samples, seed `42`.
- Báo cáo absolute score cùng delta; không chỉ công bố phần trăm cải thiện.
- Không xem CI chồng lấp là bằng chứng equivalence.
- Multiple arms tạo multiple comparisons; kết luận chính dựa trên một primary
  metric và guardrails đã đăng ký, không chọn metric có lợi sau khi chạy.

## 9. Quy tắc chọn winner

Trên 281 câu development, một strategy chỉ hợp lệ khi so với C0:

1. Retrieval, generation và judge coverage đạt ngưỡng của C0; mục tiêu 100% sau
   retry/resume.
2. Hit@5 và Recall@5 không giảm quá `0,01`.
3. Faithfulness không giảm quá `0,02`.
4. Citation F1 không giảm quá `0,01`.
5. Citation Validity không giảm quá `0,01`.

Trong các strategy hợp lệ:

1. Chọn Answer Correctness cao nhất.
2. Nếu chênh Answer Correctness dưới `0,01`, chọn context input token thấp hơn.
3. Nếu token cost chênh dưới 5%, chọn end-to-end P95 thấp hơn.
4. Nếu vẫn hòa, chọn strategy đơn giản hơn và ít artifact/runtime dependency hơn.

Các margin là practical guardrail, không thay thế CI. `0,01` trên 281 câu tương
đương khoảng 3 câu; `0,02` tương đương khoảng 6 câu. Citation Validity dùng
margin chặt vì invalid citation trực tiếp làm người dùng không kiểm chứng được
nguồn; Faithfulness dùng `0,02` vì judge metric liên tục có độ nhiễu cao hơn.

## 10. Kiểm soát leakage và provenance

- Không dùng 284 Phase 2 held-out đã mở để chọn hoặc debug C0-C3.
- Không xem output 587 reserve trước khi winner được ký và hash.
- Exact code commit, raw revision, review/dedup hashes, tokenizer/segmenter
  version, chunk config, indexes, prompt và model parameters phải được lưu.
- Cache key bao gồm strategy, chunks hash, question, ranked contexts, expansion
  policy, context budget, prompt và model fingerprint.
- Successful records là append-only; duplicate successful record là lỗi.
- Raw generator/judge responses được giữ để audit.
- Retry chỉ xử lý transport/rate-limit/schema failure; không retry vì score thấp.
- Mọi thay đổi sau pre-registration tạo experiment version mới, không ghi đè.

## 11. Yêu cầu implementation trước notebook

Hiện tại repository mới hỗ trợ đầy đủ recursive và bước tạo child của
hierarchical. Trước khi chạy Phase 2C cần:

- thêm sentence-aware và paragraph-aware chunkers vào
  `common/newsqa_rag/ingestion/chunker.py`;
- sửa comment/config để chỉ quảng bá strategy thực sự được hỗ trợ;
- lưu parent corpus hoặc parent lookup có hash trong hierarchical artifact;
- thêm child-to-parent expansion và context deduplication;
- cập nhật scorer để chấm retrieval unit và delivered context riêng;
- bảo đảm citation builder dùng numbering của context sau expansion;
- mở rộng materialization/index scripts nhận strategy-specific config;
- thêm resumable runner và summary/paired-bootstrap exporter;
- chạy full `pytest` trước khi tạo artifact.

Test tối thiểu:

1. deterministic chunk IDs và output khi chạy lại;
2. token limit và boundary fallback;
3. exact character offsets, kể cả repeated text;
4. evidence nằm qua overlap/boundary;
5. parent expansion, ordering và duplicate removal;
6. context budget enforcement;
7. citation numbering sau expansion;
8. resume không tạo duplicate records;
9. manifest/hash mismatch phải fail-fast;
10. smoke end-to-end cho cả bốn arm.

## 12. Notebook và artifact đề xuất

| Notebook | Nền tảng | Trách nhiệm |
|---|---|---|
| `15a_phase_2c_0_preparation_kaggle.ipynb` | Kaggle GPU | Materialize C0-C3, validate, build indexes |
| `15b_phase_2c_1_retrieval_kaggle.ipynb` | Kaggle GPU | Retrieval screening 281 câu |
| `15c_phase_2c_2_e2e_screening_colab.ipynb` | Colab | E2E 80/20 cho các arm sống sót |
| `15d_phase_2c_3_finalists_colab.ipynb` | Colab/Kaggle | Full development 281 câu |
| `15e_phase_2c_4_reserve_final_kaggle.ipynb` | Kaggle | Winner trên 587 câu đúng một lần |

Mỗi strategy xuất:

```text
chunks.jsonl
parents.jsonl                 # C3
child_parent_map.jsonl        # C3
evidence_mapping.jsonl
testset_resolved.jsonl
config.yaml
variant_manifest.json
index_manifest.json
```

Mỗi experiment xuất:

```text
retrievals.jsonl
predictions.jsonl
attempts.jsonl
judge_results.jsonl
deterministic_scores.jsonl
report.json
environment.json
run_manifest.json
```

Tổng hợp Phase 2C gồm:

```text
phase2c_comparison.csv
phase2c_paired_bootstrap.json
phase2c_winner_decision.json
phase2c_final_report.md
phase2c_results_bundle.zip
```

Chỉ artifact của winner được publish như locked production artifact. Artifact
của các arm còn lại được giữ trong experiment archive để tái lập kết quả.

## 13. Điều kiện hoàn thành

- C0-C3 qua toàn bộ data-integrity tests và có reproducible manifests.
- Tất cả arm được so sánh paired trên cùng development/screening IDs.
- Winner được chọn theo đúng primary metric và guardrails đã đăng ký.
- Decision record được ký trước khi mở 587-question reserve.
- Final reserve được chạy một lần với coverage hợp lệ và báo cáo cả kết quả bất
  lợi nếu có.
- App và benchmark tải cùng winner artifact và thực hiện cùng expansion/budget
  policy.
- Báo cáo Phase 1 vẫn giữ ablation kích thước như supporting result; Phase 2C
  được báo cáo riêng như ablation chiến lược chunking end-to-end.
