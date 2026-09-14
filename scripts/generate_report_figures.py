import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = r"D:\Coding\School\Y3-K3\Text Mining\Text-Mining---NewsQA-RAG\docs\latex\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#D6DDE3'
plt.rcParams['text.color'] = '#1F2933'

# 1. GENERATE PIPELINE DIAGRAM
def generate_pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background header
    ax.text(50, 96, "KIẾN TRÚC PIPELINE RAG — HỆ THỐNG NEWSLENS", 
            ha='center', va='top', fontsize=14, fontweight='bold', color='#003F88')
    ax.text(50, 91, "Quy trình khép kín: Tiền xử lý -> Truy xuất BGE-M3 Sparse -> Reranker -> Generator Prompt P2 -> Xác thực Citation", 
            ha='center', va='top', fontsize=9, color='#52606D')

    # Offline Ingestion Box
    rect_offline = patches.FancyBboxPatch((2, 58), 96, 28, boxstyle="round,pad=1.5", 
                                          fc='#F8FAFC', ec='#94A3B8', ls='--', lw=1.5)
    ax.add_patch(rect_offline)
    ax.text(4, 84, "TẦNG CHUẨN BỊ DỮ LIỆU & CHỈ MỤC (OFFLINE INGESTION & INDEXING)", 
            fontsize=9, fontweight='bold', color='#475569')

    # Cards in Offline Box
    boxes_offline = [
        (5, 60, 26, 20, "1. NewsQA Corpus", "11.064 bài báo CNN\n200 bài evaluation\n10.864 distractors\n1.152 câu hỏi resolved", '#FFFFFF', '#D6DDE3'),
        (35, 60, 26, 20, "2. Phân đoạn (Chunking)", "Recursive Splitter\nChunk size: 512 tokens\nChunk overlap: 64 tokens\nTổng: 22.766 chunks", '#FFFFFF', '#D6DDE3'),
        (65, 60, 31, 20, "3. Chỉ mục kép (Dual Index)", "• Chỉ mục thưa: BM25S (Lexical)\n• Chỉ mục vector: Chroma DB\n(Dense Embeddings: BGE-M3)", '#F0F7FA', '#1D6F8C')
    ]
    for x, y, w, h, title, desc, fc, ec in boxes_offline:
        b = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8", fc=fc, ec=ec, lw=1.2)
        ax.add_patch(b)
        ax.text(x+1.5, y+h-2, title, fontsize=8.5, fontweight='bold', color='#1F2933', va='top')
        ax.text(x+1.5, y+h-5.5, desc, fontsize=7.5, color='#52606D', va='top', linespacing=1.3)

    # Online Stages (5 columns)
    col_w = 17.5
    gap = 2.0
    start_x = 2.0
    stages = [
        ("BƯỚC 1: TRUY VẤN", "Người dùng & Query", "Giao diện Web / API\nChuẩn hóa Standalone\n(Resolved question format)", '#FFFFFF', '#D6DDE3'),
        ("BƯỚC 2: TRUY XUẤT", "BGE-M3 Sparse", "Tìm trên 22.766 chunks\nThắng Dense +0,1634 nDCG\nTop-20 ứng viên", '#F2F9F5', '#1F7A4D'),
        ("BƯỚC 3: RERANKING", "bge-reranker-large", "Cross-Encoder scoring\nTăng +0,0659 nDCG@5\nTop-5 chunks bằng chứng", '#F2F9F5', '#1F7A4D'),
        ("BƯỚC 4: SINH ĐÁP ÁN", "Prompt P2 + Gemini", "Context [1]..[5]\nChỉ một câu đúng trọng tâm\nKèm trích dẫn [n]", '#F2F9F5', '#1F7A4D'),
        ("BƯỚC 5: XÁC THỰC", "Citation & SSE Stream", "Kiểm tra số [n] hợp lệ\nÁnh xạ bài báo, URL, ngày\nTrả về Streaming UI", '#F0F7FA', '#1D6F8C')
    ]

    for i, (badge, title, desc, fc, ec) in enumerate(stages):
        x = start_x + i * (col_w + gap)
        # Stage badge
        badge_box = patches.FancyBboxPatch((x, 48), col_w, 4.5, boxstyle="round,pad=0.4", fc='#003F88', ec='#003F88')
        ax.add_patch(badge_box)
        ax.text(x + col_w/2, 50.2, badge, ha='center', va='center', fontsize=7.5, fontweight='bold', color='white')

        # Card
        card = patches.FancyBboxPatch((x, 10), col_w, 36, boxstyle="round,pad=0.8", fc=fc, ec=ec, lw=1.5)
        ax.add_patch(card)
        ax.text(x + 1.2, 43, title, fontsize=8.5, fontweight='bold', color='#1F2933', va='top')
        ax.text(x + 1.2, 39, desc, fontsize=7.5, color='#52606D', va='top', linespacing=1.4)

        # Draw arrow to next stage
        if i < 4:
            ax.annotate('', xy=(x + col_w + gap, 28), xytext=(x + col_w, 28),
                        arrowprops=dict(arrowstyle="-|>", color='#1D6F8C', lw=2, mutation_scale=12))

    # Bottom summary bar
    bar = patches.FancyBboxPatch((2, 1.5), 96, 5.5, boxstyle="round,pad=0.5", fc='#EDF2F7', ec='#D6DDE3')
    ax.add_patch(bar)
    ax.text(4, 4.25, "Đánh giá Held-out: Answer Correctness = 0,7157 (284 câu kiểm định một lần) | Winner khóa: Sparse + Reranker + Prompt P2", 
            va='center', fontsize=8, fontweight='bold', color='#1F2933')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pipeline.png"), dpi=300)
    plt.close()
    print("Generated pipeline.png")

# 2. GENERATE CHAT INTERFACE MOCKUP
def generate_chat_mockup():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Browser window outline
    window = patches.FancyBboxPatch((2, 2), 96, 96, boxstyle="round,pad=1.0", fc='#FFFFFF', ec='#CBD5E1', lw=1.5)
    ax.add_patch(window)

    # Top title bar
    topbar = patches.FancyBboxPatch((2, 91), 96, 7, boxstyle="round,pad=0.5", fc='#1E293B', ec='#1E293B')
    ax.add_patch(topbar)
    # Circle buttons
    for i, c in enumerate(['#EF4444', '#F59E0B', '#10B981']):
        circle = plt.Circle((5 + i*2.5, 94.5), 0.8, color=c)
        ax.add_patch(circle)
    ax.text(50, 94.5, "NewsLens — Hỏi đáp Tin tức Thông minh & Trích dẫn Nguồn", 
            ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # Sub-header
    ax.text(6, 86, "NEWS CHAT", fontsize=11, fontweight='bold', color='#1F2933')
    ax.text(92, 86, "CLEAR CHAT", fontsize=8, color='#64748B', ha='right')
    line = plt.Line2D([4, 96], [83, 83], color='#E2E8F0', lw=1)
    ax.add_line(line)

    # User bubble
    user_bubble = patches.FancyBboxPatch((40, 68), 54, 11, boxstyle="round,pad=0.8", fc='#003F88', ec='#003F88')
    ax.add_patch(user_bubble)
    ax.text(42, 75.5, "User:", fontsize=8, fontweight='bold', color='#93C5FD')
    ax.text(42, 71.5, "When was Pandher sentenced to death in the Nithari serial murders case?", 
            fontsize=8, color='white')

    # Assistant bubble
    assistant_bubble = patches.FancyBboxPatch((6, 32), 80, 32, boxstyle="round,pad=0.8", fc='#F8FAFC', ec='#E2E8F0', lw=1.2)
    ax.add_patch(assistant_bubble)
    ax.text(8, 60, "NewsLens Agent [Gemini 3.1 Flash-Lite | BGE-M3 Sparse + Reranker]:", 
            fontsize=8, fontweight='bold', color='#1D6F8C')
    ax.text(8, 55, "Pandher was sentenced to death in February 2009 [1].", 
            fontsize=9.5, fontweight='bold', color='#1F2933')

    # Source citation card inside assistant bubble
    source_box = patches.FancyBboxPatch((8, 35), 76, 17, boxstyle="round,pad=0.6", fc='#FFFFFF', ec='#CBD5E1', lw=1)
    ax.add_patch(source_box)
    ax.text(10, 48.5, "▲ Nguồn trích dẫn [1]: CNN News Article · Feb 13, 2009", 
            fontsize=8, fontweight='bold', color='#003F88')
    ax.text(10, 44.5, "\"...An Indian court on Friday sentenced businessman Moninder Singh Pandher and his domestic servant...", 
            fontsize=7.5, style='italic', color='#475569')
    ax.text(10, 40.5, "to death for their role in the sensational Nithari serial murders case in February...\"", 
            fontsize=7.5, style='italic', color='#475569')
    ax.text(10, 37, "URL: http://edition.cnn.com/2009/WORLD/asiapcf/02/13/india.killings/index.html", 
            fontsize=7, color='#2563EB')

    # Chat Input bar at bottom
    input_box = patches.FancyBboxPatch((6, 8), 75, 8, boxstyle="round,pad=0.6", fc='#FFFFFF', ec='#CBD5E1')
    ax.add_patch(input_box)
    ax.text(9, 12, "Nhập câu hỏi về sự kiện, nhân vật, thời gian tin tức...", fontsize=8, color='#94A3B8')

    send_btn = patches.FancyBboxPatch((83, 8), 11, 8, boxstyle="round,pad=0.6", fc='#003F88', ec='#003F88')
    ax.add_patch(send_btn)
    ax.text(88.5, 12, "GỬI", ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chat_interface.png"), dpi=300)
    plt.close()
    print("Generated chat_interface.png")

# 3. GENERATE RETRIEVAL PLAYGROUND MOCKUP
def generate_retrieval_mockup():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    window = patches.FancyBboxPatch((2, 2), 96, 96, boxstyle="round,pad=1.0", fc='#FFFFFF', ec='#CBD5E1', lw=1.5)
    ax.add_patch(window)

    topbar = patches.FancyBboxPatch((2, 91), 96, 7, boxstyle="round,pad=0.5", fc='#1E293B', ec='#1E293B')
    ax.add_patch(topbar)
    ax.text(50, 94.5, "NewsLens — Retrieval Playground & Inspector", 
            ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    ax.text(6, 86, "RETRIEVAL PLAYGROUND", fontsize=11, fontweight='bold', color='#1F2933')
    
    # Stats badge
    stats = patches.FancyBboxPatch((6, 77), 88, 6.5, boxstyle="round,pad=0.5", fc='#F1F5F9', ec='#CBD5E1')
    ax.add_patch(stats)
    ax.text(8, 80.2, "Collection: newsqa_512_64 — 22.766 chunks indexed | Embedding: BGE-M3 | Index: BM25S + Chroma", 
            fontsize=8, color='#334155', fontweight='bold')

    # Query Input & algorithm selection
    ax.text(6, 72, "Truy vấn kiểm thử:", fontsize=8, fontweight='bold', color='#1F2933')
    q_box = patches.FancyBboxPatch((6, 63), 56, 7, boxstyle="round,pad=0.5", fc='#FFFFFF', ec='#94A3B8')
    ax.add_patch(q_box)
    ax.text(8, 66.5, "Pandher death sentence Nithari serial murders", fontsize=8, color='#1F2933')

    algo_box = patches.FancyBboxPatch((64, 63), 20, 7, boxstyle="round,pad=0.5", fc='#F0F7FA', ec='#1D6F8C')
    ax.add_patch(algo_box)
    ax.text(74, 66.5, "Thuật toán: LOCKED (Sparse+Rerank)", ha='center', fontsize=7.5, fontweight='bold', color='#1D6F8C')

    search_btn = patches.FancyBboxPatch((86, 63), 8, 7, boxstyle="round,pad=0.5", fc='#1F7A4D', ec='#1F7A4D')
    ax.add_patch(search_btn)
    ax.text(90, 66.5, "TÌM", ha='center', fontsize=8, fontweight='bold', color='white')

    # Timing Bar
    ax.text(6, 58, "Kết quả truy xuất: 5 chunks (Độ trễ: Sparse 12ms + Reranker 498ms = Tổng 510ms)", 
            fontsize=8, style='italic', color='#475569')

    # Result Item 1 (Rank 1 - Gold Chunk)
    card1 = patches.FancyBboxPatch((6, 32), 88, 23, boxstyle="round,pad=0.6", fc='#F2F9F5', ec='#1F7A4D', lw=1.5)
    ax.add_patch(card1)
    ax.text(8, 51.5, "#1  Score: 0.9842 (Reranker) | Chunk ID: c_724f_001 | Trạng thái: [GOLD EVIDENCE]", 
            fontsize=8, fontweight='bold', color='#1F7A4D')
    ax.text(8, 47, "Bài báo: Businessmen sentenced to death in India serial killings case (CNN)", 
            fontsize=7.5, fontweight='bold', color='#1F2933')
    ax.text(8, 43, "\"...NEW DELHI, India (CNN) -- An Indian court on Friday sentenced businessman Moninder Singh Pandher...", 
            fontsize=7.5, color='#334155')
    ax.text(8, 39, "and his domestic servant Surender Koli to death for their role in the sensational Nithari serial murders...\"", 
            fontsize=7.5, color='#334155')
    ax.text(8, 35, "Vị trí ký tự: [261 - 270] -> Khớp nhãn bằng chứng: 'February.'", fontsize=7.5, fontweight='bold', color='#003F88')

    # Result Item 2
    card2 = patches.FancyBboxPatch((6, 9), 88, 20, boxstyle="round,pad=0.6", fc='#FFFFFF', ec='#E2E8F0', lw=1)
    ax.add_patch(card2)
    ax.text(8, 25.5, "#2  Score: 0.7410 (Reranker) | Chunk ID: c_724f_002", fontsize=8, fontweight='bold', color='#475569')
    ax.text(8, 21.5, "\"...The special Central Bureau of Investigation court in Ghaziabad near New Delhi handed down the verdict...", 
            fontsize=7.5, color='#64748B')
    ax.text(8, 17.5, "after convicting both men on charges of murder, rape and criminal conspiracy...\"", 
            fontsize=7.5, color='#64748B')
    ax.text(8, 13, "Từ khóa khớp: 'court', 'Ghaziabad', 'verdict', 'convicting', 'murder'", fontsize=7, color='#64748B')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "retrieval_playground.png"), dpi=300)
    plt.close()
    print("Generated retrieval_playground.png")

# 4. GENERATE EVALUATION DASHBOARD MOCKUP
def generate_eval_mockup():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    window = patches.FancyBboxPatch((2, 2), 96, 96, boxstyle="round,pad=1.0", fc='#FFFFFF', ec='#CBD5E1', lw=1.5)
    ax.add_patch(window)

    topbar = patches.FancyBboxPatch((2, 91), 96, 7, boxstyle="round,pad=0.5", fc='#1E293B', ec='#1E293B')
    ax.add_patch(topbar)
    ax.text(50, 94.5, "NewsLens — Evaluation Desk & Experiment Dashboard", 
            ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    ax.text(6, 86, "EVALUATION DESK", fontsize=11, fontweight='bold', color='#1F2933')

    # Experiment selector box
    sel_box = patches.FancyBboxPatch((6, 74), 88, 9, boxstyle="round,pad=0.6", fc='#F8FAFC', ec='#CBD5E1')
    ax.add_patch(sel_box)
    ax.text(8, 79.5, "Thí nghiệm đã chọn: phase2_generation_comparison.yaml", fontsize=8.5, fontweight='bold', color='#003F88')
    ax.text(8, 76, "Tập dữ liệu: Development (281 câu hỏi / 50 bài báo) | Metric chính: Answer Correctness | Trạng thái: COMPLETE", 
            fontsize=7.5, color='#475569')

    # Metric Cards Row
    metrics = [
        ("Answer Correctness", "0,8015", "+0,1043 so với P0", '#F2F9F5', '#1F7A4D'),
        ("Faithfulness", "0,9603", "Vượt guardrail (-0,02)", '#F2F9F5', '#1F7A4D'),
        ("Citation F1", "0,8441", "Vượt guardrail (-0,01)", '#F2F9F5', '#1F7A4D'),
        ("Citation Validity", "0,9786", "Vượt guardrail (-0,01)", '#F2F9F5', '#1F7A4D')
    ]
    card_w = 20.5
    for i, (m_title, m_val, m_sub, fc, tc) in enumerate(metrics):
        cx = 6 + i * (card_w + 2.0)
        c_box = patches.FancyBboxPatch((cx, 55), card_w, 16, boxstyle="round,pad=0.6", fc=fc, ec='#D6DDE3', lw=1.2)
        ax.add_patch(c_box)
        ax.text(cx+1.5, 68, m_title, fontsize=7.5, fontweight='bold', color='#1F2933')
        ax.text(cx+1.5, 62, m_val, fontsize=13, fontweight='bold', color=tc)
        ax.text(cx+1.5, 57.5, m_sub, fontsize=6.8, color='#52606D')

    # Chart Section: Comparison across Prompts
    chart_bg = patches.FancyBboxPatch((6, 10), 54, 42, boxstyle="round,pad=0.6", fc='#FFFFFF', ec='#CBD5E1')
    ax.add_patch(chart_bg)
    ax.text(8, 48.5, "Biểu đồ so sánh Answer Correctness theo Prompt (Development)", fontsize=8, fontweight='bold', color='#1F2933')

    # Bar chart inside
    prompts = ['P0-depth5\n(Baseline)', 'P1-depth5\n(Grounding)', 'P2-depth3\n(Trượt Guard)', 'P2-depth5\n(WINNER)', 'P3-depth5\n(Disambig)']
    scores = [0.6308, 0.6120, 0.8375, 0.8015, 0.6480]
    bar_x = [11, 21, 31, 41, 51]
    for px, sc, pr in zip(bar_x, scores, prompts):
        h = (sc - 0.5) * 60  # scaled
        color = '#1F7A4D' if 'WINNER' in pr else ('#B3261E' if 'Trượt' in pr else '#003F88')
        bar = patches.Rectangle((px - 3, 18), 6, h, fc=color, ec=color)
        ax.add_patch(bar)
        ax.text(px, 18 + h + 1, f"{sc:.4f}", ha='center', fontsize=6.8, fontweight='bold', color=color)
        ax.text(px, 14.5, pr.split('\n')[0], ha='center', fontsize=6.8, color='#334155')

    # Failure Analysis Panel on right
    fail_bg = patches.FancyBboxPatch((62, 10), 32, 42, boxstyle="round,pad=0.6", fc='#F8FAFC', ec='#CBD5E1')
    ax.add_patch(fail_bg)
    ax.text(64, 48.5, "Phân tích thất bại (Failure Analysis)", fontsize=8, fontweight='bold', color='#1F2933')
    fail_items = [
        ("Retrieval Miss (Hit@5 fail):", "12 câu (4,27%)"),
        ("Generator Hallucination:", "5 câu (1,78%)"),
        ("Citation Index Out-of-bounds:", "6 câu (2,14%)"),
        ("Format / Over-length:", "8 câu (2,85%)"),
        ("Semantic Match (Judge Disagree):", "18/30 câu audit")
    ]
    for j, (k, v) in enumerate(fail_items):
        ax.text(64, 44 - j*6.5, k, fontsize=7, color='#475569')
        ax.text(64, 41 - j*6.5, v, fontsize=7.5, fontweight='bold', color='#003F88')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_dashboard.png"), dpi=300)
    plt.close()
    print("Generated evaluation_dashboard.png")

if __name__ == '__main__':
    generate_pipeline_figure()
    generate_chat_mockup()
    generate_retrieval_mockup()
    generate_eval_mockup()
    print("All figures generated successfully!")
