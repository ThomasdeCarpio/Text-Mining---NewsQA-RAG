2. PRODUCT FEATURES & USER EXPERIENCE (UI/UX)

The application will have two distinct user roles: Standard User and Admin.

USER ROLES & INTERACTIONS:

1. Standard User (News Reader / Analyst)
- Authentication: Basic Log in / Log out functionality.
- Chat Interface: Users can type natural language questions.

  * CORE FUNCTIONALITY: Fact-Finding (Single-Source)
    - Example: "What was the reported unemployment rate according to the Labor Department article?"
    - Focus: Extracting a specific fact from a single document.
    - Evaluation: We will use the NewsQA dataset to benchmark the system's precision in finding a "needle in a haystack" within the entire corpus.

  * ADVANCED FUNCTIONALITY (Planned/Experimental): Multi-Source Synthesis & Comparison
    - Example 1 (Synthesis): "Summarize all latest reports regarding the Haiti earthquake from all available sources."
    - Example 2 (Comparison): "Contrast the reactions of the Wall Street Journal and the AP regarding the new jobs report."
    - Focus: Determining if the Agent can successfully retrieve multiple relevant documents and merge/compare their contents.
    - Evaluation: Pending successful modification of the Multi-News dataset. We will first establish a baseline using Single-Source Fact-Checking before implementing these complex multi-source logic loops.

- Source Transparency: Users can click on "Citations" or "Sources" to view the exact text chunks and article links the Agent used to generate the answer.
- Session Memory: Users can view their current chat history within the session.

2. Administrator (Evaluator / Developer)
- Authentication: Secure Admin login.
- Evaluation Dashboard: A dedicated page to view the system's performance metrics (Hit Rate, Faithfulness, Context Precision) visualized via charts.
- Pipeline Logs: A view to see the Agent's "Thought Process" (which tools it decided to use and what queries it sent to the database).
- Manual Crawler Trigger (Optional): A button to manually trigger the live crawler to fetch today's news and update the database.

REQUIRED UI ELEMENTS (STREAMLIT LAYOUT):

1. Navigation & Authentication (Sidebar)
- Login/Logout form.
- Page Router: Toggle between "News Chat" (Standard User) and "Evaluation Dashboard" (Admin only).
- Chat History list (Clear chat button).

2. Main Chat View (News Chat Page)
- Chat Message Bubbles: Alternating user questions and AI responses.
- Citation Expanders: Hidden drop-down boxes under the AI's response that say "View Sources". Clicking expands to show the retrieved chunks.
- Chat Input Bar: Fixed at the bottom for typing questions.
- Loading State/Spinner: Text that says "Agent is thinking... Searching database..." while the hybrid retrieval runs.

3. Admin Evaluation View (Dashboard Page)
- Metric Cards: Big numbers showing the overall system scores (e.g., MRR: 0.85, Faithfulness: 92%).
- Performance Graphs: Bar charts comparing the Vector Search performance vs. Hybrid Search performance.
- Failure Analysis Table: A data table showing questions where the AI failed to retrieve the correct evidence, helping the team debug.
