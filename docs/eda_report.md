# NewsQA 200/11064 — what the data actually looks like

**What this report is for.** Before we can say "our system scores X", we have to
know what the data underneath that score is like. If the data has problems, the
score means something different from what it looks like. So this report is not
"here is a description of the dataset" — it is **"here is what our results are
allowed to claim, and why"**.

Everything here comes from a script in `scripts/eda/`, and each script saves its
numbers to `scripts/eda/out/*.json`. `notebooks/06_dataset_eda.ipynb` runs them
in order with explanations. To reproduce:

```bash
python scripts/eda/01_profile.py        # basic counts, cleanliness, consistency
python scripts/eda/05_reason_codes.py   # what was wrong with the questions
python scripts/eda/06_near_duplicates.py
python scripts/eda/08_truncation_gap.py # ~14 min: how much text is missing
python scripts/eda/07_figures.py        # draws every chart below
```

Charts (300 DPI, saved to `docs/figures/eda/`): `fig1_truncation`,
`fig2_evidence_position`, `fig3_question_repair`, `fig4_retrieval_difficulty`,
`fig5_restoration`, `fig6_truncation_gap`.

---

## 0. Terms used in this report

Read this once and the rest will make sense. Terms are also re-explained where
they first appear.

### The data

| Term | What it means here |
|---|---|
| **Corpus** | All the articles the system can search through — 11,064 of them. |
| **Evaluation article** | One of the 200 articles that actually holds an answer to a question. |
| **Distractor** | One of the 10,864 articles that holds **no** answer. They exist to make the search realistic — a real system searches a big pile, not 200 hand-picked articles. |
| **Chunk** | Articles are cut into pieces before being searched. A chunk is one piece — here about 512 tokens with 64 tokens of overlap between neighbours. The system retrieves *chunks*, not whole articles. |
| **Token** | Roughly a word-piece. "unbelievable" might be 3 tokens. Models count tokens, not words. |
| **Gold chunk** | The chunk that contains the correct answer. "Gold" = the officially correct one, used to mark a retrieval right or wrong. |
| **Ground truth** | The official correct answer text for a question. |
| **Evidence span** | The exact position of the answer inside the article, stored as *character offsets* — e.g. characters 770 to 791. This is why we cannot casually edit article text: editing moves those positions. |
| **Variant** | A version of the question set. `original` = questions as NewsQA wrote them. `resolved` = the same questions after our review repaired them. (Two intermediate variants exist; see section 1.) |

### The search system

| Term | What it means here |
|---|---|
| **Retriever** | The component that finds candidate chunks for a question. |
| **Sparse retrieval** (BM25) | Matches on *words that literally appear* in both the question and the chunk. |
| **Dense retrieval** | Matches on *meaning* using embeddings — it can match "car" to "vehicle". |
| **Hybrid** | Both together, results merged. |
| **Reranker** | A slower, more accurate model that re-orders the top candidates the retriever returned. A cross-encoder reads the question and chunk together, which is why it is more accurate and more expensive. |
| **First-stage retrieval** | The fast first pass, before any reranking. |

### The measurements

| Term | What it means here |
|---|---|
| **IDF** / **rare term** | A score for how unusual a word is across the corpus. A "rare term" here is a word appearing in at most ~46 of 19,263 chunks. Fully explained with examples in section 5. |
| **Median** | The middle value. Half the items are below it, half above. More honest than an average when a few extreme values would drag the average around. |
| **Quartile** | A quarter of the data, sorted by size. "The longest quartile" = the longest 25% of articles. |
| **p90** | The 90th percentile — 90% of cases are at or below this value. Shows the bad-but-not-freak case. |
| **Precision / recall** | For a rough test: **precision** = of the things it flagged, how many were really right. **Recall** = of the things that were really right, how many it caught. A test can be good at one and bad at the other. |
| **Jaccard similarity** | How much two sets of words overlap, from 0 (nothing shared) to 1 (identical). 0.70 means they share about 70% of their words. |
| **Baseline** | A deliberately dumb comparison, to prove a result is real. Here: comparing a question against a *random* chunk instead of its correct one. |
| **False negative** | The system found a correct answer, but our scoring marked it wrong because it wasn't the officially labelled one. |
| **Closed-world assumption** | The scoring pretends exactly one chunk is correct and everything else is wrong. Section 7 measures how often this is false. |
| **Hit@K / MRR@K** | Scoring measures. **Hit@K** = was a correct chunk anywhere in the top K results. **MRR@K** = how high up it was (higher is better, being 1st beats being 5th). |

### Other words that appear

| Term | What it means here |
|---|---|
| **Truncation** | Text being cut off partway through. Section 3. |
| **Prefix** | Text A is a prefix of text B if B starts with exactly A and then continues. If the benchmark article is a prefix of the real article, the benchmark version was cut off. |
| **Page furniture** (or *chrome*) | Bits of a web page that are not the article: "E-mail to a friend", video captions, copyright footers. Our text extraction accidentally kept some of these. |
| **Provenance** | Where the data came from and whether it was altered on the way. |
| **Byte-identical** | Exactly the same, character for character, with no difference at all. |
| **Standalone question** | A question that can be understood on its own. *"Who won the Gabon election?"* is standalone; *"Who won it?"* is not. |
| **Coreference** | A word pointing at something mentioned earlier — "it", "he", "the company". "Unresolved coreference" means the thing being pointed at was never stated. |

---

## 1. What the dataset is, and where it came from

| | |
|---|---|
| Source | `lucadiliello/newsqa` on Hugging Face, version `728e5292…` |
| Evaluation articles (contain answers) | 200 |
| Distractor articles (contain none) | 10,864 |
| Corpus total | 11,064 articles → **19,263 chunks** |
| Questions | 1,340 selected → 1,336 kept (4 removed during review) |
| Data split | 50 articles reserved for development, random seed 42 |

> **Note — why the split is at the article level.** All questions about one
> article go to the same side of the split. If they were split individually, a
> question in the test set could be about an article we had already tuned on,
> which quietly inflates the score. This is called *evidence leakage*.

**We checked where the data came from instead of assuming.**
`01l_verify_source.py` re-downloads the dataset from Hugging Face and matches it
against our copy using a fingerprint of each article's text. Six sampled
articles came back **byte-identical** — not one character different — and the
longest article is exactly 4,595 characters in both.

**Why this matters:** anything wrong with the article text was already wrong in
the published benchmark. We did not break it. That is the first thing to be able
to say if someone challenges our choice of dataset.

### How big are the articles and questions?

| | evaluation | distractor |
|---|---|---|
| median length in characters | 3,118 | 3,448 |
| median length in tokens | 658 | 722 |
| longest article | 4,549 | 4,595 |

Cutting the articles into chunks gives **1.74 chunks per article** (2,971
articles become 1 chunk, 7,987 become 2, and 106 become 3).

> **What this implies.** Almost every article is one or two chunks. So
> "find the right chunk" and "find the right article" are nearly the same task
> here. **An experiment comparing chunking strategies has very little room to
> change the results** — worth knowing before spending a whole tournament round
> on it.

Question length is the clearest difference between the four question versions:

| version | median words | count | what it is |
|---|---|---|---|
| `original` | 6 | 1,340 | NewsQA's questions, untouched |
| `reviewed_original` | 6 | 1,336 | same, minus the 4 removed questions |
| **`resolved`** | **11** | 1,336 | the repaired questions — our main set |
| `clarified` | 12 | 1,078 | only the questions that actually needed repair |

`resolved` is `reviewed_original` with each broken question swapped for its
repaired version. Sections 5 and 6 explain what was broken and what repairing it
cost us.

---

## 2. Is the data clean?

**Mechanically clean, but not clean in meaning.**

**What is fine:** no leftover HTML tags, no invisible control characters, no
stray tabs, no empty articles. All 1,340 answer positions point at the right
text. Every gold chunk actually exists. No question is left without a gold chunk.

**What is not fine — leftover page furniture.** When the articles were extracted
from CNN web pages, parts of the page that are *not* the article came along with
them (`01f_html_residue.py`, across all 11,064 articles):

| leftover | articles affected |
|---|---|
| runs of 4+ blank lines (leftover block boundaries) | 9,990 (90%) |
| the `(CNN) --` dateline | 10,216 |
| video captions ending in `»`, e.g. *"Watch Bush talk about 9/11 »"* | 2,826 |
| the "E-mail to a friend" button text | 580 |
| copyright / publisher footers | 166 |

This matters because the search system treats these strings exactly like article
text — it chunks them, indexes them, and can retrieve them. Section 8 covers
what we removed and what we deliberately left alone.

**What is not fine — 225 duplicate articles.** 224 groups of articles are the
same text differing only in spacing (`01c_clarified_and_dupes.py`). They add 386
duplicate chunks (2.0% of the index). Importantly they affect **zero gold
chunks** — so they make the search pile slightly bigger without changing any
answer.

---

## 3. The articles are cut off partway through

This is the biggest finding and the one most likely to be challenged, so the
evidence is laid out in the order we established it — including the argument we
had to throw away.

![Truncation evidence](figures/eda/fig1_truncation.png)

**The argument we rejected first.** Our initial reasoning was: *"the longest
article is just under 4,600 characters, so there must be a cut-off."* **That
does not work.** If there were a hard character limit, we would see many
articles piling up right at the limit. We don't — each length near 4,595 occurs
exactly once (left chart). A distribution that simply ends is not evidence.

**Four things that do hold:**

**1. There is a cap — measured in words, not characters** (middle chart).
Characters hid it, because articles differ in average word length. In words the
ceiling is obvious: **30% of the corpus (3,323 articles) stops between 640 and
680 words**, then it falls off a cliff — 579 articles reach 680 words or more,
and only **64 go past 700**. Real article lengths don't spike at one width and
stop. This is the single strongest piece of evidence, and it corrects our earlier
"there is no pile-up" claim: there is one, it just isn't visible on the character
axis.

**2. Long articles stop mid-sentence** (`01d_truncation.py`, right chart). We
counted how often an article ends with a full stop, question mark or exclamation
mark:

| article length group | ends properly |
|---|---|
| shortest 25% | **88.4%** |
| middle 50% | 59.9% |
| longest 25% | **14.7%** |
| within 200 characters of the longest | 7.3% |

Short articles end in a full stop. Long ones stop in the middle of a sentence.
**Ending mid-sentence more often the longer you get is exactly what a cut-off
looks like** — nothing else explains that pattern.

**3. Direct comparison with the original web pages**
(`01e_truncation_proof.py`). We have the archived CNN pages (see section 4). Of
94 articles we could pair with their original page, 67 are shorter than the
original, and **50 are an exact prefix** — the benchmark version is
character-for-character the beginning of the real article, then just stops. A
prefix relationship *is* being cut off; there is nothing to interpret.

**4. Reading the articles.** The strongest check is the least technical. Two
articles, benchmark version against the cleaned original (full text in section 4):

> ours ends *"…they're walking and walking and walking — **but**"*
> the original continues *"**I do think that people have no excuse for bad hair.**"*

A short article in the same sample is byte-identical and ends on a full stop.

*A fifth line of evidence — live re-fetching of the CNN URLs — was withdrawn.
It came from `01k_live_verify.py`, which used the same faulty extraction
described in section 4, so it could not be relied on and the script was
deleted.*

### How much is affected?

4,548 of 11,064 articles (41%) end on a lowercase word or a comma. **This is an
upper bound, not the truncation rate** — it also catches things like
unpunctuated bylines. Checked against the 94 verified pairs, that rough
text-pattern test is right about 67.5% of the articles it flags (precision) and
catches 65.9% of the truly cut ones (recall).

We also tested whether "longer than 3,800 characters" could stand in as a simple
rule (`01n_threshold_validity.py`): 77.4% precision, 60.0% recall. **40% of
genuinely truncated articles are shorter than 3,800 characters**, so we cannot
use a length cut-off to declare everything else intact.

### Does it actually hurt our scores?

![Evidence position](figures/eda/fig2_evidence_position.png)

**Much less than the 41% headline suggests** (`01m_truncation_impact.py`). We
measured where the answer sits inside its article. The median answer is at the
**18th percentile** — i.e. answers cluster near the *beginning*. Truncation
removes the *end*. The two barely overlap.

Only 34 questions have their answer in the last 10% of the article, and only 9
of those are in articles at risk of being cut. In total 52 evaluation articles
carry any risk, covering 321 questions.

**Conclusion.** The corpus is systematically cut off, we inherited that from the
published benchmark, and it damages retrieval far less than 41% sounds — because
NewsQA answers live near the top of the article.

---

## 4. How much text is missing, and can we restore it?

`data/cnn_downloads.tgz` holds **92,579 archived CNN web pages** — the same
pages the benchmark was built from, already in this repository.
`08_truncation_gap.py` pairs each benchmark article with its original page and
measures the gap.

> **Note — this section was recomputed after an extraction bug.** The first
> version used a hand-rolled parser that selected `p.cnn_storypgraphtxt`, a
> class that appears in **none** of these pages, and so fell through to "take
> every `<p>`" — sweeping in CNN's sign-up form, weather widget and topic tags
> as if they were article text. Because those are fixed-length template blocks,
> hundreds of articles appeared to be missing byte-for-byte identical amounts
> (1,736 characters × 675 articles). That script is deleted. The numbers below
> come from `NewsCleaner`, the newspaper3k-based extractor the project already
> uses to build `data/processed/`. Section 4b measures what the bug cost.

| | |
|---|---|
| archived pages scanned | 92,579 |
| benchmark articles paired with their original page | **9,468 / 11,064 (85.6%)** |
| intact — the original is no longer than ours | 2,693 (28.4%) |
| **truncated — ours is an exact prefix of the original** | **4,745 (50.1%)** |
| diverged — differs mid-text, not countable as truncation | 2,030 (21.4%) |
| …of the truncated, page furniture only (≤40 chars) | 908 |
| **…of the truncated, real content missing** | **3,837** |

**How much is lost**, counting only real content:

| | characters | share of the article |
|---|---|---|
| median | **540** | **12%** |
| mean | 1,069 | 17.2% |
| p90 | 2,886 | 42% |
| max | 7,537 | 66% |

| severity | articles | share |
|---|---|---|
| about a sentence (41–200 chars) | 1,552 | 40.4% |
| a few paragraphs (200–1k) | 775 | 20.2% |
| **a large section (1k–3k)** | **1,164** | **30.3%** |
| **most of the story (3k+)** | **346** | **9.0%** |

Total across the corpus: **4.1 million characters**.

**On the evaluation set specifically** — the 200 articles that hold answers, of
which 187 paired: **38 lose more than 200 characters** (median 1,128) and **21
lose more than 1,000** (median 1,827). The blast radius on scoring stays small.

### Verified by reading, not just by metric

Two articles, benchmark version against the cleaned original:

> **`00a39c13…` — truncated.** Benchmark 3,496 chars, original 5,906.
> Ours ends: *"…they're walking and walking and walking — **but**"*
> The original continues: *"**I do think that people have no excuse for bad
> hair.** Because you know what? There's a hat…"* — then two more interview
> exchanges. It stops mid-sentence and loses real journalism.

> **`00a2aef1e18d…` — complete.** Benchmark 1,656 chars, original 1,656.
> Ends: *"…deep in debt after paying the school a large amount of money to board
> his son."* A full stop, and byte-identical.

Length predicts it, which ties this back to the word cap in section 3:

| benchmark length | articles | cut | rate | median loss |
|---|---|---|---|---|
| under 400 words | 35 | 7 | 20% | 45 chars |
| 400–600 words | 21 | 4 | 19% | 70 chars |
| **600–640 words** | 10 | 6 | **60%** | **1,383 chars** |
| **640+ (at the cap)** | 32 | 23 | **72%** | **1,116 chars** |

Short articles end properly; articles at the 640–680 word ceiling are cut, and
lose over a thousand characters when they are. The 20% "cut" rate in the short
rows is an artefact: newspaper3k leaves CNN's trailing `"All About <topic>"` tag,
worth 45–70 characters. That is why gaps of 40 characters or less are counted as
furniture, not truncation.

![Restoration reach](figures/eda/fig5_restoration.png)

Because our text is an exact **prefix** of the archived text, restoring an
article means simply *appending* the missing end. Nothing before the join moves,
so **all the existing answer positions stay valid** and no ground truth has to
be re-labelled.

This is the answer to *"did you fix the defect, or did you just try to
re-crawl?"* — the material is already in the repo, and restoring costs minutes
of CPU rather than a crawl.

**Recommended approach** (scoped, not yet built): restore the distractors —
that only makes the search *harder*, so it cannot flatter our score — and treat
the 38 affected evaluation articles as a separate, clearly-flagged comparison.
Lengthening an article that holds an answer risks adding *more* answer text
that the official labels don't know about (see section 7).

> **Restoration cannot be run blind.** The cleaner is not clean. In 24 of 98
> checked articles it injects CNN promo boxes *into the middle of the body* —
> *"…Taiwan's Health Ministry said Monday. **Impact Your World See how you can
> make a difference in children's lives** And a second child in Hong Kong…"*,
> or *"**Don't Miss** Obama threatens dissenting Democrats"*. It also leaves a
> trailing `"All About <topic>"` tag. Restored text must have those stripped
> first, by the same rules `scripts/clean_corpus.py` applies to the benchmark
> corpus. One thing the cleaner never does is *lose* article text: in 98 checks
> there were zero cases where its output was shorter than ours.

### 4b. What the extraction bug cost

`09_extractor_check.py` reproduces the deleted extractor deliberately, runs both
it and `NewsCleaner` over the same pages, and compares verdicts. On a stratified
sample of 281 articles:

| the broken extractor said | → intact | → furniture | → truncated | → diverged |
|---|---|---|---|---|
| **truncated** (250) | **69** | 25 | **124** | 32 |
| **intact** (31) | 27 | 0 | 0 | 4 |

**Verdict unchanged: 151 of 281 (53.7%).** Barely half of what it called
truncated actually was — **69 of 250 were not truncated at all**. Its
signature is visible in the raw gaps: three separate articles were each reported
as missing exactly 110 characters, and the cleaner says all three are missing
nothing.

Aggregate effect of the fix:

| | broken | **correct** |
|---|---:|---:|
| articles paired | 9,846 | **9,468** |
| intact | 164 | **2,693** |
| truncated | 4,229 | **4,745** |
| diverged (not countable) | 5,453 | **2,030** |
| real content missing | 3,231 | **3,837** |

Both large moves have the same cause: the broken extractor *added* boilerplate
to every page, so almost nothing could come out shorter than the benchmark
(intact 164 → 2,693) and most pages differed mid-text (diverged 55% → 21%).
Note the direction — **truncation is slightly more common than the broken run
claimed**, not less. The bug inflated individual gaps while hiding cases.

*Caveat: the reproduction is not a byte-exact replay of the deleted script
(it omits HTML-entity unescaping), so the table shows the character of the
failure rather than an exact re-run.*

### What is still not known

- **"Intact" is not "verified complete".** It means the archived page is no
  longer than ours. The archived page could itself be a partial capture; that
  was never checked.
- **1,596 articles (14.4%) never paired with any page.** Their status is
  unknown, not intact.
- **2,030 articles (21.4%) diverge mid-text.** Because the cleaner injects promo
  boxes, we cannot tell whether text is missing or the cleaner added something.
  They are excluded from the truncation count, which makes **4,745 a lower
  bound**.

---

## 5. What was wrong with the questions

During review, each question got one or more **reason codes** — a short label
saying what was wrong with it (`05_reason_codes.py`, 1,340 questions):

| reason code | count | share | plain meaning |
|---|---|---|---|
| `missing_subject` | 688 | 51.3% | doesn't say *who or what* it is about |
| `underspecified_event` | 139 | 10.4% | doesn't say *which* event |
| `generic_reference` | 98 | 7.3% | says "the man", "the company" with no name |
| `wrong_evidence` | 87 | 6.5% | the marked answer location was wrong |
| `malformed_question` | 71 | 5.3% | broken grammar or not a question |
| `truncated_answer` | 59 | 4.4% | the labelled answer was itself cut off |
| `unresolved_coreference` | 58 | 4.3% | "it"/"he" pointing at nothing stated |
| `missing_location` | 55 | 4.1% | asks *where* without saying where-ish |
| `wrong_answer` | 35 | 2.6% | the labelled answer is incorrect |
| `multiple_corpus_matches` | 27 | 2.0% | more than one article answers it |
| `missing_time` | 19 | 1.4% | no time frame given |
| others (4 codes) | 13 | 1.0% | |

![Question defects and repair](figures/eda/fig3_question_repair.png)

957 questions have exactly one code, 174 have two or more, and 209 have none —
those were fine as written.

**Why these are facts, not opinions.** Every code names something you can *check
against the article*: is a subject stated or not; does "it" refer to something
named or not. None of them says "this question is too hard". That is what makes
the repair auditable — anyone can re-check any question.

The most common problem, in over half the set, is simply that the question never
says what it is about. NewsQA questions were written by people looking at the
article, so *"Who was found dead?"* felt complete to them. Pulled out of that
context and pointed at 11,064 articles, it isn't.

### How can an answer be "wrong"? (`wrong_answer`, 35 questions)

This is the code that most looks like reviewer opinion, so here is exactly what
those 35 cases are. Each keeps the original answer, the original answer
position, and a written reason, so anyone can re-check it against the article in
under a minute.

Of the 31 where the answer text actually changed:

| what changed | count | is it a real correction? |
|---|---|---|
| replaced with completely different text | 22 | **yes** |
| trimmed (new answer sits inside the old) | 7 | no — just tightening the boundary |
| expanded (old answer sits inside the new) | 2 | no — just tightening the boundary |

The 22 real replacements fall into three kinds, none needing a judgement call:

- **The labelled answer isn't an answer at all.** *"How many driverless pods
  were being tested?"* → the NewsQA answer was `'are'`. Whoever highlighted it
  selected a verb.
- **The answer type doesn't match the question word.** *"**When** were the first
  impeachment charges brought?"* → `'vote-tampering.'` — that's a charge, not a
  date. *"What **country** were the passengers from?"* → `'Chinese nationals.'` —
  a nationality, not a country.
- **The stated fact is the wrong one.** *"How many Canadian troops?"* →
  `'35,000.'`, which is the total for all NATO allies. The article says *"more
  than 2,800 Canadian troops"*.

> **Two caveats we should state, not hide.** First, 20 of the 31 revised answers
> are copied word-for-word from their article, but **11 are not** — they are
> tidied-up versions like `'China'` instead of `'Chinese nationals'`. If we ever
> score by locating the answer inside the article, those 11 can no longer be
> found. Check them first. Second, the 7 trims and 2 expansions aren't error
> corrections at all — counting them under `wrong_answer` slightly overstates
> the code.

### Did a human actually check the machine's proposals?

Yes, and disagreed often. A model proposed labels; a human reviewer decided.

| | |
|---|---|
| model proposed "not standalone" | 979 |
| human final decision "not standalone" | 1,078 |
| **proposals the human overturned or amended** | **262 (19.6%)** |
| answers the human modified | 298 (22.2%) |
| questions removed entirely | 4 |

Nearly one proposal in five was changed. This was not a rubber stamp.

### Did repairing the questions actually help retrieval?

First, the term this depends on.

> #### What "rare term" means
>
> **IDF** (inverse document frequency) scores how unusual a word is across the
> corpus: `IDF = log(19,263 / (1 + number of chunks containing it))`. A word in
> almost every chunk scores near 0; a word in one chunk scores high.
>
> Throughout this report, **"rare term" means IDF ≥ 6.0 — a word appearing in at
> most about 46 of 19,263 chunks (0.24% of them)**.
>
> | word | appears in | IDF | rare? |
> |---|---|---|---|
> | `the` | 19,207 chunks | 0.00 | no |
> | `said` | 14,592 chunks | 0.28 | no |
> | `police` | 2,871 chunks | 1.90 | no |
> | `hurricane` | 230 chunks | 4.42 | no |
> | `gabon` | 13 chunks | 7.23 | **yes** |
> | `wozniak` | 8 chunks | 7.67 | **yes** |
> | `cocodrie` | 1 chunk | 9.17 | **yes** |
>
> In practice a rare term is a **name or a specific label** — a person, place,
> organisation, or technical term appearing in only a handful of articles. It is
> the thing that lets a word-matching retriever jump straight to the right chunk
> instead of just finding the right *topic*. Rare terms shown in bold:
>
> > Which crimes was **Theoneste Bagosora** convicted of by the Rwanda tribunal?
> > In how many states were cases of the **Listeria monocytogenes** outbreak reported?
> > How many militants attacked military **checkposts** in the **Mohmand** agency?
>
> The 6.0 cut-off is a choice, not a law — it was picked so "rare" means roughly
> *"in under a quarter of one percent of the corpus"*. Every measurement in this
> report uses the same value, so comparisons stay consistent even if the exact
> number is arbitrary.

Now: how many rare terms did each kind of repair add to the question?

| reason code | questions | words added | rare terms added | gained ≥1 rare term |
|---|---|---|---|---|
| `underspecified_event` | 139 | 5.67 | 0.99 | 64.0% |
| `generic_reference` | 98 | 6.11 | 0.97 | 60.2% |
| `unresolved_coreference` | 57 | 4.51 | 0.91 | 63.2% |
| `wrong_evidence` | 86 | 3.71 | 0.77 | 50.0% |
| `missing_subject` | 688 | 4.33 | 0.73 | 47.4% |
| `missing_time` | 19 | 3.68 | 0.32 | 31.6% |
| **(no code — untouched)** | **209** | **0.00** | **0.00** | **0.0%** |

Repairs that name a *thing* — an event, a reference, a pronoun's target — add
the most searchable signal, which makes sense: naming something adds its name.

**The last row is the control.** Questions with no defect were left completely
alone: zero words added. If resolution had been a blanket rewrite of everything,
that row would not be zero.

---

## 6. `original` vs `resolved` — which set do we report, and what does it cost?

### The finding that settles it (`06_near_duplicates.py`)

We looked for questions that are near-copies of each other:

| | `original` | `resolved` |
|---|---|---|
| groups of word-for-word identical questions | 6 | 47 |
| …where the group spans **different articles** | **5** | **0** |
| near-identical pairs (70%+ word overlap) | 94 | 150 |
| …pointing at different answers **in different articles** | **34** | **0** |
| questions caught in such a conflict | **49 (3.7%)** | **6 (0.4%)** |

In the original set, 34 pairs of near-identical questions point at *different
articles*. For example, **"what does faa say"** appears three times, with three
different correct articles.

> **Why this is fatal.** These are not hard questions — they are **impossible**
> questions. The retriever sees identical text and is expected to return
> different articles. No system can do that. Every retriever we test loses the
> same points on them, so they add noise and measure nothing.

Repairing the questions removes this class completely: 34 → 0.

### What repair adds (`03_lexical_overlap.py`)

| | `original` | `resolved` |
|---|---|---|
| question words that also appear in the correct chunk | 66.1% | 78.4% |
| …same test against a **random** chunk (the baseline) | 7.5% | 5.8% |
| rare terms shared with the correct chunk | 0.33 | **0.89** |
| …against a random chunk | 0.000 | 0.001 |
| questions with at least one rare term to anchor on | 27.7% | **57.7%** |

Repair more than doubles the number of rare-term anchors, and the random-chunk
baseline stays near zero — so this is real signal, not an artefact.

> **The honest caveat.** Rare-term anchors are exactly what **word-matching
> (sparse) retrieval** feeds on. Dense retrieval, which matches meaning, gains
> less from them. So **comparing sparse against dense using only the `resolved`
> set tilts the comparison toward sparse.** This is the main reason both
> versions must be run and reported together.

### What repair costs

47 groups of originally-different questions ended up **word-for-word identical**
after being clarified — 96 questions (7.2%), which is **49 duplicate queries**:

> *"Where did the nightmare day take place?"* and *"Where did the shooting take
> place?"* both became
> *"Where did the mass shooting involving Spc. Logan Burnette take place?"*

All 47 groups stay inside one article and share the same correct chunk, so
nothing becomes impossible — but 49 questions are now asking the identical
question twice, which gives those 47 articles double weight in the score.
**Effective number of distinct questions: 1,287, not 1,336.** Worth a footnote
whenever a score is reported.

### Verdict

- **`resolved` is the realistic set.** A real user types *"who won the Gabon
  election"*, not *"who won it"*. This is what the system would actually face.
- **`original` is the floor** — the pessimistic bound, including questions no
  system can answer.

**Report both as a range, and say which end is which.** Publishing either one
alone is misleading: `original` alone understates the system, `resolved` alone
overstates it and quietly favours sparse retrieval.

---

## 7. How hard is this search task, and is the "one correct answer" assumption true?

From `04_distractor_collision.py`, measured on the `resolved` set.

![Retrieval difficulty](figures/eda/fig4_retrieval_difficulty.png)

### How much competition is there?

**What we measured, step by step.** Take each rare term in the question, look up
every chunk containing it, pool those together, and remove the correct chunk.
What remains is the set of wrong chunks a word-matching retriever still has to
choose between — the **competitors**. On a real question:

```
Q: What made landfall near Cocodrie, Louisiana?

  every chunk in the corpus                          19,263
  question words:      cocodrie, landfall, louisiana, made
  of those, RARE:      cocodrie, landfall

    chunks containing 'cocodrie'          1
    chunks containing 'landfall'         46      (pooled: 46)

  chunks sharing at least one rare term                  46
  minus the 1 correct chunk  ->  COMPETITORS             45
```

> **Important: BM25 does not actually do this.** A real word-matching retriever
> scores *every* chunk and returns a ranked list; it never discards anything.
> What this measures is **how much distinguishing power the rare words carry** —
> an optimistic best case. Real retrieval does worse, because a chunk with no
> rare-term match can still outrank the correct one on common-word overlap.
> Treat the numbers below as a ceiling on what word matching can do.

Across all questions:

| | |
|---|---|
| median competitors | **20** |
| 90th percentile | 49 |
| worst case | 171 |
| questions narrowed to 10 competitors or fewer | only **30.7%** |
| **questions with no rare term at all** | **493 (37%)** |

> **What this justifies.** Even in the optimistic case above, rare terms narrow
> the field from 19,263 chunks to about 20 — an enormous reduction, but 20 is
> still not 1. **The fast first pass gets close and cannot finish the job.**
> That is a measured, data-level argument for using a reranker, and it holds
> independently of whatever our tournament results said.
>
> The 37% with no rare term at all is the other side: for those questions,
> word-matching has nothing sharp to grab, which is where dense retrieval should
> earn its place.

### Is exactly one chunk really the only correct answer?

Our scoring assumes it is. We tested that by searching every *wrong* chunk for
the correct answer text.

A raw text match says 23.1% of questions have a distractor containing the gold
answer — but **containing the answer text is not the same as answering the
question**. "German authorities" turns up in plenty of unrelated articles. So we
graded each match by how many of the question's rare terms that distractor also
shares:

| match strength | share of questions |
|---|---|
| no wrong chunk contains the answer | 76.9% |
| 1 shared rare term — probably coincidence | 16.6% |
| 2 shared — probably the same news story | 4.5% |
| 3+ shared — almost certainly the same news story | 2.0% |

**About 6.5% of questions have a wrong-labelled chunk that plausibly answers
them** — usually the same news event covered in a second article. For example:
*"What made landfall near Cocodrie, Louisiana?"* → *Hurricane Gustav* — which
also appears in a second Gustav article that is not marked as correct.

> **What this means for every score we report.** If the system retrieves that
> second Gustav article, our scoring calls it wrong even though a user would be
> satisfied. So **every retrieval score carries a built-in error of roughly
> 6.5% (strong cases) to 23.1% (absolute upper bound)**. Two systems whose
> scores differ by less than that are not meaningfully different.
>
> For comparison, the human review flagged only 27 questions as having multiple
> matching articles — found by spot-checking. Searching systematically finds
> around 87.

*Limitation: this is an indicator, not verified reading. We checked for shared
rare terms plus the answer text appearing — not whether the chunk truly answers
the question. Turning 6.5% into a confirmed number needs a human to read a
sample of the 46 strong cases.*

---

## 8. What we cleaned

`scripts/clean_corpus.py` (run with `--apply`) writes a **separate** cleaned
copy at `data/evaluation/newsqa_200_11064/cleaned/`. **The locked benchmark in
`final/` is not touched**, so nothing already measured is invalidated.

| removed | articles |
|---|---|
| video captions (`Watch … »`) | 2,186 |
| "E-mail to a friend" text and whatever followed it | 579 |
| copyright / publisher footers | 141 |
| subscribe footers | 1 |
| runs of 3+ blank lines, collapsed to one | all |

2,757 articles changed (24.9%); 167,061 characters removed.

### The answer positions survive

This is the risky part. Answer positions are stored as *character offsets* — so
deleting 40 characters from the middle of an article silently breaks every
position after it.

The script therefore records every deletion in an offset map, shifts each
position by the right amount, **and then re-checks that the position still
quotes the same text**. Result on the `resolved` set: **1,338 positions exact, 3
relocated by searching, 0 broken.** A position that cannot be resolved is
flagged `evidence_spans_broken`, never silently dropped.

`python scripts/clean_corpus.py --selfcheck` runs a small test of the offset
logic on its own.

### What we deliberately did *not* remove

Our first scan flagged an "All About" pattern (145 articles) and a bullet
character (123) as probable page furniture. **Reading the actual matches showed
this was wrong** — most are ordinary prose (*"it's all about the arms"*, *"what
life is all about"*) or genuine list markers. Removing them would have deleted
real article content, which is a worse mistake than leaving some furniture
behind.

The `(CNN) --` dateline is also kept: it is part of the published article text,
appears in 92% of articles, and therefore helps no retriever tell articles
apart.

The 225 duplicate articles are **not** removed either. Removing them would
change chunk IDs and invalidate the locked Phase-2 artifacts, in exchange for a
2.0% smaller index that affects no correct answer. Note it; don't act on it in
the middle of the study.

---

## 9. What this means for how we run and report the experiments

1. **Run both question versions and report a range.** Repair changes the
   searchable signal enough (0.33 → 0.89 rare anchors) that one number isn't
   defensible — and it specifically favours sparse retrieval.
2. **The reranker is justified by the data, not just by the tournament.** Median
   20 competitors after rare-term filtering; only 31% of questions get down to
   10 or fewer.
3. **Chunking experiments have little room to move.** 1.74 chunks per article
   means chunk retrieval and article retrieval are nearly the same task here.
4. **@7 does not exist.** `score_benchmark_predictions.py` produces k ∈ {1, 3,
   5, 10}, limited by `top_n`. Report @3/@5/@10, or change the scorer — but do
   not report a @7 that was never computed.
5. **State the built-in error.** Every Hit@K carries a 6.5–23% unlabelled-answer
   error. Differences smaller than that are not real differences.
6. **Footnote the 49 duplicate questions** (1,287 genuinely distinct queries).
7. **The judge must not be the generator.** The benchmark notebook previously
   set `JUDGE_MODEL = GENERATOR_MODEL` with `ALLOW_SAME_JUDGE = True` — i.e. the
   model graded its own answers — while the text three cells above said to use a
   different judge. Now fixed and guarded.

---

## 10. Still open

- **A second benchmark on restored text** — planned in section 4, not built.
  Restore the distractors (stripping the promo boxes the cleaner injects),
  re-chunk, re-map the correct chunk IDs, check the 38 affected evaluation
  articles for newly-introduced unlabelled answers, then run the same locked
  configuration on both corpora and report the pair.
- **Whether the archived pages are themselves complete.** Every truncation
  number is relative to them; nobody has checked them against a third source.
- **The 1,596 articles that never paired**, and the 2,030 that diverge mid-text.
  Both are unknown rather than intact, which is why 4,745 is a lower bound.
- **Human validation** of a sample of the 46 strong unlabelled-answer cases, to
  turn the 6.5% indicator into a measured rate.
- **The Phase 1 result files** (`round1/2/3.csv`) are missing from the
  repository, so we cannot verify that the documented MiniLM reranker choice
  matches what the selection code actually picked.
