# AI Banking Support & Advisory Agent (Non-Transactional)

Capstone Project | Prithwiraj Choudhury | 06-10-2026

---

## 1. Root folder


Capstone_Project_PrithwirajChoudhury/

    PHASE[1-4]                              # Phases 1–4 (framing, baseline, LLM, RAG)

    AI Banking Support & Advisory Agent     # Phases 5–9 (final agent)

    demo_scripts                            # Demo script & logs

    Problem Framing Document.pdf

    Engineering and Product Justification.pdf

    Prompt Comparison Table.pdf

    Evaluation Report.pdf

    Screenshots Streamlit & LangSmith.pdf  # Screenshots from streamlit app and Langsmith


### PHASE[1-4]:


PHASE-1  : Problem Framing Document 

PHASE-2  : (rule-based baseline) : `python python_based_banking.py`

PHASE-3  : (LLM, no retrieval) : Set `version` in `banking_support_llm.py` (`version-1`, `version-2` or `version-3`), then `python banking_support_llm.py`

PHASE-4  : (RAG + Streamlit) : `pip install -r requirements.txt`<br>`python ingest.py` (builds `vectorstore/`)<br>`streamlit run app.py`

RAG pipeline document  : `PHASE-4.pdf`


### PHASE[5-9]: AI Banking Support & Advisory Agent

Run these from inside the `AI Banking Support & Advisory Agent/` folder.

```bash
pip install -r requirements.txt         
python rag/ingestion_pipeline/ingest.py   
streamlit run app/app.py
```

| Item | Location |
| --- | --- |
| Interaction logs | `logs/user_interaction/user_interaction_prompt_<version>.log` |
| User feedback | `logs/feedback/feedback_log.json` |
| Prompt versions | `src/agent/prompts/`|

---

## 2. Documents

| Document | Content |
| --- | --- |
| `Problem Framing Document.pdf` | Persona, problem statement, scope, constraints and success criteria. |
| `Engineering and Product Justification.pdf` | Architecture, Design decisions, trade-offs, safety approach and deployment assumptions. Acknowledgement & References |
| `Prompt Comparison Table.pdf` | Three prompt variants (version-1, version-2, version-3.2) on the same test set, with results and insights. |
| `Evaluation Report.pdf` | Evaluation Report (Evaluation method, Changes , Results , Debugged Failures, Next step improvements). |
| `Screenshots Streamlit & LangSmith.pdf` | Screenshots (Streamlit App, Langsmith- Traces, Runs and Monitoring). |

---

## 3. Demo script

Run from the root folder:

```bash
python demo_scripts/demo_script.py
```

It runs the 5 demo questions through prompt versions `version-1`, `version-2` and `version-3.2`.

Logs are in `demo_scripts/`, one file per version (cleared at the start of each run):

- `user_interaction_prompt_version-1.log`
- `user_interaction_prompt_version-2.log`
- `user_interaction_prompt_version-3.2.log`

---

## 4. Evaluation

Run from the `AI Banking Support & Advisory Agent/` folder:

```bash
python evaluation/evaluation.py
```

Evaluation Results are in `evaluation/eval_results`
