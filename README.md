FinReg Navigator

AI-Powered Regulatory Intelligence Engine for Fintech & Digital Banking

🚀 What It Is

FinReg Navigator is a Retrieval-Augmented Generation (RAG) system designed to help fintech companies assess regulatory compliance across multiple jurisdictions.

It enables automated regulatory reasoning over structured legal corpora including:

🇵🇰 Pakistan (SBP, FBR, SRB frameworks)

🇦🇪 UAE (ADGM virtual asset framework)

🇬🇧 UK (FCA & EMI regulations)

Why It Exists

Fintech companies often struggle with:

EMI licensing requirements

Digital banking eligibility

AML/CFT obligations

Taxation compliance

Cross-border expansion

Regulatory comparison between jurisdictions

FinReg Navigator allows compliance teams to query regulatory texts semantically and receive structured responses grounded in official regulatory documents.

Example Use Cases
1️⃣ Pakistani EMI expanding to UAE

Compare:

SBP EMI Regulations 2023

ADGM Virtual Asset Guidance

2️⃣ New fintech entering Pakistan

Evaluate:

EMI capital requirements

Customer onboarding framework

AML compliance obligations

3️⃣ Tax impact analysis

Assess:

Finance Act 2025 amendments

Sales Tax Act 1990 updates

Sindh Sales Tax on Services

Architecture

Intent Agent
→ Retrieval Agent (ChromaDB + Embeddings)
→ Web Agent (fallback if no strong semantic match)
→ Final Answer Agent

Tech stack:

ChromaDB (Vector Store)

Sentence Transformers (MiniLM)

Ollama (Local LLM)

LangGraph (Agent orchestration)

Streamlit (UI)