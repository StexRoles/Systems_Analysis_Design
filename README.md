## WORKSHOP 2 LINK
[Go to the final report PDF](./Workshop_2/Workshop_2_SystemsAnalysis.pdf)

## Project Overview

This repository contains the results and insights from Workshop 2, where we designed a system to detect mathematical misconceptions in multiple-choice questions, based on the Kaggle competition: [EEDI - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview).

Our goal was to predict the affinity between distractor options (wrong answers) and potential misconceptions, contributing to better AI-supported educational tools.

## 🔍 Initial Analysis – Workshop 1 Review

We began this workshop by revisiting the findings from Workshop 1. There, we analyzed the competition and broke down the main components of the system:

- *Key Elements:* Questions, correct answers, distractors, and misconceptions.
- *Sensitivity:* The system is highly sensitive to small input changes (e.g., wording of questions).
- *Structure:* Questions are strictly mathematical with one correct answer and three distractors per question.

We observed that changes in distractors or phrasing had a nonlinear impact on system output due to the complexity of natural language processing. As a result, strategies such as ensemble modeling and label smoothing were proposed to improve robustness.

## 📋 System Requirements

### Functional Requirements
- *RF1:* Load multiple-choice questions.
- *RF2:* Load a database of misconceptions.
- *RF3:* Analyze linguistic/conceptual patterns in distractors.
- *RF4:* Generate at least 25 plausible misconceptions per distractor.
- *RF5:* Label each misconception with the corresponding distractor.
- *RF6:* Export the labeled misconceptions to formats like CSV or JSON.

### Non-Functional Requirements
- *RNF1:* Fast response time for generating misconceptions.
- *RNF2:* Scalable processing for large datasets.
- *RNF3:* Support for integration with modern LLMs (e.g., GPT, Claude).

## 🧠 System Architecture

We created a high-level architecture diagram showing data flow through various components:

1. *Trainer Data:* Input of questions and misconceptions.
2. *Filter Affinity:* Identifies the most relevant misconceptions.
3. *Less Similar:* Discards low-affinity associations.
4. *NLP Model:* Associates misconceptions with distractors using affinity scoring.
5. *Verification:* Validates associations; if scores are low, retraining is triggered.
6. *Distractor Tagged:* Final output with high-affinity, validated misconceptions.

This iterative approach ensures quality control and continuous model improvement.

## ⚖️ Reducing Sensitivity

To prevent chaotic outputs, we implemented several control mechanisms:

- Filtered out irrelevant or malformed data before training.
- Avoided semantic and syntactical noise in question phrasing.
- Added a verification stage to validate outputs before final labeling.
- Created feedback loops to retrain the model when predictions didn't meet thresholds.

This layered strategy helped stabilize system performance against small input changes.

## 🛠️ Tech Stack

The technologies selected for implementation were:

- *Python:* For general system development and ML workflows.
- *Claude 3.5 Sonnet (Anthropic API):* Used to generate and analyze misconceptions due to its strong reasoning capabilities.
- *GPT-4 (OpenAI):* Used in the verification stage to ensure label quality and concept clarity.
- *JSON/CSV:* For exporting results in readable formats.

This combination offered flexibility, strong NLP capabilities, and ease of integration.