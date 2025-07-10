# In-Depth Analysis of MedGemma: An Interpretability and Adversarial Testing Notebook

This repository contains a Jupyter/Colab notebook detailing my comprehensive investigation into the behavior, knowledge structure, and safety alignment of the MedGemma-4b-it large language model.

The goal of this project was to move beyond surface-level prompting and apply a suite of advanced interpretability and testing techniques to build a deep "character profile" of a specialized medical AI.

---

## Model Used

All experiments were conducted on [**google/medgemma-4b-it**](https://huggingface.co/google/medgemma-4b-it), a 4-billion parameter medical language model fine-tuned by Google. I extend my sincere thanks to the Google team for making this powerful and remarkably robust model available to the research community.

---

## Summary of Investigations and Key Findings

This notebook is structured as a series of investigative "frontiers," each building upon the last.

### 1. Confidence and Knowledge Analysis (White-Box Probing)

I began by analyzing the model's internal confidence scores and concept vectors.

*   **Key Finding 1 (Confidence Patterns):** The model is most confident when reciting specific, factual data (e.g., drug dosages) and least confident when explaining complex, open-ended mechanisms (e.g., disease etiology).
*   **Key Finding 2 (Knowledge Structure):** Its internal vector representations are organized by practical associations, not just textbook definitions. For instance, I found the concept of "pneumonia" to be more similar to its common symptomatic treatment ("ibuprofen") than to its direct cause ("Streptococcus").

### 2. Adversarial Testing and Safety Alignment

I stress-tested the model's safety guardrails and logical consistency.

*   **Key Finding 3 (Safety Robustness):** The model demonstrated state-of-the-art safety alignment. It successfully refused to generate harmful content, even when faced with sophisticated "jailbreaking" prompts involving role-playing and hypothetical scenarios.
*   **Key Finding 4 (Logical & Factual Consistency):** The model proved to be highly consistent. It correctly identified and refuted logical fallacies in conversation and refused to be misled by factual misinformation, correcting the user's premise instead of accepting it.
*   **Key Finding 5 (Persona Vulnerability):** I discovered a single weakness where the model's stated helpful persona could be overridden by a direct instruction to adopt a cynical, conflicting persona.

### 3. Mechanistic Interpretability (Visualizing the "Thought Process")

Using advanced white-box techniques, I visualized the model's internal reasoning pathways.

*   **Key Finding 6 (Logit-Lens Analysis):** I found clear, visual evidence that the model uses different internal pathways for different cognitive tasks. Simple factual recall was a fast, direct process where the answer "crystallized" in the mid-to-late layers. In contrast, comparative reasoning was a slower, more deliberative process.

### 4. Causal Intervention (Attempted Model Editing)

In the final and most advanced experiment, I attempted to surgically alter the model's memory by patching its internal activation states.

*   **Key Finding 7 (Resilience to Editing):** Direct intervention, both on a single layer and across a multi-layer block, failed to change the model's factual recall. Instead, the intervention caused a "coherence collapse." This profound negative result suggests that factual knowledge in this model is a highly distributed and resilient property, not a simple, localized memory that can be easily edited.

## Conclusion

MedGemma proved to be a remarkably knowledgeable, safe, and consistent model. This investigation serves as my comprehensive case study on applying modern interpretability and testing techniques to understand and verify the behavior of specialized AI systems.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
