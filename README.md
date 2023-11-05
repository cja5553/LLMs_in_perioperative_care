# Prescribing Large Language Models (LLMs) for Perioperative Care: What’s The Right Dose for Pretrained Models?

## Goal: 
- Experiment the use of LLMs across different fine-tuning strategies in surgical outcomes of Perioperative Care. The following strategies were experimented: (1) pretrained models alone, (2) using finetuning, (3) semi-supervised fine-tuning with the labels, and (4) foundational model strategy where a multi-task learning framework was employed. 3 primary models were used (1) bioGPT, (2) ClinicalBERT, and (3) bioclinicalBERT.

## Dataset:
- We used 48,875 clinical notes from patients from the Barnes Jewish Center Hospital (BJC) hospital system in St Louis, MO. The following outcomes were used: (1) Death in 30 days, (2) Deep vein thrombosis (DVT), (3) pulmonary embolism (PE), (4) Pneumonia, (5) Acute Knee Injury, and (6) delirium

## To use:
- You should be able to run the codes as it is on the Jupyter notebook files provided. For the semi-supervised and foundational version, you may need to clone the `transformers` package from `huggingface`'s github profile and slot the relevant files in the same folders of which they appear in the local folders of this github profile. 
