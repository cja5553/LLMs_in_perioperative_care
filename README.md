# Prescribing Large Language Models (LLMs) for Perioperative Care: What’s The Right Dose for Pretrained Models?

## Goal: 
- Experiment the use of pretrained LLMs across different fine-tuning strategies in surgical outcomes of Perioperative Care.
- The following strategies were experimented: (1) using pretrained models alone, (2) applying finetuning, (3) applying semi-supervised fine-tuning with the labels, and (4) a foundational model where a multi-task learning strategy was employed.
- 3 primary models were used (1) bioGPT, (2) ClinicalBERT, and (3) bioclinicalBERT.

## Dataset:
- We used 84,875 clinical notes from patients spanning the Barnes Jewish Center Hospital (BJC) hospital system in St Louis, MO.
  - The following outcomes were used: (1) Death in 30 days, (2) Deep vein thrombosis (DVT), (3) pulmonary embolism (PE), (4) Pneumonia, (5) Acute Knee Injury, and (6) delirium

## To use:
- You should be able to run the codes as it is on the Jupyter notebook files provided (of course with your own dataset)
- For the semi-supervised and foundational version, you may need to clone the `transformers` package from `huggingface`'s github profile and slot the relevant files in the same folders of which they appear in the `local_transformers` folders of this github repo.

## Questions? 
Contact me at alba@wustl.edu
