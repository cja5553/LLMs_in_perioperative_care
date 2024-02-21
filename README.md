# Prescribing Large Language Models (LLMs) for Perioperative Care: What’s The Right Dose for Pretrained Models?

Our best performing finetuned models are available at 🤗 Huggingface

### [`cja5553/BJH-perioperative-notes-bioClinicalBERT`](https://huggingface.co/cja5553/BJH-perioperative-notes-bioClinicalBERT)

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("cja5553/BJH-perioperative-notes-bioClinicalBERT")
model = AutoModel.from_pretrained("cja5553/BJH-perioperative-notes-bioClinicalBERT")
```

### [`cja5553/BJH-perioperative-notes-bioGPT`](https://huggingface.co/cja5553/BJH-perioperative-notes-bioGPT)

## Goal: 
- Experiment the use of pretrained LLMs across different fine-tuning strategies in surgical outcomes of Perioperative Care.
- The following strategies were experimented:
  1. using pretrained models alone
  2. applying finetuning
  3. applying semi-supervised fine-tuning with the labels
  4. foundational model where a multi-task learning strategy was employed.
 
     
- 3 primary models were used for prediction
  1. bioGPT
  2. ClinicalBERT
  3. bioclinicalBERT.

## Dataset:
- We used 84,875 clinical notes from patients spanning the Barnes Jewish Center Hospital (BJC) hospital system in St Louis, MO.
  - The following outcomes were used: 
    1. Death in 30 days
    2. Deep vein thrombosis (DVT)
    3. pulmonary embolism (PE)
    4. Pneumonia
    5. Acute Knee Injury
    6. delirium
  
 - Characteristics:
   - vocabulary size 3203
   - averaging 8.9 words per case,
   - all single sentenced clinical notes

## To use:
- You should be able to run the codes as it is on the Jupyter notebook files provided (of course with your own dataset)
- For the semi-supervised and foundational version, you may need to clone the `transformers` package from `huggingface`'s github profile and slot the relevant files in the same folders of which they appear in the `local_transformers` folders of this github repo.

## Questions? 
Contact me at alba@wustl.edu
