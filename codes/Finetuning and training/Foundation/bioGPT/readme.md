- `local_transformers` contains the modules (edited from the `transformers` package) to faciliate multi-task learning architecture. When implemented, it can be treated similarly to the `transformers` package, and can be deployed using the `trainer` module.   
- `implementation.ipynb` is the implementation, where the modules utilized to finetune and test on our unseen fold.
- `printing_results.ipynb` prints the results upon the finetuning.  

To use:
1. Git Clone the `transformers` package from `HuggingFace` (https://github.com/huggingface/transformers)
2. Change folder name `transformers` to `local_transformers`. 
3. Insert the following python files (`.py`) in the respective folders. 