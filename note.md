https://cactuscompute.com/blog/finetuning-whistle

I want you use Thai commonvoice corpus (v27) to finetuning whistle asr model (follow https://cactuscompute.com/blog/finetuning-whistle) but It needs some preprcoessing.

"""
Text that mapping to file:
- these texts want to preprcoessing follow /workspace/whistle-thai/thai_commonvoice_dataset/make_datasets/2-1-clean-validated.ipynb
- skip all file that text has dight or non-thai after preprcoessing
"""

It is meaning if text is not in thai_commonvoice_dataset/make_datasets/db-lastest.db, it stil keep if text do not has dight or non-thai.