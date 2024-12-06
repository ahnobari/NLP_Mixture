import huggingface_hub as hf

hf.snapshot_download(repo_id="ahn1376/LLM_Tests", local_dir='.', repo_type='dataset')