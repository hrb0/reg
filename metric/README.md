# REG2025 
<img a=https://github.com/user-attachments/assets/513dbb0b-d30c-4eb5-abbd-c1e3fc7c3234>

You can test by replacing the file in the `reg/metric/json` directory with the JSON file you want to evaluate.

```
python test.py
```

## Metrics
1. Install python pakcage in your environment (docker, conda, ...)
```
pip install -r requirements.txt
```
- Any Embedding Models can used for simple test when training, but BioLLM will be used for Test Phase.
- Ref: <a href=https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B> Llama3-OpenBioLLM-8B </a>

2. Download Scispacy Model.
```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```
- SpaCy models for biomedical text processing.</br>
- Ref: <a href=https://allenai.github.io/scispacy/>scispacy</a>
