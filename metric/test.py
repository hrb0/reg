import json
from eval import REG_Evaluator

GT_PATH = './json/ground-truth.json'
PRED_PATH = './json/sample.json'

EMBEDDING_MODEL = 'dmis-lab/biobert-v1.1'
SPACY_MODEL = 'en_core_sci_lg'

evaluator = REG_Evaluator(embedding_model=EMBEDDING_MODEL, spacy_model=SPACY_MODEL)

with open(GT_PATH, 'r') as f:
    gt_data = json.load(f)

with open(PRED_PATH, 'r') as f:
    pred_data = json.load(f)

pred_dict = {item['id']: item['report'] for item in pred_data}

eval_pairs = []
for item in gt_data:
    gt_id = item['id']
    gt_report = item['report']
    
    pred_report = pred_dict.get(gt_id, "") 
    
    eval_pairs.append((gt_report, pred_report))

score = evaluator.evaluate_dummy(eval_pairs)

print(f"\n Average Ranking Score: {score:.4f}")
