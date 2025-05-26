from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

class EmbeddingEvaluator:
    def __init__(self, model_name):
        assert model_name in [
            'dmis-lab/biobert-v1.1',
            'aaditya/Llama3-OpenBioLLM-8B',
            'openai-communitya/gpt2',
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'NeuML/pubmedbert-base-embeddings'
        ], 'Not Possible model_name.'
        self.model_name = model_name
        self._get_embedding_model()
    
    def _get_embedding_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
    
    @torch.no_grad()
    def get_embedding(self, text):
        inputs  = self.tokenizer(text, return_tensors='pt',padding=False,truncation=True,max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def get_score(self, ref_text, hyp_text, scale=.5):
        ref_embedding = self.get_embedding(ref_text)
        hyp_embedding = self.get_embedding(hyp_text)
        score = cosine_similarity([ref_embedding],[hyp_embedding])[0][0]
        if (scale != 0) and (score>scale):
            score = (score-scale)/(1-scale)
        return score
    
import spacy
import scispacy
from spacy.tokens import Doc
from typing import List, Tuple

class KeywordEvaluator:
    def __init__(self, model_name='en_core_sci_lg'):
        self.nlp = spacy.load(model_name)
        
    def get_keywords(self, text: str, min_length: int = 3) -> List[str]:
        doc = self.nlp(text)
        keywords = []
        for ent in doc.ents:
            if len(ent.text) >= min_length:  
                keywords.append(ent.text.lower())
        return list(set(keywords))  
    
    @staticmethod
    def get_jaccard(list1: List[str], list2: List[str]) -> float:
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0
    
    def get_score(self, ref_text: str, hyp_text: str, min_length: int = 3) -> Tuple[float, List[str], List[str]]:
        ref_keywords = self.get_keywords(ref_text, min_length)
        hyp_keywords = self.get_keywords(hyp_text, min_length)
        
        score = self.get_jaccard(ref_keywords, hyp_keywords)

        return score


class REG_Evaluator:
    def __init__(self, embedding_model, spacy_model='en_core_sci_lg', api_key=None): 
        self.embedding_eval = EmbeddingEvaluator(embedding_model)
        
        self.key_eval = KeywordEvaluator(spacy_model)
    
    @staticmethod
    def get_bleu4(ref_text, hyp_text):
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()

        ref_fourgrams = [' '.join(ref_words[i:i+4]) for i in range(len(ref_words)-3)]
        hyp_fourgrams = [' '.join(hyp_words[i:i+4]) for i in range(len(hyp_words)-3)]

        count = 0
        total = 0

        for fourgram in hyp_fourgrams:
            count += min(hyp_fourgrams.count(fourgram), ref_fourgrams.count(fourgram))
            total += 1

        if total == 0:
            return 0.0

        precision = count / total
        return precision


    @staticmethod
    def get_rouge(ref_text, hyp_text):
        
        def lcs(X, Y):
            m = len(X)
            n = len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i-1] == Y[j-1]:
                        L[i][j] = L[i-1][j-1] + 1
                    else:
                        L[i][j] = max(L[i-1][j], L[i][j-1])
            return L[m][n]
        
        ref_tokens = ref_text.lower().split()
        hyp_tokens = hyp_text.lower().split()

        lcs_length = lcs(ref_tokens, hyp_tokens)

        ref_length = len(ref_tokens)
        hyp_length = len(hyp_tokens)

        precision = lcs_length / hyp_length if hyp_length > 0 else 0
        recall = lcs_length / ref_length if ref_length > 0 else 0

        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
    
    def evaluate_text(self, ref_text, hyp_text):
        emb_score   = self.embedding_eval.get_score(ref_text,hyp_text)
        key_score   = self.key_eval.get_score(ref_text,hyp_text)
        bleu_score  = self.get_bleu4(ref_text,hyp_text)
        rouge_score = self.get_rouge(ref_text,hyp_text)

        ranking_score= 0.15*(rouge_score+bleu_score) + 0.4*key_score + 0.3*emb_score
        return ranking_score
    
    def evaluate_dummy(self, eval_lists):
        ''' list of tuples(pairs) '''
        score = 0
        pbar = tqdm(eval_lists, total=len(eval_lists))
        for i, (ref_text, hyp_text) in enumerate(pbar):
            score += self.evaluate_text(ref_text,hyp_text)
        score/=len(eval_lists)
        return score
    
    
    
    
    
    