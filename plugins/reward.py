# all reward funcs I used

import os
import re
import math
from typing import List, Sequence
from swift.rewards import ORM, orms

# supposed to be a fast checker backend?

try:
    from rapidfuzz.distance import Levenshtein as _rf_levenshtein
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import Levenshtein

def get_dominant_script(text: str) -> str:
    """
    tests a couple multilingual inp->english out bugs I had in the P99
    """
    
    if not text: return "None"
    
    cyrillic = len(re.findall(r'[\u0400-\u04FF]', text)) # Cyrillic range: \u0400-\u04FF
    latin = len(re.findall(r'[a-zA-Z]', text)) # Latin range: a-zA-Z
    cjk = len(re.findall(r'[\u4e00-\u9fff]', text)) # CJK range

    counts = {"Cyrillic": cyrillic, "Latin": latin, "CJK": cjk}
  
    return max(counts, key=counts.get)

def _get_clean_texts(completions, kwargs):
    """
    text grab/normalize helper
    """
  
    def extract_ocr(text):
        if text is None: return ''
        match = re.search(r'<ocr>(.*?)</ocr>', text, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else text

    def normalize(text):
        if text is None: return ''
        return ' '.join(text.split())

    solution = kwargs.get('solution') or kwargs.get('response') or kwargs.get('ground_truth')
    if not solution: return None, None

    gt_raw = solution[0] if isinstance(solution, list) else solution
    if isinstance(gt_raw, dict): gt_raw = gt_raw.get('content', '')
    elif isinstance(gt_raw, list): gt_raw = gt_raw[-1].get('content', '') if gt_raw else ''

    gt_text = normalize(extract_ocr(str(gt_raw)))
    pred_texts = [normalize(extract_ocr(c)) for c in completions]
    return pred_texts, gt_text

def _fast_cer(s1: str, s2: str) -> float:
    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 1.0
    if _HAS_RAPIDFUZZ:
        dist = _rf_levenshtein.distance(s1, s2)
    else:
        dist = Levenshtein.distance(s1, s2)
    return dist / max(len(s1), len(s2))

# rewards

class OCR_CER_Reward(ORM):
    """
    exponentially scaled CER component (70% of the reward)
    """
    def __call__(self, completions, **kwargs) -> List[float]:
        preds, gt = _get_clean_texts(completions, kwargs)
        if gt is None: return [0.0] * len(completions)

        rewards = []
        for p in preds:
            cer = _fast_cer(p, gt)
            # 70% of total reward budget
            score = 0.7 * math.exp(-1500.0 * (cer ** 2))
            rewards.append(max(0.01, score))
        
        return rewards

class OCR_Survival_Reward(ORM):
    """
    Logs how far the model gets perfectly from the start, by letter (Max 0.3).
    """
  
    def __call__(self, completions, **kwargs) -> List[float]:
        preds, gt = _get_clean_texts(completions, kwargs)
        if not gt: return [0.3] * len(completions)
        if preds is None: return [0.0] * len(completions)

        rewards = []
        for p in preds:
            match_count = 0
            for char_p, char_g in zip(p, gt):
                if char_p == char_g: 
                  match_count += 1
                else: 
                  break

            # 30% of total reward
            survival_rate = match_count / len(gt)
            rewards.append(0.3 * survival_rate)
        return rewards

class OCR_Moonshot_Reward(ORM):
    """
    perfection bonus (0.5 if perfect, else 0)
    """
    
    def __call__(self, completions, **kwargs) -> List[float]:
        preds, gt = _get_clean_texts(completions, kwargs)
        if gt is None: return [0.0] * len(completions)

        rewards = []
        for p in preds:
            # perfection check
            if p == gt and len(gt) > 0:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        return rewards

class OCR_Length_Reward(ORM):
    """
    rewards the model for matching the exact length of the gt
    """
    def __call__(self, completions, **kwargs) -> List[float]:
        preds, gt = _get_clean_texts(completions, kwargs)
        if gt is None: return [0.0] * len(completions)

        rewards = []
        gt_len = len(gt)
        for p in preds:
            if gt_len == 0:
                rewards.append(1.0 if len(p) == 0 else 0.0)
                continue

            # len ratio (0.0 to 1.0)
            # perfect len match gets 0.2
            diff = abs(len(p) - gt_len)
            
            # fast decay falls off fast if length is wrong
            score = 0.2 * math.exp(-0.5 * diff)
            rewards.append(score)
        
        return rewards

class OCR_AntiYap_Reward(ORM):
    """
    stop infinite loops and hallucinations
    1. Penalty for length > 1.5x gt
    2. Massive penalty for length > 2x gt
    3. Penalty for n-gram repetition (loops)
    """
    def __call__(self, completions, **kwargs) -> List[float]:
        preds, gt = _get_clean_texts(completions, kwargs) # Using helper from prev turn
        if gt is None: return [0.0] * len(completions)

        rewards = []
        gt_len = len(gt)

        for p in preds:
            penalty = 0.0
            p_len = len(p)

            if gt_len > 0:
                ratio = p_len / gt_len
                if ratio > 2.0: penalty -= 2.0  # mass hit
                elif ratio > 1.3: penalty -= 0.5 # Smaller hit

            # if the same 10-char sequence repeats more than 3 times, it's a loop
            found_reps = 0
            for i in range(len(p) - 20):
                chunk = p[i:i+10]
                if p.count(chunk) > 3:
                    found_reps += 1
                    break
            if found_reps > 0: penalty -= 1.5

            rewards.append(penalty)
        return rewards

class OCR_ScriptGuard_Reward(ORM):
    """
    Punishes this russian->eng thing I saw, If GT is Cyrillic and Pred is Latin,
    the ViT isn't grounded, drop reward
    """
    def __call__(self, completions, **kwargs) -> List[float]:
        preds, gt = _get_clean_texts(completions, kwargs)
        if not gt or len(gt) < 5: return [0.0] * len(completions)

        gt_script = get_dominant_script(gt)
        rewards = []

        for p in preds:
            if len(p) < 5:
                rewards.append(0.0)
                continue

            pred_script = get_dominant_script(p)

            if gt_script != "None" and pred_script != gt_script:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)
        return rewards

# rewards
orms['ocr_cer'] = OCR_CER_Reward
orms['ocr_survival'] = OCR_Survival_Reward
orms['ocr_moonshot'] = OCR_Moonshot_Reward
orms['ocr_length'] = OCR_Length_Reward
orms['ocr_anti_yap'] = OCR_AntiYap_Reward
orms['ocr_script_guard'] = OCR_ScriptGuard_Reward
