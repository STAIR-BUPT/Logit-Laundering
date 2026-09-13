
import gensim.downloader
from tqdm import tqdm
import numpy as np
from textattack.utils import Logger, to_string, truncation,find_homo, random_keyboard_neighbor, load_json
import datetime
from copy import deepcopy
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from copy import copy
from paraphraser import DipperParaphraser, Authormist
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import Levenshtein
import random
import language_tool_python
from text_OCR import text_OCR_text
import os

def check_num(x_str):
    num_str_list=[str(idx) for idx in range(10)]
    if x_str in num_str_list:
        return True
    return False

def split_sentence(sentence):
        sentence=sentence.replace('\n', ' ')
        last_idx=0
        sub_sentence_list=[]
        for idx in range(len(sentence)):
            char_=sentence[idx]
            if idx>0 and idx<len(sentence)-1 and char_=='.' and check_num(sentence[idx-1]) and check_num(sentence[idx+1]):
                continue
            if char_ in ['.', '?', ';', '!', '•']:#,' – ',',"'
                sub_sentence=sentence[last_idx:idx+1]
                if sub_sentence[0]==' ':
                    sub_sentence=sub_sentence[1:]
                if sub_sentence[-1]==' ':
                    sub_sentence=sub_sentence[:-1]
                if len(sub_sentence)<=20:
                    continue
                last_idx=idx+1
                sub_sentence_list.append(sub_sentence)
        if last_idx<len(sentence)-1:
            sub_sentence_list.append(sentence[last_idx:])
        return sub_sentence_list

def belu_func(ref_sen, ground_sen):
    reference = [ref_sen.split(' ')]
    candidate = ground_sen.split(' ')

    smooth = SmoothingFunction().method1  
    score = sentence_bleu(reference, candidate, smoothing_function=smooth)
    return score

def rouge_f1(ground_sen, candi_sen):
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(ground_sen, candi_sen)
    return scores["rouge1"].fmeasure

class GensimModel:

    def __init__(self):
        self.model_name_list=[
            'glove-twitter-25',
        ]
        self.low_model_list=[
            'glove-twitter-25',
            'glove-wiki-gigaword-50',
        ]
        self.high_model_list=[
            'word2vec-google-news-300',
            # 'glove-twitter-100',
            # 'glove-wiki-gigaword-100',
        ]
        # Lazy-load models with error handling and progress indication
        self.model_dict={}
        self.simi_words_dict={
            model_name:{}
            for model_name in self.model_name_list
        }
        self._load_models()
    
    def _load_models(self):
        """Load gensim models with error handling."""
        for model_name in self.model_name_list:
            try:
                print(f"Downloading/loading Gensim model: {model_name} (this may take a while...)")
                self.model_dict[model_name] = gensim.downloader.load(model_name)
                print(f"Successfully loaded model: {model_name}")
            except Exception as e:
                print(f"Warning: failed to load model {model_name}: {e}")
                print("If there are network issues, download the model manually or check your connection.")
                # On failure, set to None and skip later
                self.model_dict[model_name] = None

    def find_simi_words(self, target_word, simi_num=10):
        found_words=[]
        words_scores=[]
        for model_name in self.model_name_list:
            try:
                # Skip if model failed to load
                if self.model_dict.get(model_name) is None:
                    continue
                    
                if target_word in self.simi_words_dict[model_name]:
                    similar_words=self.simi_words_dict[model_name][target_word]
                else:
                    similar_words = self.model_dict[model_name].most_similar(target_word)
                    self.simi_words_dict[model_name][target_word]=similar_words
                for (word, score) in similar_words:
                    if word not in found_words:
                        found_words.append(word)
                        words_scores.append((word, score))
                    if len(found_words)>simi_num:
                        break
            except Exception as e:
                # print(model_name+' do not find')
                continue
        if len(found_words)>simi_num:
            found_words=found_words[0:simi_num]
        return found_words

zwsp = chr(0x200B)
zwj = chr(0x200D)
ZWNJ=chr(0x200C)
WJ=chr(0x2060)
VS16=chr(0xFE0F)

class RandomAttack:
    def __init__(
        self,
        tokenizer,
        logger=None,
        ref_model=None,
        device='cuda',
        wm_name='',
        ori_flag=False,
        wm_detector=None,
        ppl_checker=None,
        char_op=2,
        def_stl='',
    ):
        self.char_op=char_op
        # If char_op <= 0, enable random operation selection automatically
        self.char_op_random = (char_op <= 0)
        self.gensimi=None 
        self.paraphraser=None
        self.token_len_flag=True
        self.simi_num_for_token=5
        self.special_char=[zwsp, zwj]#, zwj, ZWNJ' ',, [WJ]#, VS16, zwj, ZWNJ,
        self.device=device if torch.cuda.is_available() else "cpu"
        self.wm_detector=wm_detector
        self.ppl_checker=ppl_checker
        self.ori_flag=ori_flag
        self.def_stl=def_stl
        # Initialize ref_model attribute
        self.ref_model = None
        # self.spellchecker = SpellChecker()
        # Lazily initialize spellchecker only when needed (operation==6 and def_stl is set)
        self.spellchecker = None
        if isinstance(tokenizer,str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.vocab_size=self.tokenizer.vocab_size
        try:
            if isinstance(ref_model, str):
                self.ref_model = AutoModelForSequenceClassification.from_pretrained(ref_model)
                self.ref_model.to(self.device)
        except:
            self.ref_model=None
        
        if logger is None:
            log_path='attack_log/Rand/'
            os.makedirs(log_path, exist_ok=True)
            self.log=Logger(
                log_path+'RandomAttack'+'-'.join([
                    wm_name, 
                ])+'-'+str(datetime.datetime.now())[0:-7]+'.log',
                level='debug', 
                screen=False
            )
        else:
            self.log=logger
        self.log_info('\n')

    def log_info(self, info=''):
        if isinstance(info, str):
            self.log.logger.info(info)
        elif isinstance(info, dict):
            keys='\t'.join([key for key in info])
            values='\t'.join([str(info[key]) for key in info])
            self.log.logger.info(keys)
            self.log.logger.info(values)
        else:
            info=to_string(info)
            self.log.logger.info(info)

    def truncation(self, text, max_token_num=100):
        new_text, token_num=truncation(
            text, 
            # self.tokenizer, 
            max_token_num=max_token_num,
            # min_num=max_token_num*0.65
        )
        if token_num<max_token_num*0.65:
            return '', 0
        return new_text, token_num

    def substitute(self, token):
        if self.gensimi is None:
            self.gensimi=GensimModel()
        space_flag=False
        if ' ' in token:
            space_flag=True
        token=token.replace(' ', '').lower()
        if self.token_len_flag:
            if len(token)<=2:
                return []
            # if token in ['that', 'the', 'to', 'of', 'is', 'are','be','on','in','it','an','and','for']:
            #     return []
        # simi_token_ids=[]
        simi_tokens=self.gensimi.find_simi_words(token, simi_num=self.simi_num_for_token)

        if hasattr(self, 'token_fq_path'):
            # filename='_'.join([
            #     'TokenFq',wm_name, 
            #     dataset_name.replace('/','_'), 
            #     llm_name.replace('/','_'), 
            # ])+'.json'
            if hasattr(self, 'token_fq')==False:
                self.token_fq=load_json(self.token_fq_path)
            simi_tokens_fq={}
            for tmp in simi_tokens:
                tmp=tmp.lstrip("Ġ▁")
                simi_tokens_fq[tmp]=self.token_fq.get(tmp, 0)
            simi_tokens = sorted(simi_tokens_fq, key=simi_tokens_fq.get)#, reverse=True

        if space_flag:
            for idx in range(len(simi_tokens)):
                if simi_tokens[idx][0]!=' ':
                    simi_tokens[idx]=' '+simi_tokens[idx]
        
        return simi_tokens
    
    def ref_score(self, sentences, target_class):
        if self.ori_flag:
            if isinstance(sentences, str):
                sentences=[sentences]
            return [
                self.wm_detector(sentence)['score']
                for sentence in sentences
            ]
        # Use getattr for safe access; fall back to None if attribute is missing
        ref_model = getattr(self, 'ref_model', None)
        if ref_model is None:
            # Return a zero array matching the length of sentences
            if isinstance(sentences, str):
                return [0]
            return [0] * len(sentences)
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(self.device) for k, v in inputs.items()}
        # Evaluate fitness using the fine-tuned classification model
        with torch.no_grad():
            outputs = ref_model(**batch)
        logits = outputs.logits
        fitness=logits
        # predictions = torch.softmax(logits, dim=1)

        # Define a fitness value based on the target misclassification
        # fitness = predictions[:, target_class]

        # return fitness.item()
        return fitness.cpu().detach().numpy().reshape((-1))

    def get_adv(
        self, sentence, atk_style,
        max_edit_rate=0.3,
        atk_times=1, target_class=0,
        batch_size=8
    ): 
        self.atk_style=atk_style
        adv_sentence_list=[]
        if atk_style=='char':
            atk_method=self.char_attack1
        elif atk_style=='token':
            atk_method=self.token_attack
        elif atk_style=='token_fq':
            atk_method=self.token_attack
        elif atk_style=='agg_sentence':
            atk_method=self.agg_sentence
        elif atk_style=='sentence':
            atk_method=self.sentence_attack
        elif atk_style=='ende':
            atk_method=self.ende_attack
        elif atk_style=='low':
            atk_method=self.low_attack
        elif atk_style=='mix_char':
            atk_method=self.char_attack1
            sentence=self.low_attack(sentence)[0]
            sentence=self.ende_attack(sentence)[0]
        elif atk_style=='mix_token':
            atk_method=self.token_attack
            sentence=self.low_attack(sentence)[0]
            sentence=self.ende_attack(sentence)[0]
        elif 'sand' in atk_style:
            atk_method=self.sand_attack
        else:
            import textattack
            # import textattack.attack_sems
            self.model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(self.ref_model, self.tokenizer)
            self.grad_method = getattr(textattack.attack_recipes, atk_style).build(
                self.model_wrapper, 
                query_budget=50,
                # temperature=self.temperature,
                # max_single_query=20,
                # edit_distance=len(sentence)*max_edit_rate,
            )
            atk_method=self.grad_attack
            
        # Not using ref_model for scoring; if atk_times > 1, run only once
        if atk_times > 1:
            atk_times = 1
        
        for idx in range(atk_times):
            adv_sentence, edit_dist=atk_method(
                sentence, 
                max_edit_rate=max_edit_rate
            )
            tmp_rlt={
                'sentence':adv_sentence, 
                'edit_dist':edit_dist,
            }
            adv_sentence_list.append(tmp_rlt)
        
        # Return the first (and only) attack result
        return adv_sentence_list[0]

    def token_attack(
        self, sentence, 
        max_edit_rate=0.3,
    ):
        # token_ids=self.tokenizer.encode(sentence)
        # token_ids=remove_repeat(token_ids)
        token_list=sentence.split()#[self.tokenizer.decode(t, skip_special_tokens=True) for t in token_ids]
        token_num=len(token_list)
        
        max_edit_dist=round(token_num*max_edit_rate)
        rand_tokens=np.random.choice(token_num, min(token_num, max_edit_dist*2), replace=False)

        edit_dist=0
        new_token_list=deepcopy(token_list)
        for token_id in rand_tokens:
            tmp_token=token_list[token_id]
            tmp_subst=self.substitute(tmp_token)
            if len(tmp_subst)>0:
                new_token_list[token_id]=tmp_subst[0]
                edit_dist+=1
            # else:
            #     print()
            if edit_dist>=max_edit_dist:
                break

        new_sentence=' '.join(new_token_list)

        return new_sentence, edit_dist
    
    def char_attack(
        self, sentence, 
        max_edit_rate=0.3,
    ):
        token_ids=self.tokenizer.encode(sentence, add_special_tokens=False)
        token_ids_dict={idx:t for idx, t in enumerate(token_ids)}
        token_dict={idx:self.tokenizer.decode(t, skip_special_tokens=True) for idx, t in enumerate(token_ids)}
        token_num=len(token_ids)
        
        max_edit_dist=round(token_num*max_edit_rate)
        rand_tokens=np.random.choice(token_num, min(token_num, max_edit_dist*3), replace=False)
        rand_opt=np.random.randint(0, 4, min(token_num, max_edit_dist*3))

        edit_dist=0
        new_token_ids_dict=deepcopy(token_ids_dict)
        for ids, token_idx in enumerate(rand_tokens):
            tmp_token=token_dict[token_idx]
            
            half_token_len=len(tmp_token)//2
            if half_token_len<=1:
                continue
            
            tmp_opt=2#rand_opt[ids]
            rand_idx=np.random.randint(2, len(tmp_token)-1)
            rand_char_id=np.random.choice(len(self.special_char))
            rand_char=self.special_char[rand_char_id]
            
            if tmp_opt==0: # delete
                tmp_token=tmp_token[0:rand_idx]+tmp_token[rand_idx+1:]
            elif tmp_opt==1: # insert
                tmp_token=tmp_token[0:rand_idx]+rand_char+tmp_token[rand_idx:]
            elif tmp_opt==2: # substitute
                tmp_char=tmp_token[rand_idx]
                tmp_token=tmp_token[0:rand_idx]+find_homo(tmp_char)+tmp_token[rand_idx+1:]
            elif tmp_opt==3: # reorder
                tmp_token=tmp_token[0:rand_idx-1]+tmp_token[rand_idx+1]+tmp_token[rand_idx]+tmp_token[rand_idx+2:]

            new_token_ids_dict[token_idx]=self.tokenizer.encode(tmp_token, add_special_tokens=False)
            edit_dist+=1
            if edit_dist>=max_edit_dist:
                break
        new_token_ids=[]
        for idx in new_token_ids_dict:
            if isinstance(new_token_ids_dict[idx], list):
                new_token_ids+=new_token_ids_dict[idx]
            else:
                new_token_ids.append(new_token_ids_dict[idx])
        new_sentence=self.tokenizer.decode(new_token_ids)
        return new_sentence, edit_dist 
    
    def char_attack1(
        self, sentence, 
        max_edit_rate=0.3,
    ):
        tokens = sentence.split()
        edited_sentence = list(tokens)
        selected_tokens = []
        token_num=len(tokens)
        max_edit_dist=round(token_num*max_edit_rate)
        solution=np.random.choice(token_num, min(token_num, max_edit_dist*2), replace=False)

        for i in solution:
            if len(selected_tokens) < max_edit_dist:  # Enforce max edits

                len_t=len(edited_sentence[i])
                sep_len=3
                m_num=len_t//sep_len
                if m_num==0:
                    continue
                m_locs=[int(len_t//2)]
                # m_locs=[int((m_num/len_t)*(j)*len_t)+(sep_len//2) for j in range(m_num)]
                # half_token_len=len(edited_sentence[i])//2
                # if half_token_len<=1:
                #     continue

                selected_tokens.append(i)

                # # Treat operation as part of the solution
                # operation = solution[len(self.tokens) + i]  # Operation encoded in the extended solution
                # operation=gene
                # If char_op <= 0, randomly choose an operation type (1-5)
                if self.char_op_random or self.char_op <= 0:
                    operation = random.choice([1, 2, 3, 4, 5])  # randomly select one of the five ops
                else:
                    operation = self.char_op  # use the specified operation type

                tmp_token=copy(edited_sentence[i])
                for m_loc in m_locs:
                    if m_loc>=(len_t-1):
                        continue
                    if operation == 1:  # Delete
                        tmp_token = tmp_token[:m_loc] + tmp_token[m_loc+1:]
                    elif operation == 2:  # homo
                        tmp_char=tmp_token[m_loc]
                        tmp_token = tmp_token[:m_loc] +find_homo(tmp_char)+ tmp_token[m_loc+1:] 
                    elif operation == 3:  # Insert
                        tmp_token = tmp_token[:m_loc] + random.choice(self.special_char)+ tmp_token[m_loc:]
                    elif operation == 4:  # swap
                        tmp_token = tmp_token[:m_loc] + tmp_token[m_loc+1]+tmp_token[m_loc]+ tmp_token[m_loc+2:]
                    elif operation == 5:  # typo
                        tmp_char=tmp_token[m_loc]
                        tmp_token = tmp_token[:m_loc] + random_keyboard_neighbor(tmp_char)+ tmp_token[m_loc+1:]
                    elif operation == 6:  # c homo
                        tmp_char=tmp_token[m_loc]
                        tmp_token_list=[
                            tmp_token[:m_loc] + tmp_token[m_loc+1]+find_homo(tmp_char)+ tmp_token[m_loc+2:], #2 4
                            tmp_token[:m_loc] + find_homo(random_keyboard_neighbor(tmp_char))+ tmp_token[m_loc+1:], #2 5
                            tmp_token[:m_loc]+ random.choice(self.special_char) + find_homo(tmp_token[m_loc])+ tmp_token[m_loc+1:],
                        ]
                        dist_dict={}
                        for idx, word in enumerate(tmp_token_list):
                            # c_word=''#self.spellchecker.correction(word)
                            if 'ocr' in self.def_stl:
                                c_word=text_OCR_text(word, style='ocr_t')
                            else:
                                # Lazy-initialize spellchecker
                                if self.spellchecker is None:
                                    try:
                                        self.spellchecker = language_tool_python.LanguageTool('en-US')
                                    except Exception as e:
                                        # On failure (e.g. incompatible Java version), keep original word
                                        print(f"Warning: LanguageTool initialization failed ({e}); skipping spell check.")
                                        self.spellchecker = False  # mark as unavailable

                                if self.spellchecker is False:
                                    # LanguageTool unavailable; use original word
                                    c_word = word
                                else:
                                    matches = self.spellchecker.check(word)
                                    c_word=language_tool_python.utils.correct(word, matches)
                                    if c_word is None:
                                        c_word=word
                            c_word=c_word.lower()
                            word=word.lower()
                            d0=Levenshtein.distance(word, c_word)
                            d1=Levenshtein.distance(word, tmp_token)
                            d2=Levenshtein.distance(c_word, tmp_token)
                            # if d1*d2==0:
                            #     dc=d0+len(c_word)-len(word)
                            # else:
                            #     dc=d0+len(c_word)-len(word)-1/(d1*d2)
                            dc=d1*0.5+d2#
                            dist_dict[idx]=(dc, tmp_token, word, c_word, d0, d1, d2)
                        sorted_keys_desc = sorted(dist_dict.keys(), key=lambda k: dist_dict[k][0], reverse=True)#
                        tmp_token=tmp_token_list[sorted_keys_desc[0]]#
                edited_sentence[i]=tmp_token
        # Reconstruct sentence
        new_sentence = " ".join(edited_sentence)
        edit_dist=len(selected_tokens)
        return new_sentence, edit_dist 
    
    def ende_attack(
        self, sentence, max_edit_rate=None
    ):
        if self.tokenizer is not None:
            tmp_ids=self.tokenizer.encode(sentence, add_special_tokens=False)
            new_sentence=self.tokenizer.decode(tmp_ids, skip_special_tokens=True)
        return new_sentence, 0 
    
    def low_attack(
        self, sentence, max_edit_rate=None
    ):
        new_sentence=sentence.lower()
        return new_sentence, 0 
    
    def sand_attack(self, sentence, max_edit_rate=None):
        T=10#min(int(max_edit_rate/0.02),10)
        new_sentence=deepcopy(sentence)
        if self.ori_flag:
            ori_score=self.wm_detector(new_sentence)['score']
        else:
            ori_score=self.ref_score(new_sentence, 0)
            # ori_score=self.ppl_checker(new_sentence)
        ori_ids=self.tokenizer.encode(sentence, add_special_tokens=False)
        for idx in range(T):
            if 'token' in self.atk_style:
                tmp_sentence, _=self.token_attack(sentence=new_sentence, max_edit_rate=min(max_edit_rate/T,0.02))
            else:
                tmp_sentence, _=self.char_attack1(sentence=new_sentence, max_edit_rate=min(max_edit_rate/T,0.02))
            if self.ori_flag:
                tmp_score=self.wm_detector(tmp_sentence)['score']
                if tmp_score<ori_score:
                    ori_score=tmp_score
                    new_sentence=tmp_sentence
            else:
                tmp_score=self.ref_score(tmp_sentence, 0)
                if tmp_score<ori_score:
                    ori_score=tmp_score
                    new_sentence=tmp_sentence
                # tmp_score=self.ppl_checker(tmp_sentence)
                # if (tmp_score-ori_score)/ori_score<1.5:
                #     new_sentence=tmp_sentence
            
            new_ids=self.tokenizer.encode(new_sentence, add_special_tokens=False)
            edit_dist=  Levenshtein.distance(ori_ids, new_ids)
            # if edit_dist/len(ori_ids)>=max_edit_rate:
            #     break
        return new_sentence, edit_dist
    
    def agg_sentence(self, sentence, max_edit_rate=None):
        if self.paraphraser is None:
            self.paraphraser=Authormist()

        new_sentence = self.paraphraser.paraphrase(sentence)
        ori_ids=self.tokenizer.encode(sentence, add_special_tokens=False)
        new_ids=self.tokenizer.encode(new_sentence, add_special_tokens=False)
        edit_dist=  Levenshtein.distance(ori_ids, new_ids)
        return new_sentence, edit_dist

    def sentence_attack(
        self, sentence, max_edit_rate=None
    ):
        if self.paraphraser is None:
            # Check for device override
            paraphraser_devices = getattr(self, 'paraphraser_devices', None)
            # CUDA_VISIBLE_DEVICES already restricts visible devices; pass device_map="auto"
            self.paraphraser = DipperParaphraser(device_map="auto")

        # Map max_edit_rate to lex_diversity and order_diversity
        # max_edit_rate range: 0.0-1.0
        # lex_diversity allowed values: 0, 20, 40, 60, 80, 100 (multiples of 20)
        # order_diversity allowed values: 0, 20, 40, 60, 80, 100 (multiples of 20)
        if max_edit_rate is None:
            max_edit_rate = 0.3  # default

        # Map max_edit_rate (0.0-1.0) to lex_diversity (0-100, multiples of 20)
        # e.g. 0.0->0, 0.2->20, 0.4->40, 0.6->60, 0.8->80, 1.0->100
        lex_diversity = int(round(max_edit_rate * 100 / 20) * 20)
        lex_diversity = max(0, min(100, lex_diversity))  # clamp to 0-100

        # order_diversity can be adjusted; keeping it at 0 preserves sentence order
        # Uncomment below to scale order_diversity with max_edit_rate:
        # order_diversity = int(round(max_edit_rate * 100 / 20) * 20)
        # order_diversity = max(0, min(100, order_diversity))
        order_diversity = 0  # keep original order
        
        sentence=sentence.replace(';', '.')
        new_sentence=self.paraphraser.paraphrase(
            sentence, lex_diversity=lex_diversity, order_diversity=order_diversity, sent_interval=1, 
            do_sample=True, top_p=0.75, top_k=None, max_length=512
        )
        
        ori_ids=self.tokenizer.encode(sentence, add_special_tokens=False)
        new_ids=self.tokenizer.encode(new_sentence, add_special_tokens=False)
        edit_dist=  Levenshtein.distance(ori_ids, new_ids)
        return new_sentence, edit_dist
    
    def grad_attack(
        self, sentence, max_edit_rate=None
    ):
        attacked=self.grad_method.attack(
            sentence, 0, 
        )
        
        cls_score=attacked.perturbed_result.score
        num_queries=attacked.perturbed_result.num_queries
        budget=attacked.original_result.attacked_text.words_diff_num(attacked.perturbed_result.attacked_text)
        attacked_text=attacked.perturbed_result.attacked_text.text
        # attacked_rlt=self.wm_detector(attacked_text)
        return attacked_text, budget