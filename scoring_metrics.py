#!/usr/bin/env python3
"""
Scoring metrics for evaluation including BLEU, ROUGE, and plan quality
"""

import re
import logging
from typing import Dict, List, Any
from collections import Counter
import math

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate various evaluation metrics"""
    
    def __init__(self):
        pass
    
    def calculate_bleu_score(self, candidate: str, reference: str, n_gram: int = 4) -> float:
        """Calculate BLEU score between candidate and reference"""
        def get_ngrams(text: str, n: int) -> List[tuple]:
            words = text.lower().split()
            return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        def calculate_precision(candidate_ngrams: List, reference_ngrams: List) -> float:
            if not candidate_ngrams:
                return 0.0
            
            candidate_counts = Counter(candidate_ngrams)
            reference_counts = Counter(reference_ngrams)
            
            clipped_counts = {}
            for ngram in candidate_counts:
                clipped_counts[ngram] = min(candidate_counts[ngram], reference_counts.get(ngram, 0))
            
            return sum(clipped_counts.values()) / sum(candidate_counts.values())
        
        # Calculate precision for n-grams from 1 to n_gram
        precisions = []
        for n in range(1, n_gram + 1):
            candidate_ngrams = get_ngrams(candidate, n)
            reference_ngrams = get_ngrams(reference, n)
            precision = calculate_precision(candidate_ngrams, reference_ngrams)
            precisions.append(precision)
        
        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0
        
        # Brevity penalty
        candidate_length = len(candidate.split())
        reference_length = len(reference.split())
        
        if candidate_length > reference_length:
            bp = 1.0
        else:
            bp = math.exp(1 - reference_length / candidate_length) if candidate_length > 0 else 0.0
        
        # BLEU score
        log_precision_sum = sum(math.log(p) for p in precisions if p > 0)
        bleu = bp * math.exp(log_precision_sum / len(precisions))
        
        return bleu
    
    def calculate_rouge_scores(self, candidate: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
        def get_tokens(text: str) -> List[str]:
            return text.lower().split()
        
        def calculate_rouge_n(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> Dict[str, float]:
            def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
                return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            
            candidate_ngrams = get_ngrams(candidate_tokens, n)
            reference_ngrams = get_ngrams(reference_tokens, n)
            
            if not reference_ngrams:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            candidate_counts = Counter(candidate_ngrams)
            reference_counts = Counter(reference_ngrams)
            
            overlap = 0
            for ngram in candidate_counts:
                overlap += min(candidate_counts[ngram], reference_counts.get(ngram, 0))
            
            precision = overlap / sum(candidate_counts.values()) if candidate_counts else 0.0
            recall = overlap / sum(reference_counts.values()) if reference_counts else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {'precision': precision, 'recall': recall, 'f1': f1}
        
        def calculate_rouge_l(candidate_tokens: List[str], reference_tokens: List[str]) -> Dict[str, float]:
            """Calculate ROUGE-L using LCS"""
            def lcs_length(seq1: List[str], seq2: List[str]) -> int:
                m, n = len(seq1), len(seq2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if seq1[i-1] == seq2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                return dp[m][n]
            
            lcs_len = lcs_length(candidate_tokens, reference_tokens)
            
            if not reference_tokens:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            precision = lcs_len / len(candidate_tokens) if candidate_tokens else 0.0
            recall = lcs_len / len(reference_tokens) if reference_tokens else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {'precision': precision, 'recall': recall, 'f1': f1}
        
        candidate_tokens = get_tokens(candidate)
        reference_tokens = get_tokens(reference)
        
        rouge1 = calculate_rouge_n(candidate_tokens, reference_tokens, 1)
        rouge2 = calculate_rouge_n(candidate_tokens, reference_tokens, 2)
        rougeL = calculate_rouge_l(candidate_tokens, reference_tokens)
        
        return {
            'rouge1': rouge1['f1'],
            'rouge2': rouge2['f1'],
            'rougeL': rougeL['f1']
        }
    
    def score_plan_quality(self, response: str, expected_commands: List[str]) -> float:
        """Score plan quality on 0-2 scale"""
        score = 0.0
        
        # Check if response contains a structured plan
        has_steps = bool(re.search(r'(step|Step|\d+\.|\d+\))', response))
        if has_steps:
            score += 0.5
        
        # Check if response explains the approach
        has_explanation = len(response.split()) > 20  # At least some explanation
        if has_explanation:
            score += 0.5
        
        # Check if commands are present and reasonable
        commands_in_response = self.extract_commands_for_scoring(response)
        if commands_in_response:
            score += 0.5
            
            # Check if commands are relevant to expected commands
            if expected_commands:
                relevance_score = self.calculate_command_relevance(commands_in_response, expected_commands)
                score += 0.5 * relevance_score
        
        return min(score, 2.0)
    
    def extract_commands_for_scoring(self, text: str) -> List[str]:
        """Extract commands for scoring purposes"""
        commands = []
        
        # Look for code blocks
        code_blocks = re.findall(r'```(?:bash|shell|sh)?\n(.*?)\n```', text, re.DOTALL)
        for block in code_blocks:
            commands.extend([cmd.strip() for cmd in block.split('\n') if cmd.strip() and not cmd.strip().startswith('#')])
        
        # Look for commands in backticks or after $
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('$ '):
                commands.append(line[2:])
            elif line.startswith('`') and line.endswith('`') and len(line) > 2:
                commands.append(line[1:-1])
        
        return commands
    
    def calculate_command_relevance(self, extracted: List[str], expected: List[str]) -> float:
        """Calculate how relevant extracted commands are to expected ones"""
        if not expected or not extracted:
            return 0.0
        
        total_relevance = 0.0
        for exp_cmd in expected:
            exp_tokens = set(exp_cmd.lower().split())
            best_match = 0.0
            
            for ext_cmd in extracted:
                ext_tokens = set(ext_cmd.lower().split())
                if exp_tokens and ext_tokens:
                    intersection = exp_tokens.intersection(ext_tokens)
                    union = exp_tokens.union(ext_tokens)
                    similarity = len(intersection) / len(union) if union else 0.0
                    best_match = max(best_match, similarity)
            
            total_relevance += best_match
        
        return total_relevance / len(expected)
    
    def calculate_command_similarity(self, extracted: List[str], expected: List[str]) -> float:
        """Calculate overall command similarity"""
        if not expected and not extracted:
            return 1.0
        if not expected or not extracted:
            return 0.0
        
        # Use Jaccard similarity on command tokens
        all_extracted_tokens = set()
        all_expected_tokens = set()
        
        for cmd in extracted:
            all_extracted_tokens.update(cmd.lower().split())
        
        for cmd in expected:
            all_expected_tokens.update(cmd.lower().split())
        
        intersection = all_extracted_tokens.intersection(all_expected_tokens)
        union = all_extracted_tokens.union(all_expected_tokens)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_all_metrics(self, evaluation_results: Dict) -> Dict:
        """Calculate all metrics for the evaluation results"""
        metrics = {
            'individual_scores': [],
            'averages': {}
        }
        
        # Process fine-tuned model results
        if 'finetuned_model' in evaluation_results and evaluation_results['finetuned_model']:
            test_cases = evaluation_results['finetuned_model'].get('test_cases', [])
            
            bleu_scores = []
            rouge_scores = []
            plan_quality_scores = []
            command_similarity_scores = []
            
            for test_case in test_cases:
                response = test_case.get('response', '')
                expected_commands = test_case.get('expected_commands', [])
                extracted_commands = test_case.get('extracted_commands', [])
                
                # Create reference text from expected commands
                reference_text = ' '.join(expected_commands)
                
                # Calculate BLEU
                bleu = self.calculate_bleu_score(response, reference_text)
                bleu_scores.append(bleu)
                
                # Calculate ROUGE
                rouge = self.calculate_rouge_scores(response, reference_text)
                rouge_scores.append(rouge)
                
                # Calculate plan quality
                plan_quality = self.score_plan_quality(response, expected_commands)
                plan_quality_scores.append(plan_quality)
                
                # Calculate command similarity
                cmd_similarity = self.calculate_command_similarity(extracted_commands, expected_commands)
                command_similarity_scores.append(cmd_similarity)
                
                # Store individual scores
                individual_score = {
                    'test_id': test_case.get('test_id', 'unknown'),
                    'bleu': bleu,
                    'rouge': rouge,
                    'plan_quality': plan_quality,
                    'command_similarity': cmd_similarity
                }
                metrics['individual_scores'].append(individual_score)
            
            # Calculate averages
            if bleu_scores:
                metrics['averages']['bleu'] = sum(bleu_scores) / len(bleu_scores)
            
            if rouge_scores:
                avg_rouge = {
                    'rouge1': sum(r['rouge1'] for r in rouge_scores) / len(rouge_scores),
                    'rouge2': sum(r['rouge2'] for r in rouge_scores) / len(rouge_scores),
                    'rougeL': sum(r['rougeL'] for r in rouge_scores) / len(rouge_scores)
                }
                metrics['averages']['rouge'] = avg_rouge
            
            if plan_quality_scores:
                metrics['averages']['plan_quality'] = sum(plan_quality_scores) / len(plan_quality_scores)
            
            if command_similarity_scores:
                metrics['averages']['command_similarity'] = sum(command_similarity_scores) / len(command_similarity_scores)
        
        return metrics