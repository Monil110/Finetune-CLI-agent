#!/usr/bin/env python3
"""
Evaluation script to compare base vs fine-tuned model outputs.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, base_model_path, adapter_path=None, max_samples=None, batch_size=4):  # Increased batch size
        """
        Initialize evaluator with base model and optional adapter.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to LoRA adapter (optional)
            max_samples: Maximum number of samples to evaluate
            batch_size: Batch size for generation (default: 4)
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)  # Use fast tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading base model with optimizations...")
        # Use 4-bit quantization for faster inference and lower memory usage
        bnb_config = None
        torch_dtype = torch.float32
        
        if self.device == "cuda":
            bnb_config = {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': torch.bfloat16,
                'bnb_4bit_use_double_quant': True,
            }
            torch_dtype = torch.bfloat16
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2" if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
        )
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            self.base_model = torch.compile(self.base_model, mode="reduce-overhead", fullgraph=True)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")
        
        if adapter_path and os.path.exists(adapter_path):
            print("Loading fine-tuned model...")
            self.finetuned_model = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.finetuned_model.eval()
        else:
            print("No adapter provided, using base model only")
            self.finetuned_model = None
        
        self.base_model.eval()
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
        
        # Enable torch optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    def format_prompt(self, instruction):
        """Format instruction as chat prompt."""
        prompt = f"""<|user|>
{instruction}

Please provide a step-by-step plan to accomplish this task. If shell commands are needed, start each command line with the actual command.
<|assistant|>
"""
        return prompt
    
    def generate_response(self, model, instructions, max_length=128, temperature=0.3):
        """Generate responses from model in batches."""
        if isinstance(instructions, str):
            instructions = [instructions]
            
        # Batch process prompts
        prompts = [self.format_prompt(instr) for instr in instructions]
        
        # Tokenize all prompts at once
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=384,  # Further reduced context length
            padding=True,
            add_special_tokens=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Optimized generation settings for maximum speed
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,  # Fastest option
                top_p=0.95,   # Slightly more focused than before
                top_k=30,     # Reduced from 50 for speed
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,  # Prevent repetition
                early_stopping=False  # Not needed for greedy search
            )
        
        # Decode all outputs at once
        generated_texts = []
        for i in range(len(instructions)):
            output = outputs[i][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(
                output,
                skip_special_tokens=True
            )
            generated_texts.append(generated_text.strip())
        
        return generated_texts[0] if len(generated_texts) == 1 else generated_texts
    
    def extract_commands(self, text):
        """Extract shell commands from generated text."""
        commands = []
        lines = text.split('\n')
        
        command_patterns = [
            r'^(git|bash|sh|ls|cd|mkdir|rm|cp|mv|tar|gzip|gunzip|grep|find|chmod|chown)\s+',
            r'^(python|pip|conda|virtualenv|source)\s+',
            r'^(sudo|su)\s+',
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s+',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            clean_line = re.sub(r'^Step\s+\d+:\s*', '', clean_line, flags=re.IGNORECASE)
            
            for pattern in command_patterns:
                if re.match(pattern, clean_line):
                    commands.append(clean_line)
                    break
        
        return commands
    
    def compute_bleu(self, reference, candidate):
        """Compute BLEU score between reference and candidate."""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not cand_tokens:
            return 0.0
        
        return sentence_bleu(
            [ref_tokens], 
            cand_tokens, 
            smoothing_function=self.smoothing.method1
        )
    
    def compute_rouge(self, reference, candidate):
        """Compute ROUGE scores."""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def score_plan_quality(self, instruction, response):
        """
        Score plan quality on a scale of 0-2.
        0: Poor (irrelevant, incorrect, or nonsensical)
        1: Fair (partially relevant, some correct elements)
        2: Good (relevant, mostly correct, well-structured)
        """
        score = 0
        instruction_lower = instruction.lower()
        response_lower = response.lower()
        
        # Check relevance to instruction
        key_terms = []
        if 'git' in instruction_lower:
            key_terms.extend(['git', 'commit', 'push', 'pull', 'clone', 'branch'])
        if 'python' in instruction_lower or 'virtual' in instruction_lower:
            key_terms.extend(['python', 'pip', 'virtualenv', 'venv', 'activate'])
        if 'file' in instruction_lower or 'directory' in instruction_lower:
            key_terms.extend(['ls', 'mkdir', 'cd', 'cp', 'mv', 'rm'])
        if 'compress' in instruction_lower or 'archive' in instruction_lower:
            key_terms.extend(['tar', 'gzip', 'zip'])
        if 'search' in instruction_lower or 'find' in instruction_lower:
            key_terms.extend(['grep', 'find', 'search'])
        
        # Basic relevance check
        relevance_score = 0
        if key_terms:
            matches = sum(1 for term in key_terms if term in response_lower)
            relevance_score = min(matches / len(key_terms), 1.0)
        else:
            # Generic relevance check
            relevance_score = 0.5 if len(response.split()) > 10 else 0.2
        
        # Check for command structure
        commands = self.extract_commands(response)
        command_score = min(len(commands) / 3, 1.0) if commands else 0
        
        # Check for step-by-step structure
        structure_score = 0
        if any(marker in response_lower for marker in ['step', '1.', '2.', 'first', 'then', 'next']):
            structure_score = 0.5
        if len(response.split('\n')) >= 3:  # Multi-line response
            structure_score += 0.3
        
        # Combine scores
        total_score = (relevance_score * 0.5 + command_score * 0.3 + structure_score * 0.2)
        
        # Convert to 0-2 scale
        if total_score >= 0.8:
            return 2
        elif total_score >= 0.4:
            return 1
        else:
            return 0
    
    def evaluate_on_prompts(self, test_prompts, reference_responses=None):
        """
        Evaluate models on test prompts.
        
        Args:
            test_prompts: List of test instructions
            reference_responses: Optional list of reference responses
            
        Returns:
            DataFrame with evaluation results
        """
        # Limit samples if specified
        if self.max_samples:
            test_prompts = test_prompts[:self.max_samples]
            if reference_responses:
                reference_responses = reference_responses[:self.max_samples]
        
        results = []
        
        for i, prompt in enumerate(tqdm(test_prompts, desc="Evaluating prompts")):
            print(f"\nEvaluating prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            # Generate responses
            base_response = self.generate_response(self.base_model, prompt)
            
            if self.finetuned_model:
                ft_response = self.generate_response(self.finetuned_model, prompt)
            else:
                ft_response = None
            
            # Compute metrics
            result = {
                'prompt_id': i,
                'prompt': prompt,
                'base_response': base_response,
                'finetuned_response': ft_response,
                'base_plan_quality': self.score_plan_quality(prompt, base_response),
                'base_commands_count': len(self.extract_commands(base_response))
            }
            
            if ft_response:
                result.update({
                    'finetuned_plan_quality': self.score_plan_quality(prompt, ft_response),
                    'finetuned_commands_count': len(self.extract_commands(ft_response))
                })
                
                # Compare responses
                bleu_score = self.compute_bleu(base_response, ft_response)
                rouge_scores = self.compute_rouge(base_response, ft_response)
                
                result.update({
                    'bleu_score': bleu_score,
                    'rouge1': rouge_scores['rouge1'],
                    'rouge2': rouge_scores['rouge2'],
                    'rougeL': rouge_scores['rougeL']
                })
            
            # If reference responses provided, compute metrics against them
            if reference_responses and i < len(reference_responses):
                ref_response = reference_responses[i]
                
                base_vs_ref_bleu = self.compute_bleu(ref_response, base_response)
                base_vs_ref_rouge = self.compute_rouge(ref_response, base_response)
                
                result.update({
                    'reference_response': ref_response,
                    'base_vs_ref_bleu': base_vs_ref_bleu,
                    'base_vs_ref_rouge1': base_vs_ref_rouge['rouge1'],
                    'base_vs_ref_rouge2': base_vs_ref_rouge['rouge2'],
                    'base_vs_ref_rougeL': base_vs_ref_rouge['rougeL']
                })
                
                if ft_response:
                    ft_vs_ref_bleu = self.compute_bleu(ref_response, ft_response)
                    ft_vs_ref_rouge = self.compute_rouge(ref_response, ft_response)
                    
                    result.update({
                        'ft_vs_ref_bleu': ft_vs_ref_bleu,
                        'ft_vs_ref_rouge1': ft_vs_ref_rouge['rouge1'],
                        'ft_vs_ref_rouge2': ft_vs_ref_rouge['rouge2'],
                        'ft_vs_ref_rougeL': ft_vs_ref_rouge['rougeL']
                    })
            
            results.append(result)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and i % 5 == 0:
                torch.cuda.empty_cache()
        
        return pd.DataFrame(results)
    
    def generate_report(self, results_df, output_path):
        """Generate evaluation report."""
        report = []
        report.append("# Model Evaluation Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append("")
        
        if 'finetuned_response' in results_df.columns and results_df['finetuned_response'].notna().any():
            # Base model stats
            base_quality_mean = results_df['base_plan_quality'].mean()
            base_commands_mean = results_df['base_commands_count'].mean()
            
            # Fine-tuned model stats
            ft_quality_mean = results_df['finetuned_plan_quality'].mean()
            ft_commands_mean = results_df['finetuned_commands_count'].mean()
            
            report.append(f"**Base Model:**")
            report.append(f"- Average Plan Quality: {base_quality_mean:.2f} / 2.0")
            report.append(f"- Average Commands per Response: {base_commands_mean:.1f}")
            report.append("")
            
            report.append(f"**Fine-tuned Model:**")
            report.append(f"- Average Plan Quality: {ft_quality_mean:.2f} / 2.0")
            report.append(f"- Average Commands per Response: {ft_commands_mean:.1f}")
            report.append("")
            
            # Improvement metrics
            quality_improvement = ft_quality_mean - base_quality_mean
            commands_improvement = ft_commands_mean - base_commands_mean
            
            report.append(f"**Improvements:**")
            report.append(f"- Plan Quality: {quality_improvement:+.2f}")
            report.append(f"- Commands Count: {commands_improvement:+.1f}")
            report.append("")
            
            # BLEU/ROUGE scores
            if 'bleu_score' in results_df.columns:
                bleu_mean = results_df['bleu_score'].mean()
                rouge1_mean = results_df['rouge1'].mean()
                rougeL_mean = results_df['rougeL'].mean()
                
                report.append(f"**Similarity Metrics (Base vs Fine-tuned):**")
                report.append(f"- BLEU Score: {bleu_mean:.3f}")
                report.append(f"- ROUGE-1: {rouge1_mean:.3f}")
                report.append(f"- ROUGE-L: {rougeL_mean:.3f}")
                report.append("")
        
        # Individual results
        report.append("## Individual Results")
        report.append("")
        
        for idx, row in results_df.iterrows():
            report.append(f"### Prompt {idx + 1}")
            report.append(f"**Instruction:** {row['prompt']}")
            report.append("")
            
            report.append(f"**Base Model Response:**")
            report.append(f"```")
            report.append(row['base_response'])
            report.append(f"```")
            report.append(f"- Plan Quality: {row['base_plan_quality']}/2")
            report.append(f"- Commands Extracted: {row['base_commands_count']}")
            report.append("")
            
            if 'finetuned_response' in row and pd.notna(row['finetuned_response']):
                report.append(f"**Fine-tuned Model Response:**")
                report.append(f"```")
                report.append(row['finetuned_response'])
                report.append(f"```")
                report.append(f"- Plan Quality: {row['finetuned_plan_quality']}/2")
                report.append(f"- Commands Extracted: {row['finetuned_commands_count']}")
                
                # Add detailed metrics
                if 'bleu_score' in row:
                    report.append(f"**Metrics:**")
                    report.append(f"- BLEU Score: {row['bleu_score']:.3f}")
                    report.append(f"- ROUGE-1: {row['rouge1']:.3f}")
                    report.append(f"- ROUGE-2: {row['rouge2']:.3f}")
                    report.append(f"- ROUGE-L: {row['rougeL']:.3f}")
                
                # Add reference comparison metrics if available
                if 'reference_response' in row and pd.notna(row['reference_response']):
                    report.append(f"\n**Reference Comparison:**")
                    report.append(f"- BLEU (vs Reference): {row.get('ft_vs_ref_bleu', 0):.3f}")
                    report.append(f"- ROUGE-1 (vs Reference): {row.get('ft_vs_ref_rouge1', 0):.3f}")
                    report.append(f"- ROUGE-2 (vs Reference): {row.get('ft_vs_ref_rouge2', 0):.3f}")
                    report.append(f"- ROUGE-L (vs Reference): {row.get('ft_vs_ref_rougeL', 0):.3f}")
                
                report.append("")
            
            report.append("---")
            report.append("")
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to: {output_path}")


def get_default_test_prompts():
    """Get default test prompts for evaluation."""
    return [
        "Create a new Git repository and make an initial commit",
        "Set up a Python virtual environment and install pandas",
        "Find all Python files in the current directory that contain 'import numpy'",
        "Create a compressed archive of all .txt files in the current directory",
        "List all files modified in the last 24 hours, sorted by modification time",
        # Two additional edge cases
        "Recursively search for files larger than 100MB and move them to a backup directory",
        "Set up a cron job to automatically backup a directory every hour using tar and gzip"
    ]


def main():
    parser = argparse.ArgumentParser(description='Evaluate Model Performance')
    parser.add_argument('--base-model', 
                       default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Base model path')
    parser.add_argument('--adapter-path', 
                       help='Path to fine-tuned adapter')
    parser.add_argument('--test-prompts', 
                       help='JSON file with test prompts')
    parser.add_argument('--reference-responses', 
                       help='JSON file with reference responses')
    parser.add_argument('--output-dir', 
                       default='results',
                       help='Output directory for results')
    parser.add_argument('--output-name', 
                       default='evaluation',
                       help='Output file name prefix')
    parser.add_argument('--max-samples', 
                       type=int,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=1,
                       help='Batch size for generation')
    parser.add_argument('--max-length', 
                       type=int, 
                       default=256,
                       help='Maximum generation length')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load test prompts
    if args.test_prompts and os.path.exists(args.test_prompts):
        with open(args.test_prompts, 'r') as f:
            test_prompts = json.load(f)
    else:
        print("Using default test prompts")
        test_prompts = get_default_test_prompts()
    
    # Load reference responses if provided
    reference_responses = None
    if args.reference_responses and os.path.exists(args.reference_responses):
        with open(args.reference_responses, 'r') as f:
            reference_responses = json.load(f)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        args.base_model, 
        args.adapter_path, 
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    print(f"Evaluating on {len(test_prompts)} prompts...")
    results_df = evaluator.evaluate_on_prompts(test_prompts, reference_responses)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{args.output_name}_{timestamp}.csv"
    json_path = output_dir / f"{args.output_name}_{timestamp}.json"
    report_path = output_dir / f"{args.output_name}_report_{timestamp}.md"
    
    # Save CSV
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save JSON
    results_json = results_df.to_dict('records')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {json_path}")
    
    # Generate report
    evaluator.generate_report(results_df, report_path)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    if 'finetuned_response' in results_df.columns and results_df['finetuned_response'].notna().any():
        base_quality = results_df['base_plan_quality'].mean()
        ft_quality = results_df['finetuned_plan_quality'].mean()
        print(f"Base Model - Average Plan Quality: {base_quality:.2f}/2.0")
        print(f"Fine-tuned Model - Average Plan Quality: {ft_quality:.2f}/2.0")
        print(f"Improvement: {ft_quality - base_quality:+.2f}")
        
        if 'bleu_score' in results_df.columns:
            bleu_mean = results_df['bleu_score'].mean()
            print(f"Average BLEU Score: {bleu_mean:.3f}")
    else:
        base_quality = results_df['base_plan_quality'].mean()
        print(f"Base Model - Average Plan Quality: {base_quality:.2f}/2.0")


if __name__ == "__main__":
    main()