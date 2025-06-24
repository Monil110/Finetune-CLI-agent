#!/usr/bin/env python3
"""
Dynamic evaluation framework for CLI agents.
Provides utilities for comprehensive evaluation of command-line task performance.
"""

import os
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

import pandas as pd
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    prompt_id: str
    instruction: str
    generated_plan: str
    extracted_commands: List[str]
    plan_quality_score: float
    execution_results: List[Dict[str, Any]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class CommandValidator:
    """Validates and categorizes shell commands."""
    
    def __init__(self):
        self.safe_commands = {
            'read_only': ['ls', 'pwd', 'whoami', 'date', 'cat', 'head', 'tail', 'wc', 'grep', 'find'],
            'git': ['git status', 'git log', 'git diff', 'git branch', 'git remote'],
            'python': ['python --version', 'pip list', 'pip show'],
            'system_info': ['df -h', 'free -h', 'ps aux', 'top', 'uname -a']
        }
        
        self.potentially_dangerous = [
            'rm', 'rmdir', 'mv', 'cp', 'chmod', 'chown', 'sudo', 'su',
            'dd', 'mkfs', 'fdisk', 'mount', 'umount', 'kill', 'killall'
        ]
    
    def categorize_command(self, command: str) -> str:
        """Categorize a command by safety level."""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return 'empty'
        
        base_command = cmd_parts[0]
        
        # Check if it's a safe read-only command
        for category, commands in self.safe_commands.items():
            if any(command.startswith(safe_cmd) for safe_cmd in commands):
                return f'safe_{category}'
        
        # Check if potentially dangerous
        if base_command in self.potentially_dangerous:
            return 'dangerous'
        
        # Check common patterns
        if base_command in ['python', 'pip', 'conda']:
            return 'python_related'
        elif base_command == 'git':
            return 'git_related'
        elif base_command in ['tar', 'gzip', 'gunzip', 'zip', 'unzip']:
            return 'compression'
        elif base_command in ['mkdir', 'touch', 'echo']:
            return 'file_creation'
        
        return 'unknown'
    
    def is_safe_to_execute(self, command: str) -> bool:
        """Check if command is safe to execute in sandbox."""
        category = self.categorize_command(command)
        return category.startswith('safe_') or category in ['python_related', 'file_creation']


class SandboxEnvironment:
    """Manages a sandbox environment for safe command execution."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(tempfile.mkdtemp(prefix='cli_eval_'))
        self.base_dir.mkdir(exist_ok=True)
        self.original_cwd = os.getcwd()
        
        # Create some test files and directories
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Set up test files and directories."""
        # Create test directories
        (self.base_dir / 'test_dir').mkdir(exist_ok=True)
        (self.base_dir / 'backup').mkdir(exist_ok=True)
        
        # Create test files
        test_files = {
            'test.txt': 'This is a test file\nWith multiple lines\n',
            'data.csv': 'name,age,city\nJohn,25,NYC\nJane,30,LA\n',
            'script.py': 'import numpy as np\nprint("Hello World")\n',
            'README.md': '# Test Project\nThis is a test project.\n'
        }
        
        for filename, content in test_files.items():
            (self.base_dir / filename).write_text(content)
    
    def execute_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute command in sandbox environment."""
        try:
            # Change to sandbox directory
            os.chdir(self.base_dir)
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_dir
            )
            
            return {
                'command': command,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': time.time(),
                'status': 'completed'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'command': command,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'execution_time': timeout,
                'status': 'timeout'
            }
        except Exception as e:
            return {
                'command': command,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0,
                'status': 'error'
            }
        finally:
            # Return to original directory
            os.chdir(self.original_cwd)
    
    def cleanup(self):
        """Clean up sandbox environment."""
        try:
            os.chdir(self.original_cwd)
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not clean up sandbox: {e}")


class DynamicEvaluator:
    """Dynamic evaluation framework for CLI agents."""
    
    def __init__(self, agent_class, agent_kwargs: Dict[str, Any]):
        """
        Initialize evaluator with agent class and configuration.
        
        Args:
            agent_class: The agent class to evaluate
            agent_kwargs: Keyword arguments for agent initialization
        """
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.validator = CommandValidator()
        
        # Initialize agent
        self.agent = agent_class(**agent_kwargs)
    
    def extract_commands_from_plan(self, plan_text: str) -> List[str]:
        """Extract commands from generated plan."""
        commands = []
        lines = plan_text.split('\n')
        
        # Command patterns
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
            
            # Clean up line (remove step numbers, etc.)
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            clean_line = re.sub(r'^Step\s+\d+:\s*', '', clean_line, flags=re.IGNORECASE)
            clean_line = re.sub(r'^[-*]\s*', '', clean_line)  # Remove bullet points
            
            # Check if it matches command patterns
            for pattern in command_patterns:
                if re.match(pattern, clean_line):
                    commands.append(clean_line)
                    break
        
        return commands
    
    def evaluate_plan_quality(self, instruction: str, plan: str) -> Dict[str, float]:
        """Evaluate plan quality with multiple metrics."""
        scores = {}
        
        # 1. Relevance Score
        scores['relevance'] = self._compute_relevance_score(instruction, plan)
        
        # 2. Command Appropriateness
        commands = self.extract_commands_from_plan(plan)
        scores['command_appropriateness'] = self._compute_command_appropriateness(instruction, commands)
        
        # 3. Structure Score
        scores['structure'] = self._compute_structure_score(plan)
        
        # 4. Completeness Score
        scores['completeness'] = self._compute_completeness_score(instruction, plan)
        
        # 5. Safety Score
        scores['safety'] = self._compute_safety_score(commands)
        
        # Overall score (weighted average)
        weights = {
            'relevance': 0.3,
            'command_appropriateness': 0.25,
            'structure': 0.2,
            'completeness': 0.15,
            'safety': 0.1
        }
        
        scores['overall'] = sum(scores[metric] * weight for metric, weight in weights.items())
        
        return scores
    
    def _compute_relevance_score(self, instruction: str, plan: str) -> float:
        """Compute relevance of plan to instruction."""
        instruction_lower = instruction.lower()
        plan_lower = plan.lower()
        
        # Extract key terms from instruction
        key_terms = []
        
        # Technology-specific terms
        tech_terms = {
            'git': ['git', 'repository', 'commit', 'push', 'pull', 'clone', 'branch'],
            'python': ['python', 'pip', 'virtual', 'environment', 'venv', 'conda'],
            'file': ['file', 'directory', 'folder', 'ls', 'find', 'search'],
            'compress': ['compress', 'archive', 'tar', 'gzip', 'zip'],
            'backup': ['backup', 'copy', 'sync', 'archive']
        }
        
        for tech, terms in tech_terms.items():
            if tech in instruction_lower:
                key_terms.extend(terms)
        
        # Generic command terms
        if 'create' in instruction_lower:
            key_terms.extend(['mkdir', 'touch', 'new'])
        if 'delete' in instruction_lower or 'remove' in instruction_lower:
            key_terms.extend(['rm', 'delete', 'remove'])
        if 'list' in instruction_lower:
            key_terms.extend(['ls', 'list', 'show'])
        
        if not key_terms:
            return 0.5  # Neutral score if no specific terms found
        
        # Count matches
        matches = sum(1 for term in key_terms if term in plan_lower)
        return min(matches / len(key_terms), 1.0)
    
    def _compute_command_appropriateness(self, instruction: str, commands: List[str]) -> float:
        """Compute appropriateness of extracted commands."""
        if not commands:
            return 0.1
        
        appropriate_count = 0
        
        for command in commands:
            category = self.validator.categorize_command(command)
            
            # Check if command category matches instruction intent
            if self._command_matches_intent(instruction, command, category):
                appropriate_count += 1
        
        return appropriate_count / len(commands)
    
    def _command_matches_intent(self, instruction: str, command: str, category: str) -> bool:
        """Check if command matches the intent of instruction."""
        instruction_lower = instruction.lower()
        command_lower = command.lower()
        
        # Git-related
        if 'git' in instruction_lower and category == 'git_related':
            return True
        
        # Python-related
        if any(term in instruction_lower for term in ['python', 'pip', 'virtual', 'environment']) and category == 'python_related':
            return True
        
        # File operations
        if any(term in instruction_lower for term in ['file', 'directory', 'create', 'list']) and category in ['file_creation', 'safe_read_only']:
            return True
        
        # Compression
        if any(term in instruction_lower for term in ['compress', 'archive', 'tar', 'gzip']) and category == 'compression':
            return True
        
        # Generic match
        if any(term in command_lower for term in instruction_lower.split() if len(term) > 3):
            return True
        
        return False
    
    def _compute_structure_score(self, plan: str) -> float:
        """Compute structural quality of plan."""
        lines = [line.strip() for line in plan.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return 0.2
        
        score = 0.0
        
        # Check for step indicators
        step_indicators = sum(1 for line in lines if re.match(r'^\d+[\.\)]|^Step\s+\d+|^[-*]\s+', line))
        if step_indicators > 0:
            score += 0.4
        
        # Check for logical flow words
        flow_words = ['first', 'then', 'next', 'after', 'finally', 'lastly']
        flow_count = sum(1 for word in flow_words if word in plan.lower())
        if flow_count > 0:
            score += 0.3
        
        # Check for reasonable length
        if 3 <= len(lines) <= 10:
            score += 0.3
        
        return min(score, 1.0)
    
    def _compute_completeness_score(self, instruction: str, plan: str) -> float:
        """Compute completeness of plan."""
        # Check if plan addresses all parts of instruction
        instruction_parts = instruction.lower().split(' and ')
        
        if len(instruction_parts) == 1:
            return 0.8  # Single task, assume mostly complete
        
        # Multi-part instruction
        addressed_parts = 0
        for part in instruction_parts:
            if any(word in plan.lower() for word in part.split() if len(word) > 3):
                addressed_parts += 1
        
        return addressed_parts / len(instruction_parts)
    
    def _compute_safety_score(self, commands: List[str]) -> float:
        """Compute safety score based on commands."""
        if not commands:
            return 1.0
        
        dangerous_count = sum(1 for cmd in commands if self.validator.categorize_command(cmd) == 'dangerous')
        safe_ratio = 1.0 - (dangerous_count / len(commands))
        
        return safe_ratio
    
    def evaluate_execution(self, commands: List[str], sandbox: SandboxEnvironment) -> List[Dict[str, Any]]:
        """Evaluate command execution in sandbox."""
        results = []
        
        for command in commands:
            if self.validator.is_safe_to_execute(command):
                result = sandbox.execute_command(command)
                result['safe_execution'] = True
            else:
                result = {
                    'command': command,
                    'return_code': -1,
                    'stdout': '',
                    'stderr': 'Command not executed for safety reasons',
                    'execution_time': 0,
                    'status': 'skipped_unsafe',
                    'safe_execution': False
                }
            
            # Add command analysis
            result['command_category'] = self.validator.categorize_command(command)
            results.append(result)
        
        return results
    
    def evaluate_single_prompt(self, instruction: str, prompt_id: str = None) -> EvaluationResult:
        """Evaluate agent on a single prompt."""
        if prompt_id is None:
            prompt_id = f"prompt_{int(time.time())}"
        
        # Generate plan using agent
        if hasattr(self.agent, 'generate_plan'):
            plan = self.agent.generate_plan(instruction)
        elif hasattr(self.agent, 'process_instruction'):
            result = self.agent.process_instruction(instruction)
            plan = result.get('plan', '')
        else:
            raise ValueError("Agent must have either 'generate_plan' or 'process_instruction' method")
        
        # Extract commands
        commands = self.extract_commands_from_plan(plan)
        
        # Evaluate plan quality
        quality_scores = self.evaluate_plan_quality(instruction, plan)
        
        # Create sandbox and evaluate execution
        sandbox = SandboxEnvironment()
        try:
            execution_results = self.evaluate_execution(commands, sandbox)
        finally:
            sandbox.cleanup()
        
        # Compute additional metrics
        metrics = {
            'num_commands': len(commands),
            'num_safe_commands': sum(1 for cmd in commands if self.validator.is_safe_to_execute(cmd)),
            'successful_executions': sum(1 for result in execution_results if result.get('return_code') == 0),
            'execution_success_rate': sum(1 for result in execution_results if result.get('return_code') == 0) / max(len(execution_results), 1)
        }
        metrics.update(quality_scores)
        
        return EvaluationResult(
            prompt_id=prompt_id,
            instruction=instruction,
            generated_plan=plan,
            extracted_commands=commands,
            plan_quality_score=quality_scores['overall'],
            execution_results=execution_results,
            metrics=metrics,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'agent_class': self.agent_class.__name__,
                'agent_kwargs': self.agent_kwargs
            }
        )
    
    def evaluate_batch(self, instructions: List[str]) -> List[EvaluationResult]:
        """Evaluate agent on a batch of instructions."""
        results = []
        
        for i, instruction in enumerate(tqdm(instructions, desc="Evaluating instructions")):
            try:
                result = self.evaluate_single_prompt(instruction, f"prompt_{i}")
                results.append(result)
            except Exception as e:
                print(f"Error evaluating instruction {i}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    prompt_id=f"prompt_{i}",
                    instruction=instruction,
                    generated_plan="",
                    extracted_commands=[],
                    plan_quality_score=0.0,
                    execution_results=[],
                    metrics={'error': str(e)},
                    metadata={'timestamp': datetime.now().isoformat(), 'status': 'error'}
                )
                results.append(error_result)
        
        return results
    
    def generate_comprehensive_report(self, results: List[EvaluationResult], output_path: Path):
        """Generate comprehensive evaluation report."""
        report_lines = []
        
        # Header
        report_lines.extend([
            "# Comprehensive CLI Agent Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Agent: {results[0].metadata.get('agent_class', 'Unknown') if results else 'Unknown'}",
            ""
        ])
        
        # Summary statistics
        if results:
            overall_scores = [r.plan_quality_score for r in results]
            command_counts = [r.metrics.get('num_commands', 0) for r in results]
            success_rates = [r.metrics.get('execution_success_rate', 0) for r in results]
            
            report_lines.extend([
                "## Summary Statistics",
                f"- Total Prompts Evaluated: {len(results)}",
                f"- Average Plan Quality Score: {sum(overall_scores) / len(overall_scores):.3f}",
                f"- Average Commands per Plan: {sum(command_counts) / len(command_counts):.1f}",
                f"- Average Execution Success Rate: {sum(success_rates) / len(success_rates):.3f}",
                ""
            ])
            
            # Quality distribution
            quality_distribution = {
                'Excellent (>0.8)': sum(1 for s in overall_scores if s > 0.8),
                'Good (0.6-0.8)': sum(1 for s in overall_scores if 0.6 <= s <= 0.8),
                'Fair (0.4-0.6)': sum(1 for s in overall_scores if 0.4 <= s < 0.6),
                'Poor (<0.4)': sum(1 for s in overall_scores if s < 0.4)
            }
            
            report_lines.extend([
                "## Quality Distribution",
                *(f"- {category}: {count} ({count/len(results)*100:.1f}%)" 
                  for category, count in quality_distribution.items()),
                ""
            ])
        
        # Individual results
        report_lines.append("## Individual Results")
        report_lines.append("")
        
        for result in results:
            report_lines.extend([
                f"### {result.prompt_id}",
                f"**Instruction:** {result.instruction}",
                f"**Overall Quality Score:** {result.plan_quality_score:.3f}",
                "",
                "**Generated Plan:**",
                "```",
                result.generated_plan,
                "```",
                "",
                f"**Extracted Commands ({len(result.extracted_commands)}):**",
                *(f"- `{cmd}`" for cmd in result.extracted_commands),
                "",
                "**Quality Metrics:**",
                *(f"- {metric}: {value:.3f}" for metric, value in result.metrics.items() 
                  if isinstance(value, (int, float)) and metric != 'error'),
                "",
                "**Execution Results:**"
            ])
            
            for exec_result in result.execution_results:
                status = exec_result.get('status', 'unknown')
                return_code = exec_result.get('return_code', -1)
                report_lines.append(f"- `{exec_result['command']}` → {status} (exit code: {return_code})")
            
            report_lines.extend(["", "---", ""])
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comprehensive report saved to: {output_path}")


def create_test_scenarios() -> List[Dict[str, Any]]:
    """Create comprehensive test scenarios."""
    scenarios = [
        {
            'category': 'git_basics',
            'instructions': [
                "Initialize a new git repository and make first commit",
                "Create a new branch called 'feature' and switch to it",
                "Add all Python files to git and commit with message 'Add Python files'"
            ]
        },
        {
            'category': 'python_environment',
            'instructions': [
                "Create a Python virtual environment and activate it",
                "Install pandas and numpy in a virtual environment",
                "Check which Python packages are installed in current environment"
            ]
        },
        {
            'category': 'file_operations',
            'instructions': [
                "Find all .py files in current directory and subdirectories",
                "Create a backup directory and copy all .txt files to it",
                "List all files larger than 1MB in the current directory"
            ]
        },
        {
            'category': 'text_processing',
            'instructions': [
                "Search for lines containing 'import' in all Python files",
                "Count the number of lines in all .py files",
                "Find and replace 'old_name' with 'new_name' in all text files"
            ]
        },
        {
            'category': 'compression',
            'instructions': [
                "Create a compressed tar archive of all .log files",
                "Extract a tar.gz file to a specific directory",
                "Compress all files in a directory using gzip"
            ]
        },
        {
            'category': 'system_monitoring',
            'instructions': [
                "Show disk usage for all mounted filesystems",
                "Display the top 10 processes by CPU usage",
                "Find files modified in the last 24 hours"
            ]
        },
        {
            'category': 'edge_cases',
            'instructions': [
                "Recursively find files larger than 100MB and move them to /tmp/large_files",
                "Set up a cron job to backup /home/user/documents every day at 2 AM",
                "Create a script that monitors log files for error patterns and sends alerts"
            ]
        }
    ]
    
    return scenarios


if __name__ == "__main__":
    # Example usage
    print("Dynamic Evaluation Framework for CLI Agents")
    print("This module provides comprehensive evaluation capabilities.")
    print("Import and use the DynamicEvaluator class in your evaluation scripts.")