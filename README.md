Sure! Here's the full `README.md` content in **one copy-paste block**:

---

````markdown
# Finetune CLI Agent

A command-line agent that generates shell commands based on natural language prompts. Built using fine-tuning techniques to improve command generation accuracy.

## Features

- Generates accurate shell commands from natural language descriptions  
- Supports various command categories (git, search, archive, python, logs, etc.)  
- Includes comprehensive evaluation framework  
- Fine-tuned model for better command generation  
- Role-based prompt handling  

## Prerequisites

- Python 3.8+  
- Git  
- Required Python packages (see `requirements.txt`)  

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Monil110/finetune-cli-agent.git
cd finetune-cli-agent
````

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
finetune-cli-agent/
├── agent.py                # Command-line agent implementation
├── evaluate.py             # Evaluation framework
├── evaluation_framework.py # Core evaluation logic
├── scoring_metrics.py      # Metric calculation
├── test_prompts.json       # Test cases for evaluation
├── results/                # Directory for evaluation results
└── logs/                   # Directory for execution logs
```

## Running the Agent

1. Basic usage:

```bash
python agent.py "Your command description here"
```

2. With specific options:

```bash
python agent.py "Your command" --model-path ./lora_adapter --dry-run
```

## Evaluation

To run the full evaluation suite:

```bash
python evaluate.py
```

To run a specific test case:

```bash
python evaluate.py --test-id git_branch_create
```

## Fine-tuning

The agent uses a fine-tuned model located at `./lora_adapter`. To fine-tune your own model:

1. Prepare training data in the format shown in `test_prompts.json`
2. Update the model path in `evaluate.py`
3. Run the fine-tuning process

## Configuration

The evaluation framework can be configured through command-line arguments:

```bash
python evaluate.py \
    --test-file test_prompts.json \
    --dry-run \
    --test-id specific_test_id
```

## Metrics

The evaluation framework measures:

* Command accuracy
* BLEU score
* ROUGE scores (1, 2, L)
* Plan quality
* Execution time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

* Thanks to the open-source community for contributing to the development tools used in this project
* Special thanks to the developers of the evaluation metrics used
