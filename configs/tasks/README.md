# Tasks Configuration Guide

## Overview
This directory contains configuration files for all MLGym tasks. Each task is defined through a YAML configuration file that specifies its properties, requirements, and evaluation criteria.

## Configuration Structure

### Task Configuration
The task configuration is defined by the `TaskConfig` class with the following attributes:

#### Required Fields
- `id`: Unique task identifier (string)
- `name`: Display name of the task (string)
- `description`: Detailed task description and goals (string)
- `task_entrypoint`: Task implementation class name (string)
- `evaluation_paths`: Paths to evaluation scripts (list[str])

#### Optional Fields
- `dataset_configs`: List of paths to dataset configuration files (list[str]). If the task doesn't require any dataset, skip this field.
- `training_timeout`: Maximum time for a single [model training](#training-detection) run in seconds. The environment will detect when the agent launches a model training and set the timeout correctly. (int)
- `requirements_path`: Path to custom conda environment requirements.txt (str)
- `use_generic_conda`: Whether to use default MLGym conda env (bool, default=True)
- `starter_code`: List of starter code file paths (list[str])
- `baseline_paths`: Paths to baseline implementation scripts (list[str])
- `baseline_scores`: Reference performance metrics (list[dict])
- `sample_submission`: Path to sample submission file (str)
- `evaluation_read_only`: Whether evaluation script is read-only (bool, default=False)
- `memory_path`: Path to a memory file. (str)

#### Extra Fields (Not in use currently)
- `secret_files`: Files that should be inaccessible to agent (list[str])
- `benchmark_name`: Name of benchmark to associate with task (str). NOT BEING USED CURRENTLY

### Dataset Configuration 
The dataset configuration is defined by the `DatasetConfig` class:

#### Required Fields
- `name`: Name of the dataset (string)
- `description`: Description of dataset format (string) 
- `data_path`: Path or HuggingFace repo ID (string)

#### Optional Fields
- `is_local`: Whether dataset is stored locally (bool, default=False)
- `train_split`: Training split configuration (SplitConfig)
- `valid_split`: Validation split configuration (SplitConfig)
- `test_split`: Test split configuration (SplitConfig)

### Split Configuration
The `SplitConfig` class defines dataset splits:

- `name`: Name of the split (string)
- `file_regex`: Regex pattern to match files in split (string)

## Task Classes

### Base Task Class
All task implementations must inherit from `AbstractMLTask` and implement required methods:

```python
class AbstractMLTask(metaclass=AbstractMLTaskMeta):
    def __init__(self, seed: int, args: TaskConfig, task_workspace: str, 
                 _communicate: Callable, _communicate_with_handling: Callable):
        """Initialize task with configuration
        
        Args:
            seed: Random seed for reproducibility
            args: Task configuration object
            task_workspace: Path to task workspace directory
            _communicate: Function to communicate with container
            _communicate_with_handling: Function to communicate with error handling
        """
        pass

    @abstractmethod 
    def evaluate(self) -> tuple[dict[str, Any], str]:
        """Evaluate submission and return metrics with submission path
        
        Returns:
            tuple containing:
                - dict: Evaluation metrics
                - str: Path to submission artifact
        """
        raise NotImplementedError
```

### Container Communication
Task classes can communicate with the Docker container through two provided methods:

1. **_communicate(command: str, timeout_duration: int | None = None) -> str**
   - Executes a command in the container and returns output
   - Use for basic commands that are expected to succeed
   - Returns: Command output as string
   - Example:
   ```python
   def _get_submission_file(self) -> str | None:
       # List files in workspace
       files = self._communicate(f"ls {self.task_workspace}").strip().split("\n")
       if "submission.csv" in files:
           return f"{self.task_workspace}/submission.csv"
   ```

2. **_communicate_with_handling(command: str, timeout_duration: int | None = None, error_msg: str | None = None) -> str**
   - Executes command with error handling and timeout
   - Use for commands that might fail or timeout
   - Raises RuntimeError with custom message on failure
   - Example:
   ```python
   def evaluate(self) -> tuple[dict[str, Any], str]:
       eval_script = self._get_evaluation_paths()[0]
       output = self._communicate_with_handling(
           f"python {eval_script}",
           timeout_duration=self.args.training_timeout,
           error_msg="Evaluation script failed"
       )
   ```

Common Use Cases:
- File operations: `ls`, `find`, `cat`
- Running scripts: `python script.py`
- GPU queries: `nvidia-smi`
- Environment setup: `pip install`, `conda install`

Best Practices:
- Always use workspace-relative paths
- Set appropriate timeouts for long-running commands
- Handle command output parsing carefully
- Use error handling for critical operations

### Parsing Evaluation Metrics

While the `evaluate()` method must return metrics as a dictionary, the evaluation script's output format is flexible. The task class is responsible for correctly parsing this output into the required dictionary format.

1. **Recommended: JSON Output**
   - While not required, having evaluation scripts output a single JSON line is recommended for simplicity:
   ```python
   def evaluate(self) -> tuple[dict[str, Any], str]:
       eval_script = self._get_evaluation_paths()[0]
       output = self._communicate(f"python {eval_script}")
       
       try:
           metrics = json.loads(output)
       except json.JSONDecodeError:
           # Handle custom parsing if not JSON
           metrics = self._parse_custom_output(output)
           
       return metrics, submission_path
   ```

2. **Custom Output Parsing**
   - Task classes can implement custom parsing for any output format:
   ```python
   def _parse_custom_output(self, output: str) -> dict[str, Any]:
       """Example: Parse space-separated key-value pairs"""
       metrics = {}
       for line in output.split('\n'):
           if ':' in line:
               key, value = line.split(':', 1)
               try:
                   metrics[key.strip()] = float(value.strip())
               except ValueError:
                   continue
       return metrics
   ```

3. **Expected Return Format**
   - Regardless of evaluation script output, `evaluate()` must return a dictionary with numeric values
   - Example return formats:
   ```python
   # Classification metrics
   {
       "accuracy": 0.85,
       "f1_score": 0.83
   }
   
   # Regression metrics
   {
       "rmse": 1.234,
       "r2": 0.89
   }
   ```

4. **Error Handling**
   - Implement robust parsing for your evaluation script's output format
   - Validate final metrics dictionary before returning
   - Provide clear error messages if parsing fails
   ```python
   def _validate_metrics(self, metrics: dict) -> None:
       if not metrics:
           raise EvaluationFormatError("No metrics parsed from evaluation output")
       for key, value in metrics.items():
           if not isinstance(value, (int, float)):
               msg = f"Metric {key} has non-numeric value: {value}"
               raise EvaluationFormatError(msg)
   ```

Best Practices:
- While evaluation script output format is flexible, using JSON is recommended for simplicity
- Document expected output format in evaluation script
- Implement robust parsing in task class
- Always validate final metrics dictionary
- Include error handling for malformed output

### Available Task Types

1. **CSVSubmissionTasks**
   - Purpose: For tasks requiring predictions in CSV format
   - Submission: Must generate `submission.csv` file
   - Evaluation: Compares predictions against ground truth
   - Examples: Image Classification, Regression, Image Captioning
   - Key Methods:
     - `evaluate()`: Validates and evaluates submission.csv
     - `_get_submission_file()`: Locates submission file

2. **ModelSubmissionTasks** 
   - Purpose: For tasks requiring trained model checkpoints
   - Submission: Must save model checkpoints and config
   - Evaluation: Loads and evaluates model performance
   - Examples: Reinforcement Learning tasks
   - Key Methods:
     - `evaluate()`: Loads and evaluates model checkpoints
     - `_get_submission_file()`: Finds model config file

3. **LMSubmissionTasks**
   - Purpose: Specialized for language model tasks
   - Features: Built-in GPU support, distributed training
   - Submission: Must save model and tokenizer
   - Examples: Next Token Prediction, Text Generation
   - Key Methods:
     - `evaluate()`: Handles distributed model evaluation
     - `setup()`: Configures GPU environment

4. **PythonSubmissionTasks**
   - Purpose: For tasks requiring Python function implementations
   - Submission: Must implement function in target.py
   - Evaluation: Tests function behavior
   - Examples: Game Theory, Algorithm Design
   - Key Methods:
     - `evaluate()`: Tests implemented function
     - `_get_submission_file()`: Locates target.py

## Integrating a new ML Task in MLGym

### Adding a New Task Type

1. Create new class in `mlgym/environment/tasks.py`:
```python
class NewTaskType(AbstractMLTask):
    def evaluate(self) -> tuple[dict[str, Any], str]:
        """Implementation of evaluation logic"""
        # Your evaluation code here
        return metrics_dict, submission_path
        
    def _get_submission_file(self) -> str | None:
        """Optional: Custom submission file handling"""
        pass
```

2. Key considerations:
   - Inherit from AbstractMLTask
   - Implement `setup()` method if the task requires special environment variables.
   - Implement required `evaluate()` method
   - Add task-specific helper methods
   - Handle submission artifacts appropriately
   - Return standardized metrics format

### Adding a New Task

1. Create new YAML file in `configs/tasks/`:
```yaml
id: unique_task_id
name: Task Name
description: |
  Detailed task description
  Include {dataset_docs} string to autoload dataset description from dataset config.
  Specify submission format requirements
task_entrypoint: TaskClassName
evaluation_paths:
  - path/to/evaluate.py # relative to /home/agent/workspace
dataset_configs:  # Optional
  - datasets/your_dataset.yaml
```

2. Create task files:
   - Evaluation script
   - Starter code (optional)
   - Baseline implementation (optional)
   - Sample submission (optional)

### Adding a New Dataset

1. Create new YAML file in `configs/datasets/`:
```yaml
name: dataset_name
description: |
  Dataset format description
  Input/output specifications
  Any special considerations
data_path: path/or/huggingface/id
is_local: false  # Set true for local datasets
train_split:  # Optional
  name: train
  file_regex: "train.*"
valid_split:  # Optional
  name: valid
  file_regex: "valid.*"
test_split:  # Optional
  name: test
  file_regex: "test.*"
```

2. Data preparation:
   - Ensure data is accessible
   - Structure files according to splits
   - If using regex, verify file patterns match regex

## Best Practices

1. Task Configuration:
   - Use clear, descriptive IDs
   - Provide comprehensive descriptions
   - Include all necessary starter files
   - Set appropriate timeouts

2. Dataset Configuration:
   - Document data format clearly
   - Use consistent split patterns
   - Verify data accessibility

3. Task Implementation:
   - Handle errors gracefully through provided error classes.
   - Validate submissions thoroughly
   - Return standardized metrics as a dictionary
   - Document special requirements

## Implementation Notes

### Training Detection
Training timeout is applied to the commands by detecting known patterns. Currently we support training commands include `[torchrun, python/python3, accelerate, deepspeed]`.
If you would like a new command to be added, please open an issue.