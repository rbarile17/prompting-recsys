## Prompting RecSys
### Finetuning and evaluation
`python -m src.models.finetune`
`python -m src.models.eval`

Both files requires .json parameters file
The paths file should be specified directly in the code, in the params_list variable

### Code structure
* package src.data: 
  - data preprocessing
  - specializations of pytorch dataset class in which we handle the addition of prompts to he input examples
  - preprocessing scripts to convert datasets and predictions in the formats required by the framework elliot
* package src.models: contains scripts to execute main operations: finetuning and eval, few observations:
  - during finetuning it is handled also a validation datasets to evaluate on after a fixed numer of steps (e.g. one epoch), but this evaluation is only in terms of accuracy, the evaluation is more detailed in the eval file that handles the test set
  - tha paths of parameters json files needs to be specified directly in the code, notice that multiple parameters files can be specified, this is useful to run several experiments 
  - package src.utilities: setup_parameters.py contains parameters definition, in particular it contains definitions for parameter in addition tothe default ones in huggingface, so only to what is specific to prompt based finetuning, e.g. the template of the prompt
