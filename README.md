To install do
```
chmod +x install
./install
```

## Training the models

To train a model, on Great Lakes run
```bash
# Use context window of 1 (default)
sbatch run_sft_context_preceding.sh
```

This took 2 hours. This will output a model under the outputs folder in your GreatLakes directory.

## Evaluating the models

To evaluate a trained model, run the predict script with the model directory as an argument:

```bash
# Basic usage (context_window defaults to 1)
sbatch run_predict_context_preceding.sh "outputs/granite-friends-context-20251206-195720/checkpoint-41374"

```

The script takes two arguments:
1. **model_dir** (required): Path to the checkpoint you want to evaluate
2. **context_window** (optional, default=1): Number of preceding dialogue lines to use as context

This script only takes into account preceding lines currently.
