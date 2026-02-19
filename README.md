To install do
```
chmod +x install
./install
```

## Training the models

To train a model, on Great Lakes run
```bash
# Use context window of 3 (default)
sbatch run_sft_context_preceding.sh

# Use context window of 5
sbatch run_sft_context_preceding.sh 5
```

This should take between 2-4 hours. This will output a model under the outputs folder.