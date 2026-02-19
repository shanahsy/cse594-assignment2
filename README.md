To install do
```
chmod +x install
./install
```

## Training the models

To train a model, on Great Lakes run
```bash
# Use context window of 2 (default)
sbatch run_sft_context_preceding.sh
```

This took 2 hours. This will output a model under the outputs folder in your GreatLakes directory.
