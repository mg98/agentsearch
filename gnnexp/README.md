# Edge Classification with Graph Neural Networks


## Quick Start
To run full training with the configuration in `config.yaml`:

```bash
python -m gnnexp.main
```

## Setup Wandb

```bash
# Login to wandb (first time only)
wandb login

# Or set environment variable
export WANDB_API_KEY=your_api_key
```

## Config

The `/config/config.yaml` file contains all configuration parameters:

Command line arguments are not yet implemented.


### Edge Splitting
- Edge_label are randomly split into train/val/test sets
- **THIS IS BAD; THE MODEL OVERFITS, WE SHOULD ONLY GIVE ACCESS TO TRAINING EDGES**
- Splits maintain the same random seed for reproducibility

### Logarithmic Label Normalization
- Trust scores are log-transformed: `log(score + ε)`
- Standardized within each split to have mean=0, std=1
- Applied separately to train/val/test sets to prevent data leakage
