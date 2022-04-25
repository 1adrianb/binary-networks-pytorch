# Something similar to pytorch lighting functionality 

Trainer takes:
- model
- dataloader (train / val)
- metrics
- configuration

Train performs:
- training
- validation
- model checkpointing 

Use case:
1. specify config file
2. instantiate dataloader, 