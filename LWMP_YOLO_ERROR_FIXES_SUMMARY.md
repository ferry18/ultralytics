
## Error 6: `SyntaxError: 'callbacks' is not a valid YOLO argument`

### Root Cause
The training script was passing `callbacks` as a parameter to `model.train()`, but this is not a valid training argument in Ultralytics YOLO.

### Fix Applied
Changed from passing callbacks as a parameter to using the `model.add_callback()` method:

```python
# INCORRECT - callbacks as parameter
results = model.train(
    data='dataset.yaml',
    epochs=100,
    callbacks={
        'on_pretrain_routine_start': setup_awloss
    }
)

# CORRECT - using add_callback method
model.add_callback('on_pretrain_routine_start', setup_awloss)
results = model.train(
    data='dataset.yaml',
    epochs=100
)
```

### Why This Matters
- Ultralytics YOLO uses a callback system that requires registration through `add_callback()`
- Callbacks cannot be passed as training parameters
- The callback system allows extending functionality without modifying core code

### Key Takeaway
- Use `model.add_callback(event_name, function)` to register callbacks before training
- Do not pass `callbacks` as a parameter to `train()`
- Common callback events include:
  - `'on_pretrain_routine_start'` - Called before training starts
  - `'on_epoch_end'` - Called after each epoch
  - `'on_train_end'` - Called after training completes