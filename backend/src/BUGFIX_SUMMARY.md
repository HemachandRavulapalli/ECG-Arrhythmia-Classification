# Bug Fix Summary

## Issue
During training, encountered shape mismatch error in build_cnn_2d:
```
ValueError: Inputs have incompatible shapes. Received shapes (50, 5, 192) and (1, 1, 256)
```

## Root Cause
The channel attention mechanism was using a fixed filter size (256, 512, 1024) instead of the actual number of channels in the tensor `x` at each stage of the residual blocks.

## Fix Applied
Modified `backend/src/cnn_models.py` in the `build_cnn_2d` function:

**Before:**
```python
for i, filters in enumerate([256, 512, 1024]):
    # Channel attention
    channel_attention = layers.GlobalAveragePooling2D()(x)
    channel_attention = layers.Dense(filters//4, activation='relu')(channel_attention)
    channel_attention = layers.Dense(filters, activation='sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
    x = layers.Multiply()([x, channel_attention])
```

**After:**
```python
for i, filters in enumerate([256, 512, 1024]):
    # Get current number of channels
    current_channels = x.shape[-1]
    
    # Channel attention
    channel_attention = layers.GlobalAveragePooling2D()(x)
    channel_attention = layers.Dense(max(current_channels//4, 16), activation='relu')(channel_attention)
    channel_attention = layers.Dense(current_channels, activation='sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1, 1, current_channels))(channel_attention)
    x = layers.Multiply()([x, channel_attention])
```

## Changes
1. Added `current_channels = x.shape[-1]` to get actual channel count
2. Changed attention dimensions from `filters` to `current_channels`
3. Added `max(current_channels//4, 16)` to prevent division issues

## Verification
✓ Model compiles successfully
✓ No shape mismatch errors
✓ Ready for training

## Training Command
```bash
cd backend/src
python train_pipeline.py --epochs 50 --normalize --batch_size 64
```
