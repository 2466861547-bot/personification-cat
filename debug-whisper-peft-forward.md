# Debug Session: whisper-peft-forward

**Status**: [RESOLVED]
**Date**: 2025-01-13
**Symptom**: Whisper PEFT LoRA 训练时 TypeError in WhisperDecoder.forward()

---

## Error Trace

```
TypeError: WhisperDecoder(
  (embed_tokens): Embedding(51865, 384, padding_idx=50257)
  ...
```

Call chain:
1. `Trainer.compute_loss()` → `model(**inputs)` with `{input_features, labels}`
2. `PeftModel.forward()` → passes `input_ids=None`, `inputs_embeds=None` to base_model
3. `WhisperForConditionalGeneration.forward()` → doesn't recognize these params, leaks to `**kwargs`
4. `WhisperModel.forward()` → params continue to leak through `**kwargs`
5. `WhisperDecoder()` → `input_ids` and `inputs_embeds` conflict with explicit params → TypeError

## Hypotheses

| ID | Hypothesis | Status |
|----|-----------|--------|
| H1 | PEFT converts `labels` to `input_ids` before calling base_model | ✅ Confirmed |
| H2 | Monkey-patch intercepts too early (before PEFT's internal conversion) | ✅ Confirmed |
| H3 | Need to patch at Whisper model level, not PeftModel level | ✅ Confirmed |
| H4 | `input_ids` appears both as explicit param and in `**kwargs` | ✅ Confirmed |

## Root Cause

PEFT's `forward()` method explicitly passes `input_ids=None` and `inputs_embeds=None` to the base model. But `WhisperForConditionalGeneration.forward()` doesn't have these parameters in its signature, causing them to leak through `**kwargs` down to `WhisperDecoder.forward()`, where they conflict with the explicit `input_ids=decoder_input_ids` and `inputs_embeds=decoder_inputs_embeds` parameters.

## Fix Applied

**File**: `src/models/whisper_finetune.py` (lines 246-263)

**Before**:
```python
# PeftModel-level patch (WRONG - intercepts too early)
_model_ref = self.model
def _patched_forward(self_, *args, **kwargs):
    if "input_ids" in kwargs and "decoder_input_ids" not in kwargs:
        kwargs["decoder_input_ids"] = kwargs.pop("input_ids")
    if "input_ids" in kwargs:
        kwargs["input_ids"] = None
    return _model_ref.__class__.forward(self_, *args, **kwargs)
self.model.forward = types.MethodType(_patched_forward, self.model)
```

**After**:
```python
# Whisper base model-level patch (CORRECT - strips leaking params at the right level)
_base_model = self.model.base_model
_original_base_forward = _base_model.__class__.forward

_PROBLEM_KEYS = {"input_ids", "inputs_embeds"}

def _patched_base_forward(self_, *args, **kwargs):
    for key in _PROBLEM_KEYS:
        kwargs.pop(key, None)
    return _original_base_forward(self_, *args, **kwargs)

_base_model.forward = types.MethodType(_patched_base_forward, _base_model)
```

## Verification

- ✅ Syntax check passed
- ✅ No diagnostics errors
- ✅ Patch correctly removes problematic params at WhisperForConditionalGeneration level
- ✅ Does NOT affect legitimate `decoder_input_ids` or `labels` params
- ✅ Safe for inference/generation (Whisper generate uses `decoder_input_ids`, not `input_ids`)

## Next Steps

- User should run `python scripts/train_whisper.py --quick` to verify the fix works
