# TODO List

## High Priority

### Add Progress Bars for Training Operations

**Problem**: When training tokenizers or language models, there's no visibility into training progress. Users don't know how much time is left or how many iterations have been completed.

**Affected Components**:
- `src/composennent/nlp/tokenizers/wordpiece.py` - WordPieceTokenizer training
- `src/composennent/nlp/tokenizers/sentencepiece.py` - SentencePiece training
- Any LLM training loops in `src/composennent/training/`

**Recommended Solution**:
Use `tqdm` library for progress bars:

```python
from tqdm import tqdm

# For iterating over training data
for item in tqdm(data, desc="Training tokenizer"):
    # training logic
    pass

# For known total iterations
for epoch in tqdm(range(num_epochs), desc="Training"):
    # epoch logic
    pass
```

**Implementation Steps**:
1. Add `tqdm` to `requirements.txt` and `pyproject.toml` dependencies
2. Add progress bars to tokenizer training methods:
   - WordPieceTokenizer.train()
   - SentencePieceTokenizer.train()
3. Add progress bars to LLM training loops
4. Add progress bars for vocabulary building iterations
5. Test with different dataset sizes to ensure proper progress tracking

**Benefits**:
- Better user experience during long training runs
- Ability to estimate remaining training time
- Visual feedback that training is progressing (not frozen)
- Easy to see training speed (iterations/second)

**Example Output**:
```
Training tokenizer: 100%|████████████| 1000/1000 [00:45<00:00, 22.1it/s]
Building vocabulary: 100%|█████████| 30000/30000 [00:12<00:00, 2415.3it/s]
```
