from composennent.nlp.tokenizers import WordPieceTokenizer
from datasets import load_dataset
dataset = load_dataset("truongpdd/vietnamese-10classes-clf")
texts = list(dataset["train"].shuffle(seed=1234).select(range(1000))["text"])



# Initialize and train
tokenizer = WordPieceTokenizer(vocab_size=30000)
tokenizer.train(texts)

# Encode text
ids = tokenizer.encode("hello world")

# Decode back
text = tokenizer.decode(ids)

# Save and load
tokenizer.save('./models/tokenizer/wordpiece_model.json')
loaded = WordPieceTokenizer.from_pretrained('./models/tokenizer/wordpiece_model.json')