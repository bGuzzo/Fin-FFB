from transformers import AutoTokenizer


TOKENIZER_HF_PATH = "albert-base-v2"
TEXT = "Lore ipsum lorem ipsum dolor sit amet, consectetur adipiscing elit."
MAX_LENGTH = 128


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_HF_PATH)

tokenized_batch = tokenizer(
    [TEXT],
    padding="max_length",
    truncation=True, # Important, chunk the text if len(token) > max_length
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

# Test only, confirm that the <PAD> token is 0
print(tokenized_batch)