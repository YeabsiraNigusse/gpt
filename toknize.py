import urllib.request
import re

from SimpleTokenizerV1 import SimpleTokenizerV1

url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")

file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


print("Total number of character:", len(raw_text))
print(raw_text[:99])

# separating the raw text into single word array
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

print(preprocessed[:30])

# unifing repeated words and sorting based on vocablery order
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# converting sorted word into id dictionery
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 5:
        break


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(vocab["the"])
print(vocab["you"])

# the process of spliting raw text into token(individual word) removing all the whitspace
# assign them id and put them as dict
# write a encoder - recive a text and return the id by refring the previous dict/vocablery
# write a decode - recive a collation of id and return the text



