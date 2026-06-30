from importlib.metadata import version
import urllib.request
import tiktoken
import re

from SimpleTokenizer import SimpleTokenizerV1, SimpleTokenizerV2

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


#  once yo have the text data processed(toknized) and given id in sorted manner
#  you can use the classes below to encode and get thire id's

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

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # adding two additional words in our vocab so we can use those words 
                                                #  for signaling end of some text and default reprsentation of unknown word

vocab = {token:integer for integer,token in enumerate(all_tokens)} 
print(len(vocab.items()))


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokeniz = SimpleTokenizerV2(vocab)
print(tokeniz.encode(text))


# since we already covered how toknizers work we can use already builtin libreries from python


print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace Akwirwier"
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)


strings = tokenizer.decode(integers)
print(strings)
# Akwirw ier
# 50,257 total words


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))

#  Before going to embading process there is one step remaining - preparing input - target for our model 
#  as we know the model main goal is pridicting the next word correctl from a given text 


enc_sample = enc_text[50:]


context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y: {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# maybe doing some sliding window leetcode is good

