# -*- coding: utf-8 -*-
# Define the punctuation characters
PUNCTUATION = set('.,/><?)(;:\'"\[!\]@#$%^&*')
from collections import Counter
def wordTokenizer(text):
    words = []
    current_word = ""
    i = 0
    while i < len(text):
        # Handle contractions
        if text[i:].startswith("n't"):
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append("n't")
            i += 3  # Skip over 'n't'
        elif text[i] == "'":
            if i > 0 and text[i - 1] == 'n':
                words.append(current_word)
                current_word = ""
            current_word += text[i]
            i += 1
        elif text[i] in PUNCTUATION:
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(text[i])
            i += 1
        elif text[i:].startswith(":)") or text[i:].startswith(":-)") or text[i:].startswith(":(") or text[i:].startswith(":-("):
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(text[i:i + 2])
            i += 2
        elif text[i] == '-':
            if i > 0 and text[i - 1] not in PUNCTUATION and text[i + 1] not in PUNCTUATION:
                current_word += text[i]
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(text[i])
            i += 1
        elif text[i] == '$' or text[i] == '#' or text[i] == '@':
            if i < len(text) - 1 and text[i + 1] not in PUNCTUATION:
                current_word += text[i:i + 2]
                i += 2
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(text[i])
                i += 1
        elif text[i] == '.':
            if i > 0 and text[i - 1].isdigit() and i < len(text) - 1 and text[i + 1].isdigit():
                current_word += text[i]
                i += 1
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(text[i])
                i += 1
        else:
            current_word += text[i]
            i += 1

    if current_word:
        words.append(current_word)

    return words

def spacelessBPELearn(docs, max_vocabulary=1000):
    # Initialize vocabulary with all ASCII letters as words
    vocabulary = set(chr(i) for i in range(128))
    print(vocabulary,"===================================")
    # Convert all non-ASCII characters into "?"
    docs = [doc.encode('ascii', 'replace').decode() for doc in docs]

    # Run until the vocabulary size reaches max_vocabulary
    while len(vocabulary) < max_vocabulary:
        # Count pairs of characters in the corpus
        pairs_count = {}
        for doc in docs:
            for i in range(len(doc) - 1):
                pair = doc[i:i + 2]
                if pair not in vocabulary:
                    pairs_count[pair] = pairs_count.get(pair, 0) + 1

        # Get the most frequent pair
        most_frequent_pair = max(pairs_count, key=pairs_count.get)

        # Add the most frequent pair to the vocabulary
        vocabulary.add(most_frequent_pair)

        # Merge occurrences of the most frequent pair in the corpus
        new_docs = []
        for doc in docs:
            new_doc = doc.replace(most_frequent_pair, most_frequent_pair[0])
            new_docs.append(new_doc)
        docs = new_docs

    return vocabulary

def spacelessBPETokenize(text, vocab):
    tokens = []
    i = 0
    while i < len(text):
        j = i + 1
        while j <= len(text):
            if text[i:j] in vocab:
                tokens.append(text[i:j])
                i = j
                break
            j += 1
        if j == len(text) + 1:
            tokens.append(text[i:j])
            break
    return tokens

with open('a1_tweets.txt', 'r', encoding='utf-8') as f:
    tweets = f.readlines()

print("Checkpoint 1.1 - Output of wordTokenizer on the first 5 documents and the last document:")
for tweet in tweets[:5]:
    print(wordTokenizer(tweet))
print(wordTokenizer(tweets[-1]))

print("\nCheckpoint 1.2 - Output of spacelessBPELearn and spacelessBPETokenize:")
vocab = spacelessBPELearn(tweets)
print("Top five most frequent pairs at iterations 0, 1, 10, 100, and 500:")
for i in [0, 1, 10, 100, 500]:
    pairs_count = Counter()
    for doc in tweets:
        for j in range(len(doc) - 1):
            pair = doc[j:j+2]
            if pair in vocab:
                pairs_count[pair] += 1
    print(f"Iteration {i}: {pairs_count.most_common(5)}")

print("Final vocabulary:")
print(vocab)

print("Tokenization of the first 5 documents and the last document:")
for tweet in tweets[:5]:
    print(spacelessBPETokenize(tweet, vocab))
print(spacelessBPETokenize(tweets[-1], vocab))

# if __name__ == "__main__":
#     # Read input file
#     with open('a1_tweets.txt', 'r', encoding='utf-8') as f:
#         tweets = f.readlines()
#
#     # Checkpoint 1.1
#     with open('a1_p1_kumar_115317804_OUTPUT.txt', 'w', encoding='utf-8') as out:
#         out.write("Checkpoint 1.1 - Output of wordTokenizer on the first 5 documents and the last document:\n")
#         for tweet in tweets[:5]:
#             out.write(str(wordTokenizer(tweet)) + '\n')
#         out.write(str(wordTokenizer(tweets[-1])) + '\n')
#
#     # Checkpoint 1.2
#     with open('804_OUTPUT.txt', 'a', encoding='utf-8') as out:
#         vocab = spacelessBPELearn(tweets)
#         out.write("\nCheckpoint 1.2 - Output of spacelessBPELearn and spacelessBPETokenize:\n")
#         out.write("Top five most frequent pairs at iterations 0, 1, 10, 100, and 500:\n")
#         for i in [0, 1, 10, 100, 500]:
#             pairs_count = {}
#             for doc in tweets:
#                 for j in range(len(doc) - 1):
#                     pair = doc[j:j + 2]
#                     if pair in vocab:
#                         pairs_count[pair] = pairs_count.get(pair, 0) + 1
#             most_common_pairs = sorted(pairs_count.items(), key=lambda x: x[1], reverse=True)[:5]
#             out.write(f"Iteration {i}: {most_common_pairs}\n")
#
#         out.write("Final vocabulary:\n")
#         out.write(str(vocab) + '\n')
#
#         out.write("Tokenization of the first 5 documents and the last document:\n")
#         for tweet in tweets[:5]:
#             out.write(str(spacelessBPETokenize(tweet, vocab)) + '\n')
#         out.write(str(spacelessBPETokenize(tweets[-1], vocab)) + '\n')
