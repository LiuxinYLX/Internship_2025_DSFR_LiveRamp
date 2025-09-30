import json

input_path = '/Users/liuxinyang/Documents/kaikki.org-dictionary-French.jsonl'
output_path = '/Users/liuxinyang/Documents/Code/data/french_dictionary.txt'


word_counts = {}

with open(input_path, 'r', encoding='utf-8') as fin:
    for line in fin:
        data = json.loads(line.strip())
        if data.get('lang') == 'French':
            word = data.get('word')
            if word:
                word = word.lower()
                word_counts[word] = word_counts.get(word, 0) + 1

with open(output_path, 'w', encoding='utf-8') as fout:
    for word, count in word_counts.items():
        if " " not in word:
            fout.write(f"{word} {count}\n")

print("french_dictionary.txt est généré avec succès.")

