from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
import nltk.data

def deromanize(number):
    number = number.upper().replace(" ", "")
    numerals = { 1000: "M", 900: "CM", 500: "D", 400: "CD", 100: "C", 90: "XC", 50: "L", 40: "XL", 10: "X", 9: "IX", 5: "V", 4: "IV", 1: "I" }
    result = 0
    for value in sorted(numerals, reverse=True):
        key = numerals[value]
        while (number.find(key) == 0):
            result += value
            number = number[len(key):]   
    return result

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
titles = ["Great Expectations.txt", "The Communist Manifesto.txt", "The Great Gatsby.txt", "The Picture of Dorian Gray.txt", "The Scarlet Letter.txt", "Wuthering Heights.txt" ,"War and Peace.txt", "Jane Eyre.txt", "Pride and Prejudice.txt"]
for t in titles:
    with open("./novels/" + t, "r", encoding = "UTF-8") as f:
        d = f.read()
    tok_lines = tokenizer.tokenize(d)

    nl = []
    tok_remove = ["\n", "\ufeff",]
    #import pdb; pdb.set_trace()
    num_chaps = 0
    for l in tok_lines:
        n = l.replace("\n", " ")
        n = n.replace("\ufeff", "")
        n = n.replace("\t", " ")
        if "CHAPTER" in n or "Chapter" in n:
            s = word_tokenize(n)
            c_indices = []
            indices = []
            for i, tok in enumerate(s):
                if tok == "CHAPTER" or tok == "Chapter":
                    c_indices.append(True)
                    indices.append(i+1)
            for i in indices:
                if isinstance(deromanize(s[i]), int) or isintance(int(s[i]), int):
                    del(s[i-1]) # remove chapter
                    del(s[i]) # remove chapter #
                
            n = TreebankWordDetokenizer().detokenize(s)   
        #n = n.replace("CHAPTER", "") 
        #n = n.replace("Chapter", "")
        n = n.replace("--", " - ")
        n = n.replace("_", "")
        n = n.strip()
        
        #n = " ".join(n.split())
        nl.append(n)

    tok_title = tokenizer.tokenize(t.lower())
    output_title = []
    for i, v in enumerate(tok_title):
        output_title.append(v)
    output_title = "".join(output_title)
    output_title = output_title.replace(" ", "_")
    print(output_title)
    with open(output_title, "w", encoding = "UTF-8") as f:
        for l in nl:
            f.write(l + "\n")
