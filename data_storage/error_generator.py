import argparse
import random
import nltk
import time
import os
import glob
from unidecode import unidecode
from multiprocessing import Process, Manager
from ftfy.badness import sequence_weirdness
from pattern.en import pluralize, singularize, lexeme, suggest, tenses, conjugate
from collections import OrderedDict

DROPOUT_TOKENS = ["a", "an", "the", "'ll", "'s", "'m", "'ve"] + lexeme("be")

REPLACEMENTS = {"there": "their", "their": "there", "then": "than", "than": "then", 'on': 'in', 'in': 'on', 'the': 'a', 'an': 'the',
                'might': 'would', 'would': 'might', 'could': 'might', 'can': 'may', 'may': 'can','inside': 'in', 'besides': 'beside', 'beside': 'besides',
                'towards': 'toward', 'toward': 'towards', 'till': 'until', 'until': 'till', 'to': 'through', 'through': 'to', 'its': "it's", "it's": "its", "off": "of", 
                "of": "off", "effect": "affect", "affect": "effect", ";":",", ",":";", ".":"?", "?":".", "cat": "car", "car": "cat", "red": "reading", "reading": "red",
                "who":"that", "that":"who", "woman":"female", "female":"woman", "man":"male", "male":"man", "lead":"led", "led":"lead", "breath":"breathe", "breathe":"breath", "disinterested": "uninterested",
                "uninterested":"disinterested", "e.g.":"i.e.", "i.e.":"e.g.", "farther":"further", "further":"farther",
                "fewer":"less", "less":"fewer", "imply":"infer", "infer":"imply", "lay":"lie", "lie":"lay", "your":"you're",
                "accept":"except", "except":"accept", "altogether":"all together", "all together":"altogether", "bad":"badly", "badly":"bad", "lose":"loose", "loose":"lose", "who":"whom", "whom":"who", "all right": "alright", 
                "alright":"all right", "alumna":"alumnae", "alumnae":"alumna", "alumnus":"alumni", "alumnni":"alumnus", "principal":"principle",
                "principle":"principal", "to":"too", "too":"to", "advice":"advise", "advise":"advice", "already":"all ready", "all ready": "already",
                "among":"between", "between":"among","weather":"whether", "whether":"weather", "which":"that", "that":"which", "assure":"ensure", "ensure":"assure",
                "ensure":"insure", "insure":"ensure", "assure":"insure", "insure":"assure", "compliment":"complement", "complement":"compliment", "emigrate":"immigrate",
                "immigrate":"emigrate", "sympathy":"empathy", "empathy":"sympathy", "flaunt":"flout", "flout":"flaunt", "gaff":"gaffe", "gaffe":"gaff", "historic":"historical",
                "historical":"historic", "stationary":"stationery", "stationery":"stationary", "whose":"who's", "public":"pubic", "pubic":"public", "council":"counsel",
                "counsel":"council", "thorough":"though", "though":"thorough", "thru":"threw", "threw":"thru", "there":"they're",
                "stalking":"stocking", "stocking":"stalking", "role":"roll", "roll":"role", "rain":"reign", "reign":"rain", "reign":"rein", "rein":"reign", "rain":"rein", "rein":"rain",
                "precede":"proceed","proceed":"precede", "peak":"peek", "peek":"peak", "peak":"pique","pique":"peak", "peek":"pique", "pique":"peek", "past":"passed", "passed":"past",
                "pass":"past","past":"pass", "pain":"pane","pane":"pain", "main":"mane", "mane":"main", "interment":"internment", "internment":"interment",
                "internship":"internment", "internment":"internship", "hear":"here", "here":"hear", "formally":"formerly", "formerly":"formally", "former":"formal",
                "formal":"former", "for":"fore", "fore":"for", "for":"four", "four":"for", "four":"fore", "fore":"four", "flair":"flare","flare":"flair",
                "flew":"flue","flue":"flew","flew":"flu","flu":"flew","flu":"fleu","fleu":"flu","ellicit":"illicit","illicit":"elicit","desert":"dessert",
                "dessert":"desert","eminent":"immanent","immanent":"eminent","eminent":"imminent","imminent":"eminent","immanent":"imminent","imminent":"immanent",
                "dam":"damn", "damn":"dam","conscious":"conscience", "conscience":"conscious", "cite":"site", "site":"cite", "cite":"sight", "sight":"cite", "site":"sight",
                "sight":"site", "capitol":"capital", "capital":"capitol", "break":"brake", "brake":"break", "cent":"sent", "sent":"cent", "cent":"scent", "scent":"cent",
                "cent":"sense", "sense":"cent", "sent":"scent","scent":"sent","sent":"scent", "scent":"sent", "scent":"sense","sense":"scent", "border":"boarder", "boarder":"border",
                "bear":"bare", "bare":"bear", "ascent":"assent", "assent":"ascent", "allusion":"illusion","illusion":"allusion", "aloud":"allowed","aloud":"allowed",
                "adverse":"averse", "averse":"adverse", "appraise":"apprise", "apprise":"appraise", "bemused":"amused","amused":"bemused", "credible":"credulous", "credulous":"credible",
                "criteria":"criterion", "criterion":"criteria", "depreciate":"deprecate", "deprecate":"depreciate", "dichotomy":"discrepancy", "discrepancy":"dichotomy", 
                "enervate":"energize", "energize":"enervate", "enormity":"enormousness", "enormousness":"enormity", "flounder":"founder", "founder":"flounder", "fortuitous":"fortunate",
                "fortunate":"fortuitous", "fulsome":"full", "full":"fulsome", "inter":"intern", "intern":"inter", "luxuriant":"luxurious", "luxurious":"luxuriant",
                "meritorious":"meretricious", "meretricious":"meritorous", "mitigate":"militate", "militate":"mitigate", "noisome":"noisy", "noisy":"noisome",
                "practicable":"practical", "practical":"practicable", "prescribe":"proscribe", "proscribe":"prescribe", "protagonist":"proponent","proponent":"protagonist",
                "reticent":"reluctant", "reluctant":"reticent", "simple":"simplistic","simplistic":"simple", "staunch":"stanch", "stanch":"staunch", "tortuous":"torturous",
                "torturous":"tortuous", "unexceptionable":"unexceptional", "unexceptional":"unexceptionable", "verbal":"oral", "oral":"verbal", "allude":"elude", "elude":"allude",
                "aggravate":"irritate", "irritate":"aggravate", "inflammable":"flammable", "regardless":"irregardless", "regretful":"regrettable", "regrettable":"regretful"}

PREPOSITIONS = ["aboard", "about", "above", "across", "after", "against", "along", "amid", "among", "anti", "around", "as", "at", 
"before", "behind", "below", "beneath", "besides", "beside", "between", "beyond", "but", "by", "concerning", "considering", "despite", 
"down", "during", "except", "excepting", "excluding", "following", "for", "from", "in", "inside", "into", "like", "minus", "near", "of", "off", 
"on", "onto", "opposite", "outside", "over", "past", "per", "plus", "regarding", "round", "since", "than", "through", "to", "toward", "towards", 
"under", "underneath", "unlike", "until", "up", "upon", "versus", "via", "with", "without"]

COMPARATIVES = ["more", "less", "the most", "the least"]

MODAL = {'can': 'could', 'could': 'can',
         'may': 'might', 'might': 'may',
         'will': 'would', 'would': 'will'}

VERBS_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

NOUN_TAGS = ['NN', 'NNP', 'NNPS', 'NNS']

PREPOSITION_TAGS = ['IN']

ADVERB_TAGS = ['RB']

ADJECTIVE_TAGS = ['JJ']

def change_pluralization(token):
    singularForm = singularize(token)
    pluralForm = pluralize(token)
    if token == singularForm:
        return pluralForm
    else:
        return singularForm


def change_tense(token):
        
    nt = tenses(token)
    #print(nt)
    if len(nt) != 0:
        current_tense = nt[0][0]
    else:
        return token
    #if any(isinstance(i, tuple) for i in nt):
    #    current_tense = nt[0][0]
    #else:
    #    current_tense = nt[0]
    p_conj = []
    for p in range(1,4):
        n_conj = conjugate(token, tense = current_tense, person = p)
        if n_conj != token:
            return n_conj
        p_conj.append(n_conj)

    return p_conj[0]

    # return random.choice(lexeme(token))

def noise_generator(original_sentence_list, i, process_dict):
    noised_sentence_list = []
    for target_sentence in original_sentence_list:
        generated_source = []

        # x = x.lower() # this cause some error ignorance (Mec)
        x_split = nltk.word_tokenize(target_sentence)
        x_pos = nltk.pos_tag(x_split)

        # avoid too much error creation
        similar_flag = False
        replace_flag = False
        proposition_flag = False
        plural_flag = False
        tense_flag = False
        modal_flag = False
        incorrect_comparative_flag = False

        for token, pos in x_pos:

            similar_token = (pos in NOUN_TAGS and random.random() < 0.3 and not similar_flag) 

            dropout_token = (token in DROPOUT_TOKENS and
                             random.random() < 0.3)

            incorrect_comparative_token = (pos in ADJECTIVE_TAGS and
                                random.random() < 0.3 and not incorrect_comparative_flag)

            preposition_token = (pos in PREPOSITION_TAGS and 
                                random.random() < 0.3
                                and not proposition_flag)

            replace_token = (token in REPLACEMENTS and
                             random.random() < 0.3 and
                             not replace_flag)

            pos_plural_token = (pos in NOUN_TAGS and
                                random.random() < 0.3 and
                                not plural_flag)

            pos_tense_token = (pos in VERBS_TAGS and
                               random.random() < 0.3 and
                               not tense_flag)

            pos_modal_token = (token in MODAL and
                               random.random() < 0.3 and
                               not modal_flag)

            if replace_token:
                generated_source.append(REPLACEMENTS[token])
                replace_flag = True
            elif similar_token:
                temp = token[:-1] + "_"
                cand_list = suggest(temp)
                cand = random.choice(cand_list)[0]
                generated_source.append(cand)
                similar_flag = True
            elif preposition_token:
                generated_source.append(random.choice(PREPOSITIONS))
                proposition_flag = True
            elif incorrect_comparative_token:
                generated_source.append(random.choice(COMPARATIVES) + " " + token)
                incorrect_comparative_flag = True
            elif pos_plural_token:
                token = change_pluralization(token)
                generated_source.append(token)
                plural_flag = True
            elif pos_tense_token:
                token = change_tense(token)
                generated_source.append(token)
                tense_flag = True
            elif not dropout_token:
                generated_source.append(token)
            elif pos_modal_token:
                generated_source.append(MODAL[token])
                modal_flag = True

        noised_sentence_list.append(" ".join(generated_source))
    process_dict[i]=noised_sentence_list




def main():
    ignore_files = ["train_correct.txt", "train_error.txt", "validation_correct.txt", "validation_error.txt"]
    lines = []
    dirs = glob.glob("*.txt")
    for f in ignore_files:
        dirs.remove(f)
    print("Successfully read lines from the following files:")
    for filename in dirs:
        d = os.path.join(os.getcwd(),filename)
        with open(d, "r", encoding = 'UTF-8') as f:
            if filename != "nmt_combined.txt":
                for r_l in f:
                    split_l = r_l.split("\t")
                    line = unidecode(split_l[0].strip())
                    lines.append(line)
            else:
                spl_lines = f.read().splitlines()
                for l in spl_lines:
                    lines.append(unidecode(l))
        
        print(filename)
    

    lines = list(OrderedDict.fromkeys(lines)) # removes all duplicates
    corrupted_lines = []
    for i in range(len(lines)-1, -1, -1): # remove all corrupted lines
        if sequence_weirdness(lines[i]) != 0:
            corrupted_lines.append(lines[i])
            del(lines[i])
    with open("corrupted_lines", "w") as f:
        for i in corrupted_lines:
            f.write(i + "\n")
    print("Removed {} corrupted lines".format(len(corrupted_lines)))
    print("Corrupted lines extracted and written to corrected_lines")
    print("Extracted {} unique clean lines".format(len(lines)))
    #import pdb; pdb.set_trace()
    num_dups = 3
    lines = [val for val in lines for _ in range(num_dups)] # create duplicates

    print("Creating {} duplicates for every clean line".format(num_dups))
    print("Total clean lines now: {}".format(len(lines)))
    #lines = lines[:1000]
    noised_lines = []
    chunk_size = int(max(len(lines)/10,1))
    manager = Manager()
    process_dict = manager.dict()
    process_list = []
    for i in range(0, len(lines), chunk_size):
        process_dict[i] = []
        index = min(i+chunk_size, len(lines))
        p = Process(target=noise_generator, args=(lines[i:index],i, process_dict))
        process_list.append(p)
        
    print("Generating noise - {} processes spawned for noise generation...".format(len(process_list)))
    
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    for k, v in process_dict.items():
        noised_lines.extend(v)

    c = list(zip(noised_lines, lines))
    random.shuffle(c)
    noised_lines, lines = zip(*c)

    train_thres = int(len(lines)*(85/100))

    with open("validation_error.txt", "w", encoding = "utf-8") as f, open("validation_correct.txt", "w", encoding = "utf-8") as f2:
        for l in noised_lines[train_thres:]:
            f.write(l.strip() + "\n")

        for l in lines[train_thres:]:
            f2.write(l.strip() + "\n")

    with open("train_error.txt", "w", encoding = "utf-8") as f, open("train_correct.txt", "w", encoding = "utf-8") as f2:
        for l in noised_lines[:train_thres]:
            f.write(l.strip() + "\n")

        for l in lines[:train_thres]:
            f2.write(l.strip() + "\n")


    print("Created the following files: validation_error.txt, validation_correct.txt, train_correct.txt, train_error.txt")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("The application took {} seconds = {} minutes".format(end-start, (end-start)/60))