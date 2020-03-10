import argparse
from io import open
import pyconll
import csv
import os

def create_fn(filename, add, extension):
    base = os.path.splitext(filename)[0]
    return "{0}_{1}{2}".format(base, add, extension)

def create_fn_from_url(url, add, extension):
    filename = url[url.rfind("/")+1:]
    base = os.path.splitext(filename)[0]
    return "{0}_{1}{2}".format(base, add, extension)

def token_info(token): # Format token information in the required format
    token_char = list()
    token_char.append(token.form.replace(" ", ""))
    string_to_append = token.upos
    for k, v in token.feats.items():
        if token.upos == 'NOUN' or token.upos == 'PROPN':
            if k == 'Case' or k == 'Gender' or k == 'Number':
                feat_to_append = ";{}={}".format(k, ','.join(v))
                string_to_append += feat_to_append
        else:
            feat_to_append = ";{}={}".format(k, ','.join(v))
            string_to_append += feat_to_append
    token_char.append(string_to_append)
    return token_char

def create_input(file_path, context):
    file = pyconll.load_from_url(file_path)
    statistics = set()    
    bos = list()
    bos.append("-START-") # Beginning of sentence marker
    eos = list()
    eos.append("-END-") # End of sentence marker

    with open(create_fn_from_url(file_path, "result", ".tsv"), "w+", newline='', encoding="utf-8") as f:
        for sentence in file:
            for token in sentence:
                if token.upos == 'NOUN' or token.upos == 'PROPN':
                    token_id = int(token.id)
                    id = token_id
                    token_list = list();

                    # Take 'context' amount of words before target
                    for i in range(1, context + 1):
                        if id - i > 0:
                            try:
                                while sentence[str(id-i)].upos == 'PUNCT' or sentence[str(id-i)].form == '_': # Avoiding punctuation or 'empty' words (_)
                                    id -= 1
                            except KeyError:
                                pass
                            if id - i > 0:
                                token_list.append(token_info(sentence[str(id - i)]))
                            else:
                                token_list.append(bos)
                        else:
                            token_list.append(bos)

                    token_list.reverse()
                    token_list.append(token_info(sentence[str(token_id)]))

                    # Take 'context' amount of words after target
                    id = token_id
                    for i in range(1, context + 1):
                        end = int(sentence[len(sentence)-1].id)
                        if id + i <= end:
                            try:
                                while sentence[str(id+i)].upos == 'PUNCT' or sentence[str(id+i)].form == '_': # Avoiding punctuation or 'empty' words (_)
                                        id += 1
                            except KeyError:
                                pass
                            if id + i <= end:
                                token_list.append(token_info(sentence[str(id + i)]))
                            else:
                                token_list.append(eos)
                        else:
                            token_list.append(eos)

                    tsv = csv.writer(f, delimiter='|')  
                    for list_elem in token_list:
                        tsv.writerow(list_elem)
                    tsv.writerow([])

def create_stats(file):
    with open(file, 'r', encoding="utf-8") as f:
        tsv_in = csv.reader(f, delimiter='\n')
        tags = list()
        for row in tsv_in:
            try:
                row = row[0].split("|", maxsplit=1)[1].replace("PROPN", "NOUN")
                tags.append(row)
            except IndexError:
                pass
        stats_after = set(tags)
        print(len(stats_after))
        with open(create_fn(file, "stats", ".tsv"), "w+", newline='', encoding="utf-8") as out:
            stats = csv.writer(out, delimiter='\n')
            stats.writerow([len(stats_after),])
            for element in stats_after:
                stats.writerow([element])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", help="Word count before and after target")
    parser.add_argument("--file", help="File that contains sentences in CoNLL-U format")
    args = parser.parse_args()   
    create_input(args.file, int(args.context))
    create_stats(create_fn_from_url(args.file, "result", ".tsv"))

if __name__ == "__main__":
    main()