# source: https://github.com/fastai/course-nlp/blob/master/nlputils.py

from fastai2.basics import *
import re
import pandas as pd


def get_wiki(path,lang):
    name = f'{lang}wiki'
    if (path/name).exists():
        print(f"{path/name} already exists; not downloading")
        return

    xml_fn = f"{lang}wiki-latest-pages-articles.xml"
    zip_fn = f"{xml_fn}.bz2"

    if not (path/xml_fn).exists():
        print("downloading...")
        download_url(f'https://dumps.wikimedia.org/{name}/latest/{zip_fn}', path/zip_fn)
        print("unzipping...")
        bunzip(path/zip_fn)

    # Change working directory to `path`
    prev_cwd = Path.cwd()
    os.chdir(path)
    
    # Get wikiextractor
    if not (path/'wikiextractor').exists(): os.system('git clone https://github.com/attardi/wikiextractor.git && cd wikiextractor && git checkout 16186e290d9eb0eb3a3784c6c0635a9ed7e855c3')

    # Extraction
    print("extracting...")
    os.system("python wikiextractor/WikiExtractor.py --processes 4 --no_templates " +
            f"--min_text_length 1800 --filter_disambig_pages --log_file log -b 100G -q {xml_fn}")
    shutil.move(str(path/'text/AA/wiki_00'), str(path/name))
    shutil.rmtree(path/'text')
    
    # Return working directory to previous
    os.chdir(prev_cwd)

def split_wiki(path,lang):
    dest = path/'docs'
    name = f'{lang}wiki'
    if dest.exists():
        print(f"{dest} already exists; not splitting")
        return dest

    dest.mkdir(exist_ok=True, parents=True)
    title_re = re.compile(rf'<doc id="\d+" url="https://{lang}.wikipedia.org/wiki\?curid=\d+" title="([^"]+)">')
    lines = (path/name).open()
    f=None

    for i,l in enumerate(lines):
        if i%100000 == 0: print(i)
        if l.startswith('<doc id="'):
            title = title_re.findall(l)[0].replace('/','_')
            if len(title)>150: continue
            if f: f.close()
            f = (dest/f'{title}.txt').open('w')
        else: f.write(l)
    f.close()
    return dest

def clean_files(dest):

    doc_re = re.compile(rf'([\w\W]*)<\/doc>') # delete </doc>
    
    for i,l in enumerate(dest.ls()):
        # open file and get content without first line which is the title
        f = l.open('r+', encoding="utf-8")
        f.readline()
        text = f.read()
        # get content without </doc> and delete empty line and whitespaces at the head and tail
        text = doc_re.findall(text)[0].strip()
        # delete file content
        f.seek(0)
        f.truncate()
        # write modificated text in file
        f.write(text)
        f.close()
        
def get_one_clean_file(dest,lang):

    fname = f'all_texts_{lang}wiki.txt'
    doc_re = re.compile(rf'([\w\W]*)<\/doc>') # delete </doc>
    
    all_texts = ''
    for i,l in enumerate(dest.ls()):
        # open file and get content without first line which is the title
        f = l.open('r+', encoding="utf-8")
        f.readline()
        text = f.read()
        f.close()
        # get content without </doc> and delete empty line and whitespaces at the head and tail
        text = doc_re.findall(text)[0].strip()
        # concatenate text
        all_texts += text
        all_texts += "\n"
        if not (i % 1000): print(i)
  
    with open (dest.parent/fname, 'w') as fp: 
        fp.write(all_texts)
    print(f"all texts from wikipedia {lang} in the file {dest.parent/fname}\n")

def get_one_clean_csv_file(dest,lang):    
                         
    fname = f'all_texts_{lang}wiki.csv'
    doc_re = re.compile(rf'([\w\W]*)<\/doc>') # delete </doc>
    
    all_texts = list()
    for i,l in enumerate(dest.ls()):
        # open file and get content without first line which is the title
        f = l.open('r+', encoding="utf-8")
        f.readline()
        text = f.read()
        f.close()
        # get content without </doc> and delete empty line and whitespaces at the head and tail
        text = doc_re.findall(text)[0].strip()
        # append text
        all_texts.append(text)
  
    # Create the pandas DataFrame 
    df = pd.DataFrame(all_texts, columns = ['text'])
    
    # save
    df.to_csv(dest.parent/fname, index=False)  
    print(f"all texts from wikipedia {lang} in the file {dest.parent/fname}\n")
                         
def get_num_tokens(dest):
    
    # Getting an idea of the number of words
    files = dest.ls()
    num_tokens = 0

    for i,l in enumerate(files):
        f = l.open('r', encoding="utf-8")
        words = f.read()
        num_tokens += len(words.split())
        f.close()
        
    num_files = i+1
    
    return num_files, num_tokens
