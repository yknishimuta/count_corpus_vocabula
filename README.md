# count_corpus_vocabula

**count_corpus_vocabula** is a lightweight toolkit for counting vocabulary items and generating frequency tables from text corpora.  
It is designed mainly for Latin text processing but can be adapted to other languages depending on the NLP pipeline used.

---

## Features

- Vocabulary counting for one or multiple corpora (**group** / **groups**)
- Lemma-based counting via Stanza (default) and configurable language/package
- Optional preprocessing step (**preprocess**) to run a cleaner before counting
- Batch processing with a single command
- Output frequency tables in CSV format and summary statistics
- Simple exclusion list support via `config/exclude_lemmas.txt`

---

## Installation

```bash
git clone https://github.com/yknishimuta/count_corpus_vocabula.git
cd count_corpus_vocabula

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Installation (Full Setup)

count_corpus_vocabula depends on nlpo_toolkit.

####   Step 1 — Install nlpo_toolkit
##### Option A: Install from GitHub (recommended)

```
pip install git+https://github.com/yknishimuta/nlpo_toolkit.git
```

##### Option B: Install from source

```
git clone https://github.com/yknishimuta/nlpo_toolkit.git
cd nlpo_toolkit
pip install -r requirements.txt
pip install .
```

#### Step 2 — Install count_corpus_vocabula

```
git clone https://github.com/yknishimuta/count_corpus_vocabula.git
cd count_corpus_vocabula
pip install -r requirements.txt
```

#### Step 3 — Install Stanza models (first time only)

```
import stanza
stanza.download("la", package="perseus")
```


## Requirements / Environment

- **Python 3.8+** is recommended.

Typical dependencies:

- **stanza** — used for tokenization, lemmatization, and other NLP processing  

## Usage

### 1. Prepare your text files

Place your corpus text files (UTF-8 plain text) anywhere on your filesystem.
 You may organize them by author, work, or time period, for example:

```
/corpora/
  chapter1/*.txt
  chapter2/*.txt
```

These paths will later be referenced in the YAML configuration.

------

### 2. Define groups (single or multiple)

Create or edit `config/groups.config.yml`.

#### Option A: Single group (shortcut)

```yaml
group:
  name: text
  files:
    - corpora/chapter1/*.txt

out_dir: output
language: la
stanza_package: perseus
cpu_only: true
```

#### Option B: Multiple groups

```
groups:
  chapter1:
    files:
      - corpora/chapter1/*.txt
  chapter2:
    files:
      - corpora/chapter2/*.txt

out_dir: output
language: la
stanza_package: perseus
cpu_only: true
```

Each group collects one or more text files using glob patterns.

------

### 3. Run the vocabulary counter

```
python count_corpus_vocabula_local.py 
```

------

## Preprocess (Cleaner)

You can optionally run a cleaner before counting by adding a `preprocess` block.
This enables a single-command workflow:

1. Run cleaner
2. Generate cleaned text files
3. Count vocabulary on the selected cleaned groups

### Example: cleaner + single group (all cleaned files)

```yaml
preprocess:
  kind: cleaner
  config: cleaners/config/sample.yml

group:
  name: cleaned_all
  files:
    - cleaned/*.txt

out_dir: output/cleaned_vocab
language: la
stanza_package: perseus
cpu_only: true
```



### Example: cleaner + multiple groups (split cleaned outputs)

```
preprocess:
  kind: cleaner
  config: cleaners/config/sample.yml

groups:
  left:
    files:
      - cleaned/ST-I_left_*.txt
  right:
    files:
      - cleaned/ST-I_right_*.txt

out_dir: output/cleaned_vocab
language: la
stanza_package: perseus
cpu_only: true
```

## Exclude list

To exclude specific lemmas from the final frequency tables (e.g., `idest`),
create `config/exclude_lemmas.txt` (one lemma per line):

```txt
idest
ides
```

## License

This project is released under the **MIT License**.

## Author

**yknishimuta**
 GitHub: https://github.com/yknishimuta