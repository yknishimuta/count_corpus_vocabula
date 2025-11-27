# count_corpus_vocabula

**count_corpus_vocabula** is a lightweight toolkit for counting vocabulary items and generating frequency tables from text corpora.  
It is designed mainly for Latin text processing but can be adapted to other languages depending on the NLP pipeline used.

---

## Features

- Vocabulary counting for multiple corpora or text groups  
- Support for lemmatized or surface-form counting (via external NLP tools such as Stanza)  
- Group configuration via a YAML file (`groups.config.yml`)  
- Batch processing through a command-line entry script  
- Output of frequency tables in CSV formats

---

## Installation

```bash
git clone https://github.com/yknishimuta/count_corpus_vocabula.git
cd count_corpus_vocabula

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
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

### 2. Define corpus groups

Create or edit `groups.config.yml`:

```
groups:
  - name: "Group1"
    path: "/path/to/chapter1/*.txt"
  - name: "Group2"
    path: "/path/to/chapter2/*.txt"
```

Each group collects one or more text files using a glob pattern.

------

### 3. Run the vocabulary counter

```
python count_corpus_vocabula_local.py 
```

## License

This project is released under the **MIT License**.

## Author

**yknishimuta**
 GitHub: https://github.com/yknishimuta