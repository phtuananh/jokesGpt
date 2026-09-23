# Joke GPT From Scratch

Minimal joke-only chatbot trained from scratch with:

* raw joke dataset
* data cleaner
* tokenizer Hugging Face
* small GPT model
* chat interface

No pretrained model.
Only PyTorch + Hugging Face tokenizer.

Goal: a sample chatbot for fun. For a minimalist GPT example, use this [repo](https://github.com/phtuananh/sampleGpt)

## Project structure

```text
joke_gpt/
  data.txt
  clean_data.py
  tokenizer.py
  model.py
  train.py
  chat.py
  requirements.txt
  README.md
```
## Create virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```
## Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```txt
torch
```

---

## Prepare dataset

Input file:

```text
data.txt
```

Format:

```text
Why did the chicken cross the road? To get to the other side.
I told my computer I needed a break, and it said no problem.
Why do programmers prefer dark mode? Because light attracts bugs.
```

One raw joke per line.

---

## Clean dataset

```bash
python clean_data.py --input data.txt --output data_clean.txt
```

This creates:

```text
data_clean.txt
```

The cleaner:

* removes IDs
* normalizes text
* removes duplicates
* removes broken/too short/too long lines
* converts jokes into chat training format

Example output:

```text
<|bos|>
<|user|>Tell me a joke.
<|assistant|>
Why did the chicken cross the road? To get to the other side.
<|eos|>
```

---

## Train model

### Default training

```bash
python train.py --data data_clean.txt --out_dir models/jokes_bpe
```

This outputs:

```text
models/jokes_bpe/
  model.pt
  tokenizer.json
  config.json
  data_clean.txt
```
Training detect automatically if device is CPU or GPU then take default parameters

Training time estimation: 1h40' on an old CPU only (i7-5820K at 60%)

## Chat with the model

```bash
python chat.py --model_dir models/jokes_bpe
```

Example:

```text
You: tell me a programming joke

Bot: Why did the programmer quit his job? Because he didn't get arrays.
```

Exit:

```text
exit
```

or:

```text
quit
```

---

## Chat optional parameters

```bash
python chat.py \
  --model_dir models/jokes_bpe \
  --attempts 20 \
  --max_new_tokens 100 \
  --temperature 0.9 \
  --top_k 50
```

Meaning:

| Parameter          | Meaning                                      |
| ------------------ | -------------------------------------------- |
| `--attempts`       | generates many candidates and keeps the best |
| `--max_new_tokens` | max answer length                            |
| `--temperature`    | randomness                                   |
| `--top_k`          | restricts token choices                      |

Good values:

```text
attempts = 20
max_new_tokens = 80 to 120
temperature = 0.8 to 1.0
top_k = 40 to 80
```

## Training: optional parameters

```bash
python train.py \
  --data data_clean.txt \
  --out_dir models/jokes_bpe \
  --bpe_vocab_size 2000 \
  --block_size 128 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --batch_size 32 \
  --max_iters 5000
```

Windows PowerShell:

```powershell
python train.py `
  --data data_clean.txt `
  --out_dir models/jokes_bpe `
  --bpe_vocab_size 2000 `
  --block_size 128 `
  --n_layer 4 `
  --n_head 4 `
  --n_embd 256 `
  --batch_size 32 `
  --max_iters 5000
```

## Training parameters

| Parameter          | Meaning                      |
| ------------------ | ---------------------------- |
| `--bpe_vocab_size` | BPE vocabulary size          |
| `--block_size`     | context length in tokens     |
| `--n_layer`        | number of Transformer blocks |
| `--n_head`         | number of attention heads    |
| `--n_embd`         | embedding size               |
| `--batch_size`     | batch size                   |
| `--max_iters`      | training steps               |
| `--learning_rate`  | optimizer learning rate      |
| `--dropout`        | dropout probability          |

---

## Good first settings

Small CPU model:

```text
bpe_vocab_size = 2000
block_size = 128
n_layer = 4
n_head = 4
n_embd = 256
batch_size = 32
max_iters = 5000
```

Better GPU model:

```text
bpe_vocab_size = 2000
block_size = 128
n_layer = 6
n_head = 6
n_embd = 384
batch_size = 64
max_iters = 10000
```

Larger GPU model:

```text
bpe_vocab_size = 4000
block_size = 192
n_layer = 8
n_head = 8
n_embd = 512
batch_size = 32
max_iters = 20000
```

---

## Full pipeline

```bash
python clean_data.py --input data.txt --output data_clean.txt

python train.py \
  --data data_clean.txt \
  --out_dir models/jokes_bpe \
  --bpe_vocab_size 2000 \
  --block_size 128 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --batch_size 64 \
  --max_iters 10000

python chat.py --model_dir models/jokes_bpe
```

---

## Troubleshooting

### Bad output

Try:

```bash
python chat.py --model_dir models/jokes_bpe --temperature 0.8 --top_k 40
```

### Repetitive output

Reduce temperature and max tokens:

```bash
python chat.py --model_dir models/jokes_bpe --temperature 0.75 --max_new_tokens 80
```

### Output too short

Increase max tokens:

```bash
python chat.py --model_dir models/jokes_bpe --max_new_tokens 140
```

### Training too slow

Use smaller model:

```bash
python train.py \
  --data data_clean.txt \
  --out_dir models/jokes_bpe \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --batch_size 32 \
  --max_iters 5000
```

### CUDA out of memory

Reduce batch size:

```bash
python train.py --data data_clean.txt --out_dir models/jokes_bpe --batch_size 16
```

---

## 14. Notes

This model is trained from scratch.

It will not understand language like a pretrained LLM.

It can become a decent joke generator if:

* dataset is clean
* duplicates are removed
* training is long enough
* BPE tokenizer is used
* several candidates are generated and filtered

Best quality improvement:

```text
more clean jokes > bigger model > longer training
```

---

## 15. Acknowledgements

This project is heavily inspired by the educational work of Andrew Karpathy.

The Transformer architecture, training loop structure, attention implementation, GPT block organization, and overall learning approach are based on concepts presented in:

* https://github.com/karpathy/minGPT
* https://github.com/karpathy/nanoGPT
* https://github.com/karpathy/ng-video-lecture

Karpathy's repositories are among the best resources for understanding GPT models from first principles.

This project intentionally keeps the same educational philosophy:

* minimal code
* minimal dependencies
* transparent implementation
* learning-focused design

Differences from minGPT/nanoGPT:

* uses a BPE tokenizer built with the `tokenizers` library
* stores tokenizer and configuration separately
* includes a simple interactive chat program
* simplified folder structure for experimentation
* designed as a learning project rather than a production training framework

Credit goes to Andrew Karpathy for the original educational GPT implementations and teaching material and to ChatGPT 5.5.
