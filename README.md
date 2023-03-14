# LLaMA. Simple. Using HuggingFace.

### What is this all about?

- Do you also want a "private GPT-3" at home?
- It also annoys you that people on the internet are excited about "llama weights" and yet there is no interface or any guide for how to use them?
- You also sick of dealing with all kinds of people on the Internet who play around with tensors then upload a code that no one can really use?

**I prepared a single repo for you with EVERYTHING you need to run LLaMA.**

Here is Everything you need for running (and training!) LLaMA using Hugging Face interface 👌

### TL;DR:

```python
tokenizer = llama.LLaMATokenizer.from_pretrained('decapoda-research/llama-7b-hf')
model = llama.LLaMAForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
print(tokenizer.decode(model.generate(tokenizer('Yo mama', return_tensors = "pt")["input_ids"])[0]))
```

Yeah. No overengineering bullshit.

> Also: No need to clone a huge custom `transformers` repo that you later on stuck with maintaining and updating yourself. 


# What is LLaMA?

**TL;DR:** GPT model by [meta](https://ai.facebook.com/research/publications/llama-open-and-efficient-foundation-language-models/) that surpasses GPT-3, released to selected researchers but [leaked to the public](https://analyticsindiamag.com/metas-llama-leaked-to-the-public-thanks-to-4chan/).

LLaMA is a large language model [trained by Meta AI](https://ai.facebook.com/research/publications/llama-open-and-efficient-foundation-language-models/) that surpasses GPT-3 in terms of accuracy and efficiency while being 10 times smaller.

> **Paper Abstract:**
>
> We introduce LLaMA, a collection of founda- tion language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla- 70B and PaLM-540B. We release all our models to the research community.
>   

# How can I use LLaMA?

## Installation

```bash
#git clone https://github.com/ypeleg/llama
git clone https://github.com/vrangayyan6/llama
```

```bash
conda create -n llama python=3.10
conda activate llama
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

cd into this repository
```bash
pip install -e .
```


## Usage

### 1. Import the library and choose model size

```python
import llama
MODEL = 'decapoda-research/llama-7b-hf'
```

**We currently support the following models sizes:**
- 
- Options for `MODEL`:
    - `decapoda-research/llama-7b-hf`
    - `decapoda-research/llama-13b-hf`
    - `decapoda-research/llama-30b-hf`
    - `decapoda-research/llama-65b-hf`

**Note:** The model size is the number of parameters in the model. The larger the model, the more accurate the model is, but the slower, heavier and more expensive it is to run. 

### 2. Load the tokenizer and model

```python
tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
model = llama.LLaMAForCausalLM.from_pretrained(MODEL)
model.to('cuda')
```

### 3. Encode the prompt

> For example, we will use the prompt: "Yo mama"
>   
> We will use the `tokenizer` to encode the prompt into a tensor of integers.

```python
PROMPT = 'Yo mama'
encoded = tokenizer(PROMPT, return_tensors = "pt")
```

### 4. Generate the output

> We will use the `model` to generate the output.

```python
generated = model.generate(encoded["input_ids"].cuda())[0])
``` 

### 5. Decode the output
```python
decoded = tokenizer.decode(generated)
```

### 6. Print the output

```python
print(decoded)
```

**Expected output:** "Yo mama is so fat, she has to buy two seats on the plane."
