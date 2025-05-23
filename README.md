[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-purple)](https://huggingface.co/Jakh0103/lid)

# Language Identification for 2000 languages: Optimized for low-resource langauges

TL;DR: The model is trained on [GlotLID-C](https://huggingface.co/datasets/cis-lmu/glotlid-corpus), and supports 2099 languages.

### 🛠️ Setup
```
pip install -r requirements.txt
```

### 🤖 Usage

**Download the model**
```
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Jakh0103/lid", local_dir="checkpoint")
```

**Use the model**
```
from model import LID
model = LID.from_pretrained(dir='checkpoint')

# print the supported labels
print(model.get_labels())
## ['aai_Latn', 'aak_Latn', 'aau_Latn', 'aaz_Latn', 'aba_Latn', ...]

# prediction
model.predict("The cat climbed onto the roof to enjoy the warm sunlight peacefully!")
# (['eng_Latn'], [0.970989465713501])

model.predict("The cat climbed onto the roof to enjoy the warm sunlight peacefully!", k=3)
## (['eng_Latn', 'sco_Latn', 'jam_Latn'], [0.970989465713501, 0.006496887654066086, 0.00487488554790616])
```

### 🎯 TODO
- [ ] Release the training code
- [ ] Release the evaluation code
- [ ] Optimize the inference using parallel tokenization