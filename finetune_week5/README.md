Huggingface peft fine tuned model link: https://huggingface.co/NuppuCat/llama31b-finetuned

Errors encountered & fixes if applied

all bugs fiexd as the notebook

1.#bug: module 'rouge_score' has no attribute '__version__'

solution
from importlib.metadata import version

try:
    rouge_score_version = version('rouge-score')
    print(f'rouge-score version: {rouge_score_version}')
except Exception as e:
    print(f'Error retrieving version: {e}')

2.duiring the finetune, the GPU memory not enough, CUDA out of memory
solved as

per_device_train_batch_size=2,

gradient_accumulation_steps=8,

3. changed the model followed the instructions to llama3 1b, alter the train setting and the location of the dir. etc.
