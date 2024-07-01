#!/bin/sh

# NOTE: This script assumes that you've already downloaded your weights from https://llama.meta.com/llama-downloads.
# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/
# Output path example: llama-2-7b-hf

if [ $# -ne 3 ]; then
    echo "Usage: $0 tokenizer_path model_dir_path model_size"
    echo "Example: $0 llama llama-2-7b 7B"
    exit 1
fi

# install some necessary packages
pip install transformers
pip install accelerate
pip install sentencepiece
pip install -U bitsandbytes


# set model path
tokenizer_path="$1"
model_dir="$2"
model_size="$3"

# cp/link tokenizer to model path
ln $tokenizer_path/tokenizer.model $model_dir/tokenizer.model

# convert llama weights to Hugging Face compatible ones.
# 13GB+ of available RAM required.
TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
pip install protobuf && python $TRANSFORM --input_dir $model_dir --model_size $model_size --output_dir $model_dir-hf
