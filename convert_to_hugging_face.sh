#!/bin/sh

# NOTE: This script assumes that you've already downloaded your weights from https://llama.meta.com/llama-downloads.
# https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/
# Output path example: llama-2-7b-hf
# ./convert_to_hugging_face.sh --tokenizer_path ~/models/llama --model_dir_path ~/models/llama/llama-2-7b --model_size 7B --llama_version 2

usage() {
  echo "Usage: $0 --tokenizer_path <tokenizer_path> --model_dir_path <model_dir_path> --model_size <model_size> --llama_version <llama_version >"
  echo "Example: $0 --tokenizer_path /llama --model_dir_path /llama/llama-2-7b --model_size 7B --llama_version 2"
  exit 1
}

# check for missing arguments
if [ $# -ne 8 ]; then
    usage
fi

# install some necessary packages
pip install transformers
pip install accelerate
pip install sentencepiece
pip install -U bitsandbytes
pip install protobuf
pip install blobfile
pip install tiktoken


# set model path
tokenizer_path=""
model_dir=""
model_size=""
llama_version=""

# parse input
while [ "$1" != "" ]; do
    case $1 in
        --tokenizer_path )         shift
                         tokenizer_path=$1
                         ;;
        --model_dir_path )          shift
                         model_dir=$1
                         ;;
        --model_size  )      shift
                         model_size=$1
                         ;;
        --llama_version  )      shift
                         llama_version=$1
                         ;;
        -h | --help )    usage
                         ;;
        * )              usage
                         ;;
    esac
    shift
done


# Validate that all required arguments are provided
if [ -z "$tokenizer_path" ] || [ -z "$model_dir" ] || [ -z "$model_size" ] || [ -z "$llama_version" ]; then
    usage
fi

# cp/link tokenizer to model path
ln $tokenizer_path/tokenizer.model $model_dir/tokenizer.model

# convert llama weights to Hugging Face compatible ones.
# 13GB+ of available RAM required.
echo "13GB+ of RAM required"
TRANSFORM=`python -c "import transformers;print('/'.join(transformers.__file__.split('/')[:-1])+'/models/llama/convert_llama_weights_to_hf.py')"`
python $TRANSFORM --input_dir $model_dir --model_size $model_size --output_dir $model_dir-hf --llama_version $llama_version
echo "Saved at $model_dir-hf"
