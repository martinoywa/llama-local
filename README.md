# llama-local
Run LLama-2/3 Weights Locally

# Set-up
1. The bash script assumes that you've already downloaded your weights from https://llama.meta.com/llama-downloads.
2. Run bash script.
```
chmod +x convert_to_hugging_face.sh
./convert_to_hugging_face.sh /home/llama /home/llama/llama-2-7b-chat 7B
```
3. Run python script.
```
python local_llama.py
```