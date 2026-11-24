CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server --model-path openai/gpt-oss-20b --host 127.0.0.1 --port 30000 --tp-size 1 --attention-backend triton --disable-radix-cache --disable-cuda-graph --log-level info --reasoning-parser gpt-oss 
