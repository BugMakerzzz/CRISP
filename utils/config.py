# different template format of the commonsense example generation
CASE_FORMAT_LIST = ['..., then ...', '..., so ...', '..., and ...', '..., but ...', '..., after that, ...', 'Before ..., ...']

# openai config
OPENAI_API_KEY = ''
MAX_REQUESTS_PER_MINUTE = 3500 # 3_000 * 0.5
MAX_TOKENS_PER_MINUTE = 90000 #250_000 * 0.5
REQUEST_URL = ''

llama2_7b_path = '/mnt/{dir}/huggingface/llama-2-7b-hf'
# llama2_7b_path = '/netcache/huggingface/llama-2-7b-hf'
llama2_7b_chat_path = '/mnt/{dir}/huggingface/llama-2-7b-chat-hf'
llama2_13b_path = '/mnt/{dir}/huggingface/Llama-2-13b-hf'
llama2_13b_chat_path = '/mnt/{dir}/huggingface/Llama-2-13b-chat-hf'
llama3_8b_path = '/mnt/{dir}/huggingface/Meta-Llama-3-8B'
llama3_8b_chat_path = '/mnt/{dir}/huggingface/Meta-Llama-3-8B-Instruct'
llama3_1_8b_chat_path = '/mnt/{dir}/huggingface/Meta-Llama-3.1-8B-Instruct-new'
llama_moe_path = '/mnt/{dir}/huggingface/LLaMA-MoE-v1-3_5B-4_16'

mistral_7b_path = '/mnt/{dir}/huggingface/Mistral-7B-v0.1'
mistral_7b_chat_path = '/mnt/{dir}/huggingface/Mistral-7B-Instruct-v0.2'

vicuna_7b_path = '/mnt/{dir}/huggingface/vicuna-7b'
vicuna_13b_path = '/mnt/{dir}/huggingface/vicuna-13b'

qwen1_8b_path = '/mnt/{dir}/huggingface/Qwen-1_8B'
qwen2_5_3b_chat_path =  '/mnt/{dir}/huggingface/Qwen2.5-3B-Instruct'
qwen2_5_7b_chat_path =  '/mnt/{dir}/huggingface/Qwen2.5-7B-Instruct'
qwen2_5_14b_chat_path = '/mnt/{dir}/huggingface/Qwen2.5-14B-Instruct'

qwen2_5_math_1_5b_path = '/mnt/{dir}/huggingface/Qwen2.5-Math-1.5B'
qwen2_5_math_1_5b_chat_path = '/mnt/{dir}/huggingface/Qwen2.5-Math-1.5B-Instruct'
qwen2_5_math_7b_path = '/mnt/{dir}/huggingface/Qwen2.5-Math-7B'
qwen2_5_math_7b_chat_path = '/mnt/{dir}/huggingface/Qwen2.5-Math-7B-Instruct'

phi2_path = '/mnt/{dir}/huggingface/phi-2'
phi3_path = '/mnt/{dir}/huggingface/Phi-3-small-8k-instruct'

yi_1_5_6b_chat_path = '/mnt/{dir}/huggingface/Yi-1.5-6B-Chat'

gemma_2_9b_path = '/mnt/{dir}/huggingface/gemma-2-9b'
gemma_2_9b_chat_path = '/mnt/{dir}/huggingface/gemma-2-9b-it'

qwq_32b_path = '/mnt/{dir}/huggingface/QwQ-32B-Preview'

deepseek_14b_path = '/mnt/{dir}/huggingface/DeepSeek-R1-Distill-Qwen-14B'

reward_skywork_path = '/mnt/{dir}/huggingface/Skywork-Reward-Llama-3.1-8B-v0.2'
reward_shepherd_path = '/mnt/{dir}/huggingface/math-shepherd-mistral-7b-prm'
reward_armorm_path = '/mnt/{dir}/huggingface/ArmoRM-Llama3-8B-v0.1'
reward_grm_path = '/mnt/{dir}/huggingface/GRM-gemma2-2B-rewardmodel-ft'
reward_skyworko1_path = '/mnt/{dir}/huggingface/Skywork-o1-Open-PRM-Qwen-2.5-7B'
reward_genprm_path = '/mnt/{dir}/huggingface/GenPRM-7B'
reward_qwen_math_path = '/mnt/{dir}/huggingface/Qwen2.5-Math-PRM-7B'
reward_gemma_path = '/mnt/{dir}/huggingface/RM-Gemma-2B'

figure_colors = ['#90BCD5', '#E76254', '#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6']