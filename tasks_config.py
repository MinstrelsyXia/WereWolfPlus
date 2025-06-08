# WANDB_API_KEY='0df2bb005b0367568689f62cc7dcd9e521f4e70b'
# WANDB_API_KEY='5e4de12fa847ce69f658bd4cd6ef1819aa110ed5'
WANDB_API_KEY = 'your wandb api key here'
WANDB_TIMEOUT = '60'
WANDB_ENTITY = 'your entity'
WEAVE_OPEN = False
TASKS_CFG_PATH = 'configs/eval_configs/'

TASKS = {
"deepseek-v1_vs_doubao_12":"deepseek-v1_vs_doubao_12.yaml",
"deepseek-v1_vs_doubao_9":"deepseek-v1_vs_doubao_9.yaml",
"deepseek-v1_vs_doubao_12_helms":"deepseek-v1_vs_doubao_12_helms.yaml",
"deepseek-v1_vs_doubao_9_helms":"deepseek-v1_vs_doubao_9_helms.yaml",
"deepseek-v1_vs_gpt_4o":"deepseek-v1_vs_gpt_4o.yaml",
"doubao_vs_deepseek-v1":"doubao_vs_deepseek-v1.yaml",
"doubao_vs_gpt_4o":"doubao_vs_gpt_4o.yaml",
"gpt_4o_vs_deepseek-v1":"gpt_4o_vs_deepseek-v1.yaml",
"gpt_4o_vs_doubao":"gpt_4o_vs_doubao.yaml",
}
