## Introduction

This project has plenty modification based on DSGBench, and realized the pipeline of
 werewolf game, with flexible game set-up, LLM model configuration and evaluation.
## Quick Start

### DSGBench
First, you should refer to `README.md` to download dsgbench environment. 

- Replace the prerequisites's download with `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
Our experiment is conducted on Windows, so there might be some errors.

- If you encounter `ERROR: No matching distribution found for black`, try running 

    ```shell
    python -m pip install --upgrade pip
    pip install black -i https://pypi.tuna.tsinghua.edu.cn/simple

    ```

- If you encounter `TypeError: encode() argument 'encoding' must be str, not None`, when installing jax with `pip install -U jax`, check [JAX](https://docs.jax.dev/en/latest/quickstart.html) official website.

If you want to realize the UI provided by DSGBench, you can refer to `games/werewolf/README.md` and install npm. 

- Make sure the game info is under `/visual/logs/{session_id}`
- Your final UI will be `http://10.181.133.1:8081/?session_id={session_id}` once you run:

```shell
cd games/werewolf/visual  
npm run start
```

### Message Pool

Since huggingface is not stable domestically, you can refer to `my_warewolf/download_sentence_transformer.py` to download faster using snapshot_download.

### Config structure

- For all your llm model api and setting, you should put it under `werewolf\configs\llm_configs` , which carries the form of :

  ```yaml
  model_name: claude-3-7-sonnet-20250219
  api_key: your-api-key
  api_url: https://api.toiotech.com/v1
  max_tokens: 1024
  timeout: 30
  temperature: 0.2
  model_type: LLMModel
  extra_body:
    thinking:
      type: enabled
      budget_tokens: 20
  ```


- For all your game setting, you should put it under `werewolf\config\eval_configs`

  ```yaml
  # refer to configs\eval_configs\eval_deepseek-v25_werewolf_scene1_deepseek-v25_vs_gpt-4o-mini.yaml
  use_message_pool: false # if is set to false, the game will not use message pool
  message_pool_args:
    human_in_combat: false
    load_exps_from: ""
    logs_path_to: "logs"
    exps_path_to: "checkpoints/deepseek_vs_doubao"
    # current_game_number: 1
    use_crossgame_exps: true
    who_use_exps: 
      - "werewolf"
      - "villager"
      - "seer"
      - "witch"
      - "guard"
    exps_retrieval_threshold: 0.85
    similar_exps_threshold: 0.1

  has_sheriff: True # if is set to True, the game will have sheriff role

  eval:
    num_matches: 10
    output_path: ./output_my/deepseek-v25/werewolf/scene1/deepseek-v1_vs_doubao
    weave_prj_name: my_deepseek_r1_vs_gpt_4o
  game:
    game_name: WereWolfEnv
    good_model_config: loguru_deepseek-r1-250120.yaml
    bad_model_config: loguru_doubao-1-5-pro-256k-250115.yaml
  agent:
  - agent_name: WereWolfAgent
    agent_nick: Seer
    agent_model: LLMModel
    agent_model_config: loguru_deepseek-r1-250120.yaml
    use_message_pool: true # if is set to true, the agent will use message pool
  - agent_name: WereWolfAgent
    agent_nick: Doctor
    agent_model: LLMModel
    agent_model_config: loguru_deepseek-r1-250120.yaml
  - agent_name: WereWolfAgent
    agent_nick: Villager
    agent_model: LLMModel
    agent_model_config: loguru_deepseek-r1-250120.yaml
  - agent_name: WereWolfAgent
    agent_nick: Villager
    agent_model: LLMModel
    agent_model_config: loguru_deepseek-r1-250120.yaml
  - agent_name: WereWolfAgent
    agent_nick: Villager
    agent_model: LLMModel
    agent_model_config: loguru_deepseek-r1-250120.yaml
  - agent_name: WereWolfAgent
    agent_nick: Villager
    agent_model: LLMModel
    agent_model_config: loguru_deepseek-r1-250120.yaml
  - agent_name: WereWolfAgent
    agent_nick: Werewolf
    agent_model: LLMModel
    agent_model_config: loguru_doubao-1-5-pro-256k-250115.yaml
  - agent_name: WereWolfAgent
    agent_nick: Werewolf
    agent_model: LLMModel
    agent_model_config: loguru_doubao-1-5-pro-256k-250115.yaml

  ```


- Then put the yaml file under `configs\llm_configs`, and config it first at `tasks_config.py`,  then at `run_tasks_parallel.py`:

  ```python
  # tasks_config.py
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
  ```

  ```python
  # run_tasks_parallel.py
  tasks = [
      # "--tasks deepseek-v1_vs_doubao",
      # "--tasks deepseek-v1_vs_gpt_4o",
      # "--tasks doubao_vs_gpt_4o",
      # "--tasks doubao_vs_deepseek-v1",
      # "--tasks gpt_4o_vs_deepseek-v1",
      "--tasks gpt_4o_vs_doubao",
      # "--tasks gpt_4o_vs_deepseek-v1",
  ]
  ```

  Where your log will be saved at `output_my` and the print file will be saved at `output_backup`.

- Start the evaluation:

   ```shell
    python run_tasks_parallel.py
    ```


