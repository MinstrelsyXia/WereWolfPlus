import os
import logging
import random
import numpy as np
import yaml
import concurrent
import json

from concurrent.futures import ThreadPoolExecutor
from box import Box
import games
from agent_manager import llm_models,agents,prompts
from tqdm.contrib.logging import logging_redirect_tqdm
from logging import Logger

def load_game(game_config_path, *args):
    game_config = Box.from_yaml(
        filename=game_config_path, Loader=yaml.FullLoader)
    return getattr(games, game_config.game_name)(args[0])

def load_env(args):
    if args.game.game_name=="WelfareDiplomacyEnv" and args.game.get('game_summarizer_model'):
        llm_config = Box.from_yaml(filename=os.path.join('configs/llm_configs/', args.game.get('game_summarizer_model_config')),
                                   Loader=yaml.FullLoader)
        llm_model = getattr(llm_models, args.game.get('game_summarizer_model'))(llm_config)
        args.game_summarizer_model=llm_model
        return getattr(games, args.game.game_name)(args)

    else:
        return getattr(games, args.game.game_name)(args)

def load_config_old(config_path):
    config = Box.from_yaml(
        filename=config_path, Loader=yaml.FullLoader)

    return config


def load_agent(args, **kwargs):
    agents_ret = []
    idx=0
    for agent in args.agent:
        if agent.get('agent_prompt'):
            if args.game.get('players'):
                if "_VS_" in args.game.players:
                    players = args.game.players.split("_VS_")
                prompt = getattr(prompts, agent.get('agent_prompt'))(players[idx])
            else:
                prompt = getattr(prompts, agent.get('agent_prompt'))()
        else:
            prompt=None
        if agent.get('agent_prompt_opp'):
            prompt_opp = getattr(prompts, agent.get('agent_prompt_opp'))()
        else:
            prompt_opp=None
        # this way:
        if agent.get('agent_model_config'):
            llm_config = Box.from_yaml(filename=os.path.join('configs/llm_configs/', agent.get('agent_model_config')),
                                       Loader=yaml.FullLoader)
            llm_model = getattr(llm_models, agent.get('agent_model'))(llm_config)
        else:
            llm_model=None
        if agent.get('agent_model_config_opp'):
            llm_config_opp = Box.from_yaml(filename=os.path.join('configs/llm_configs/', agent.get('agent_model_config_opp')),
                                       Loader=yaml.FullLoader)
            llm_model_opp = getattr(llm_models, agent.get('agent_model_opp'))(llm_config_opp)
        else:
            llm_model_opp=None

        base_config=Box({'prompt': prompt, 'llm_model': llm_model})
        if prompt_opp is not None and llm_model_opp is not None:
            base_config = Box({'prompt': prompt, 'llm_model': llm_model,'prompt_opp': prompt_opp, 'llm_model_opp': llm_model_opp})
        agent_instance = getattr(agents, agent.get('agent_name'))(base_config,args=args,idx=idx)
        agents_ret.append(agent_instance)
        idx+=1
    return agents_ret

def load_agent_new(config_path, key, *args):
    agents_ret=[]
    agent_config = Box.from_yaml(
        filename=config_path, Loader=yaml.FullLoader)
    if agent_config.get(key) is not None:
        for agent in agent_config.get(key):
            prompt=getattr(prompts, agent.get('agent_prompt'))(agent.get('agent_race'))
            llm_config = Box.from_yaml(filename=os.path.join('configs/llm_configs/',agent.get('agent_model_config')), Loader=yaml.FullLoader)
            llm_model=getattr(llm_models, agent.get('agent_model'))(llm_config)
            agent_instance=getattr(agents, agent.get('agent_name'))(Box({'prompt':prompt,'llm_model':llm_model}))
            agents_ret.append(agent_instance)
    return agents_ret

def load_model(model_config_path):
    model_config = Box.from_yaml(
        filename=model_config_path, Loader=yaml.FullLoader)
    return getattr(llm_models, model_config.model_type)(model_config)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_logger(logger_path, debug=False, rm_existed=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    if rm_existed and os.path.exists(logger_path):
        os.remove(logger_path)

    fh = logging.FileHandler(logger_path)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    return logger


def parallel_func(worker, arg_list, num_workers=20):
    results = []
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, arg in enumerate(arg_list):
            futures.append(executor.submit(worker, arg))

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def load_jsonl(path):
    result = []
    with open(path, 'r') as f:
        for l in f.readlines():
            r = json.loads(l)
            result.append(r)
    return result


def save_jsonl(results, path):
    with open(path, 'w') as f:
        for r in results:
            f.writelines(json.dumps(r) + '\n')


class LLMBenchLogger:
    _instance = None

    def __new__(cls, logger_path, debug=False, rm_existed=False):
        if cls._instance is None:
            cls._instance = super(LLMBenchLogger, cls).__new__(cls)
            cls._instance.logger = cls._configure_logger(
                logger_path, debug, rm_existed)
        return cls._instance.logger

    @staticmethod
    def _configure_logger(logger_path, debug, rm_existed):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

        if rm_existed and os.path.exists(logger_path):
            os.remove(logger_path)

        fh = logging.FileHandler(logger_path)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        return logger

def compose_print(x):
    return f"\033[{x}m"

def print_step(*args):
    PRINT_RESUME = compose_print(0)
    PRINT_STEP = compose_print(1) + compose_print(46)
    print(PRINT_STEP, *args, PRINT_RESUME)
    args_str = "".join(args)
    print_str = f"{PRINT_STEP} {args_str} {PRINT_RESUME}"
    return print_str

def log_info(logger: Logger, message: str) -> None:
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.info(message)


def log_warning(logger: Logger, message: str) -> None:
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.warning(message)


def log_error(logger: Logger, message: str) -> None:
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.error(message)
