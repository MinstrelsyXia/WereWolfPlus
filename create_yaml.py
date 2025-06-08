import copy
import os.path

import yaml
LLM_model=["gpt-35-turbo-0125.yaml","GPT-4o.yaml","deepseek-v25.yaml","qwen-25.yaml","gemini-15-flash.yaml","o1-mini.yaml","llama31-70b.yaml","llama31-8b.yaml","gpt-4o-mini.yaml"]
starcraft2_agent_vs_computer_params={
        "num_matches":5,
        "game_map": "Ancient Cistern LE",
        "game_difficulty":"medium",
        # "asynch_mode":[True,False],
        "agent_model_config/0":LLM_model,
        # "players":["Protoss_VS_Protoss"],
}


stratego_agent_vs_agent_params={
        "num_matches":10,
        "agent_model_config/0":LLM_model[:-1],
        "agent_model_config_opp/0":"gpt-4o-mini.yaml"

}

streetfight3_agent_vs_agent_params={
        "num_matches":10,
        "agent_model_config/0":LLM_model[:-1],
        "agent_model_config_opp/0":"gpt-4o-mini.yaml"

}

civ_agent_vs_agent_params={
        "num_matches":10,
        "game_turn":400,
        "agent_model_config/0":LLM_model,

}

werewolf_agent_vs_agent_params1={
        "num_matches":10,
        "good_model_config":LLM_model[:-1],
        "bad_model_config":"gpt-4o-mini.yaml"

}
werewolf_agent_vs_agent_params2={
        "num_matches":10,
        "good_model_config":"gpt-4o-mini.yaml",
        "bad_model_config":LLM_model[:-1]

}

welfare_diplomacy_agent_vs_agent_params={
        "num_matches":10,
        "game_max_message_rounds":3,
        "game_max_years": 10,
        "game_exploiter_model":LLM_model[:-1],
        "game_super_exploiter_model":"gpt-4o-mini.yaml"

}

def read_yaml(path):
    with open(path, 'r') as f:
        load_yaml = yaml.load(f,Loader=yaml.FullLoader )
        return load_yaml



def merge(dict_1,dict_2):
    return {**dict_1,**dict_2}

def comb_lst(mutli_value):
    '''
    combination list
    example  input: [[{k1:v1},{k2:v2}],[{ak1:av1},{ak2:av2}]]
            comb_lst(input)
            output:[{k1:v1,ak1:av1},{k1:v1,ak2:av2},{k2:v2,ak2:av2},{k2:v2,ak2:av2}]
    '''
    if len(mutli_value)==1:
        print(f"final combination result:{mutli_value[0]}")
        return mutli_value[0]
    elif len(mutli_value)==0:
        print(f"final combination result:[]")
        return []
    elif len(mutli_value)>=2:
        new_comb=[]
        for i in range( len(mutli_value[0])):
            for j in range(len(mutli_value[1])):
                new_comb.append(merge(mutli_value[0][i],mutli_value[1][j]))
        print(f"===============combination: {mutli_value[0]}    ||    {mutli_value[1]}")
        print(f"================{new_comb}")
        if len(mutli_value)==2:
            return comb_lst([new_comb])
        else:
            return comb_lst([new_comb]+mutli_value[2:])

def merge_args(mutli_dic_all,args):
    merge_args_lst=[]
    for cus_cfg in mutli_dic_all:
        base_args = copy.deepcopy(args)
        for key, value in cus_cfg.items():
            flag=False
            for k,v in base_args.items():
                if isinstance(v,dict):
                    for sk,sv in v.items():
                        if key==sk:
                            base_args[k][sk] = value
                            print(f"base_args[{k}][{sk}]={value}")
                            flag = True
                        if flag:
                            break
                elif isinstance(v, list):
                    if '/' in key:
                        ele_name,ele_num=key.split('/')
                        ele_num=int(ele_num)
                        if ele_num<len(v):
                            for sk, sv in v[ele_num].items():
                                if ele_name == sk:
                                    if not isinstance(value, list):
                                        base_args[k][ele_num][sk] = value
                                        print(f"base_args[{k}][{ele_num}][{sk}]={value}")
                                    flag = True
                                if flag:
                                    break
                if flag:
                    break
        conver_value_params(base_args)
        merge_args_lst.append(base_args)
        root_path="./configs/test/"
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        save_name=base_args['eval']['weave_prj_name']
        save_path=os.path.join(root_path,save_name+'.yaml')
        if os.path.exists(save_path):
            os.remove(save_path)
        save_yaml(base_args,save_path)
    return merge_args_lst
def conver_value_params(base_args):
    for key in ['weave_prj_name','output_path']:
        value=base_args['eval'][key]
        while value.count("$"):
            start = value.index("$")
            end = value.index("}") + 1
            print(value[start:end])
            name = value[start + 2:end - 1]
            print(f"save_name before:======{value}")
            value = value.replace(value[start:end], get_args_value(name, base_args).replace(".yaml",""))
            print(f"save_name after:======{value}")
        base_args['eval'][key]=value
    if base_args['eval']['output_path'].__contains__("/werewolf/") or base_args['eval']['output_path'].__contains__("/welfare_diplomacy/"):
        for agent in base_args['agent']:
            key='agent_model_config'
            value = agent[key]
            while value.count("$"):
                start = value.index("$")
                end = min(value.index("}") + 1,len(value))
                print(value[start:end])
                name = value[start + 2:end - 1]
                print(f"save_name before:======{value}")
                value = value.replace(value[start:end], get_args_value(name, base_args))
                print(f"save_name after:======{value}")
            agent[key] = value
def get_args_value(key,args):
    if "/" in key:
        ele_name, ele_num = key.split('/')
        ele_num = int(ele_num)
        for k, v in args.items():
            if isinstance(v, list):
                    for sk, sv in v[ele_num].items():
                        if ele_name == sk:
                            return args[k][ele_num][sk].replace(".yaml","")
    else:
        for k ,v in args.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if key == sk:
                        return str(args[k][sk])
def save_yaml(multi_conf,save_path):
    # Writing data to a YAML file
    with open(save_path, 'w') as file:
        yaml.dump(multi_conf, file, default_flow_style=False,sort_keys=False)

def gen_mutli_yamls(base_yaml,custom_cfg):
    base_args=read_yaml(base_yaml)

    mutli_dic_lst=[]
    base_custom_cfg={}
    # Split multi-value kv and single-value kv
    for k,v in custom_cfg.items():
        if isinstance(v,list):
            if len(v)==0:
                raise Exception(f" wrong value {k}:{v}")
            value_lst=[]
            for v_inner in v:
                value_lst.append({k:v_inner})
            mutli_dic_lst.append(value_lst)
        else:
            base_custom_cfg[k]=v
    print("====")
    # Combine multi-value kv into multiple single-value kv
    mutli_dic_lst_comb=comb_lst(mutli_dic_lst)
    # Merge completed kv
    mutli_dic_all=[]
    for one_config in mutli_dic_lst_comb:
        new_one_config={**base_custom_cfg,**one_config}
        mutli_dic_all.append(new_one_config)
    print(mutli_dic_all)
    # Generate multiple yaml files
    merge_args_lst= merge_args(mutli_dic_all, base_args)
    print("Generate successfully!!!")


def gen_starcraft2_yamls(root_path):
    base_yamls = ["eval_starcraft2_agent_vs_computer_scene1.yaml",
                  "eval_starcraft2_agent_vs_computer_scene2.yaml",
                  "eval_starcraft2_agent_vs_computer_scene3.yaml",
                  "eval_starcraft2_agent_vs_computer_scene4.yaml",
                  "eval_starcraft2_agent_vs_computer_scene5.yaml",
                  "eval_starcraft2_agent_vs_computer_scene6.yaml"
                  ]
    for base_yaml in base_yamls:
        gen_mutli_yamls(os.path.join(root_path, base_yaml), starcraft2_agent_vs_computer_params)

def gen_stratego_yamls(root_path):
    base_yamls = ["eval_stratego_agent_vs_agent_scene1.yaml",
                  "eval_stratego_agent_vs_agent_scene2.yaml"
                  ]
    for base_yaml in base_yamls:
        gen_mutli_yamls(os.path.join(root_path, base_yaml), stratego_agent_vs_agent_params)

def gen_streetfight3_yamls(root_path):
    base_yamls = ["eval_streetfight3_agent_vs_agent_scene1.yaml",
                  "eval_streetfight3_agent_vs_agent_scene2.yaml"
                  ]
    for base_yaml in base_yamls:
        gen_mutli_yamls(os.path.join(root_path, base_yaml), streetfight3_agent_vs_agent_params)

def gen_civ_yamls(root_path):
    base_yamls = ["eval_civ_agent_vs_agent_scene1.yaml",
                  "eval_civ_agent_vs_agent_scene2.yaml",
                  "eval_civ_agent_vs_agent_scene3.yaml"
                  ]
    for base_yaml in base_yamls:
        gen_mutli_yamls(os.path.join(root_path, base_yaml), civ_agent_vs_agent_params)

def gen_werewolf_yamls(root_path):
    base_yamls = ["eval_werewolf_agent_vs_agent_scene1.yaml"]
    for base_yaml in base_yamls:
        gen_mutli_yamls(os.path.join(root_path, base_yaml), werewolf_agent_vs_agent_params1)
    base_yamls = ["eval_werewolf_agent_vs_agent_scene2.yaml"]
    for base_yaml in base_yamls:
        gen_mutli_yamls(os.path.join(root_path, base_yaml), werewolf_agent_vs_agent_params2)

def gen_welfare_diplomacy_yamls(root_path):
    base_yamls = ["eval_welfare_diplomacy_agent_vs_agent_scene1.yaml"]
    for base_yaml in base_yamls:
        gen_mutli_yamls(os.path.join(root_path, base_yaml), welfare_diplomacy_agent_vs_agent_params)


if __name__ == '__main__':
    root_path="./configs/eval_config_base"
    gen_starcraft2_yamls(root_path)
    # gen_stratego_yamls(root_path)
    # gen_streetfight3_yamls(root_path)
    # gen_civ_yamls(root_path)
    # gen_werewolf_yamls(root_path)
    # gen_welfare_diplomacy_yamls(root_path)
    #