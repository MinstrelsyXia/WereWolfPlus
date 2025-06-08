import os.path

import wandb
import numpy as np
import csv
from box import Box
import yaml
from tasks_config import WANDB_ENTITY,TASKS_CFG_PATH,TASKS
from dsgbench_ability_calc.ability_scores import calc_ability_score
# 1. Record multiple game results for each project 记录多轮游戏结果
# 2. Extract the summary results of each project into 6 in-game csv files 提取结果
# 3. Execute the ability calculation script to get the final ability result csv file 计算能力，存到csv

starcraft2_strategic_planning={'Strategic planning capabilities/RPM':'RPM','Strategic planning capabilities/EER':'EER',
                               'Strategic planning capabilities/SUR':'SUR','Strategic planning capabilities/TRR':'TRR'}
starcraft2_real_decision={'real_decision_capability/APM_avg':'APM','real_decision_capability/EPM_avg':'EPM'}
starcraft2_learning= {'learning_capability/match_grounding_acc':'GA','player_win_rate':'WR'}
other= {'_runtime':'RUNTIME', '_step':'STEP'}

stratego_strategic_planning= {"live_pieces_rate":"PCR","live_pieces_score":"PCS","critical_live_pieces_rate":"KPR"}
stratego_learning= {"grounding_acc":"GA","player_win_rate":"WR","match_turns_avg":"TURNS"}

streetfight3_strategic_planning= {"AHR":"AHR","SMHR":"SMHR","HCR":"HCR"}
streetfight3_learning= {"grounding_acc":"GA","player_win_rate":"WR","match_turns_avg":"TURNS"}

civ_strategic_planning= {"Myagent/egr":"EGR","Myagent/cer":"CER","Myagent/trp":"TRP","Myagent/lur":"LUR","Myagent/mgr":"MGR","Myagent/wbr":"WBR"}
civ_learning={"grounding_acc":"GA"}

werewolf_social_reasoning= {"irp":"IRP"}
werewolf_teamwork={"ksr":"KRS","vss":"VSS"}
werewolf_learning={"player_win_rate":"WR"}

welfare_diplomacy_units=["score/units/AUS","score/units/ENG","score/units/FRA","score/units/GER","score/units/ITA","score/units/RUS","score/units/TUR",]
welfare_diplomacy_welfares=["score/welfare/AUS","score/welfare/ENG","score/welfare/FRA","score/welfare/GER","score/welfare/ITA","score/welfare/RUS","score/welfare/TUR",]
welfare_diplomacy_centers=["score/centers/AUS","score/centers/ENG","score/centers/FRA","score/centers/GER","score/centers/ITA","score/centers/RUS","score/centers/TUR",]


welfare_diplomacy_ability={ 'units_score':"UNITS", 'opp_units_score':"OUNITS", 'welfare_score':"WFS", 'opp_welfare_score':"OWFS",
                            'centers_score':"CNT", 'opp_centers_score':"OCNT", 'social/betrayal_rate':"BTR",
                            'score/model_allied_avg_turns/opp_player':"OALT",
                            'social/allied_succ_rate':"ASR", 'score/model_allied_avg_turns/player':"ALT"}



def get_wandb_project_runs(project_name):
    # initialization wandb API
    api = wandb.Api()
    # Get the runs in the project
    runs = api.runs(project_name, filters={'state': 'finished'})
    try:
        runs_num = len(runs)
    except:
        runs_num = 0
    print(f"{project_name}==================runs_num {runs_num}")
    return runs,runs_num

def get_lastest_value(history,cap,exclude=[]):
    non_max_idx = history[cap].size - 1
    for i in range(history[cap].size - 1, -1, -1):
        if not np.isnan(history[cap].iloc[i]):
            if history[cap].iloc[i] in exclude:
                continue
            non_max_idx = i
            break
    return non_max_idx,history[cap].iloc[non_max_idx]

def get_mean_value(history,cap,exclude=[]):
    non_max_idx = history[cap].size - 1
    all_data=[]
    ret_val=0
    for i in range(history[cap].size - 1, -1, -1):
        if not np.isnan(history[cap].iloc[i]):
            if history[cap].iloc[i] in exclude:
                continue
            all_data.append(history[cap].iloc[i])
    if len(all_data)>0:
        ret_val=np.sum(all_data)/len(all_data)
    return len(all_data),ret_val

def get_mean_value2(history,cap,exclude=[]):
    non_max_idx = history[cap].size - 1
    all_data=[]
    ret_val=0
    for i in range(history[cap].size - 1, -1, -1):
        if not np.isnan(history[cap].iloc[i]):
            if history[cap].iloc[i] in exclude:
                continue
            if len(all_data)==0 or (history[cap].iloc[i]!=all_data[-1]):
                all_data.append(history[cap].iloc[i])
    if len(all_data)>0:
        ret_val=np.sum(all_data)/len(all_data)
    return len(all_data),ret_val
def calc_model_match_scene_score_perproj(project_name, save_path):
    try:
        runs,runs_num = get_wandb_project_runs(project_name)

        if runs_num==0:
            return None

        s_mj = {}
        all_mj=[]
        for run in runs:
            try:
                calc_cap= {}
                if project_name.__contains__("starcraft2"):
                    calc_cap={**starcraft2_strategic_planning,**starcraft2_real_decision,**starcraft2_learning}
                elif project_name.__contains__("stratego"):
                    calc_cap={**stratego_strategic_planning,**stratego_learning}
                elif project_name.__contains__("streetfight3"):
                    calc_cap={**streetfight3_strategic_planning,**streetfight3_learning}
                elif project_name.__contains__("civ"):
                    calc_cap = {**civ_strategic_planning,**civ_learning}
                elif project_name.__contains__("werewolf"):
                    calc_cap = {**werewolf_social_reasoning,**werewolf_teamwork,**werewolf_learning}
                elif project_name.__contains__("welfare_diplomacy"):
                    calc_cap = {**welfare_diplomacy_ability}
                calc_cap={**calc_cap,**other}
                max_step=run.scan_history(keys=[list(calc_cap.keys())[0]]).max_step
                history = run.history(samples=max_step)
                tmp_mj= {}
                tmp_mj["match_id"]=run

                # Calculate single run score
                if project_name.__contains__("welfare_diplomacy"):
                    ## units
                    pos_units=[]
                    for cap in ["score/units/AUS", "score/units/ENG", "score/units/FRA","score/units/GER"]:
                        # get lastest one
                        non_max_idx, value = get_lastest_value(history, cap)
                        pos_units.append(value)
                    pos_units_mean=np.mean(pos_units)
                    neg_units=[]
                    for cap in ["score/units/ITA", "score/units/RUS", "score/units/TUR"]:
                        non_max_idx, value = get_lastest_value(history, cap)
                        neg_units.append(value)
                    neg_units_mean=np.mean(neg_units)
                    tmp_mj[calc_cap["units_score"]] = pos_units_mean
                    tmp_mj[calc_cap["opp_units_score"]] = neg_units_mean

                    ## welfare
                    pos_welfare = []
                    for cap in ["score/welfare/AUS", "score/welfare/ENG", "score/welfare/FRA", "score/welfare/GER"]:
                        non_max_idx, value = get_lastest_value(history, cap)
                        pos_welfare.append(value)
                    pos_welfare_mean = np.mean(pos_welfare)
                    neg_welfare = []
                    for cap in ["score/welfare/ITA", "score/welfare/RUS", "score/welfare/TUR"]:
                        non_max_idx, value = get_lastest_value(history, cap)
                        neg_welfare.append(value)
                    neg_welfare_mean = np.mean(neg_welfare)
                    tmp_mj[calc_cap["welfare_score"]] = pos_welfare_mean
                    tmp_mj[calc_cap["opp_welfare_score"]] = neg_welfare_mean

                    ## centers
                    pos_centers = []
                    for cap in ["score/centers/AUS","score/centers/ENG","score/centers/FRA","score/centers/GER"]:
                        non_max_idx, value = get_lastest_value(history, cap)
                        pos_centers.append(value)
                    pos_centers_mean = np.mean(pos_centers)
                    neg_centers = []
                    for cap in ["score/centers/ITA","score/centers/RUS","score/centers/TUR"]:
                        non_max_idx, value = get_lastest_value(history, cap)
                        neg_centers.append(value)
                    neg_centers_mean = np.mean(neg_centers)
                    tmp_mj[calc_cap["centers_score"]]=pos_centers_mean
                    tmp_mj[calc_cap["opp_centers_score"]]=neg_centers_mean

                    cap_name=''
                    for key in history.keys():
                        if key.__contains__("score/model_allied_avg_turns/") and key!="score/model_allied_avg_turns/gpt-4o-mini":
                            cap_name =key
                            break
                    for cap in ['social/betrayal_rate','score/model_allied_avg_turns/gpt-4o-mini',
                                'social/allied_succ_rate','_step','_runtime',cap_name]:
                        # get lastest one
                        non_max_idx, value = get_lastest_value(history, cap)
                        if cap in ['social/betrayal_rate']:
                            value=1-value # 1- betr_nums/alli_nums
                        # record into s_mj
                        if cap == cap_name:
                            tmp_mj[calc_cap["score/model_allied_avg_turns/player"]] = value
                        elif cap=="score/model_allied_avg_turns/gpt-4o-mini":
                            tmp_mj[calc_cap["score/model_allied_avg_turns/opp_player"]] = value
                        else:
                            tmp_mj[calc_cap[cap]] = value
                    # calc_cap=list(tmp_mj.keys())
                    # calc_cap=[cap for cap in calc_cap if cap !="match_id"]
                else:
                    for cap,cap_mapping in calc_cap.items():
                        value=0
                        # ====== starcraft2 ======
                        if cap=="Strategic planning capabilities/RPM":
                            # get lastest one
                            non_max_idx,last_value=get_lastest_value(history,cap)
                            value=last_value/non_max_idx
                        elif  cap in ['Strategic planning capabilities/EER']:
                            # get mean one
                            _,value = get_mean_value(history, cap)
                            value=value/100
                        elif  cap in ['Strategic planning capabilities/SUR','real_decision_capability/APM_avg','real_decision_capability/EPM_avg',
                                    ]:
                            # get mean one
                            _,value = get_mean_value(history, cap)
                        elif cap in ['Strategic planning capabilities/TRR','learning_capability/match_grounding_acc',
                                   "AHR", "SMHR","match_turns_avg",
                                   'grounding_acc','player_win_rate','_runtime', '_step']:
                            # get lastest one
                            non_max_idx,value=get_lastest_value(history,cap)
                        # ====== streetfight3 ======
                        elif cap in ["HCR"]:
                            # get mean one
                            _,value=get_mean_value2(history,cap,exclude=[0])
                            value=(1-value/160)  #1-(总血量-当前血量)/耗时/总血量

                        # ====== civ ======
                        elif cap in {**civ_strategic_planning,**civ_learning}.keys():
                            # get lastest one
                            non_max_idx,value=get_lastest_value(history,cap)
                        # ====== stratego ======
                        elif cap in {**stratego_strategic_planning,**stratego_learning}.keys():
                            # get lastest one
                            non_max_idx, value = get_lastest_value(history, cap)
                        # ====== werewolf ======
                        elif cap in {**werewolf_social_reasoning ,**werewolf_teamwork}.keys():
                            # get lastest one
                            non_max_idx, value = get_lastest_value(history, cap)

                        # 记录到s_mj中
                        tmp_mj[cap_mapping] = value

                all_mj.append(tmp_mj)
            except Exception as e:
                print("this run has something wrong ,skip it ",e)
                continue

        # calc mean result
        merge_ret={}


        if len(all_mj)>0:
            for key in all_mj[0].keys():
                if key =='match_id':
                    continue
                merge_ret[key]=[]
                for i in range(len(all_mj)):
                    merge_ret[key].append(all_mj[i][key])
        else:
            return None
        mean_ret={}
        mean_ret['match_id']='summ'
        for k,v in merge_ret.items():
            if k=='player_win_rate'or k=='WR':
                mean_ret[k]=merge_ret[k][-1]
            else:
                mean_ret[k]=round(np.mean(merge_ret[k]),4)

        all_mj.append(mean_ret)

        # save result
        proj_name = project_name.split('/')[-1]
        filename = os.path.join(save_path, proj_name + "_ret_step1.csv")
        header=['match_id']
        for k,v in calc_cap.items():
            header.append(v)
        with open(filename, 'w', newline='') as file:
            fieldnames = header
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_mj)
        return mean_ret
    except Exception as e:
        print(f"calc_cap_socre({project_name}) wrong:{e}")

def calc_model_match_scene_score(save_path):

    tasks_dic = {}
    for k, v in TASKS.items():
        # game_name="welfare_diplomacy"
        for game_name in ["starcraft2", "stratego", "streetfight3", "civ", "werewolf", "welfare_diplomacy"]:
            if game_name not in tasks_dic.keys():
                tasks_dic[game_name] = []
            if k.__contains__(game_name):
                tasks_dic[game_name].append(k)
                break

    for game_name, tasks in tasks_dic.items():
        if len(tasks) == 0:
            continue
        project_names = tasks
        entity = WANDB_ENTITY
        model_game_ability = []
        for project_name in project_names:
            config_file = TASKS[project_name]
            config_file_path = os.path.join(TASKS_CFG_PATH, config_file)
            config = Box.from_yaml(filename=config_file_path, Loader=yaml.FullLoader)
            output_path = config.eval.output_path
            output_folders = output_path.split("/")
            if len(output_path.split("/")) == 6:
                model_name = output_folders[2]
                scene_name = output_folders[4]
            else:
                raise (f"task config {config_file_path} eval/output is not valid ")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            full_project_name = entity + "/" + project_name
            sum_ret = calc_model_match_scene_score_perproj(full_project_name, output_path)
            if sum_ret is None:
                continue
            sum_ret_prefix = {'Model': model_name, 'Game': game_name, 'Scenario': scene_name}
            sum_ret['match_id'] = project_name
            model_game_ability.append({**sum_ret_prefix, **sum_ret})
        # save result
        # save_folder = os.path.join(output_path[:output_path.find(model_name)], "ability_score")
        save_folder = save_path
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        filename = os.path.join(save_folder, game_name + ".csv")
        header = []
        if len(model_game_ability) > 0:
            for k, v in model_game_ability[0].items():
                header.append(k)
            with open(filename, 'w', newline='') as file:
                fieldnames = header
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(model_game_ability)

def calc_model_ability_score(output):
    detail_save_path=os.path.join(output, "ability_detail_score")
    calc_model_match_scene_score(detail_save_path)
    calc_ability_score(detail_save_path,output)

if __name__ == '__main__':
    ability_save_path=os.path.join("./output", "ability_score")
    calc_model_ability_score(ability_save_path)




