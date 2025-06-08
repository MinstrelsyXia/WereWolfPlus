
from typing import Union, TypedDict

class Action(TypedDict):
    from_: str
    role: str
    step: str
    content: str
    other_content: dict


class StateInfo(TypedDict):
    from_: str
    role: str
    step: str
    content: str
    system_content:str
    user_content:str

class Reward(TypedDict):
    from_: str
    role: str
    score: float


Trajectory = list[Union[StateInfo, Action, Reward]]

def set_action_info(from_,role,step,content,other_content={}):
    ret: Action = {"from_": from_, "role": role, "step": step, "content": content,
                      "other_content": other_content}
    return ret

def set_state_info(from_,role,step,content,system_content,user_content):
    ret: StateInfo = {"from_": from_,"role": role,"step": step, "content": content,"system_content": system_content, "user_content": user_content}
    return ret

def set_reward(env,role,score):
    ret: Reward = {"from_": env,"role":role,"score": score}
    return ret


if __name__ == '__main__':
    action:Action={}
    action["role"]="seer"
    action["element_name"]="deepseek"
    action["step"]="1"
    action["content"]="content"
    action["other_content"]= {}
    traj:Trajectory=[]
    traj.append(action)
    state_info: StateInfo = {}
    state_info["env"] = "env"
    state_info["step"] = "step"
    state_info["content"] = "content"
    state_info["system_content"] = "system_content"
    state_info["user_content"] = "user_content"
    traj.append(state_info)

    print(traj)