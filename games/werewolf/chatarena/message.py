from typing import List, Union, Tuple
from dataclasses import dataclass
import time
from uuid import uuid1
import hashlib

from sentence_transformers import SentenceTransformer
import torch
import re
import pickle
import os
import math
import sys
import threading
import msvcrt  # Windows文件锁定


SYSTEM_NAME="System"

def _hash(input: str):
    hex_dig = hashlib.sha256(input.encode()).hexdigest()
    return hex_dig


@dataclass
class Message:
    agent_name: str
    content: Union[str, List[Union[str, int]]]
    # content: str
    turn: int
    timestamp: int = time.time_ns()
    visible_to: Union[str, List[str]] = 'all'
    msg_type: str = "text"
    importance: int = 1
    logged: bool = False
    embedding: torch.FloatTensor = torch.zeros((2,), dtype=torch.float32)
    stage: str = "vote"
    role: str = ""
    think: str = ""
    reward: int = 0
    
    def __hash__(self):
        return int(self.msg_hash, 16)
    
    def __eq__(self, other):
        if isinstance(other, Message):
            return self.msg_hash == other.msg_hash
        return False

    @property
    def msg_hash(self):
        # Generate a unique message id given the content, timestamp and role
        return _hash(
            f"agent: {self.agent_name}\ncontent: {self.content}\ntimestamp: {str(self.timestamp)}\nturn: {self.turn}\nmsg_type: {self.msg_type}\nstage: {self.stage}\nrole: {self.role}")


class MessagePool():
    """
    A message pool to manage the messages. This allows a unified treatment of the visibility of the messages.
    Draft design:
    The message pool is a list of (named) tuples, where each tuple has (turn, role, content).

    There should be two potential configurations for step definition: multiple players can act in the same turn (rock-paper-scissors).
    The agents can only see the messages that
    1) before the current turn, and
    2) visible to the current role
    """

    def __init__(self, args):
        self.args = args
        self.conversation_id = str(uuid1())
        self._last_message_idx = 0
        self._messages: List[Message] = []
        self._lock = threading.Lock()  # 添加线程锁
        self._file_lock = threading.Lock()  # 添加文件锁
        
        # 初始化数据库文件
        self.db_path = os.path.join(self.args.exps_path_to, "exps.pkl")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 加载初始数据
        self._load_from_db()
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.model_qa = SentenceTransformer('checkpoints\model_qa')
        self.model_sym = SentenceTransformer('checkpoints\model_sym')
        # self.model_qa = SentenceTransformer('D:\脚痛大学/basics\大三下\博弈论与多智能体\大作业\DSGBench\checkpoint\local_models\model_qa')
        # self.model_sys = SentenceTransformer('D:\脚痛大学/basics\大三下\博弈论与多智能体\大作业\DSGBench\checkpoint\local_models\model_sys')

    def _load_from_db(self):
        """从数据库加载数据"""
        try:
            if os.path.exists(self.db_path):
                with self._file_lock:  # 使用文件锁保护读取
                    with open(self.db_path, 'rb') as f:
                        self._messages = pickle.load(f)
        except Exception as e:
            print(f"Error loading from database: {e}")
            self._messages = []

    def _save_to_db(self):
        """保存数据到数据库"""
        try:
            with self._file_lock:  # 使用文件锁保护写入
                # 创建临时文件
                temp_path = self.db_path + '.tmp'
                with open(temp_path, 'wb') as f:
                    pickle.dump(self._messages, f)
                
                # 如果原文件存在，先删除
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                
                # 重命名临时文件
                os.rename(temp_path, self.db_path)
        except Exception as e:
            print(f"Error saving to database: {e}")
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def save_exps_to(self, is_incremental=False):
        """保存经验到数据库"""
        if is_incremental:
            exps = [exp for exp in self._messages if exp.msg_type == "exp"]
            self._save_to_db()
            
            # 同时保存文本版本用于调试
            txt_path = os.path.join(self.args.exps_path_to, "exps.txt")
            with self._file_lock:  # 使用文件锁保护文本文件写入
                with open(txt_path, "w", encoding='utf-8') as f:
                    for exp in exps:
                        f.write("Reflexion: " + exp.content[0] + '\n')
                        f.write("Talking content: " + exp.content[1] + '\n')
                        f.write("Reward: " + str(exp.content[2]) + '\n')
                        f.write("IsChoose: " + str(exp.content[3]) + '\n\n')

    def reset(self):
        # self._messages = []
        pass
        
    def give_importance(self, message: Message):
        content = message.content if message.msg_type == "text" or message.msg_type == "ref" else message.content[0]
        if message.agent_name != "Moderator" and message.importance == 1:
            identity_pattern = r"(?:A|a)s(?:\s(?:a|an)\s(?:villager|werewolf|guard|seer|witch))|(?:I|i)\s?a(?:'?m| was)?(?:\s(?:a|an|the)\s(?:villager|werewolf|guard|seer|witch))"
            role_pattern = r'\b(?:P|p)layer(?:\s?[0-9]{1,2})?\s(?:is|are)\s(?:villager|villagers|werewolf|werewolves|guard|seer|witch)\b'
            if re.search(identity_pattern, content) or re.search(role_pattern, content):
                message.importance = 5
                print(f"    give importance 5: {content}", file=sys.stderr)

    def append_message(self, message: Message):
        """添加消息并实时保存到数据库"""
        with self._lock:  # 使用线程锁保护写入操作
            if self.args and self.args.human_in_combat and message.msg_type == "text" and ('Player 1' in message.visible_to or message.visible_to == 'all'):
                print(f"{message.agent_name} -> {message.visible_to}: {message.content}")
            
            if message.importance == 0:
                return
                
            content = message.content if message.msg_type == "text" or message.msg_type == "ref" else message.content[0]
            message.embedding = torch.from_numpy(self.model_qa.encode(content)) if message.msg_type == "text" or message.msg_type == "ref" \
                else torch.from_numpy(self.model_sym.encode(content))
            self.give_importance(message)
            self._messages.append(message)
            
            # 如果是经验消息，立即保存到数据库
            if message.msg_type == "exp":
                # 在锁内直接调用_save_to_db，避免嵌套锁
                self._save_to_db()
                
                # 保存文本版本
                txt_path = os.path.join(self.args.exps_path_to, "exps.txt")
                with self._file_lock:
                    with open(txt_path, "w", encoding='utf-8') as f:
                        exps = [exp for exp in self._messages if exp.msg_type == "exp"]
                        for exp in exps:
                            f.write("Reflexion: " + exp.content[0] + '\n')
                            f.write("Talking content: " + exp.content[1] + '\n')
                            f.write("Reward: " + str(exp.content[2]) + '\n')
                            f.write("IsChoose: " + str(exp.content[3]) + '\n\n')
            
            if self.args and message.agent_name == "Moderator":
                with open(os.path.join(self.args.logs_path_to, str(self.args.current_game_number) + ".md"), "a") as f:
                    output = f"**{message.agent_name} (-> {str(message.visible_to)})**: {message.content}"
                    f.write(output + "  \n")
        
    def append_message_at_index(self, message: Message, index: int):
        message.embedding = torch.from_numpy(self.model_qa.encode(message.content))
        self.give_importance(message)
        self._messages.insert(index, message)

    def print(self):
        for message in self._messages:
            print(f"[{message.agent_name}->{message.visible_to}]: {message.content}")

    @property
    def last_turn(self):
        if len(self._messages) == 0:
            return 0
        else:
            for msg in reversed(self._messages):
                if msg.msg_type == "text":
                    return msg.turn

    @property
    def last_message(self):
        if len(self._messages) == 0:
            return None
        else:
            return self._messages[-1]

    def get_all_messages(self) -> List[Message]:
        return self._messages

    def get_visible_messages(self, agent_name, turn: int) -> List[Message]:
        """
        get the messages that are visible to the agents before the specified turn
        """

        # Get the messages before the current turn
        prev_messages = [message for message in self._messages if message.turn <= turn and message.importance > 0 
                         and (message.msg_type == "text" or message.msg_type == "ref")]

        visible_messages = []
        for message in prev_messages:
            if message.visible_to == "all" or agent_name in message.visible_to or agent_name == "Moderator":
                visible_messages.append(message)
            
        return visible_messages
    
    def get_last_k_messages(self, agent_name, turn: int, k: int) -> List[Message]:
        visible_messages = self.get_visible_messages(agent_name, turn)
        important_k = math.ceil(k * 0.66)
        important_messages = [msg for msg in visible_messages if msg.importance >= 3]
        # this implemantation considers the importance of messages
        if len(visible_messages) <= k:
            return visible_messages
        filtered_message_set = set(visible_messages[-k:]) | set(sorted(important_messages, key=lambda x: x.importance, reverse=True)[:important_k])
        return [message for message in visible_messages if message in filtered_message_set]
    
    def find_k_most_similar(self, agent_name, query_sentence, k):                   # for qa
        def _cosine_similarity(a, b):
            dot_product = torch.dot(a, b)
            norm_a = torch.norm(a)
            norm_b = torch.norm(b)
            cosine_s = dot_product / (norm_a * norm_b)
            return cosine_s if cosine_s > 0.5 else -1
        
        query_embedding = torch.from_numpy(self.model_qa.encode(query_sentence))
        # print(query_embedding.shape)
        visible_messages = self.get_visible_messages(agent_name, self.last_turn)
        similarities = torch.tensor([_cosine_similarity(query_embedding, msg.embedding) for msg in visible_messages])
        topk_values, topk_indices = torch.topk(similarities, k)
        res = [visible_messages[i].content for sim, i in zip(topk_values.tolist(), topk_indices.tolist()) if sim > 0.5]
        # print(topk_values)
        print(res, file=sys.stderr)
        return res
    
    def get_best_experience(self, query_reflexion, role, branch=0, threshold=0.85, topk=50):
        def _cosine_similarity(a, b):
            dot_product = torch.dot(a, b)
            norm_a = torch.norm(a)
            norm_b = torch.norm(b)
            cosine_s = dot_product / (norm_a * norm_b)
            return cosine_s
        
        def are_close(values, threshold=0.1):
            if self.args and self.args.similar_exps_threshold:
                threshold = self.args.similar_exps_threshold
            return all(abs(values[i] - values[j]) < threshold for i in range(len(values)) for j in range(i+1, len(values)))
        
        if branch == 1 or branch == 0:
            prev_experiences = [exp for exp in self._messages if exp.msg_type == "exp" and exp.turn < self.args.current_game_number and exp.content[3] == branch]
        else:
            role_u = "As the " + role
            role_l = "as the " + role
            
            prev_experiences = [exp for exp in self._messages if exp.msg_type == "exp" and exp.turn < self.args.current_game_number and exp.content[3] == branch and (role_u in exp.content[0].split(',')[0] or role_l in exp.content[0].split(',')[0])]
        
        query_embedding = torch.from_numpy(self.model_sym.encode(query_reflexion))
        similar_exps = []
        
        for msg in prev_experiences:
            sim = _cosine_similarity(query_embedding, msg.embedding)
            # print(sim, file=sys.stderr)
            if sim >= threshold:
                similar_exps.append((msg, sim))
        
        if not similar_exps:
            return None
        similar_exps.sort(key=lambda x: x[1], reverse=True)
        # top k good experiences: value in 993-995
        similar_exps = similar_exps[:topk]
        similar_exps.sort(key=lambda x: x[0].content[2], reverse=True)
        # one bad experience: similar but value is low
        bad_exp = similar_exps.pop()
        similar_exps = [(exp, sim) for exp, sim in similar_exps if 993 <= exp.content[2] <= 995]
        similar_exps = similar_exps[:5]
        num_results = len(similar_exps)
        if branch > 0:
            res = [exp.content[1].split('.')[0] for exp, sim in similar_exps]
        else:
            res = [exp.content[1] for exp, sim in similar_exps]
        return res, bad_exp[0].content[1]
        
    def give_rewards(self, winner_names):
        current_game_number = 0 if not self.args.current_game_number else self.args.current_game_number
        for exp in reversed(self._messages):
            if exp.msg_type == "exp":
                if exp.turn == current_game_number:
                    if exp.agent_name in winner_names:
                        exp.content[2] = 1000 - self.last_turn
                    else:
                        exp.content[2] = self.last_turn
                else:
                    break

    def get_vote_experience(self, agent_name: str, role: str, query: str, k: int = 5):
        """获取投票经验，每次调用都重新加载数据库"""
        with self._lock:  # 使用线程锁保护读取操作
            # 重新加载数据库
            self._load_from_db()
            
            def _cosine_similarity(a, b):
                dot_product = torch.dot(a, b)
                norm_a = torch.norm(a)
                norm_b = torch.norm(b)
                cosine_s = dot_product / (norm_a * norm_b)
                return cosine_s
            
            # 只检索该玩家的经验
            player_exps = [exp for exp in self._messages 
                          if exp.msg_type == "exp" 
                          and exp.agent_name == agent_name
                          and exp.stage == "vote"
                          and exp.role == role]
            
            # 使用向量相似度进行检索
            query_embedding = torch.from_numpy(self.model_sym.encode(query))
            similar_exps = []
            
            for msg in player_exps:
                # 直接使用已计算好的embedding
                sim = _cosine_similarity(query_embedding, msg.embedding)
                if sim >= 0.3:
                    similar_exps.append((msg, sim))
            
            if not similar_exps:
                return None
                
            similar_exps.sort(key=lambda x: x[1], reverse=True)
            similar_exps = similar_exps[:k]
            similar_exps.sort(key=lambda x: x[0].content[2], reverse=True)
            
            top_exps = similar_exps[:5]
            bad_exp = similar_exps[-1]
            good_exps = [(exp, sim) for exp, sim in top_exps if 990 <= exp.content[2] <= 999]
            
            # 检查 bad experience 的 reward 是否小于 500
            if bad_exp[0].content[2] < 500:
                return good_exps, bad_exp
            else:
                return good_exps, []


@dataclass
class Question:
    content: str
    turn: int
    visible_to: str = 'all'
    reward: int = 0
    
    def __hash__(self):
        return int(self.msg_hash, 16)
    
    def __eq__(self, other):
        if isinstance(other, Message):
            return self.msg_hash == other.msg_hash
        return False

    @property
    def msg_hash(self):
        # Generate a unique message id given the content, timestamp and role
        return _hash(
            f"content: {self.content}\nturn: {self.turn}\nvisible_to: {self.visible_to}")


class QuestionPool():
    
    def __init__(self, args) -> None:
        
        def load_ques_from(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        
        self.args = args
        self.conversation_id = str(uuid1())
        self._last_message_idx = 0
        
        self._questions: List[Question] = self._initial_questions()
        if self.args and self.args.load_ques_from:
            self._questions += load_ques_from(self.args.load_ques_from)
    
    def save_ques_to(self, is_incremental=False):
        if is_incremental:
            ques = [que for que in self._questions if que.turn > 0]
            file_name = "ques_" + str(self.args.current_game_number) + "_incremental.pkl"
        else:
            ques = [que for que in self._questions if que.turn == self.args.current_game_number]
            file_name = "ques_" + str(self.args.current_game_number) + "_nonincremental.pkl"
        with open(os.path.join(self.args.ques_path_to, file_name), "wb") as f:
            pickle.dump(ques, f)
        file_name += ".txt"
        with open(os.path.join(self.args.ques_path_to, file_name), "w") as f:
            for que in ques:
                f.write("Question: " + que.content + '\n')
                f.write("Reward: " + str(que.reward) + '\n\n')
    
    @property
    def last_turn(self):
        if len(self._questions) == 0:
            return 0
        else:
            return self._questions[-1].turn

    def append_question(self, que: Question):
        self._questions.append(que)
    
    def get_all_questions(self):
        return self._questions
    
    def get_visible_questions(self, role):
        ques = [que for que in self._questions if que.visible_to == role or que.visible_to == 'all']
        return ques
    
    def get_best_questions(self, role, k, use_history=False):
        if use_history:
            ques = self.get_visible_questions(role)
        else:
            ques = [que for que in self.get_visible_questions(role) if que.turn == 0 or que.turn == self.last_turn]
        
        init_ques = [que for que in self.get_visible_questions(role) if que.turn == 0]
        if len(ques) <= k:
            return ques
        sorted_ques = set(sorted(ques, key=lambda x: x.reward, reverse=True)[:k]) | set(init_ques)
        return list(sorted_ques)
    
    def give_rewards(self, last_turn, camp="werewolf"):
        win_role = ["werewolf"] if camp == "werewolf" else ["villager", "seer", "witch", "guard"]
        if not self.args.current_game_number:
            return
        for que in reversed(self._questions):
            if que.turn == self.args.current_game_number:
                if que.visible_to in win_role:
                    que.reward = 1000 - last_turn
                else:
                    que.reward = last_turn
            else:
                break
    
    def get_necessary_questions(self):
        return [
            "What is my player name and what is my role? What is my final objective in this game?",
            # "Which living players could or must be my cooperators as far as I know?",
            # "Has anyone mentioned their identity during the chat? Are they my enemy or my ally?"
            "Based on the chat history, can you guess what some players' role might be?"
        ]
    
    def _initial_questions(self):
        return [
            Question(content="What is the current phase, daytime or night? what should I do at this phase according to the game rules?", turn=0, visible_to="all", reward=500),
            Question(content="Based on the current situation, what are the possible consequences if I reveal my role in the talking now?", turn=0, visible_to="all", reward=500),
            Question(content="Which player was voted for killing by my teammate just now?", turn=0, visible_to="werewolf", reward=500),
            Question(content="Is the prophet alive? Which player may be the prophet that is most threatening to us?", turn=0, visible_to="werewolf", reward=500),
            Question(content="Which player is another pretty girl in this game?", turn=0, visible_to="werewolf", reward=500),
            Question(content="Based on the conversation and my inference, who is most likely to be an alive pretty girl?", turn=0, visible_to="villager", reward=500),
            Question(content="Which player made the statement claiming to be a prophet? Can his words be trusted?", turn=0, visible_to="villager", reward=500),
            Question(content="Are there any clues or information I can refer to for special characters such as prophet, pharmacist and sentry?", turn=0, visible_to="villager", reward=500),
            Question(content="Which suspicious player should I identify?", turn=0, visible_to="seer", reward=500),
            Question(content="Which player is a pretty girl among the players I have identified? If so, how should I disclose this information?", turn=0, visible_to="seer", reward=500),
            Question(content="Should I disclose my role now?", turn=0, visible_to="seer", reward=500),
            Question(content="Based on the conversation and my inference, who is most likely to be an alive pretty girl? Should I poison him?", turn=0, visible_to="witch", reward=500),
            Question(content="Should I be using my antidote or poison at this point? If I use it now, I won't be able to use it later.", turn=0, visible_to="witch", reward=500),
            Question(content="Should I disclose my role now?", turn=0, visible_to="witch", reward=500),
            Question(content="Based on the conversation and my inference, who is most likely to be an alive pretty girl?", turn=0, visible_to="guard", reward=500),
            Question(content="Who is the possible pretty girl aggressive towards?", turn=0, visible_to="guard", reward=500),
            Question(content="Is the prophet still alive? If yes, who is the prophet?", turn=0, visible_to="guard", reward=500),
        ]
        
    def get_initial_questions(self, role):
        return [que for que in self._initial_questions if que.visible_to == role]
