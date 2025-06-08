import re


def extract_actions_from_text(text,llm_model_name,action_dict):
    action_pattern = r"(?<=Decisions)[\s\S]*"

    # Use regular expressions to find all matching decision parts from a text
    decisions_matches = re.search(action_pattern, text)

    # If there are no matching results, an empty list is returned directly
    if not decisions_matches:
        return []

    decisions_text = decisions_matches.group()
    ret_actions=[]
    for k,v in action_dict.items():
        for idx,act in v.items():
            # print(f"======{idx}===={act}")
            if act in decisions_text:
                ret_actions.append(act)
    actions=ret_actions

    return actions


def extract_actions_from_command(command, action_dict,action_extractor, empty_idx, action_db_manager,llm_model):
    llm_model_name=llm_model.model_name
    extracted_decisions = extract_actions_from_text(command,llm_model_name,action_dict)

    # If no decision was extracted, an empty action token is returned.
    if not extracted_decisions:
        return [empty_idx], ["EMPTY ACTION"]

    action_ids, valid_actions = [], []
    for decision in extracted_decisions:
        ids, actions = action_extractor.extract_and_search_actions(decision, action_db_manager)
        action_ids.extend(ids)
        valid_actions.extend(actions)

    return action_ids, valid_actions


class ActionExtractor:
    def __init__(self, action_dict):
        self.full_action_dict = {}
        for category in action_dict:
            for key, value in action_dict[category].items():
                self.full_action_dict[value.upper()] = key

    def extract_and_search_actions(self, decision, action_db_manager):
        action = decision.upper()  # 转换为大写
        if action in self.full_action_dict:
            return [self.full_action_dict[action]], [action]
        else:
            # print(f"Searching for actions similar to: {action}")
            search_results = action_db_manager.search_actions(action)
            # print("Search results:", search_results)

            if search_results and 'ids' in search_results and 'documents' in search_results:
                actions = search_results['documents']
                if actions:  
                    action_ids = search_results['ids']
                    print("vdb_return_action:", actions[0])
                    return [int(action_ids[0])], [actions[0]]  # Convert the IDs of the closest actions to integers and return them as a list

            return [], []  # If no matches are found, an empty list is returned.
