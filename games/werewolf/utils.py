# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""utility functions."""

from typing import Any
import yaml
from abc import ABC
from abc import abstractmethod
import marko
import re


def parse_json(text: str) :
    # 解析输入字符串，转换为json格式
    text = re.sub(r'\n', '', text)  # 去除换行符
    text = re.sub(r'```', '', text)
    text = re.sub(r'json', '', text)
    text = text.replace("’", "'")
    text = text.replace("—", "-")  # 替换特殊破折号为普通破折号
    text = text.replace("“", '"').replace("”", '"')  # 替换特殊双引号为普通双引号
    text = text.replace("\t", " ")
    result_json = parse_json_markdown(text)
    if not result_json:
        result_json = parse_json_str(text)
    return result_json


def parse_json_markdown(text: str) :
    # 将字符串解析为md抽象语法树SAT 
    ast = marko.parse(text)
    for c in ast.children:
            json_str = c.children[0].children
            return parse_json_str(json_str) # 找到标记为json的，使用进行解析为json/None（if没有）
    return None


def parse_json_str(text: str) :
    try:
        # use yaml.safe_load which handles missing quotes around field names.
        result_json = yaml.safe_load(text)
    except yaml.parser.ParserError:
        return None

    return result_json


class Deserializable(ABC):
    @classmethod
    @abstractmethod
    def from_json(cls, data: dict[Any, Any]):
        pass


if __name__ == "__main__":
    # Example usage
    text ='\n\n```json\n{\n"reasoning": "As the Doctor in Round 0 with no prior information, there is no strategic basis for targeting a specific player. Self-protection (Jackson) is a common first-night strategy to ensure the Doctor\'s survival, allowing future protection of key roles like the Seer. While random, prioritizing self-preservation maximizes the Doctor\'s utility in subsequent rounds.",\n"protect": "Jackson"\n}\n```'
    text = re.sub(r'\n', '', text)
    # text = re.sub(r'```', '', text)
    # text = re.sub(r'json', '', text)
    print(text)
    result = parse_json_markdown(text)
    print(result)  # Output: {'key': 'value'}