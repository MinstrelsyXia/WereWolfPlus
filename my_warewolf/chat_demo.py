import openai
import time
from loguru import logger
import sys

class DeepSeek_V3:
    """调用 DeepSeek V3 模型的类，提供简单的问答接口"""

    def __init__(self, api_key=None, base_url="https://api.siliconflow.cn/v1", timeout=30, max_retries=3):
        """
        初始化 DeepSeek V3 客户端
        
        参数:
            api_key (str): API密钥，如不提供则使用默认值
            base_url (str): API基础URL
            timeout (int): 请求超时时间（秒）
            max_retries (int): 最大重试次数
        """
        # 默认API密钥，如果没有提供则使用此密钥
        default_api_key = "api_key"
        # default_api_key = "sk-ylsmjpgdhkjlxgnogjvzajaporlkktgnakshnzgklqblfsay"  # siliconflow的API密钥

        self.api_key = api_key if api_key else default_api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_name = "deepseek-v3-250324"
        
        # 创建OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0  # 我们手动处理重试以提供更好的错误信息
        )
        
        # 设置日志
        self.log = logger.bind(model="DeepSeek V3")
        
    def _get_error_message(self, e):
        """从异常中提取错误信息"""
        if hasattr(e, 'body') and isinstance(e.body, dict):
            error_info = e.body.get('error', {})
            if isinstance(error_info, dict) and 'message' in error_info:
                return error_info['message']
            return str(e.body)
        elif hasattr(e, 'message'):
            return e.message
        return str(e)
    
    def ask(self, query, system_prompt=None, temperature=0.7, max_tokens=128):
        """
        向DeepSeek V3发送问题并获取回答
        
        参数:
            query (str): 用户问题
            system_prompt (str, optional): 系统提示，用于设置模型行为
            temperature (float): 温度参数，控制输出随机性
            max_tokens (int): 返回的最大令牌数
            
        返回:
            dict: 包含回答内容和元数据的字典
        """
        messages = []
        
        # 添加系统提示(如果有)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # 添加用户问题
        messages.append({"role": "user", "content": query})
        
        # 发送请求并处理响应
        for retry in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                
                duration = time.time() - start_time
                
                # 检查响应是否有效
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    answer = response.choices[0].message.content
                    usage = response.usage if hasattr(response, 'usage') else None
                    
                    self.log.info(f"查询成功处理 (耗时: {duration:.2f}s)")
                    
                    # 返回结果字典
                    return {
                        "success": True,
                        "answer": answer,
                        "duration": duration,
                        "usage": usage,
                        "error": None
                    }
                else:
                    self.log.error(f"响应无效或为空 (耗时: {duration:.2f}s)")
                    
            except openai.AuthenticationError as e:
                duration = time.time() - start_time
                error_msg = self._get_error_message(e)
                self.log.error(f"认证失败 - {error_msg} (耗时: {duration:.2f}s)")
                
                # 认证错误不需要重试
                return {
                    "success": False,
                    "answer": None,
                    "duration": duration,
                    "usage": None,
                    "error": f"认证失败: {error_msg}"
                }
                
            except openai.RateLimitError as e:
                duration = time.time() - start_time
                error_msg = self._get_error_message(e)
                self.log.warning(f"速率限制 - {error_msg} (耗时: {duration:.2f}s), 重试 {retry+1}/{self.max_retries}")
                
                # 速率限制错误需要等待更长时间
                wait_time = (2 ** retry) + 1
                time.sleep(wait_time)
                continue
                
            except openai.APIStatusError as e:
                duration = time.time() - start_time
                error_msg = self._get_error_message(e)
                
                # 检查是否是账户欠费问题
                if "AccountOverdueError" in error_msg or "overdue balance" in error_msg:
                    self.log.error(f"账户欠费 - {error_msg} (耗时: {duration:.2f}s)")
                    return {
                        "success": False,
                        "answer": None,
                        "duration": duration,
                        "usage": None,
                        "error": f"账户欠费，请充值: {error_msg}"
                    }
                
                self.log.error(f"API错误 (状态码 {e.status_code}) - {error_msg} (耗时: {duration:.2f}s)")
                
                # 某些HTTP错误不需要重试
                if e.status_code in [400, 401, 403, 404]:
                    return {
                        "success": False,
                        "answer": None,
                        "duration": duration,
                        "usage": None,
                        "error": f"API错误 {e.status_code}: {error_msg}"
                    }
                
                # 其他错误可以重试
                wait_time = retry + 1
                time.sleep(wait_time)
                continue
                
            except (openai.APITimeoutError, openai.APIConnectionError) as e:
                duration = time.time() - start_time
                error_type = "超时" if isinstance(e, openai.APITimeoutError) else "连接错误"
                self.log.warning(f"{error_type} - {e} (耗时: {duration:.2f}s), 重试 {retry+1}/{self.max_retries}")
                
                # 网络相关错误可以立即重试
                continue
                
            except Exception as e:
                duration = time.time() - start_time
                self.log.exception(f"未知错误 - {type(e).__name__}: {e} (耗时: {duration:.2f}s)")
                
                return {
                    "success": False,
                    "answer": None,
                    "duration": duration,
                    "usage": None,
                    "error": f"未知错误: {type(e).__name__} - {e}"
                }
        
        # 所有重试都失败
        return {
            "success": False,
            "answer": None,
            "duration": 0,
            "usage": None,
            "error": f"达到最大重试次数 ({self.max_retries})，仍未获得有效响应"
        }

    def __call__(self, query, **kwargs):
        """
        让类实例可以直接像函数一样调用
        
        参数:
            query (str): 用户问题
            **kwargs: 传递给ask方法的其他参数
            
        返回:
            str: 模型回答或错误信息
        """
        result = self.ask(query, **kwargs)
        if result["success"]:
            return result["answer"]
        else:
            return f"错误: {result['error']}"


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logger.remove()  # 移除默认处理程序
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    deepseek = DeepSeek_V3()
    
    print("\n=== DeepSeek V3 问答测试 ===\n")
    
    # 简单问题测试
    question = "请简单介绍一下你自己，你是什么模型?"
    print(f"问题: {question}\n")
    
    answer = deepseek(question)
    print(f"回答:\n{answer}\n")
    
    # # 更复杂的示例
    # question2 = "如何使用Python实现快速排序算法?"
    # print(f"问题: {question2}\n")
    
    # # 使用系统提示
    # system_prompt = "你是一个专业的编程助手，请提供简洁、高效且有详细注释的代码示例。"
    # result = deepseek.ask(question2, system_prompt=system_prompt)
    
    # if result["success"]:
    #     print(f"回答:\n{result['answer']}\n")
    #     if result["usage"]:
    #         print(f"令牌用量: 提示词 {result['usage'].prompt_tokens}, 完成 {result['usage'].completion_tokens}, 总计 {result['usage'].total_tokens}")
    # else:
    #     print(f"错误: {result['error']}")