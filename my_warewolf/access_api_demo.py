import openai
import time
import concurrent.futures
import sys
import os
from loguru import logger

# --- 配置 ---
BASE_URL = "https://api.toiotech.com/v1" # 确保末尾没有 /
API_KEYS = [
    "api_key"
]

MODELS_TO_CHECK = [
    "deepseek-r1-250120",
    "deepseek-v3-250324",
    "doubao-1-5-pro-256k-250115",
    "doubao-1-5-thinking-pro-250415",
    "gemini-2.5-flash-preview-04-17",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "o4-mini",
    "claude-3-7-sonnet-20250219",
]

MODELS_REQUIRING_THINKING_CHECK = [
    "gemini-2.5-flash-preview-04-17",
    "claude-3-7-sonnet-20250219",
]

MAX_WORKERS = 10
TIMEOUT = 30

# --- 函数定义 ---
def get_error_message(e):
    """尝试从 OpenAI 异常中提取详细错误信息"""
    if hasattr(e, 'body') and isinstance(e.body, dict):
        error_info = e.body.get('error', {})
        if isinstance(error_info, dict) and 'message' in error_info:
            return error_info['message']
        return str(e.body)
    elif hasattr(e, 'message'):
         return e.message
    return str(e)


def check_key_model_openai(api_key, model_name, base_url, use_thinking_params=False):
    """使用 openai 库和 loguru 检查 API key 对模型的访问情况"""
    mode_suffix = " (思考模式)" if use_thinking_params else ""
    key_short = f"{api_key[:6]}...{api_key[-4:]}"
    model_log_name = f"{model_name}{mode_suffix}"
    log = logger.bind(key_short=key_short, model_log=model_log_name) # 绑定上下文信息
    start_time = time.time()

    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=TIMEOUT,
            max_retries=0
        )

        create_kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
        }

        if use_thinking_params:
            if model_name == "gemini-2.5-flash-preview-04-17":
                # create_kwargs["reasoning_effort"] = "low"
                create_kwargs["extra_body"] = {"reasoning_effort": "low"}
                # log.info("添加参数 reasoning_effort='low'")
            elif model_name == "claude-3-7-sonnet-20250219":
                create_kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": 20}}
                # log.info("添加参数 extra_body={{'thinking': ...}}")
            else:
                 log.warning("标记为思考模式但无特定参数配置")

        response = client.chat.completions.create(**create_kwargs)
        duration = time.time() - start_time

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            log.success(f"{model_log_name} 成功 (耗时: {duration:.2f}s)")
            return (api_key, model_name, use_thinking_params, True, f"成功 (耗时: {duration:.2f}s)")
        else:
            log.error(f"失败: 响应内容无效或为空 (耗时: {duration:.2f}s)")
            return (api_key, model_name, use_thinking_params, False, f"失败: 响应内容无效或为空 (耗时: {duration:.2f}s)")

    except openai.AuthenticationError as e:
        duration = time.time() - start_time
        error_msg = get_error_message(e)
        log.error(f"失败: 认证失败 - {error_msg} (耗时: {duration:.2f}s)")
        return (api_key, model_name, use_thinking_params, False, f"失败: 认证失败 - {error_msg} (耗时: {duration:.2f}s)")
    except openai.RateLimitError as e:
        duration = time.time() - start_time
        error_msg = get_error_message(e)
        log.error(f"失败: 速率限制 - {error_msg} (耗时: {duration:.2f}s)")
        return (api_key, model_name, use_thinking_params, False, f"失败: 速率限制 - {error_msg} (耗时: {duration:.2f}s)")
    except openai.NotFoundError as e:
         duration = time.time() - start_time
         error_msg = get_error_message(e)
         log.error(f"失败: 模型未找到/路径错误 - {error_msg} (耗时: {duration:.2f}s)")
         return (api_key, model_name, use_thinking_params, False, f"失败: 模型未找到/路径错误 - {error_msg} (耗时: {duration:.2f}s)")
    except openai.APIStatusError as e:
        duration = time.time() - start_time
        error_msg = get_error_message(e)
        log.error(f"失败: API错误 (状态码 {e.status_code}) - {error_msg} (耗时: {duration:.2f}s)")
        return (api_key, model_name, use_thinking_params, False, f"失败: API错误 {e.status_code} - {error_msg} (耗时: {duration:.2f}s)")
    except openai.APITimeoutError:
        duration = time.time() - start_time
        log.error(f"失败: 请求超时 ({TIMEOUT}s)")
        return (api_key, model_name, use_thinking_params, False, f"失败: 请求超时 ({TIMEOUT}s)")
    except openai.APIConnectionError as e:
        duration = time.time() - start_time
        log.error(f"失败: 连接错误 - {e} (耗时: {duration:.2f}s)")
        return (api_key, model_name, use_thinking_params, False, f"失败: 连接错误 - {e} (耗时: {duration:.2f}s)")
    except Exception as e:
        duration = time.time() - start_time
        log.exception(f"失败: 未知错误 - {type(e).__name__} (耗时: {duration:.2f}s)")
        return (api_key, model_name, use_thinking_params, False, f"失败: 未知错误: {type(e).__name__} - {e} (耗时: {duration:.2f}s)")


# --- 主程序 ---
if __name__ == "__main__":
    # 为初始日志绑定默认值
    initial_log = logger.bind(key_short="SYSTEM".ljust(15), model_log="SETUP".ljust(45))
    initial_log.info(f"开始使用 openai 库和 loguru 检查 {len(API_KEYS)} 个 API Key 对模型的访问情况...")
    initial_log.info(f"基础 URL: {BASE_URL}")
    initial_log.info(f"并发数: {MAX_WORKERS}, 超时时间: {TIMEOUT}s")
    initial_log.info("-" * 30)

    tasks = []
    for key in API_KEYS:
        for model in MODELS_TO_CHECK:
            tasks.append((key, model, BASE_URL, False))
            if model in MODELS_REQUIRING_THINKING_CHECK:
                tasks.append((key, model, BASE_URL, True))

    initial_log.info(f"总计 {len(tasks)} 个检查任务.") # 继续使用 initial_log

    results = []
    start_total_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(lambda p: check_key_model_openai(*p), task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task_info = future_to_task[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                key_short = f"{task_info[0][:6]}...{task_info[0][-4:]}"
                mode_suffix = " (思考模式)" if task_info[3] else ""
                model_log_name = f"{task_info[1]}{mode_suffix}"
                # 记录执行器级别的严重错误，也绑定上下文
                logger.bind(key_short=key_short, model_log=model_log_name).critical(f"任务执行时发生意外: {exc}", exc_info=True)
                results.append((task_info[0], task_info[1], task_info[3], False, f"失败: 执行意外 {exc}"))


    end_total_time = time.time()
    total_duration = end_total_time - start_total_time

    print("-" * 30) # 使用 print 分隔日志和摘要
    print(f"所有检查完成，总耗时: {total_duration:.2f}s")
    print("-" * 30)
    print("结果摘要:")
    print("-" * 30)

    results_by_key = {}
    for key, model, use_thinking_params, success, message in results:
        if key not in results_by_key:
            results_by_key[key] = []
        results_by_key[key].append((model, use_thinking_params, success, message))

    for key in API_KEYS:
        if key in results_by_key:
            key_short = f"{key[:6]}...{key[-4:]}"
            print(f"\n--- Key: {key_short} ---")
            key_results = sorted(results_by_key[key], key=lambda x: (x[0], x[1]))
            success_count = 0
            fail_count = 0
            for model, use_thinking_params, success, message in key_results:
                 mode_suffix = " (思考模式)" if use_thinking_params else ""
                 status = "✅ 成功" if success else "❌ 失败"
                 summary_message = message.split('(耗时:')[0].strip()
                 if not success:
                     summary_message = message.split('-')[0].strip() if '-' in message else message.split('(耗时:')[0].strip()

                 print(f"  模型: {f'{model}{mode_suffix}':<45} | 状态: {status:<10} | 信息: {summary_message}")
                 if success:
                     success_count += 1
                 else:
                     fail_count += 1
            print(f"--- Key {key_short} 总结: {success_count} 成功, {fail_count} 失败 ---")
        else:
             key_short = f"{key[:6]}...{key[-4:]}"
             print(f"\n--- Key: {key_short} ---")
             print("  未找到此 Key 的检查结果 (可能在执行中出错)。")

    total_checks = len(results)
    total_success = sum(1 for r in results if r[3])
    if total_checks > 0:
        success_rate = (total_success / total_checks) * 100
        print("\n" + "=" * 30)
        print(f"总体统计:")
        print(f"  总检查次数: {total_checks}")
        print(f"  成功次数:   {total_success}")
        print(f"  失败次数:   {total_checks - total_success}")
        print(f"  成功率:     {success_rate:.2f}%")
        print("=" * 30)
    else:
        print("\n没有执行任何检查。")