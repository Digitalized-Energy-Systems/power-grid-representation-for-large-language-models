import os, asyncio, json, time

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from mango import create_tcp_container, activate
from ControllerAgent import ControllerAgent
from pandapower import runpp

from grids import small_net, simbench_net
from utils import save_dict_as_json, extract_thinking_and_toolcalls, average_tool_count, average_wrong_adjustments, average_power_at_slack
import prompts.gpt_oss_20_prompt as OSS20prompt
import prompts.gpt_oss_120_prompt as OSS120prompt


# single run settings
MODEL = 'gpt-oss:120b'
BASE_URL = os.environ['BASE_URL'] if ('BASE_URL' in os.environ) else ''
API_KEY = os.environ['API_KEY'] if ('API_KEY' in os.environ) else '123'


MODEL_CONFIG_GPT_OSS_120B = {"model": 'gpt-oss:120b', "base_url": BASE_URL, "api_key": API_KEY}
MODEL_CONFIG_GPT_OSS_20B = {"model": 'gpt-oss:20b', "base_url": BASE_URL, "api_key": API_KEY}

CONTAINER_ADDRESS = ('localhost', 5555)

CONTROLLER_CONFIG = {'data_as_text': False, # all information is given as text, if false as json
                     'data_separation': True, # data is separated from grid structure, if false combined
                     'area_info': False, # after adjustment an area around the adjusted node is returned, if false only the node itself
                     'area_radius': 3} # Amount of hops to neighbouring nodes


async def run_single_test(start_prompt, system_prompt):
    grid = small_net(plot=False)
    container = create_tcp_container(addr=CONTAINER_ADDRESS)  # one container to run all agents
    controller_agent = container.register(ControllerAgent(grid, CONTROLLER_CONFIG, {"model": MODEL, "base_url": BASE_URL, "api_key": API_KEY}, start_prompt, system_prompt), suggested_aid=f"ControllerAgent")

    async with activate(container) as c:
        future = controller_agent.agent_finished

        await asyncio.gather(future)


async def run_controller(grid, controller_config, model_config, control_prompt, start_prompt) -> dict:
    container = create_tcp_container(addr=CONTAINER_ADDRESS)  # one container to run all agents
    controller_agent = container.register(ControllerAgent(grid, controller_config, model_config, control_prompt, start_prompt), suggested_aid=f"ControllerAgent")

    async with activate(container) as c:
        future = controller_agent.agent_finished
        await asyncio.gather(future)

    return future.result()


"""
Run different LLMs by commenting in and out. 
"""
def evaluate_scenario():
    test_length = 30 # runs per model, must be multiple of 3
    result_dict = {}

    # GPT OSS 20B
    start_time = time.time()
    control_prompt_oss20 = OSS20prompt.control_prompt(data_separation=CONTROLLER_CONFIG["data_separation"])
    start_prompt_oss20 = OSS20prompt.start_prompt
    result_dict['gpt_oss_20'] = asyncio.run(run_prompt_evaluation('gpt_oss_20', MODEL_CONFIG_GPT_OSS_20B, test_length, control_prompt_oss20, start_prompt_oss20))
    gpt20_runtime = time.time() - start_time
    result_dict['gpt_oss_20']['runtime'] = f"{round(gpt20_runtime / 60, 2)}min"
    print(f"Runtime GPT OSS 20: {round(gpt20_runtime / 60, 2)}min")

    # GPT OSS 120B
    start_time = time.time()
    control_prompt_oss120 = OSS120prompt.control_prompt(data_separation=CONTROLLER_CONFIG["data_separation"])
    start_prompt_oss120 = OSS120prompt.start_prompt
    result_dict['gpt_oss_120'] = asyncio.run(run_prompt_evaluation('gpt_oss_120', MODEL_CONFIG_GPT_OSS_120B, test_length, control_prompt_oss120, start_prompt_oss120))
    gpt120_runtime = time.time() - start_time
    result_dict['gpt_oss_120']['runtime'] = f"{round(gpt120_runtime / 60, 2)}min"
    print(f"Runtime GPT OSS 120: {round(gpt120_runtime / 60, 2)}min")

    # Print all runs to a file located in the folder run_results
    save_dict_as_json(result_dict)


async def run_prompt_evaluation(model_name, model_config, test_length, control_prompt, start_prompt) -> dict:
    result_dict = {}
    bias = 1  ## change bias here
    print(f"\n---- Run {model_name} START ----\n")
    for i in range(test_length):
        print(f"\n--- Test Nr. {i} ---\n")
        result_dict[f'{model_name}_run{i}'] = {}
        if i < (1 / 3 * test_length):
        #if i < (1 / 2 * test_length):
            #grid = small_net(bias*-1)  # negative deviation
            grid = simbench_net("1-LV-rural1--0-no_sw", bias*-1)  # negative deviation
            #grid = simbench_net("1-LV-semiurb4--0-no_sw", bias*-1)  # negative deviation
            try:
                result = await run_controller(grid, CONTROLLER_CONFIG, model_config,control_prompt, start_prompt)
                if result['voltage_in_bounds']:
                    result_dict[f'{model_name}_run{i}']['voltage_in_bounds'] = True
                else:
                    result_dict[f'{model_name}_run{i}']['voltage_in_bounds'] = False
                    result_dict[f'{model_name}_run{i}']['messages'] = extract_thinking_and_toolcalls(result['llm_all_messages'])

                result_dict[f'{model_name}_run{i}']['start_bias'] = bias*-1
                result_dict[f'{model_name}_run{i}']['tool_counter'] = result['tool_counter']
                result_dict[f'{model_name}_run{i}']['power_at_slack'] = result["power_at_slack"]
            except Exception as e:
                result_dict[f'{model_name}_run{i}']['ERROR'] = str(e)
                result_dict[f'{model_name}_run{i}']['start_bias'] = bias * -1

        elif i < (2/3*test_length):
            #grid = small_net(bias=0)  # no deviation
            grid = simbench_net("1-LV-rural1--0-no_sw", bias=0)  # no deviation
            #grid = simbench_net("1-LV-semiurb4--0-no_sw", bias=0)  # no deviation
            try:
                result = await run_controller(grid, CONTROLLER_CONFIG, model_config, control_prompt, start_prompt)
                if result['voltage_in_bounds']:
                    result_dict[f'{model_name}_run{i}']['voltage_in_bounds'] = True
                else:
                    result_dict[f'{model_name}_run{i}']['voltage_in_bounds'] = False
                    result_dict[f'{model_name}_run{i}']['messages'] = extract_thinking_and_toolcalls(result['llm_all_messages'])

                result_dict[f'{model_name}_run{i}']['start_bias'] = 0
                result_dict[f'{model_name}_run{i}']['tool_counter'] = result['tool_counter']
                result_dict[f'{model_name}_run{i}']['power_at_slack'] = result["power_at_slack"]
            except Exception as e:
                result_dict[f'{model_name}_run{i}']['ERROR'] = str(e)
                result_dict[f'{model_name}_run{i}']['start_bias'] = 0

        else:
            #grid = small_net(bias=bias)  # positive deviation
            grid = simbench_net("1-LV-rural1--0-no_sw", bias=bias) # positive deviation
            #grid = simbench_net("1-LV-semiurb4--0-no_sw", bias=bias)  # positive deviation
            try:
                result = await run_controller(grid, CONTROLLER_CONFIG, model_config, control_prompt, start_prompt)
                if result['voltage_in_bounds']:

                    result_dict[f'{model_name}_run{i}']['voltage_in_bounds'] = True
                else:
                    result_dict[f'{model_name}_run{i}']['voltage_in_bounds'] = False
                    result_dict[f'{model_name}_run{i}']['messages'] = extract_thinking_and_toolcalls(result['llm_all_messages'])

                result_dict[f'{model_name}_run{i}']['start_bias'] = bias
                result_dict[f'{model_name}_run{i}']['tool_counter'] = result['tool_counter']
                result_dict[f'{model_name}_run{i}']['power_at_slack'] = result["power_at_slack"]
            except Exception as e:
                result_dict[f'{model_name}_run{i}']['ERROR'] = str(e)
                result_dict[f'{model_name}_run{i}']['start_bias'] = bias


    #print(json.dumps(result_dict, indent=3))
    print(f"\n---- Run {model_name} DONE ----\n")

    print("-------- Results --------\n")
    print(json.dumps(result_dict, indent=3))

    # count and print fails
    failed_positive = sum(1 for run in result_dict.values() if run.get("voltage_in_bounds") is False and run.get("start_bias") == 1)
    failed_negative = sum(1 for run in result_dict.values() if run.get("voltage_in_bounds") is False and run.get("start_bias") == -1)
    failed_neutral = sum(1 for run in result_dict.values() if run.get("voltage_in_bounds") is False and run.get("start_bias") == 0)
    failed_sum = failed_positive + failed_negative + failed_neutral
    print(f"Failed positive: {failed_positive} \nFailed negative: {failed_negative} \nFailed neutral: {failed_neutral}\n\nSuccess rate: {round(1 - (failed_sum/test_length), 3)}")

    # calc and print averages
    avg_iteration_count = average_tool_count(result_dict)
    print(f"Average iteration count: {round(avg_iteration_count, 3)}")
    avg_wrong_adjustments = average_wrong_adjustments(result_dict)
    print(f"Average wrong adjustments: {round(avg_wrong_adjustments, 3)}")
    avg_wrong_adjustments_ratio = avg_wrong_adjustments / avg_iteration_count
    print(f"Average wrong adjustment ratio: {round(avg_wrong_adjustments_ratio, 3)}")
    avg_power_at_slack = average_power_at_slack(result_dict)
    print(f"Average power at slack: {round(avg_power_at_slack, 3)}")

    # add failed statistics and averages to result dict
    result_dict['success_rate'] = round(1 - (failed_sum / test_length), 3)
    result_dict['avg_iteration_count'] = round(avg_iteration_count, 3)
    result_dict['avg_wrong_adjustments'] = round(avg_wrong_adjustments, 3)
    result_dict['avg_wrong_adjustments_ratio'] = round(avg_wrong_adjustments_ratio, 3)
    result_dict['avg_power_at_slack'] = round(avg_power_at_slack, 3)
    result_dict['bias'] = bias
    result_dict['grid_type'] = grid.name
    result_dict['controller_config'] = CONTROLLER_CONFIG

    return result_dict


if __name__ == "__main__":
    #asyncio.run(run_single_test()) # run single Controller
    evaluate_scenario()

    CONTROLLER_CONFIG = {'data_as_text': True,
                         'data_separation': True,
                         'area_info': False,
                         'area_radius': 1}
    # evaluate_scenario()

    # Print grids
    #small_net(plot=True)
    #simbench_net(sb_code="1-LV-rural1-1-0-no_sw", plot=True)
    #simbench_net(sb_code='1-LV-semiurb4--0-no_sw', plot=True)
