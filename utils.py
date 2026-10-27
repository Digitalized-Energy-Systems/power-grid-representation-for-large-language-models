'''
utils for converting and manage the grids
'''

from pandapower import runpp, pandapowerNet
import networkx as nx
import simbench as sb
import pandapower as pp
import json
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from datetime import datetime
from pathlib import Path


def grid_to_json(grid: pandapowerNet, add_voltage: bool = False, add_load: bool = False, add_infeed: bool = False, digits: int = 3) -> str:
    '''
    Creates a json string suited for describing the grid to a llm including the requested measurements
    '''
    json_dict = {"List of buses":{},"List of lines":{}}

    for index, _ in grid.bus.iterrows():
        generator = index in grid.sgen.bus.values # set to true if gen at bus exists else to false
        load = index in grid.load.bus.values # set to true if load at bus exists else to false

        json_dict["List of buses"]["bus" + str(index)] = {"generator": generator, "load": load} # add True values if applicable

        # add extra values as set by parameters
        if add_voltage:
            json_dict["List of buses"]["bus" + str(index)]["voltage at bus"] = round(float(
                grid.res_bus.loc[index, "vm_pu"]), digits)

        if add_infeed and generator:
            json_dict["List of buses"]["bus" + str(index)]["infeed at bus"] = round(float(
                grid.sgen.loc[grid.sgen["bus"] == index, "p_mw"].iat[0]), digits)

        if add_load and load:
            json_dict["List of buses"]["bus" + str(index)]["load at bus"] = round(float(
                grid.load.loc[grid.load["bus"] == index, "p_mw"].iat[0]), digits)


    for index, _ in grid.trafo.iterrows():
        from_bus = grid.trafo.hv_bus[index]
        to_bus = grid.trafo.lv_bus[index]
        json_dict["List of lines"]["trafo" + str(index)] = {"connecting from to": ["bus" + str(from_bus), "bus" + str(to_bus)]}

    for index, _ in grid.line.iterrows():
        from_bus = grid.line.from_bus[index]
        to_bus = grid.line.to_bus[index]
        json_dict["List of lines"]["line" + str(index)] = {"connecting from to": ["bus"+str(from_bus), "bus"+str(to_bus)]}

    return json.dumps(json_dict,indent=3)


def grid_to_text(grid: pandapowerNet, add_voltage: bool = False, add_load: bool = False, add_infeed: bool = False, digits: int = 3) -> str:
    '''
    Creates a string suited for describing the grid to a llm including the requested measurements
    '''
    prompt = ("There exists an electrical grid of the following structure of busses connected with lines. "
              "Busses can have")
    if add_voltage: prompt += " a voltage value,"
    prompt += " a generator"
    if add_infeed: prompt += " with an infeed value"
    prompt += " and/or a load"
    if add_load: prompt += " with an load value"
    prompt += ".\n"

    prompt += "The busses are:"

    for index, _ in grid.bus.iterrows():
        generator = index in grid.sgen.bus.values  # set to true if gen at bus exists else to false
        load = index in grid.load.bus.values  # set to true if load at bus exists else to false

        prompt += f" bus{index} with"
        if add_voltage:
            prompt += f" a normalized voltage of {round(float(grid.res_bus.loc[index, "vm_pu"]), digits)}V"

        if generator and load:
            prompt += ", a generator"
            if add_load: prompt += f" (infeed {round(float(grid.sgen.loc[grid.sgen["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
            prompt += " and a load"
            if add_load: prompt += f" (load {round(float(grid.load.loc[grid.load["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
        elif generator:
            if add_voltage: prompt += " and"
            prompt += " a generator"
            if add_load: prompt += f" (infeed {round(float(grid.sgen.loc[grid.sgen["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
        elif load:
            if add_voltage: prompt += " and"
            prompt += " a load"
            if add_load: prompt += f" (load {round(float(grid.load.loc[grid.load["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
        else: prompt += " nothing"
        prompt += ","
    prompt = prompt[:-1]
    prompt += ".\n"


    prompt += "(bus<id>, bus<id>) means the busses are connected bidirectionally by a line or trafo"

    prompt += ". \nThe connections with a trafo are:"
    for index, _ in grid.trafo.iterrows():
        from_bus = grid.trafo.hv_bus[index]
        to_bus = grid.trafo.lv_bus[index]
        prompt += f" (Bus{from_bus}, Bus{to_bus})"

        prompt += ","
    prompt = prompt[:-1]
    prompt += ".\n"

    prompt += "The connections with a line are:"
    for index, _ in grid.line.iterrows():
        from_bus = grid.line.from_bus[index]
        to_bus = grid.line.to_bus[index]
        prompt += f" (Bus{from_bus}, Bus{to_bus})"

        prompt += ","
    prompt = prompt[:-1]
    prompt += ".\n"

    return prompt


def area_to_json(grid: pandapowerNet, start_bus: int, range: int = 3, add_voltage: bool = True, add_load: bool = True, add_infeed: bool = True, digits: int = 3) -> str:
    '''
    Creates a json of the close by buses and lines
    '''
    #find close by buses
    nxgraph = pp.topology.create_nxgraph(grid)

    # use single-source the shortest path lengths up to range
    lengths = nx.single_source_shortest_path_length(nxgraph, start_bus, cutoff=range)
    area_nodes = sorted(int(n) for n, d in lengths.items() if d <= range)

    # create no dict for lines if range 0
    if range == 0:
        json_dict = {"List of buses": {}}
    else:
        json_dict = {"List of buses": {}, "List of lines": {}}

    for index, _ in grid.bus.iterrows():
        if index in area_nodes:
            generator = index in grid.sgen.bus.values  # set to true if gen at bus exists else to false
            load = index in grid.load.bus.values  # set to true if load at bus exists else to false

            json_dict["List of buses"]["bus" + str(index)] = {"generator": generator,
                                                              "load": load}  # add True values if applicable

            # add extra values as set by parameters
            if add_voltage:
                json_dict["List of buses"]["bus" + str(index)]["voltage at bus"] = round(float(
                    grid.res_bus.loc[index, "vm_pu"]), digits)

            if add_infeed and generator:
                json_dict["List of buses"]["bus" + str(index)]["infeed at bus"] = round(float(
                    grid.sgen.loc[grid.sgen["bus"] == index, "p_mw"].iat[0]), digits)

            if add_load and load:
                json_dict["List of buses"]["bus" + str(index)]["load at bus"] = round(float(
                    grid.load.loc[grid.load["bus"] == index, "p_mw"].iat[0]), digits)

    if range > 0: # only look at lines if range given
        for index, _ in grid.trafo.iterrows():
            from_bus = grid.trafo.hv_bus[index]
            to_bus = grid.trafo.lv_bus[index]
            if from_bus in area_nodes and to_bus in area_nodes:
                json_dict["List of lines"]["trafo" + str(index)] = {
                    "connecting from to": ["bus" + str(from_bus), "bus" + str(to_bus)]}

        for index, _ in grid.line.iterrows():
            from_bus = grid.line.from_bus[index]
            to_bus = grid.line.to_bus[index]
            if from_bus in area_nodes and to_bus in area_nodes:
                json_dict["List of lines"]["line" + str(index)] = {
                    "connecting from to": ["bus" + str(from_bus), "bus" + str(to_bus)]}

    return json.dumps(json_dict,indent=3)

def area_to_text(grid: pandapowerNet, start_bus: int, range: int = 3, add_voltage: bool = True, add_load: bool = True, add_infeed: bool = True, digits: int = 3) -> str:
    '''
        Creates a text about the close by busses and lines
    '''
    # find close by buses
    nxgraph = pp.topology.create_nxgraph(grid)

    # use single-source the shortest path lengths up to range
    lengths = nx.single_source_shortest_path_length(nxgraph, start_bus, cutoff=range)
    area_nodes = sorted(int(n) for n, d in lengths.items() if d <= range)

    # create prompt
    if range > 0: # only if range is given
        prompt = ("In close proximity the following structure of busses connected by lines is present. "
                  "Busses can have")
        if add_voltage: prompt += " a voltage value,"
        prompt += " a generator"
        if add_infeed: prompt += " with an infeed value"
        prompt += " and/or a load"
        if add_load: prompt += " with an load value"
        prompt += ".\n"

        prompt += "The busses are:"
    else:
        prompt = f"The bus is:"

    for index, _ in grid.bus.iterrows():
        if index in area_nodes:
            generator = index in grid.sgen.bus.values  # set to true if gen at bus exists else to false
            load = index in grid.load.bus.values  # set to true if load at bus exists else to false

            prompt += f" bus{index} with"
            if add_voltage:
                prompt += f" a normalized voltage of {round(float(grid.res_bus.loc[index, "vm_pu"]), digits)}V"

            if generator and load:
                prompt += ", a generator"
                if add_load: prompt += f" (infeed {round(float(grid.sgen.loc[grid.sgen["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
                prompt += " and a load"
                if add_load: prompt += f" (load {round(float(grid.load.loc[grid.load["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
            elif generator:
                if add_voltage: prompt += " and"
                prompt += " a generator"
                if add_load: prompt += f" (infeed {round(float(grid.sgen.loc[grid.sgen["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
            elif load:
                if add_voltage: prompt += " and"
                prompt += " a load"
                if add_load: prompt += f" (load {round(float(grid.load.loc[grid.load["bus"] == index, "p_mw"].iat[0]), digits)}MW)"
            else:
                prompt += " nothing"
            prompt += ","
    prompt = prompt[:-1]
    prompt += ".\n"

    if range > 0:  # only look at lines if range given
        prompt += "(bus<id>, bus<id>) means the busses are connected bidirectionally by a line or trafo"

        prompt += ". \nThe connections with a trafo are:"
        for index, _ in grid.trafo.iterrows():
            from_bus = grid.trafo.hv_bus[index]
            to_bus = grid.trafo.lv_bus[index]
            if from_bus in area_nodes and to_bus in area_nodes:
                prompt += f" (Bus{from_bus}, Bus{to_bus})"
                prompt += ","
        prompt = prompt[:-1]
        prompt += ".\n"

        prompt += "The connections with a line are:"
        for index, _ in grid.line.iterrows():
            from_bus = grid.line.from_bus[index]
            to_bus = grid.line.to_bus[index]
            if from_bus in area_nodes and to_bus in area_nodes:
                prompt += f" (Bus{from_bus}, Bus{to_bus})"
                prompt += ","
        prompt = prompt[:-1]
        prompt += ".\n"

    return prompt

def measurements_to_json(grid: pandapowerNet, digits: int=3, debug: bool = False):
    try:
        runpp(grid, max_iteration=200)  # refresh
    except:
        print(f"\033[31m Grid did not converge after 200 iterations, trying again.\033[0m")
        pass

    voltage_measurements = grid.res_bus[["vm_pu"]]
    generator_infeeds = grid.sgen[["bus", "p_mw"]]
    load_demand = grid.load[["bus", "p_mw"]]
    if debug:
        print("res_bus: ", voltage_measurements)
        print("sgen: ", generator_infeeds)
        print("load: ", load_demand)
    # adjust dicts for llm usage
    voltage_measurements = voltage_measurements.to_dict()
    voltage_measurements = voltage_measurements["vm_pu"]
    voltage_measurements = {f"bus{k}": round(v, digits) for k, v in voltage_measurements.items()}
    # generator infeeds as json
    gi_dict_save = generator_infeeds.to_dict()
    generator_infeeds = {}
    for idx, new_key in gi_dict_save['bus'].items():
        generator_infeeds[f'bus{new_key}'] = round(gi_dict_save['p_mw'][idx], digits)

    # load demand as json
    ld_dict_save = load_demand.to_dict()
    load_demand= {}
    for idx, new_key in ld_dict_save['bus'].items():
        load_demand[f'bus{new_key}'] = round(ld_dict_save['p_mw'][idx], digits)

    grid_measurements = {
        "voltage_measurements": voltage_measurements,
        "generator_infeeds": generator_infeeds,
        "load_demand": load_demand}
    return json.dumps(grid_measurements, indent=3)

def measurements_to_text(grid: pandapowerNet, digits: int=3, debug: bool = False):
    try:
        runpp(grid, max_iteration=200)  # refresh
    except:
        print(f"\033[31m Grid did not converge after 200 iterations, trying again.\033[0m")
        pass

    voltage_measurements = grid.res_bus[["vm_pu"]]
    generator_infeeds = grid.sgen[["bus", "p_mw"]]
    load_demand = grid.load[["bus", "p_mw"]]
    if debug:
        print("res_bus: ", voltage_measurements)
        print("sgen: ", generator_infeeds)
        print("load: ", load_demand)
    # adjust dicts for llm usage
    vm_dict = voltage_measurements.to_dict()
    vm_dict = vm_dict["vm_pu"]
    vm_dict = {f"bus{k}": round(v, digits) for k, v in vm_dict.items()}

    #building text for measurements
    voltage_measurements = "The voltage measurements are as follows:\n"
    for k, v in vm_dict.items():
        voltage_measurements += f"Voltage at {k} is {round(v, digits)}V.\n"
    #generator infeeds as text
    gi_dict_save = generator_infeeds.to_dict()
    gi_dict = {}
    for idx, new_key in gi_dict_save['bus'].items():
        gi_dict[f'bus{new_key}'] = round(gi_dict_save['p_mw'][idx], digits)
    generator_infeeds = "The generator infeeds are as follows:\n"
    for k, v in gi_dict.items():
        generator_infeeds += f"Generator infeed at {k} is {round(v, digits)}MW.\n"
    # load demand as text
    ld_dict_save = load_demand.to_dict()
    ld_dict = {}
    for idx, new_key in ld_dict_save['bus'].items():
        ld_dict[f'bus{new_key}'] = round(ld_dict_save['p_mw'][idx], digits)
    load_demand = "The load demands are as follows:\n"
    for k, v in ld_dict.items():
        load_demand += f"Load demand at bus {k} is {round(v, digits)}MW.\n"
    grid_measurements = {
        "voltage_measurements": voltage_measurements,
        "generator_infeeds": generator_infeeds,
        "load_demand": load_demand}
    return grid_measurements


def check_voltage_level(grid: pandapowerNet, lower_bound: float = 0.9, upper_bound: float = 1.1) -> bool:
    """Checks whether all voltage are within the defined range."""
    vm_pu = grid.res_bus[["vm_pu"]]
    return ((vm_pu >= lower_bound) & (vm_pu <= upper_bound)).all().all()

def get_all_simbench_codes() -> str:
    # Print all available simbench codes
    return sb.collect_all_simbench_codes()


def extract_thinking_and_toolcalls(llm_all_messages):
    """
    Extract only the *content* of ThinkingPart and ToolCallPart in temporal order.

    Output format:
    [
        {"type": "Thinking<id>", "content": "<string>"},
        {"type": "ToolCall<id>", "content": "<string>"},
        ...
    ]
    """

    result = {}
    idx = 0
    for entry in llm_all_messages:
        for part in entry.parts:
            cls_name = type(part).__name__

            # ThinkingPart → use its content directly (string)
            if cls_name == "ThinkingPart":
                result[f"Thinking{idx}"] = str(part.content)
                idx += 1

            # ToolCallPart → convert tool call into a single string
            elif cls_name == "ToolCallPart":
                # Tool call content rendered as "<tool_name>(<args>)"
                call_str = f"{part.tool_name}({part.args})"
                result[f"ToolCall{idx}"] =  str(call_str)
                idx += 1

    return result


def average_wrong_adjustments(runs: dict) -> float:
    """
    Calculate the average number of wrong adjustments per run.

    Rules:
    - Bias 0: every generator or load adjustment is considered wrong.
    - Bias -1: wrong if load is adjusted up, generator is adjusted down,
      or any adjustment is marked as 'same'.
    - Bias 1: wrong if load is adjusted down, generator is adjusted up,
      or any adjustment is marked as 'same'.

    Only runs with voltage_in_bounds = True are included in the calculation.
    """
    wrong_counts = []

    for run in runs.values():
        # Skip runs where voltage is out of bounds
        if not run["voltage_in_bounds"]:
            continue

        bias = run["start_bias"]
        tc = run["tool_counter"]

        if bias == -1 or -2:
            wrong = (
                tc["adjust_load_up"] +
                tc["adjust_generator_down"] +
                tc["adjust_generator_same"] +
                tc["adjust_load_same"]
            )

        elif bias == 1 or 2:
            wrong = (
                tc["adjust_load_down"] +
                tc["adjust_generator_up"] +
                tc["adjust_generator_same"] +
                tc["adjust_load_same"]
            )

        else: # bias == 0
            # every adjustment is wrong
            wrong = tc["adjust_generator"] + tc["adjust_load"]

        wrong_counts.append(wrong)

    return sum(wrong_counts) / len(wrong_counts)


def average_tool_count(data):
    """Calculate the average tool count per run where voltage is in bounds."""
    counts = [
        run["tool_counter"]["get_measurements"] +
        run["tool_counter"]["adjust_load"] +
        run["tool_counter"]["adjust_generator"]
        for run in data.values() if run["voltage_in_bounds"]
    ]
    return sum(counts) / len(counts) if counts else 0

def average_power_at_slack(data):
    """Calculate the average power at slack where voltage is in bounds."""
    power_values = [abs(run["power_at_slack"]) for run in data.values() if run["voltage_in_bounds"]]
    return sum(power_values) / len(power_values) if power_values else 0


def save_dict_as_json(data, output_dir="run_results", prefix="run_dict"):
    """
    Saves a dictionary as a JSON file with timestamped filename
    and a timestamp inside the JSON content.

    Args:
        data (dict): The dictionary to save
        output_dir (str): Directory where files are stored
        prefix (str): Filename prefix

    Returns:
        Path: Path to the written JSON file
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    readable_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # File path
    file_path = output_path / f"{prefix}_{timestamp}.json"

    # JSON structure
    output = {
        "generated_at": readable_time,
        "data": data
    }

    # Write file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    return file_path