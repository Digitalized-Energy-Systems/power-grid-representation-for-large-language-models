start_prompt = '''
You are a Distribution System Operator controlling a simulated electricity grid.
This is a mathematical and electrotechnical control task.
All generators and loads are fully flexible.

You receive per-bus voltage measurements in per-unit.
Voltages must remain strictly between 0.900 and 1.100 p.u.

Objective:
Ensure that all bus voltages satisfy 0.900 < voltage < 1.100 p.u.
The episode continues until this condition is met for every bus.

Available tools (exactly one per step):
adjust_generator(bus_id:int, new_output:float)
adjust_load(bus_id:int, new_demand:float)
request_measurements()
grid_in_bounds()

Procedure:
If you do not have current measurements, call request_measurements.
Inspect all voltages.
Identify any voltage <=0.900 or >=1.100.
Rank violations by distance from the nearest bound.
Select the violating bus or a nearby controllable bus.
Adjust exactly one generator or load to move the voltage toward the allowed range.
If the value is more than 1% (voltage <=0.890 or voltage >=1.110) off, do big adjustments to converge faster.
Use the returned measurements for the next decision.
Repeat until **no** violations remain.
When **all** voltages are within bounds, call grid_in_bounds.

Rules:
Output exactly one tool call per step.
Do not output text, explanations, or reasoning.
Do not call grid_in_bounds unless **all** voltages are within bounds.
If you call `grid_in_bounds`, do not respond further. The tool's return value is the final answer.
'''


def control_prompt(data_separation):
    if data_separation:
        adjusted_net_structure = """
Network structure:
{net_structure}

Current state:
Voltage measurements (p.u.): {voltage_measurements}
Generators (bus_id to output): {generator_infeeds}
Loads (bus_id to demand): {load_demand}
"""

    else:
        adjusted_net_structure = """
Network structure including current state:
{net_structure}
"""
    prompt = """
You are controlling a simulated electricity grid.
{adjusted_net_structure}
Goal:
Keep all voltages strictly between 0.900 and 1.100 p.u.
Any voltage outside this range must be corrected.
Do not terminate until all voltages are within bounds.

Use exactly one tool call per step.
""".format(adjusted_net_structure=adjusted_net_structure)
    return prompt
