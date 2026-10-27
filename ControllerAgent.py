import asyncio, json
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pandapower import runpp, pandapowerNet
from mango import Agent

from utils import measurements_to_json, measurements_to_text, grid_to_json, grid_to_text, area_to_json, area_to_text, check_voltage_level



class ControllerAgent(Agent):
    def __init__(self, grid: pandapowerNet, controller_config:dict, model_config: dict, control_prompt: str, start_prompt: str):
        super().__init__()
        self.debug = True # Turn on and off prints

        self.grid = grid
        self.controller_config = controller_config
        self.model_config = model_config
        self.data_as_text = controller_config["data_as_text"]
        self.data_separation = controller_config["data_separation"]
        self.area_info = controller_config["area_info"]
        self.area_radius = controller_config["area_radius"]
        self.system_prompt = {"role": "system","content":start_prompt}
        self.control_prompt = control_prompt

        self.digits = 3 # defines to how many digits a value gets rounded before added to the llm

        self.grid_balanced = asyncio.Future()
        self.agent_finished = asyncio.Future()
        #self.last_adjustments = []
        self.tool_call_counter = {'get_measurements': 0, 'adjust_generator': 0, 'adjust_generator_up': 0, 'adjust_generator_down': 0, 'adjust_generator_same': 0, 'adjust_load': 0, 'adjust_load_up': 0, 'adjust_load_down': 0, 'adjust_load_same': 0}
        self.power_at_slack = 0 # active power at external grid

        # count
        self.failed_convertion_counter = 0


    async def get_measurements(self):
        """
        Measures and returns the current voltage(vm_pu), generator infeeds, load demands and line loading values in the electricity grid.
        The values are returned in json formats with the following structure.
        voltage_measurements:
        {
          "bus<id>": value of normalized voltage for bus<id>,
          ...
        }
        generator_infeeds:
        {
          "bus<id>": value of power infeed for bus<id>,
          ...
        }
        load_demand:
        {
          "bus<id>": value of load demand for bus<id>,
          ...
        }
        """
        self._say("requesting measurements")
        if self.data_separation:
            if self.data_as_text:
                grid_measurements = measurements_to_text(self.grid, self.digits, debug=self.debug)
            else:
                grid_measurements = measurements_to_json(self.grid, self.digits, debug=self.debug)
        else:
            if self.data_as_text:
                grid_measurements = grid_to_text(self.grid, add_voltage=True, add_load=True, add_infeed=True, digits=self.digits)
            else:
                grid_measurements = grid_to_json(self.grid, add_voltage=True, add_load=True, add_infeed=True, digits=self.digits)
        self.tool_call_counter['get_measurements'] += 1
        return grid_measurements


    async def adjust_generator_to(self, bus_index: int, new_infeed: float) -> str:
        """
        Use this function to adjust the generator infeed at a bus. The correct bus needs to be selected.
        By adjusting the generator infeed to the new infeed the voltage should come between 0.9 and 1.1.

        Args:
            bus_index (int): bus index where the generator is going to be adjusted
            new_infeed (float): new_infeed: new infeed value to adjust generator infeed at a bus measured in MW. Must be equal to or above 0.

        Returns:
            str: data on changed node or proximity
        """
        self.tool_call_counter['adjust_generator'] += 1

        # Check for arguments to be correct
        if not isinstance(bus_index, int) or not isinstance(new_infeed, float):
            return "bus_index needs to be an int and new_infeed needs to be a float"

        bus_index = int(bus_index)
        if bus_index in self.grid.sgen[["bus"]].values:
            # count what kind of adjustment is made
            if new_infeed > self.grid.sgen.loc[(self.grid.sgen["bus"] == bus_index), "p_mw"].values[0]:
                self.tool_call_counter['adjust_generator_up'] += 1
            elif new_infeed < self.grid.sgen.loc[(self.grid.sgen["bus"] == bus_index), "p_mw"].values[0]:
                self.tool_call_counter['adjust_generator_down'] += 1
            elif new_infeed == self.grid.sgen.loc[(self.grid.sgen["bus"] == bus_index), "p_mw"].values[0]:
                self.tool_call_counter['adjust_generator_same'] += 1

            current_infeed = self.grid.sgen.loc[(self.grid.sgen["bus"] == bus_index), "p_mw"].values[0]
            self.grid.sgen.loc[(self.grid.sgen["bus"] == bus_index), "p_mw"] = new_infeed

            try:
                runpp(self.grid, max_iteration=200) # refresh
            except:
                self._say(f"Grid did not converge after 200 iterations, trying again.", style='\033[31m')
                self.failed_convertion_counter += 1
                if self.failed_convertion_counter > 3: return "To many convergences fails end iterations now and call grid_in_bounds."

            self._say(f"Adjusted Generator{bus_index} from {current_infeed} MW to {self.grid.sgen.loc[(self.grid.sgen["bus"] == bus_index), "p_mw"].values[0]} MW.")

        else:
            self._say(f"Generator bus {bus_index} not in sgen.")
            return f"Generator bus {bus_index} not in sgen."

        # return either the area around the adjusted node or the node only
        if self.data_separation: # decide if measurements get included
            if self.area_info and self.data_as_text:
                return area_to_text(self.grid, bus_index, self.area_radius, digits=self.digits)
            elif self.area_info and not self.data_as_text:
                return area_to_json(self.grid, bus_index, self.area_radius, digits=self.digits)
            elif self.data_as_text:
                return area_to_text(self.grid, bus_index, 0, digits=self.digits)
            else:
                return area_to_json(self.grid, bus_index, 0, digits=self.digits)
        else:
            if self.area_info and self.data_as_text:
                return area_to_text(self.grid, bus_index, self.area_radius, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)
            elif self.area_info and not self.data_as_text:
                return area_to_json(self.grid, bus_index, self.area_radius, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)
            elif self.data_as_text:
                return area_to_text(self.grid, bus_index, 0, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)
            else:
                return area_to_json(self.grid, bus_index, 0, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)


    async def adjust_load_to(self, bus_index: int, new_load: float) -> str:
        """
        Use this function to adjust the load demand at a bus. The correct bus needs to be specified.
        By adjusting the load demand to the new load the voltage should come between 0.9 and 1.1.

        Args:
            bus_index (int): bus index where the load is going to be adjusted
            new_load (float): new load value to adjust load demand at a bus measured in MW. must be equal to or above 0.

        Returns:
            str: data on changed node or proximity
        """
        self.tool_call_counter['adjust_load'] += 1

        # Check for arguments to be correct
        if not isinstance(bus_index, int) or not isinstance(new_load, float):
            return "bus_index needs to be an int and new_load needs to be a float"

        bus_index = int(bus_index)
        if bus_index in self.grid.load[["bus"]].values:
            # count what kind of adjustment is made
            if new_load > self.grid.load.loc[(self.grid.load["bus"] == bus_index), "p_mw"].values[0]:
                self.tool_call_counter['adjust_load_up'] += 1
            elif new_load < self.grid.load.loc[(self.grid.load["bus"] == bus_index), "p_mw"].values[0]:
                self.tool_call_counter['adjust_load_down'] += 1
            elif new_load == self.grid.load.loc[(self.grid.load["bus"] == bus_index), "p_mw"].values[0]:
                self.tool_call_counter['adjust_load_same'] += 1

            current_load = self.grid.load.loc[(self.grid.load["bus"] == bus_index), "p_mw"].values[0]
            self.grid.load.loc[(self.grid.load["bus"] == bus_index), "p_mw"] = new_load

            try:
                runpp(self.grid, max_iteration=200)  # refresh
            except:
                self._say(f"Grid did not converge after 200 iterations, trying again.", style='\033[31m')
                self.failed_convertion_counter += 1
                if self.failed_convertion_counter > 3: return "To many convergences fails end iterations now and call grid_in_bounds."

            self._say(f"Adjusted Load{bus_index} from {current_load} MW to {self.grid.load.loc[(self.grid.load["bus"] == bus_index), "p_mw"].values[0]} MW.")

        else:
            self._say(f"Load bus {bus_index} not in loads.")
            return f"Load bus {bus_index} not in loads."

        # return either the area around the adjusted node or only the node itself
        if self.data_separation: # decide if measurements get included
            if self.area_info and self.data_as_text:
                return area_to_text(self.grid, bus_index, self.area_radius, digits=self.digits)
            elif self.area_info and not self.data_as_text:
                return area_to_json(self.grid, bus_index, self.area_radius, digits=self.digits)
            elif self.data_as_text:
                return area_to_text(self.grid, bus_index, 0, digits=self.digits)
            else:
                return area_to_json(self.grid, bus_index, 0, digits=self.digits)
        else:
            if self.area_info and self.data_as_text:
                return area_to_text(self.grid, bus_index, self.area_radius, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)
            elif self.area_info and not self.data_as_text:
                return area_to_json(self.grid, bus_index, self.area_radius, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)
            elif self.data_as_text:
                return area_to_text(self.grid, bus_index, 0, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)
            else:
                return area_to_json(self.grid, bus_index, 0, add_voltage=True, add_infeed=True, add_load=True, digits=self.digits)


    async def grid_in_bounds(self):
        """
        Finishes the regulation process and can only be used when all conditions are met everywhere.
        Before using, ensure that the voltage in the whole grid is between 0.9 and 1.1.
        End the conversation when this function is called and give the answer.
        """
        self._say("GRID in bounds")
        if not self.grid_balanced.done():
            self.grid_balanced.set_result(True)
            self._say("Future set to True")
        else:
            self._say("Future has already been set, cannot set again. End the conversation now")

        return "Grid in bounds. Do not respond further. This tool's return value is the final answer."


    def on_ready(self):
        runpp(self.grid,max_iteration=200) # initial grid run

        self._say(f"Activated", self.model_config["model"])

        model = OpenAIChatModel(model_name=self.model_config["model"], provider=OpenAIProvider(base_url=self.model_config['base_url'], api_key=self.model_config['api_key']))
        self.pyd_agent = PydanticAgent(model,
                                       tools=[self.adjust_load_to, self.adjust_generator_to, self.get_measurements],
                                       output_type=[self.grid_in_bounds],
                                       retries=10, # give several retries for ending
                                       instructions=self.system_prompt)
        self._say(f"LLM started")
        asyncio.create_task(self.control_grid())

    async def control_grid(self):
            # select grid representation depending on given parameters
            if self.data_separation:
                if self.data_as_text:
                    grid_structure = grid_to_text(self.grid, digits=self.digits)
                else:
                    grid_structure = grid_to_json(self.grid, digits=self.digits)

            else:  # not data_separation
                if self.data_as_text:
                    grid_structure = grid_to_text(self.grid, add_voltage=True, add_load=True, add_infeed=True, digits=self.digits)
                else:
                    grid_structure = grid_to_json(self.grid, add_voltage=True, add_load=True, add_infeed=True, digits=self.digits)

            try:
                grid_measurements = await self.get_measurements()
                if self.data_separation:
                    if isinstance(grid_measurements, str):
                        grid_measurements = json.loads(grid_measurements)
                    control_prompt = self.control_prompt.format(
                        net_structure=grid_structure,
                        voltage_measurements=grid_measurements["voltage_measurements"],
                        generator_infeeds=grid_measurements["generator_infeeds"],
                        load_demand=grid_measurements["load_demand"]
                    )

                else:
                    control_prompt = self.control_prompt.format(net_structure=grid_structure)
                self.control_prompt = {"role": "assistant", "content": control_prompt}

                self.tool_call_counter['get_measurements'] -= 1  # reduce as get_measurements gets not called by the llm

                llm_answer = await self.pyd_agent.run(control_prompt)

                # Finished
                self._say(f"LLM answer is {llm_answer}")
                self._say(f"All messages: {llm_answer.all_messages()}")
                self._say(
                    f'Tools called: \n get_measurements: {self.tool_call_counter["get_measurements"]} \n adjust_generator: {self.tool_call_counter["adjust_generator"]} \n adjust_load: {self.tool_call_counter["adjust_load"]}')
                self._say(
                    f'adjust_generator_up: {self.tool_call_counter["adjust_generator_up"]}\n adjust_generator_down: {self.tool_call_counter["adjust_generator_down"]} \n adjust_load_up: {self.tool_call_counter["adjust_load_up"]} \n adjust_load_down: {self.tool_call_counter["adjust_load_down"]}')
                self._say(f'Voltage in Bounds: {check_voltage_level(self.grid)}')
                if self.debug: await self.get_measurements()  # call get_measurements to print current grid state for debug

                self.power_at_slack = self.grid.res_ext_grid.at[0, "p_mw"]
                self._say(f"Power at external grid bus is {self.power_at_slack}")

                # build return_dict
                return_dict = {'llm_answer': llm_answer,
                               'llm_all_messages': llm_answer.all_messages(),
                               'tool_counter': self.tool_call_counter,
                               'voltage_in_bounds': check_voltage_level(self.grid),
                               'power_at_slack': self.power_at_slack}

                self._say(f'Voltage in Bounds: {check_voltage_level(self.grid)}')
                self.agent_finished.set_result(return_dict)

            except Exception as e:
                self._say(f"ERROR: {e}", style='\033[31m')

                # build return_dict
                return_dict = {'llm_answer': "FAIL",
                               'llm_all_messages': "FAIL",
                               'tool_counter': self.tool_call_counter,
                               'voltage_in_bounds': check_voltage_level(self.grid),
                               'power_at_slack': self.power_at_slack}

                self.agent_finished.set_result(return_dict)
                if not self.grid_balanced.done():
                    self.grid_balanced.set_result(True)
                    self._say("Future set to True")

    def _say(self, content, style='\33[33m'):
        """
        Helper function to print per agent.
        :param content: Message to be printed.
        """
        CEND = '\033[0m'
        if self.debug:
            print(f'{style}[{self.aid}] {content}{CEND}')


