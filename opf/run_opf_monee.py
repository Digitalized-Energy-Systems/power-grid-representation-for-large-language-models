from monee.io.from_pandapower import from_pandapower_net
from monee import run_energy_flow_optimization
from monee.problem import OptimizationProblem, Objectives, Constraints, AttributeParameter
import monee.model as mm

from grids import simbench_net

CONTROLLABLE_ATTRIBUTES = [
    (
        "regulation",
        AttributeParameter(
            min=lambda attr, val: 0, max=lambda attr, val: 1, val=lambda attr, val: 1
        ),
    )
]

def create_voltage_control_ref_problem():
    problem = OptimizationProblem(debug=True)
    problem.controllable_demands(CONTROLLABLE_ATTRIBUTES)
    problem.controllable_generators(CONTROLLABLE_ATTRIBUTES)

    problem.bounds((0.9, 1.1), lambda m, _: type(m) is mm.Bus, ["vm_pu"])

    objectives = Objectives()

    def calc_weight(model):
        weight = 1
        if isinstance(model, mm.PowerLoad):
            weight = -10
        elif isinstance(model, mm.PowerGenerator):
            weight = -1
        elif isinstance(model, mm.ExtPowerGrid):
            weight = -5
        return weight

    def retrieve_power(model):
        """
        No docstring provided.
        """

        if isinstance(model, mm.PowerLoad | mm.PowerGenerator):
            return model.p_mw * model.regulation
        elif isinstance(model, mm.ExtPowerGrid):
            return model.p_mw


    def calculate_objective(model_to_data):
        """
        No docstring provided.
        """
        return sum([data * retrieve_power(model) for model, data in model_to_data.items()])


    objectives.select(lambda x: type(x) == mm.ExtPowerGrid).data(calc_weight).calculate(calculate_objective)
    objectives.with_models(problem.controllables_link).data(calc_weight).calculate(calculate_objective)

    constraints = Constraints()

    problem.constraints = constraints
    problem.objectives = objectives

biases = [-2,-1,0,1,2,3]
nets_semiurb = [from_pandapower_net(simbench_net("1-LV-semiurb4--0-no_sw", bias)) for bias in biases]
nets_rural = [from_pandapower_net(simbench_net("1-LV-rural1--0-no_sw", bias)) for bias in biases]

slacks_semi = []
slacks_rural = []

for net in nets_semiurb:
    result = run_energy_flow_optimization(net, optimization_problem=create_voltage_control_ref_problem())
    # convert to pandapower sign convention 
    slacks_semi.append(-result.dataframes["ExtPowerGrid"]["p_mw"][0])

for net in nets_rural:
    result = run_energy_flow_optimization(net, optimization_problem=create_voltage_control_ref_problem())
    # convert to pandapower sign convention 
    slacks_rural.append(-result.dataframes["ExtPowerGrid"]["p_mw"][0])

print(slacks_semi)
print(slacks_rural)