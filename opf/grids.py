"""
Create and provide different power grids vor analysis
"""

from pandapower import pandapowerNet, plotting
import pandapower.networks as pn
import matplotlib.pyplot as plt
import simbench as sb


def small_net(bias: int = 0, plot:bool = False) -> pandapowerNet:
    # Create an empty network
    net = pn.create_empty_network()
    net.name = "small_net"

    if bias == 1:
        load2 = 1
        load5 = 1 #2
        gen0 = 6.0 #5.0
        gen3 = 0.1
    elif bias == -1:
        load2 = 1
        load5 = 5
        gen0 = 0.1
        gen3 = 0.3
    else: # bias == 0:
        load2 = 2
        load5 = 2
        gen0 = 0.1
        gen3 = 0.1
    # Create buses
    bus0 = pn.create_bus(net, vn_kv=20, name="Bus 0")
    bus1 = pn.create_bus(net, vn_kv=20, name="Bus 1")
    bus2 = pn.create_bus(net, vn_kv=20, name="Bus 2")
    bus3 = pn.create_bus(net, vn_kv=20, name="Bus 3")
    bus4 = pn.create_bus(net, vn_kv=20, name="Bus 4")

    # Create generators
    pn.create_sgen(net, bus=bus0, p_mw=gen0, name="Generator 0", cost=1, p_mw_max=1)
    pn.create_sgen(net, bus=bus3, p_mw=gen3, name="Generator 3", cost=1, p_mw_max=1)

    # Create lines between the buses in a linear fashion
    pn.create_line(net, from_bus=bus0, to_bus=bus1, length_km=10, std_type="NAYY 4x50 SE", name="Line 0")
    pn.create_line(net, from_bus=bus1, to_bus=bus2, length_km=10, std_type="NAYY 4x50 SE", name="Line 1")
    pn.create_line(net, from_bus=bus2, to_bus=bus3, length_km=10, std_type="NAYY 4x50 SE", name="Line 2")
    pn.create_line(net, from_bus=bus3, to_bus=bus4, length_km=10, std_type="NAYY 4x50 SE", name="Line 3")

    # Add loads at Bus 2 and Bus 5
    pn.create_load(net, bus=bus1, p_mw=load2, name="Load at Bus 2")
    pn.create_load(net, bus=bus4, p_mw=load5, name="Load at Bus 5")

    # Add a grid connection at Bus 3
    pn.create_ext_grid(net, bus=bus2, vm_pu=1.0, name="Grid Connection")

    # Plotting grid if plot is set to true
    if plot:
        # automatically generate geo data
        net = plotting.create_generic_coordinates(net)
        # draw network
        ax = plotting.simple_plot(net, show_plot=False, plot_loads=True, plot_gens=True, plot_sgens=True)  # returns matplotlib axes

        # annotate buses with names
        for index, _ in net.bus.iterrows():
            geo = eval(net.bus.geo[index])
            x, y = geo['coordinates'][0], geo['coordinates'][1]
            label = net.bus.name[index]
            ax.text(x+0.05, y-0.05, label, fontsize=8, ha="right", va="bottom", color="black")

        # annotate lines at their midpoint
        for index, _ in net.line.iterrows():
            geo_from_bus = eval(net.bus.geo[net.line.from_bus[index]])
            geo_to_bus = eval(net.bus.geo[net.line.to_bus[index]])
            x0, y0 = geo_from_bus['coordinates'][0], geo_from_bus['coordinates'][1]
            x1, y1 = geo_to_bus['coordinates'][0], geo_to_bus['coordinates'][1]
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            label = net.line.name[index]
            ax.text(xm, ym+0.03, label, fontsize=7, ha="center", va="center", color="blue")

        plt.show()

    return net

def simbench_net(sb_code:str, bias:int = 0, plot:bool = False) -> pandapowerNet:
    # load net via the simbench code
    net = sb.get_simbench_net(sb_code)

    # set teh voltage negative or positive as needed
    if sb_code == '1-LV-rural1--0-no_sw':
        net.name = "rural1"
        if bias == 1: # set of bus10
            net.load.loc[1, "p_mw"] = 1. # load at bus7
            net.sgen.loc[2, "p_mw"] = .3 # gen at bus7
            net.sgen.loc[1, "p_mw"] = 3.5 # gen at bus10
            net.load.loc[9, "p_mw"] = .0 # load ar bus10
            net.load.loc[0, "p_mw"] = 1. # load at bus9
        elif bias == 2: # set of bus10 and bus12
            net.load.loc[1, "p_mw"] = 1. # load at bus7
            net.sgen.loc[2, "p_mw"] = .3 # gen at bus7
            net.sgen.loc[1, "p_mw"] = 3.5 # gen at bus10
            net.load.loc[9, "p_mw"] = .0 # load ar bus10
            net.load.loc[0, "p_mw"] = 1. # load at bus9
            net.sgen.loc[3, "p_mw"] = 1.  # gen at bus12
            net.load.loc[4, "p_mw"] = .8  # load at bus8
        elif bias == -1: # set of bus4
            net.load.loc[12, "p_mw"] = .32 # load at bus4
            net.load.loc[5, "p_mw"] = .0 # load at bus5
        elif bias == -2: # set of bus4 and bus8 and a bit more
            net.load.loc[12, "p_mw"] = .115 # load at bus4
            net.load.loc[5, "p_mw"] = .0 # load at bus5
            net.load.loc[4, "p_mw"] = .73 # load at bus8
            net.load.loc[11, "p_mw"] = .0 # load at bus1
            net.load.loc[3, "p_mw"] = .0  # load at bus12
            net.sgen.loc[3, "p_mw"] = .0  # gen at bus12
        elif bias == 3:
            net.load.loc[1, "p_mw"] = 1.5  # load at bus7
            net.sgen.loc[2, "p_mw"] = 0.9  # gen at bus7
            net.sgen.loc[1, "p_mw"] = 4.4  # gen at bus10
            net.load.loc[9, "p_mw"] = .0  # load ar bus10
            net.load.loc[0, "p_mw"] = 1.  # load at bus9
            net.load.loc[12, "p_mw"] = .45 # load at bus4
            net.load.loc[5, "p_mw"] = .1  # load at bus5
        else: # bias == 0
            pass

    if sb_code == '1-LV-semiurb4--0-no_sw':
        net.name = "semiurb4"
        if bias == 1:
            net.sgen.loc[0, "p_mw"] = .7 # gen at bus31
        elif bias == -1:
            net.load.loc[40, "p_mw"] = .29 # load at bus 42
        elif bias == -2:
            net.load.loc[40, "p_mw"] = .35 # load at bus 42
            #net.load.loc[0, "p_mw"] = .39 # load at bus 32
            net.load.loc[5, "p_mw"] = .40 # load at bus 16
            net.sgen.loc[0, "p_mw"] = 0.05 # gen at bus32
        elif bias == 3:
            net.sgen.loc[0, "p_mw"] = 2.1 # gen at bus32
            net.load.loc[34, "p_mw"] = .3
            net.load.loc[35, "p_mw"] = .3
        else: # bias == 0
            pass

    if plot:
        ax = plotting.simple_plot(net, show_plot=False, plot_loads=True, plot_gens=True, plot_sgens=True)

        # annotate buses with names
        for index, _ in net.bus.iterrows():
            geo = eval(net.bus.geo[index])
            x, y = geo['coordinates'][0], geo['coordinates'][1]
            label = f'bus {index}'
            ax.text(x - 0.00005, y - 0.00003, label, fontsize=8, ha="right", va="bottom", color="black")

        # annotate lines at their midpoint
        for index, _ in net.line.iterrows():
            geo_from_bus = eval(net.bus.geo[net.line.from_bus[index]])
            geo_to_bus = eval(net.bus.geo[net.line.to_bus[index]])
            x0, y0 = geo_from_bus['coordinates'][0], geo_from_bus['coordinates'][1]
            x1, y1 = geo_to_bus['coordinates'][0], geo_to_bus['coordinates'][1]
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            label = f'line {index}'
            ax.text(xm + 0.00001, ym, label, fontsize=7, ha="center", va="center", color="blue")

        plt.show()

    return net