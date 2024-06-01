import networkx as nx
import pandas as pd
import csv
import math
import random
from recordclass import recordclass
from itertools import zip_longest
from model import *

# Use recordclass over dict to eliminate visual noise
Morphology = recordclass('Morphology', ('x','y','z','type','radius'))

class SWCNeuron(object):
    """

    A cable model of t

    A node

        morphology: a recordclass 'Morphology'
            Store morphology information retrieved from SWC file.
        model: a subclass of model.Neuron
            Class of the computational model.
        params: None or a recordclass
            Parameters for the computational model specified by 'model'.
            If not present, the default parameters of the 'model' will be used.
        states: a recordclass
            States variables of the computational model.
    """
    defaultHeader = ('id', 'type', 'x', 'y', 'z', 'radius', 'parent_id')
    typeDict = {'id':int, 'x':float, 'y':float, 'z':float, 'type':int,
        'parent_id':int}

    def __init__(self, obj, **kwargs):
        """
        obj: string or a Pandas.Dataframe or a buffer
            A string specifies the path to the SWC file, a Pandas Dataframe
            contains a neuron data, or a buffer-like python object.

        Keyword arguments
        -----------------
        R: float
            Resistance between two adjecent nodes.

        axonHillockId: int
            Index of the axon hillock nodes.

        strouhal: bool
            Compute Strouhal number or not.

        passiveModel: a subclass of model.Neuron
            Default model for passive node.

        activeModel: a subclass of model.Neuron
            Default model for active node.

        swcKwargs: dict
            Keyword arguments for parsing a SWC file. See '_read_swc' below.
        """
        self.G = kwargs.pop('G', 1.)
        self.strouhal = kwargs.pop('stouhal', False)
        self.passiveModel = kwargs.pop('PassiveModel', PassiveModel)
        self.activeModel = kwargs.pop('ActiveModel', HodgkinHuxley)

        swcKwargs = kwargs.pop('swcKwargs', {})

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if type(obj) is str or hasattr(obj, 'read'):
            self.graph = self._read_swc(obj, **swcKwargs)
        elif type(obj) is pd.DataFrame:
            self.graph = self._read_dataframe(obj)
        else:
            raise TypeError('Unsupported type for SWC file/object: %r' % type(obj))

    def _read_swc(self, filename_or_buffer, **kwargs):
        """
        Read neuron model from a swc file into a NetworkX graph.

        filename_or_buffer: a string or a buffer
            Path to the swc file, or a buffer-like object containing the SWC
            content. The buffer-like object needs to support read().

        keyword arguments
        -----------------
        header: tuple or dict
            Names for each columns. Columns in a standard SWC file are assumed
            to follows the naming order,

                id, type, x, y, z, radius, parent_id

            If 'header' is a tuple, it should be a permutation of the above
            default naming order, and each of its entry indicates the name of
            column in the SWC file. In this case, the SWC file should contain no
            header row. If 'header' is a dict, the SWC file should contain a
            header row, and 'header' is a mapping between the header in the
            SWC file and the standard SWC column names. Note that some
            reconstruction packages use nonstandard headers, for example,
            CATMAID.

            If 'header' is not specified and the SWC file contains no header
            row, the default nameing order for each columns is used. If
            'header' is not specified and the SWC file contains a header row,
            the header should be a permutation of the above default naming
            order. Otherwise, an exception will be raised.

        delimiter: char
            Delimiter used in the SWC file. Default is whitespace.


        """
        header = kwargs.pop('header', self.defaultHeader)
        delimiter = kwargs.pop('delimiter', ' ')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if type(header) in (tuple, list):
            assert(set(self.defaultHeader) <= set(header))
        elif type(header) is dict:
            assert(set(self.defaultHeader) <= set(header.values()))
        else:
            raise TypeError('Unexpected header type: %r' % header)

        if type(filename_or_buffer) is str:
            with open(filename_or_buffer, 'r') as fp:
                hasHeader = csv.Sniffer().has_header(fp.read(1024))
                fp.seek(0)
                csvHeader = fp.readline().strip().split(delimiter)
        else:
            hasHeader = csv.Sniffer().has_header(filename_or_buffer.read(1024))
            filename_or_buffer.seek(0)
            csvHeader = filename_or_buffer.readline().strip().split(delimiter)
            filename_or_buffer.seek(0)

        if hasHeader:
            if type(header) in (tuple, list):
                assert(set(header) == set(csvHeader))
                header = csvHeader
            elif type(header) is dict:
                assert(set(header.values()) <= set(csvHeader))
                header = [header.get(k,k) for k in csvHeader]
        else:
            if not type(header) in (tuple, list):
                raise TypeError('If SWC file contains no header row, the ' \
                    '"header" argument should be None, a tuple, or a list.')
            if len(csvHeader) > len(header):
                n = len(csvHeader) - len(header)
                header = list(header) + ['']*n

        hasHeader = 0 if hasHeader else None
        df = pd.read_csv(filename_or_buffer, delimiter=delimiter, names=header,
            header=hasHeader)

        return self._read_dataframe(df, checkHeader=False)

    def _read_dataframe(self, df, checkHeader=True):
        if checkHeader:
            assert(set(defaultHeader) <= set(df.columns))

        for k,v in self.typeDict.items():
            df[k] = df[k].astype(v)

        graph = nx.Graph()

        # setup nodes
        for _,row in df.iterrows():
            attrs = {k:row[k] for k in ('x','y','z','radius','type')}
            graph.add_node(row['id'],
                morphology=Morphology(**attrs),
                model=self.passiveModel if attrs['type'] != 2 else self.activeModel,
                params=None,
                states=None)

            if row['parent_id'] != -1:
                graph.add_edge(row['id'], row['parent_id'], weight=0.)

        # setup edges
        for u,v,d in graph.edges(data=True):
            uu = graph.nodes[u]['morphology']
            vv = graph.nodes[v]['morphology']
            tmp = (uu.x - vv.x)**2. + (uu.y - vv.y)**2. + (uu.z - vv.z)**2.;
            d['weight'] = math.sqrt(tmp)
            d['I'] = 0.0

        return graph

    def __getitem__(self, key):
        if not hasattr(key, '__len__'):
            return self.graph.nodes[key]
        elif len(key) == 2:
            return self.graph.edge[key[0]][key[1]]
        else:
            raise KeyError('Key must be a scalar or an iterable of length 2; ' \
                'Unsupported key %r' % key)

    def __setitem__(self, key, value):
        if not hasattr(key, '__len__'):
            self.graph.nodes[key] = value
        elif len(key) == 2:
            self.graph.edge[key[0]][key[1]] = value
        else:
            raise KeyError('Key must be a scalar or an iterable of length 2; ' \
                'Unsupported key %r' % key)

    def update_edge_current(self):
        """
        Update the current flowing all edges.
        """
        for u, v, d in self.graph.edges(data=True):
            uu = self.graph.nodes[u] if u > v else self.graph.nodes[v]
            vv = self.graph.nodes[v] if u > v else self.graph.nodes[u]
            I = self.G*(uu['states'].V - vv['states'].V)
            d['I'] = I

    def update_node_states(self, dt):
        for u in self.graph.nodes():
            node = self.graph.nodes[u]
            node['states']  = node['model'].update(dt, node['states'], node['params'])

    def add_node_current(self, nodes, sti):
        for u in nodes:
            self.graph.nodes[u]['states'].I += sti

    def aggregate_node_current(self):
        """
        Aggregate current flowing from adjacent nodes.
        """
        for u, adj in self.graph.adjacency():
            uu = self.graph.nodes[u]
            for v, d in adj.items():
                sign = 1 if u < v else -1
                uu['states'].I += sign*d['I']

    def reset_node_current(self):
        for u in self.graph.nodes():
            self.graph.nodes[u]['states'].I = 0.

    def reset_all_states(self):
        for u in self.graph.nodes():
            node = self.graph.nodes[u]
            if 'initStates' in node:
                node['states'] = node['initStates']._replace()
            else:
                node['states'] = node['model'].initStates._replace()

    def create_simulator(self, dt, *args, **kwargs):
        """
        Create a simulator for given stimuli sets.

        Parameters
        ----------
        dt: float
            Time step of the simulation.

        Positional arguments
        --------------------
        The positonal arguments contain even numbers of entrie as follows,

            group_1, stimulus_1, ..., group_i, stimulus_i, ...

        group_i: a set or an iterable
            A set of indices of nodes that receives 'stimulus_i'.

        stimulus_i: an iterable
            Input stimulus applied to nodes in 'group_i'. Different 'stimulus_i'
            could have different length. If a 'stimulus_i' is exhausted during
            simulation, float(0) will be used instead.

        Keyword arguments
        -----------------
        steps: int or None
            Number of steps to run. If both 'steps' and 'duration' are given,
            the one with larger number of steps will be used. At least one
            of 'steps' or 'duration' has to be presented.

        duration: float or None
            Duration to run. The duration will be converted to number of steps
            by dividing 'duration' by 'dt'. If both 'steps' and 'duration' are
            given, the one with larger number of steps will be used. At
            least one of 'steps' or 'duration' has to present.

        Returns:
        --------
        A generator that updates the neuron and yields the neuron at each call
        of next().

        """
        assert(len(args) % 2 == 0)

        steps = kwargs.pop('steps', 0)
        duration = kwargs.pop('duration', 0)
        if duration:
            duration = duration//dt
        if duration == 0 and steps == 0:
            raise ValueError('Neither "steps" nor "duration" is specified.')
        steps = max(steps, duration)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        groups = args[::2]
        stimuli = args[1::2]
        stiGen = zip_longest(*stimuli, fillvalue=0.)

        # reset state variables
        self.reset_all_states()

        for i in range(steps):

            # reset current
            self.reset_node_current()

            # add external current
            sti = next(stiGen, [None]*len(stimuli))
            for g, s in zip(groups, sti):
                if s is not None:
                    self.add_node_current(g, s)

            # compute current between neighboring nodes
            self.update_edge_current()
            self.aggregate_node_current()

            # update nodes' state variables
            self.update_node_states(dt)

            yield self

    def set_node(self, nid, **kwargs):
        """
        Parameters
        ----------
        model: a subclass of model.Neuron
            Class of the computational model.

        params: None or an instance of 'recordclass'
            Parameters for the computational model specified by 'model'.
            If not present, the default parameters of the 'model' will be used.

        states: an instance of 'recordclass'
            Initial values for the states variables of the computational model.
        """
        params = kwargs.pop('params', None)
        states = kwargs.pop('states', None)
        model = kwargs.pop('model', None)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if model is not None:
            # TODO: assert(issubclass(Neuron, model))
            self.graph.nodes[nid]['model'] = model
            params = None or params
            states = None or states
        self.graph.nodes[nid]['params'] = params
        self.graph.nodes[nid]['initStates'] = states

    def get_no_params(self):
        """
        Get the current number of parameters.

        Returns:
        --------
        The total number of parameters in the neuron model specification.

        """
        no_params = 0
        for u,v in self.graph.nodes(data=True):
            no_params += len(v['model'].defaultParams._fields)
        return no_params

if __name__ == '__main__':

    import platform
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import numpy as np
    from io import StringIO
    import re
    from vedo import Points, show, sin
    from vedo.pyplot import DirectedGraph
    from pathlib import Path
    import matplotlib.animation

    if platform.system() == "Linux":
        plt.switch_backend("module://matplotlib-backend-kitty")

    # a simple neuron in SWC format
    # 3 - 2
    #       > 1 - 0
    # 5 - 4
    swcString = \
    "id type x y z radius parent_id\n" \
    "0  2    3 1 0 1      -1\n" \
    "1  1    2 1 0 1       0\n" \
    "2  1    1 2 0 1       1\n" \
    "3  1    0 2 0 1       2\n" \
    "4  1    1 0 0 1       1\n" \
    "5  1    0 0 0 1       4\n"
    dataString = StringIO(re.sub(' +',' ',swcString))


    neuron = SWCNeuron(dataString, swcKwargs={'delimiter':' '})
    # neuron = SWCNeuron("data/Green_19weeks_Neuron4.CNG.swc", swcKwargs={'delimiter':' '})

    dt = 1e-5
    dur = 0.5
    t = np.arange(0,dur,dt)
    stimulus = np.zeros_like(t)
    stimulus[t>0.2] = 100.

    # create a simulator
    sim = neuron.create_simulator(dt, [3,5], stimulus, steps=len(stimulus))

    # run the simulator
    V = np.zeros((neuron.graph.number_of_nodes(), len(t)))
    I = np.zeros((neuron.graph.number_of_edges(), len(t)))
    # for i, _ in enumerate(sim):
    #     for j,n in enumerate(neuron.graph.nodes()):
    #         V[j][i] = neuron[n]['states'].V
    #     for j, (u,v,d) in enumerate(neuron.graph.edges(data=True)):
    #         I[j][i] = d['I']

    # plot result

    # pos = nx.circular_layout(neuron.graph)
    # for n in neuron.graph.nodes():
    #     pos[n] = np.array([neuron[n]['morphology'].x, neuron[n]['morphology'].y])
    #
    # nx.draw(neuron.graph, pos, with_labels=True)
    # plt.show()

    # fig, axes = plt.subplots(neuron.graph.number_of_nodes(), 1, figsize=(8,16))
    #
    # for i,(ax,n) in enumerate(zip(axes, neuron.graph.nodes())):
    #     ax.plot(t,V[i])
    #     ax.set_xlabel('Time, [s]')
    #     ax.set_ylabel('Voltage, [mV]')
    #     ax.set_title('Node %r' % n)
    #     ax.grid()
    #
    # plt.tight_layout()
    # plt.savefig('swc_neuron.png', dpi=300)
    #
    # fig, axes = plt.subplots(neuron.graph.number_of_edges(), 1, figsize=(8,16))
    #
    # for i,(ax, (u,v)) in enumerate(zip(axes, neuron.graph.edges())):
    #     ax.plot(t,I[i])
    #     ax.set_xlabel('Time, [s]')
    #     ax.set_ylabel('Current, [pA]')
    #     ax.set_title('Edge %r-%r' % (u,v))
    #     ax.grid()
    #
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('swc_neuron_synapse.png', dpi=300)


    # Create some graph with nodes and edges
    # layouts: [2d, fast2d, clustering2d, circular, circular3d, cone, force, tree]
    g = DirectedGraph(layout='fast2d')

    ##################### Use networkx to create random nodes and edges
    G = neuron.graph
    for i, j in G.edges(): print(f"j: {int(j)}   i: {int(i)}")
    for i, j in G.edges(): g.add_edge(int(i),int(j))

    for n in neuron.graph.nodes():
        print(f"node: {n}")

    ##################### Manually create nodes and edges
    # for i in range(6): g.add_child(i)  # add one child node to node i
    # for i in range(3): g.add_child(i)
    # for i in range(3): g.add_child(i)
    # for i in range(7,9): g.add_child(i)
    # for i in range(3): g.add_child(12) # add 3 children to node 12
    # g.add_edge(1,16)

    ##################### build and draw
    g.build()
    graph = g[0].linewidth(4)          # get the vedo 3d graph lines
    nodes = graph.vertices             # get the 3d points of the nodes

    pts = Points(nodes, r=10).lighting('off')

    v1 = ['node'+str(n) for n in range(len(nodes))]
    v2 = [0 for x in range(len(nodes) - 1)]
    v2.insert(0, 0)
    labs1 = pts.labels(v1, scale=.02, italic=True).shift(.05,0.02,0).c('green')
    labs2 = pts.labels(v2, scale=.02, precision=3).shift(.05,-.02,0).c('red')
    intPlot = show(pts, graph, labs1, labs2, __doc__, interactive=False)
    print(f"graph.vertices: {len(nodes)}     neuron.graph.nodes: {neuron.graph.order()}")
    for i, _ in enumerate(sim):
        print(V)
        print("######")
        print(I)
        for j,n in enumerate(neuron.graph.nodes()):
            V[j][i] = neuron[n]['states'].V
        for j, (u,v,d) in enumerate(neuron.graph.edges(data=True)):
            I[j][i] = d['I']

        v2 = [V[x+1][i] for x in range(len(nodes) - 1)]
        v2.insert(0, V[1][i])
        labs1 = pts.labels(v1, scale=.02, italic=True).shift(.05,0.02,0).c('green')
        labs2 = pts.labels(v2, scale=.02, precision=3).shift(.05,-.02,0).c('red')

        # Interpolate the node value to color the edges:
        graph.cmap('viridis', v2).add_scalarbar3d(c='k')
        graph.scalarbar.shift(0.15,0,0).use_bounds(True)
        pts.cmap('viridis', v2)
        intPlot.render()
    intPlot.interactive().close()

    # This would colorize the edges directly with solid color based on a v3 array:
    # v3 = [sin(x) for x in range(graph.ncells)]
    # graph.cmap('jet', v3).add_scalarbar()

    # show(pts, graph, labs1, labs2, __doc__, zoom='tight').close()
