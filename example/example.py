from vedotest import Swc2mesh

if __name__ == '__main__':
    mesh = Swc2mesh('data/Green_19weeks_Neuron4.CNG.swc')
    mesh.build('data/Green_19weeks_Neuron4.CNG.ply')
    # mesh.build('data/Neuron.ply', compartment='cell', simplification=True)
