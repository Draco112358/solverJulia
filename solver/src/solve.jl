
function encode_complex(z)
    if z isa Complex
        return (z.re, z.im)
    else
        error("Object of type $typeof(z) is not JSON serializable")
    end
end
       
function dump_json_data(matrix_Z, matrix_S, matrix_Y)
    solver_matrices_dict = Dict(
        "matrix_Z" => 
    )
end

def dump_json_data(matrix_Z,matrix_S,matrix_Y):

    
    assert(isinstance(matrix_Z, np.ndarray))
    assert(isinstance(matrix_S, np.ndarray))
    assert(isinstance(matrix_Y, np.ndarray))


    solver_matrices_dict = {}
    
    solver_matrices_dict["matrix_Z"] = json.dumps(matrix_Z.tolist(), default=encode_complex)
    solver_matrices_dict["matrix_S"] = json.dumps(matrix_S.tolist(), default=encode_complex)
    solver_matrices_dict["matrix_Y"] = json.dumps(matrix_Y.tolist(), default=encode_complex) 

    print(solver_matrices_dict)
    
    return solver_matrices_dict
        
      
struct signal
    value
end

class signal:
    def __init__(self, dict_element):

        
        self.value = complex(float(dict_element['Re']),float(dict_element['Im']))
        assert type(self.value)==complex

        
        
struct geom_attributes
    dict_element
    radius = dict_element["radius"]
    segments = dict_element["segments"]
end
        
struct transf_params
    dict_element
    position = dict_element["position"]
    rotation = dict_element["rotation"]  
    scale = dict_element["scale"]   
end

struct element
    dict_element
    name = dict_element["name"]
    type = dict_element["type"]
    keyComponent = dict_element["keyComponent"]
    geometryAttributes = geom_attributes(dict_element["geometryAttributes"])
    transformationParams = transf_params(dict_element["transformationParams"])
end
        
            
        
struct port
    dict_element
    name = dict_element["name"]
    type = dict_element["type"]
    inputElement = element(dict_element["inputElement"])
    outputElement = element(dict_element["outputElement"])
    rlcParams = dict_element["rlcParams"]
    isSelected = dict_element["isSelected"]
end

        
class lumped_element:
    def __init__(self, dict_element):
        self.name = dict_element['name']
        self.type = dict_element['type']
        self.value = dict_element['value']

        self.inputElement = element(dict_element['inputElement'])
        assert type(self.inputElement)==element
        self.outputElement = element(dict_element['outputElement'])
        assert type(self.outputElement)==element
        self.rlcParams = dict_element['rlcParams']
        self.isSelected = dict_element['isSelected']

class material:
    def __init__(self, dict_element):
        self.name = dict_element['name']
        self.color = dict_element['color']
        self.permeability = dict_element['permeability']
        self.tangent_delta_permeability = dict_element['tangent_delta_permeability']
        self.custom_permeability = dict_element['custom_permeability']
        assert len(self.custom_permeability)==2
        self.permittivity = dict_element['permittivity']
        self.tangent_delta_permittivity = dict_element['tangent_delta_permittivity']
        self.custom_permittivity = dict_element['custom_permittivity']
        assert len(self.custom_permittivity)==2
        self.conductivity = dict_element['conductivity']
        self.tangent_delta_conductivity = dict_element['tangent_delta_conductivity']
        self.custom_conductivity = dict_element['custom_conductivity']
        assert len(self.custom_conductivity)==2
        self.epsr = None
        self.Rx = None
        self.Ry = None
        self.Rz = None
        self.Cx = None
        self.Cy = None
        self.Cz = None
        
class port_def:
    def __init__(self, inp_pos, out_pos, voxels, nodes):
        self.port_start = inp_pos
        self.port_end = out_pos
        self.port_voxels = voxels
        self.port_nodes = nodes

class le_def:
    def __init__(self, val, typ, inp_pos, out_pos, voxels, nodes):
        self.value = val
        self.type = typ
        self.le_start = inp_pos
        self.le_end = out_pos
        self.le_voxels = voxels
        self.le_nodes = nodes

# class LPWorker(Thread):
#     def __init__(self, queue):
#         Thread.__init__(self)
#         self.queue = queue

#     def run(self):
#         while True:
#             # Get the work from the queue and expand the tuple
#             (bars, sizex, sizey, sizez, dc, Lps) = self.queue.get()
#             try:
#                 Lps.append(compute_Lp_matrix(bars=bars, sizex=sizex, sizey=sizey, sizez=sizez, dc=dc))
#             finally:
#                 self.queue.task_done()
                

function read_ports(inputData:Dict):
    
    @assert inputData isa Dict
    ports = inputData["ports"]
    # """
    # A port is given by a couple of regions lying on the external surface of the compounded objects model. 
    # In particular, a port is made by a starting region and an ending region. One of these regions can be:
    # - a point
    # - a surface (disk/rectangle - dimensions: user defined)
    # - a collection of surfaces (è possibile posticipare questa versione)
    # """
    port_objects = [port(el) for el in ports]
    input_positions = []
    output_positions = []
    N_PORTS = length(port_objects)
    
    for port_object in port_objects
        @assert length(port_object.inputElement.transformationParams.position)==3
        ipos = zeros((1,3))
        ipos[1, 1] = port_object.inputElement.transformationParams.position[1]*1e-3
        ipos[1, 2] = port_object.inputElement.transformationParams.position[2]*1e-3
        ipos[1, 3] = port_object.inputElement.transformationParams.position[3]*1e-3
        push!(input_positions, ipos)
        @assert length(port_object.outputElement.transformationParams.position)==3
        opos = zeros((1, 3))
        opos[1, 1] = port_object.outputElement.transformationParams.position[1]*1e-3
        opos[1, 2] = port_object.outputElement.transformationParams.position[2]*1e-3
        opos[1, 3] = port_object.outputElement.transformationParams.position[3]*1e-3
        push!(output_positions, opos)
    @assert length(input_positions)==N_PORTS && length(output_positions)==N_PORTS
    ports_out = port_def(inp_pos=np.stack([i for i in input_positions]), out_pos=np.stack([i for i in output_positions]),voxels=np.zeros((N_PORTS, 2), dtype='int64'), nodes=np.zeros((N_PORTS, 2), dtype='int64'))
    end
    return ports_out
end


def read_lumped_elements(inputData::Dict):
    
    @assert inputData isa Dict
    
    lumped_elements = inputData["lumped_elements"]
    
    lumped_elements_objects = [lumped_element(el) for el in lumped_elements]
    input_positions = []
    output_positions = []
    values = []
    types = []
    N_LUMPED_ELEMENTS = length(lumped_elements_objects)
    if N_LUMPED_ELEMENTS == 0:
        lumped_elements_out = le_def(val=zeros(0),typ=zeros(Int64, 0),inp_pos=zeros((0, 3)),out_pos=zeros((0, 3)),voxels=zeros(Int64, (0, 2)),nodes=zeros(Int64, (0, 2)))
        @assert length(input_positions)==N_LUMPED_ELEMENTS && length(output_positions)==N_LUMPED_ELEMENTS && length(values)==N_LUMPED_ELEMENTS && length(types)==N_LUMPED_ELEMENTS
    else: 
        for lumped_element_object in lumped_elements_objects:
            @assert length(lumped_element_object.inputElement.transformationParams.position)==3
            ipos = zeros((1,3))
            ipos[1, 1] = lumped_element_object.inputElement.transformationParams.position[1]*1e-3
            ipos[1, 2] = lumped_element_object.inputElement.transformationParams.position[2]*1e-3
            ipos[1, 3] = lumped_element_object.inputElement.transformationParams.position[3]*1e-3
            push!(input_positions, ipos)
            @assert length(lumped_element_object.outputElement.transformationParams.position)==3
            opos = zeros((1, 3))
            opos[1, 1] = lumped_element_object.outputElement.transformationParams.position[1]*1e-3
            opos[1, 2] = lumped_element_object.outputElement.transformationParams.position[2]*1e-3
            opos[1, 3] = lumped_element_object.outputElement.transformationParams.position[3]*1e-3
            push!(output_positions, opos)            
            lvalue = zeros(1)
            lvalue[1] = lumped_element_object.value
            values.append(lvalue)
            
            ltype = zeros(Int64, 1)
            ltype[1] = lumped_element_object.type
            types.append(ltype)
        end
    end
            
        @assert length(input_positions)==N_LUMPED_ELEMENTS && length(output_positions)==N_LUMPED_ELEMENTS && length(values)==N_LUMPED_ELEMENTS && length(types)==N_LUMPED_ELEMENTS
    
        lumped_elements_out = le_def(val=np.stack([i for i in values]),typ=np.stack([i for i in types]),inp_pos=np.stack([i for i in input_positions]), out_pos=np.stack([i for i in output_positions]), \
                                    voxels=zeros(Int64, (N_LUMPED_ELEMENTS, 2)), nodes=zeros(Int64, (N_LUMPED_ELEMENTS, 2)))




    return lumped_elements_out

function read_materials(inputData::Dict):
    @assert inputData isa Dict
    materials = inputData["materials"]
    materials_objects = [material(el) for el in materials]
    return materials_objects
end

function read_signals(inputData::Dict)
    @assert inputData isa Dict
    signals = inputData["signals"]
    @assert signals isa 
end

function read_signals(inputData::Dict):
    @assert inputData isa Dict
    signals = inputData["signals"]
    signals_objects = [signal(Complex(Float64(el["Re"]),Float64(el["Im"]))) for el in signals]
    return signals_objects
end

    
function doSolving(mesherOutput, solverInput, solverAlgoParams):

    inputDict = solverInput
    mesherDict = mesherOutput
    
    sx, sy, sz = mesherDict["cell_size"]["cell_size_x"],mesherDict["cell_size"]["cell_size_y"],mesherDict["cell_size"]["cell_size_z"]
    # sx = sx * 10
    # sy = sy * 10
    # sz = sz * 10
    num_input_files = mesherDict["n_materials"]
    
    
    
    origin_x, origin_y, origin_z = mesherDict["origin"]["origin_x"],mesherDict["origin"]["origin_y"],mesherDict["origin"]["origin_z"]

    # origin_x = origin_x * 10
    # origin_y = origin_y * 10
    # origin_z = origin_z * 10

    origin = (origin_x,origin_y,origin_z)
    Nx, Ny, Nz = Int64(mesherDict['n_cells']['n_cells_x']), Int64(mesherDict['n_cells']['n_cells_y']), Int64(mesherDict['n_cells']['n_cells_z'])

    grids = np.stack([mesherDict['mesher_matrices'][i] for i in mesherDict['mesher_matrices']])

    assert Nx == grids[0].shape[0]
    assert Ny == grids[0].shape[1]
    assert Nz == grids[0].shape[2]
    

    frequencies = inputDict['frequencies']
    
    n_freq = length(frequencies)
    
    PORTS = read_ports(inputData=inputDict)
    
    L_ELEMENTS = read_lumped_elements(inputData=inputDict)

    
    MATERIALS = read_materials(inputData=inputDict) 
    SIGNALS = read_signals(inputData=inputDict)
    
    
    
    # START SETTINGS--------------------------------------------
    # n_freq=10
    println("numero frequenze",n_freq)
    println("frequenze",frequencies)
    # print("frequenze trasformate",freq)

    
    
    #TODO: hardcoded stuff
    
    inner_Iter = solverAlgoParams['innerIteration']
    outer_Iter = solverAlgoParams['outerIteration']
    tol = solverAlgoParams['convergenceThreshold']*ones((n_freq))
    ind_low_freq= filter(i -> !iszero(frequencies[i]), findall(frequencies<1e5, frequencies))
    tol[ind_low_freq] = 1e-7
    
    struct GMRES_set:
        Inner_Iter
        Outer_Iter
        tol
    end

    GMRES_settings = GMRES_set(inner_Iter,outer_Iter,tol)
    
    # END SETTINGS----------------------------------------------
    
    # cpu_time = time.perf_counter_ns() / 1000;


    # print(sx, sy, sz,Nx,Ny,Nz,origin)

    A, Gamma, ports, lumped_elements, sup_centers, sup_type, bars_Lp_x, bars_Lp_y, bars_Lp_z, diag_R, diag_Cd = generate_interconnection_matrices_and_centers(size_x=sx, size_y=sy, size_z=sz,
                                                                                                                                                            grid_matrix=grids, num_cel_x=Nx, num_cel_y=Ny, num_cel_z=Nz,
                                                                                                                                                            materials=MATERIALS, port_matrix=PORTS, lumped_el_matrix=L_ELEMENTS,
                                                                                                                                                            minimum_vertex=origin)  
    

    for k in range(1, stop=size(ports.voxels)[1]):
        ports.port_voxels[k,1] = ports.voxels[k,1]
        ports.port_voxels[k,2] = ports.voxels[k, 2]
        ports.port_nodes[k,1] = ports.nodes[k, 1]
        ports.port_nodes[k,2] = ports.nodes[k, 2]

    for k in range(1, stop=size(lumped_elements.voxels)[1]):
        lumped_elements.le_voxels[k,1]=lumped_elements.voxels[k,1]
        lumped_elements.le_voxels[k, 2] = lumped_elements.voxels[k, 2]
        lumped_elements.le_nodes[k, 1] = lumped_elements.nodes[k, 1]
        lumped_elements.le_nodes[k, 2] = lumped_elements.nodes[k, 2]
    
    # print("Time for mesher manipulation and data formatting:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    

    # cpu_time = time.perf_counter_ns() / 1000;
    P_mat = compute_P_matrix(sup_centers,sup_type,sx,sy,sz)
    # print("Time for computing P parallel:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    # print(P_mat)

    # cpu_time = time.perf_counter_ns() / 1000;
    # P_mat = Main.compute_P_matrix(sup_centers,sup_type,sx,sy,sz)
    # print("Time for computing P with julia:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    
    # cpu_time = time.perf_counter_ns() / 1000;
    # # note that this computation can be parallelized
    # Lp_x_mat = compute_Lp_matrix(bars=bars_Lp_x,sizex=sx,sizey=sy,sizez=sz,dc=1)
    # Lp_y_mat = compute_Lp_matrix(bars=bars_Lp_y,sizex=sx,sizey=sy,sizez=sz,dc=2)
    # Lp_z_mat = compute_Lp_matrix(bars=bars_Lp_z,sizex=sx,sizey=sy,sizez=sz,dc=3)
    # print("Time for computing Lp:", round(time.perf_counter_ns() / 1000 - cpu_time,2))


    # inputLP = [(bars_Lp_x, sx, sy, sz, 1), (bars_Lp_y, sx, sy, sz, 2), (bars_Lp_z, sx, sy, sz, 3)]
    # Lps = []
    # # cpu_time = time.perf_counter_ns() / 1000;
    # with multiprocessing.Pool() as pool:
    #     Lps = pool.starmap(compute_Lp_matrix, inputLP)
    # print("Time for computing Lp:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    # Lp_x_mat = Lps[0]
    # Lp_y_mat = Lps[1]
    # Lp_z_mat = Lps[2]
    
    # cpu_time = time.perf_counter_ns() / 1000;
    # queue = Queue()
    # # Create 8 worker threads
    # for x in range(8):
    #     worker = LPWorker(queue)
    #     # Setting daemon to True will let the main thread exit even though the workers are blocking
    #     worker.daemon = True
    #     worker.start()
    # # Put the tasks into the queue as a tuple
    # queue.put((bars_Lp_x, sx, sy, sz, 1, Lps))
    # queue.put((bars_Lp_y, sx, sy, sz, 2, Lps))
    # queue.put((bars_Lp_z, sx, sy, sz, 3, Lps))
    # # Causes the main thread to wait for the queue to finish processing all the tasks
    # queue.join()
    # print("Time for computing Lp:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    # print(Lps[0])


    # cpu_time = time.perf_counter_ns() / 1000;
    Lp_x_mat, Lp_y_mat, Lp_z_mat = compute_Lps(bars_Lp_x, bars_Lp_y, bars_Lp_z, sx, sy, sz) 
    # print("Time for computing Lp with julia:", round(time.perf_counter_ns() / 1000 - cpu_time,2))

    Z, Y, S=Quasi_static_iterative_solver(freq,A,Gamma,P_mat,Lp_x_mat,
        Lp_y_mat,Lp_z_mat,diag_R,diag_Cd,ports,lumped_elements,GMRES_settings)
    
    # Z, Y, S = solver_funcs.Quasi_static_direct_solver(freq,A,Gamma,P_mat,Lp_x_mat,\
    #     Lp_y_mat,Lp_z_mat,diag_R,diag_Cd,ports,lumped_elements)
    
    # print("Time for Solver:", round(time.perf_counter_ns() / 1000 - cpu_time,2))


    # ------------- PLOTS -----------------------------------------------
    # plt.figure(1)
    # plt.plot(freq, Z[0,0,:].real*1e3, label="PEEC - Re Z [mOhm]", linewidth=2)
    # plt.plot(freq, Z[0,0,:].__abs__()*1e3, label="PEEC - Mag Z [mOhm]", linestyle = 'dotted', linewidth=2)
    # plt.xlabel('freq')
    # plt.ylabel('R, Mag [mOhm]')
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.legend()
    # plt.draw()
    
    # # C = np.zeros((n_freq), dtype="double")
    # # for cont in range(n_freq):
    # #     C[cont]=(1/Z[0,0,cont]).imag/(2*mt.pi*freq[cont])*1e9
    
    # # plt.figure(2)
    # # plt.plot(freq, C, label="PEEC C [nF]", linewidth=2)
    # # plt.xlabel('freq')
    # # plt.xscale('log')
    # # plt.ylabel('C [nF]')
    # # plt.legend()
    # # plt.draw()
    
    # L=np.zeros((n_freq), dtype="double")
    # for cont in range(n_freq):
    #     # print("L",cont,freq[cont],(Z[0,0,cont]).imag,mt.pi,(2*mt.pi*freq[cont]),((Z[0,0,cont]).imag/(2*mt.pi*freq[cont]))*1e9)
        
    #     L[cont]=((Z[0,0,cont]).imag/(2*mt.pi*freq[cont]))*1e9
    
    # plt.figure(3)
    # plt.plot(freq, L, label="PEEC L [nH]", linewidth=2)
    # plt.xlabel('freq')
    # plt.xscale('log')
    # plt.ylabel('L [nH]')
    # plt.legend()
    # plt.draw()
    
    # # Angle=np.zeros((n_freq), dtype="double")
    # # for cont in range(n_freq):
    # #     Angle[cont]=np.angle(Z[0,0,cont], deg=True)
    
    # # plt.figure(4)
    # # plt.plot(freq, Angle, label="PEEC Angle [rad]", linewidth=2)
    # # plt.xlabel('freq')
    # # plt.xscale('log')
    # # plt.ylabel('Angle [rad]')
    # # plt.legend()
    # # plt.draw()
    
    # plt.show()
    # end
# ------------------------ END PLOTS -------------------------------------------
    return dump_json_data(matrix_Z=Z,matrix_S=S,matrix_Y=Y) 
end