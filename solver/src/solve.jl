include("lp_compute.jl")
include("solver_function.jl")
include("mesh_manipulation_functions.jl")
using JSON
using MLUtils: unsqueeze

function encode_complex(z)
    if z isa Complex
        return (z.re, z.im)
    else
        error("Object of type $typeof(z) is not JSON serializable")
    end
end
       
function dump_json_data(matrix_Z,matrix_S,matrix_Y)

    solver_matrices_dict = Dict(
        "matrix_Z" => JSON.json(matrix_Z),
        "matrix_S" => JSON.json(matrix_S),
        "matrix_Y" => JSON.json(matrix_Y)
    )

    print(solver_matrices_dict)
    
    return solver_matrices_dict
end


Base.@kwdef struct signal
    dict_element
    value = complex(float(dict_element["Re"]),float(dict_element)["Im"])
end
        
Base.@kwdef struct geom_attributes
    dict_element
    radius = dict_element["radius"]
    segments = dict_element["segments"]
end
        
Base.@kwdef struct transf_params
    dict_element
    position = dict_element["position"]
    rotation = dict_element["rotation"]  
    scale = dict_element["scale"]   
end

Base.@kwdef struct element
    dict_element
    name = dict_element["name"]
    type = dict_element["type"]
    keyComponent = dict_element["keyComponent"]
    geometryAttributes = geom_attributes(dict_element["geometryAttributes"])
    transformationParams = transf_params(dict_element["transformationParams"])
end
        
            
        
Base.@kwdef struct port
    dict_element
    name = dict_element["name"]
    type = dict_element["type"]
    inputElement = element(dict_element["inputElement"])
    outputElement = element(dict_element["outputElement"])
    rlcParams = dict_element["rlcParams"]
    isSelected = dict_element["isSelected"]
end

        
Base.@kwdef struct lumped_element
    dict_element
    name = dict_element["name"]
    type = dict_element["type"]
    value = dict_element["value"]
    inputElement = element(dict_element["inputElement"])
    outputElement = element(dict_element["outputElement"])
    rlcParams = dict_element["rlcParams"]
    isSelected = dict_element["isSelected"]
end

Base.@kwdef mutable struct material
    dict_element
    name = dict_element["name"]
    color = dict_element["color"]
    permeability = dict_element["permeability"]
    tangent_delta_permeability = dict_element["tangent_delta_permeability"]
    custom_permeability = dict_element["custom_permeability"]
    permittivity = dict_element["permittivity"]
    tangent_delta_permittivity = dict_element["tangent_delta_permittivity"]
    custom_permittivity = dict_element["custom_permittivity"]
    conductivity = dict_element["conductivity"]
    tangent_delta_conductivity = dict_element["tangent_delta_conductivity"]
    custom_conductivity = dict_element["custom_conductivity"]
    epsr = nothing
    Rx = nothing
    Ry = nothing
    Rz = nothing
    Cx = nothing
    Cy = nothing
    Cz = nothing
end
        
mutable struct port_def
    port_start
    port_end
    port_voxels
    port_nodes
end

mutable struct le_def
    value
    type
    le_start
    le_end
    le_voxels
    le_nodes
end

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
                

function read_ports(inputData::Dict)
    
    @assert inputData isa Dict
    ports = inputData["ports"]
    # """
    # A port is given by a couple of regions lying on the external surface of the compounded objects model. 
    # In particular, a port is made by a starting region and an ending region. One of these regions can be:
    # - a point
    # - a surface (disk/rectangle - dimensions: user defined)
    # - a collection of surfaces (è possibile posticipare questa versione)
    # """
    port_objects = [el for el in ports]
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
    end
    @assert length(input_positions)==N_PORTS && length(output_positions)==N_PORTS
    inp_pos = []
    for i in input_positions
        push!(inp_pos, unsqueeze([i], dims=2))
    end
    out_pos = []
    for i in output_positions
        push!(out_pos, unsqueeze([i], dims=2))
    end
    ports_out = port_def(inp_pos, out_pos,zeros(Int64, (N_PORTS, 2)),zeros(Int64,(N_PORTS, 2)))
    
    return ports_out
end


function read_lumped_elements(inputData::Dict)
    
    @assert inputData isa Dict
    
    lumped_elements = inputData["lumped_elements"]
    
    lumped_elements_objects = [el for el in lumped_elements]
    input_positions = []
    output_positions = []
    values = []
    types = []
    N_LUMPED_ELEMENTS = length(lumped_elements_objects)
    if N_LUMPED_ELEMENTS == 0
        lumped_elements_out = le_def(zeros(0),zeros(Int64, 0),zeros((0, 3)),zeros((0, 3)),zeros(Int64, (0, 2)),zeros(Int64, (0, 2)))
        @assert length(input_positions)==N_LUMPED_ELEMENTS && length(output_positions)==N_LUMPED_ELEMENTS && length(values)==N_LUMPED_ELEMENTS && length(types)==N_LUMPED_ELEMENTS
    else
        for lumped_element_object in lumped_elements_objects
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
            append!(values, lvalue)
            
            ltype = zeros(Int64, 1)
            ltype[1] = lumped_element_object.type
            push!(types, ltype)
        end
    end
            
        @assert length(input_positions)==N_LUMPED_ELEMENTS && length(output_positions)==N_LUMPED_ELEMENTS && length(values)==N_LUMPED_ELEMENTS && length(types)==N_LUMPED_ELEMENTS
    
        value = []
        for i in values
            push!(value, unsqueeze([i], dims=2))
        end

        type = []
        for i in types
            push!(type, unsqueeze([i], dims=2))
        end

        in_pos = []
        for i in input_positions
            push!(in_pos, unsqueeze([i], dims=2))
        end

        out_pos = []
        for i in output_positions
            push!(out_pos, unsqueeze([i], dims=2))
        end

        lumped_elements_out = le_def(value,type,in_pos, out_pos,zeros(Int64, (N_LUMPED_ELEMENTS, 2)), (Int64, (N_LUMPED_ELEMENTS, 2)))

    return lumped_elements_out
end

function read_materials(inputData::Dict)
    @assert inputData isa Dict
    materials = inputData["materials"]
    materials_objects = [material(dict_element=el) for el in materials]
    return materials_objects
end

function read_signals(inputData::Dict)
    @assert inputData isa Dict
    signals = inputData["signals"]
    signals_objects = [el for el in signals]
    return signals_objects
end

struct GMRES_set
    Inner_Iter
    Outer_Iter
    tol
end

    
function doSolving(mesherOutput, solverInput, solverAlgoParams)

    inputDict = Dict(solverInput)
    mesherDict = Dict(mesherOutput)
    
    sx, sy, sz = mesherDict["cell_size"]["cell_size_x"],mesherDict["cell_size"]["cell_size_y"],mesherDict["cell_size"]["cell_size_z"]
    # # sx = sx * 10
    # # sy = sy * 10
    # # sz = sz * 10
    num_input_files = mesherDict["n_materials"]
    

    origin_x = mesherDict["origin"]["origin_x"]
    origin_y = mesherDict["origin"]["origin_y"]
    origin_z = mesherDict["origin"]["origin_z"]

    # # origin_x = origin_x * 10
    # # origin_y = origin_y * 10
    # # origin_z = origin_z * 10

    origin = (origin_x,origin_y,origin_z)
    Nx = Int64(mesherDict["n_cells"]["n_cells_x"])
    Ny = Int64(mesherDict["n_cells"]["n_cells_y"])
    Nz = Int64(mesherDict["n_cells"]["n_cells_z"])

    #print(mesherDict["mesher_matrices"])

    testarray = []
    for (index, value) in mesherDict["mesher_matrices"]
        #print(copy(value))
        push!(testarray, copy(value))
    end

    grids = []

    for i in testarray
        grids = unsqueeze([i], dims=2)
        #print(grids[1][4][2][1])
    end




    #grids = unsqueeze([mesherDict["mesher_matrices"][i] for i in mesherDict["mesher_matrices"]], dims=2)

    frequencies = inputDict["frequencies"]
    
    n_freq = length(frequencies)
    
    PORTS = read_ports(inputDict)
    
    L_ELEMENTS = read_lumped_elements(inputDict)


    
    MATERIALS = read_materials(inputDict) 
    SIGNALS = read_signals(inputDict)
    
    
    
    
    # # START SETTINGS--------------------------------------------
    # # n_freq=10
    # print("numero frequenze ",n_freq)
    # print("frequenze ",frequencies)
    #print("frequenze trasformate ",freq)

    
    
    # #TODO: hardcoded stuff
    
    inner_Iter = solverAlgoParams["innerIteration"]
    outer_Iter = solverAlgoParams["outerIteration"]
    tol = solverAlgoParams["convergenceThreshold"]*ones((n_freq))
    #ind_low_freq= filter(i -> !iszero(frequencies[i]), findall(freq -> freq<1e5, frequencies))
    #tol[ind_low_freq] = 1e-7
    

    GMRES_settings = GMRES_set(inner_Iter,outer_Iter,tol)
    
    # # END SETTINGS----------------------------------------------
    
    # # cpu_time = time.perf_counter_ns() / 1000;


    # # print(sx, sy, sz,Nx,Ny,Nz,origin)

    A, Gamma, ports, lumped_elements, sup_centers, sup_type, bars_Lp_x, bars_Lp_y, bars_Lp_z, diag_R, diag_Cd = generate_interconnection_matrices_and_centers(sx, sy, sz,
                                                                                                                                                            grids, Nx, Ny, Nz,                                                                                                                                                      MATERIALS, PORTS, L_ELEMENTS,
                                                                                                                                                            origin)  
    
                                                                                                                                                            # for k in range(1, stop=size(ports.voxels)[1])
    #     ports.port_voxels[k,1] = ports.voxels[k,1]
    #     ports.port_voxels[k,2] = ports.voxels[k, 2]
    #     ports.port_nodes[k,1] = ports.nodes[k, 1]
    #     ports.port_nodes[k,2] = ports.nodes[k, 2]
    # end

    # for k in range(1, stop=size(lumped_elements.voxels)[1])
    #     lumped_elements.le_voxels[k,1]=lumped_elements.voxels[k,1]
    #     lumped_elements.le_voxels[k, 2] = lumped_elements.voxels[k, 2]
    #     lumped_elements.le_nodes[k, 1] = lumped_elements.nodes[k, 1]
    #     lumped_elements.le_nodes[k, 2] = lumped_elements.nodes[k, 2]
    # end
    
    # # print("Time for mesher manipulation and data formatting:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    

    # # cpu_time = time.perf_counter_ns() / 1000;
    #P_mat = compute_P_matrix(sup_centers,sup_type,sx,sy,sz)
    P_mat = @time compute_P_matrix(sup_centers,sup_type,sx,sy,sz)
    
    # # print("Time for computing P parallel:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    # # print(P_mat)

    # # cpu_time = time.perf_counter_ns() / 1000;
    # # P_mat = Main.compute_P_matrix(sup_centers,sup_type,sx,sy,sz)
    # # print("Time for computing P with julia:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    
    # # cpu_time = time.perf_counter_ns() / 1000;
    # # # note that this computation can be parallelized
    Lp_x_mat = @time compute_Lp_matrix_1(bars_Lp_x,sy,sz)
    Lp_y_mat = @time compute_Lp_matrix_2(bars_Lp_y,sx,sz)
    Lp_z_mat = @time compute_Lp_matrix_3(bars_Lp_z,sx,sy)
    # # print("Time for computing Lp:", round(time.perf_counter_ns() / 1000 - cpu_time,2))


    # inputLP = [(bars_Lp_x, sx, sy, sz, 1), (bars_Lp_y, sx, sy, sz, 2), (bars_Lp_z, sx, sy, sz, 3)]
    # Lps = []
    # # cpu_time = time.perf_counter_ns() / 1000;
    # with multiprocessing.Pool() as pool:
    #     Lps = pool.starmap(compute_Lp_matrix, inputLP)
    # print("Time for computing Lp:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    # Lp_x_mat = Lps[0]
    # Lp_y_mat = Lps[1]
    # Lp_z_mat = Lps[2]
    
    # # cpu_time = time.perf_counter_ns() / 1000;
    # # queue = Queue()
    # # # Create 8 worker threads
    # # for x in range(8):
    # #     worker = LPWorker(queue)
    # #     # Setting daemon to True will let the main thread exit even though the workers are blocking
    # #     worker.daemon = True
    # #     worker.start()
    # # # Put the tasks into the queue as a tuple
    # # queue.put((bars_Lp_x, sx, sy, sz, 1, Lps))
    # # queue.put((bars_Lp_y, sx, sy, sz, 2, Lps))
    # # queue.put((bars_Lp_z, sx, sy, sz, 3, Lps))
    # # # Causes the main thread to wait for the queue to finish processing all the tasks
    # # queue.join()
    # # print("Time for computing Lp:", round(time.perf_counter_ns() / 1000 - cpu_time,2))
    # # print(Lps[0])


    # # cpu_time = time.perf_counter_ns() / 1000;
    # Lp_x_mat, Lp_y_mat, Lp_z_mat = compute_Lps(bars_Lp_x, bars_Lp_y, bars_Lp_z, sx, sy, sz) 
    # # print("Time for computing Lp with julia:", round(time.perf_counter_ns() / 1000 - cpu_time,2))

    Z, Y, S = Quasi_static_iterative_solver(frequencies,A,Gamma,P_mat,Lp_x_mat,Lp_y_mat,Lp_z_mat,diag_R,diag_Cd,ports,lumped_elements,GMRES_settings)
    
    # # Z, Y, S = solver_funcs.Quasi_static_direct_solver(freq,A,Gamma,P_mat,Lp_x_mat,\
    # #     Lp_y_mat,Lp_z_mat,diag_R,diag_Cd,ports,lumped_elements)
    
    # # print("Time for Solver:", round(time.perf_counter_ns() / 1000 - cpu_time,2))

    return diag_R 
end