import os.path as osp
import os
import glob
import subprocess
import pygraphviz as pgv
import mysql.connector
from apply_directives import apply_directives
from sklearn.preprocessing import OneHotEncoder
import re
from tqdm import tqdm
import multiprocessing
from collections import defaultdict
import config
from math import ceil, floor, log2, log10
import numpy as np
import time


import torch
import torch_geometric as geo
from torch_geometric.data import Dataset, Data
from multiprocessing import Process, Manager

from dataset_gpu import CustomData


def formatForEncoder(input):
    return np.array(list(input)).reshape(-1, 1)

def tuplesToCOO(tuples, biEdges):
    sources, destinations = zip(*tuples)
    pattern = re.compile(r'\d+$')

    # Extract the numbers using the regular expression
    sources = [int(re.search(pattern, label).group()) for label in sources]
    destinations = [int(re.search(pattern, label).group()) for label in destinations]


    biSource = np.concatenate((sources, destinations))
    biDestinations = np.concatenate((destinations, sources))

    coo = [sources, destinations]
    if biEdges:
        coo = [biSource, biDestinations]

    return np.array(coo, dtype=np.int64)


class CustomDataset(Dataset):
    def __init__(self, config_name):
        self.config_name = config_name
        self.config_file = "configs/" + config_name + ".txt"
        self.saveDir = "datasets/" + config_name + "/"       
        
        os.makedirs(self.saveDir, exist_ok=True)

        self.cfg_edge_index = {}
        self.cfg_select = {}
        self.num_bbs = {}
        self.bb_batch = {}

        self.parseConfig()

        self.kernels = {'gemm' : 297}

        kernelMap = {   "gemm" : 297,
                        "get_delta_matrix_weights3" : 365,
                        "get_delta_matrix_weights1" : 366,
                        "get_delta_matrix_weights2" : 367,
                        "bfs" : 373,
                        "update" : 382,
                        "hist" : 383,
                        "init" : 384,
                        "sum_scan" : 385,
                        "last_step_scan" : 386,
                        "fft" : 389,
                        "local_scan" : 390,
                        "md_kernel" : 391,
                        "twiddles8" : 396,
                        "get_oracle_activations1" : 403,
                        "get_oracle_activations2" : 404,
                        "matrix_vector_product_with_bias_input_layer" : 405,
                        "stencil3d" : 409,
                        "ellpack" : 410,
                        "bbgemm" : 412,
                        "viterbi" : 415,
                        "aes_shiftRows" : 424,
                        "ms_mergesort" : 436,
                        "merge" : 437,
                        "add_bias_to_activations" : 440,
                        "aes256_encrypt_ecb" : 441,
                        "aes_expandEncKey" : 442,
                        "ss_sort" : 443,
                        "stencil" : 444,
                        "soft_max" : 446,
                        "take_difference" : 447,
                        "matrix_vector_product_with_bias_output_layer" : 449,
                        "update_weights" : 450,
                        "backprop" : 451,
                        "aes_addRoundKey" : 455,
                        "aes_addRoundKey_cpy" : 457,
                        "aes_mixColumns" : 459,
                        "aes_subBytes" : 460,
                        "matrix_vector_product_with_bias_second_layer" : 461,
                    }    

        with open("active_kernels.txt", 'r') as file:
            keys_from_file = [line.strip() for line in file]

        self.kernels = {}

        # Print values corresponding to the keys
        for key in keys_from_file:
            if key in kernelMap:
                self.kernels[key] = kernelMap[key]
                print(f"Using kernel {key}")
            else:
                print(f"Kernel: {key} not recognized.")
  

        self.index = 0

        super().__init__()

    @property
    def processed_file_names(self):
        return glob.glob(f"{self.saveDir}*.pt")
    
    def parseConfig(self):
        self.one_hot_node_encoders = []
        self.one_hot_edge_encoders = []
        self.normalized_node_encoders = []
        self.log2_normalized_node_encoders = []
        self.index_edge_encoders = []
        with open(self.config_file, 'r') as file:
            self.invocation = file.readline().strip()
            for line in file:
                line = line.strip()
                words = line.split()
                key = words[1]

                if words[0] == "node":
                    # config file format: node dotFileKey one_hot tag1 tag2 ...
                    if words[2] == config.one_hot_encoder_method:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
                        tags = set()
                        for word in words[3:]:
                            tags.add(word)
                        encoder.fit(formatForEncoder(tags))
                        self.one_hot_node_encoders.append({"key" :key, "encoder" : encoder})
                    # config file format: node dotFileKey normalize max
                    elif words[2] == config.normalize_float_method:
                        max = words[3]
                        self.normalized_node_encoders.append({"key" :key, "max" : max})
                    elif words[2] == config.log2_normalized_method:
                        max = words[3]
                        self.log2_normalized_node_encoders.append({"key" :key, "max" : max})

                if words[0] == "edge":
                    if words[2] == config.one_hot_encoder_method:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
                        tags = set()
                        for word in words[3:]:
                            tags.add(word)
                        encoder.fit(formatForEncoder(tags))
                        self.one_hot_edge_encoders.append({"key" :key, "encoder" : encoder})
                    if words[2] == config.index_method:
                        tags = list()
                        for word in words[3:]:
                            tags.append(word)
                        self.index_edge_encoders.append({"key":key, "tags":tags})
    def getMaxValues(self):
        cnx = mysql.connector.connect(user='user', password='password', host='localhost', auth_plugin='mysql_native_password')
        cursor = cnx.cursor()
        query = """
        SELECT 
        MAX(db4hls.resource_results.hls_lut), 
        MAX(db4hls.resource_results.hls_ff),
        MAX(db4hls.resource_results.hls_bram),
        MAX(db4hls.resource_results.hls_dsp),
        MAX(db4hls.performance_results.estimated_clock),
        MAX(db4hls.performance_results.average_latency)
        FROM db4hls.configuration
        JOIN db4hls.configuration_space ON db4hls.configuration.id_configuration_space = db4hls.configuration_space.id_configuration_space
        JOIN db4hls.implementation ON db4hls.configuration.hash_configuration = db4hls.implementation.hash_configuration
        LEFT JOIN db4hls.resource_results ON db4hls.implementation.id_resource_results = db4hls.resource_results.id_resource_result
        LEFT JOIN db4hls.performance_results ON db4hls.implementation.id_performance_results = db4hls.performance_results.id_performance_result
        WHERE db4hls.performance_results.average_latency > 0
        """

        cursor.execute(query)
        result = cursor.fetchall()

        self.max_luts = result[0][0]
        self.max_ffs = result[0][1]
        self.max_brams = result[0][2]
        self.max_dsps = result[0][3]
        self.max_clock = result[0][4]
        self.max_latency = result[0][5]

        self.max_latency_log = log2(self.max_latency) - log2(600)

    def list(self):
        cnx = mysql.connector.connect(user='user', password='password', host='localhost', auth_plugin='mysql_native_password')
        cursor = cnx.cursor()

        for kernelName, id in self.kernels.items():
            query = f"""
            SELECT 
            COUNT(*)
            FROM db4hls.configuration
            JOIN db4hls.configuration_space ON db4hls.configuration.id_configuration_space = db4hls.configuration_space.id_configuration_space
            JOIN db4hls.implementation ON db4hls.configuration.hash_configuration = db4hls.implementation.hash_configuration
            LEFT JOIN db4hls.resource_results ON db4hls.implementation.id_resource_results = db4hls.resource_results.id_resource_result
            WHERE db4hls.configuration_space.id_configuration_space = {id}
            """

            cursor.execute(query)
            result = cursor.fetchall()
            print(kernelName, result[0])

    def test(self):
        self.getMaxValues()

        cnx = mysql.connector.connect(user='user', password='password', host='localhost', auth_plugin='mysql_native_password')
        cursor = cnx.cursor()
        for kernelName, id in self.kernels.items():
            print("Testing: ", kernelName)

            cfg_edges, cfg_select, num_bbs, bb_batch = self.buildCFG(kernelName)
            self.cfg_edge_index[kernelName] = cfg_edges

            self.cfg_select[kernelName] = cfg_select
            self.num_bbs[kernelName] = num_bbs
            self.bb_batch[kernelName] = bb_batch

            query = f"""
            SELECT 
            db4hls.configuration.config_script, 
            db4hls.resource_results.hls_lut,
            db4hls.resource_results.hls_ff,
            db4hls.resource_results.hls_bram,
            db4hls.resource_results.hls_dsp,
            db4hls.performance_results.estimated_clock,
            db4hls.performance_results.average_latency
            FROM db4hls.configuration
            JOIN db4hls.configuration_space ON db4hls.configuration.id_configuration_space = db4hls.configuration_space.id_configuration_space
            JOIN db4hls.implementation ON db4hls.configuration.hash_configuration = db4hls.implementation.hash_configuration
            LEFT JOIN db4hls.resource_results ON db4hls.implementation.id_resource_results = db4hls.resource_results.id_resource_result
            LEFT JOIN db4hls.performance_results ON db4hls.implementation.id_performance_results = db4hls.performance_results.id_performance_result
            WHERE db4hls.configuration_space.id_configuration_space = {id} AND db4hls.performance_results.average_latency > 0
            """

            cursor.execute(query)
            result = cursor.fetchall()

            graph = self.processRow(kernelName, 0, 0, result[0], True)
            # print(self.get(0))

            # graph.draw(f"test_pdfs/{self.config_name}_{kernelName}.pdf", prog="dot", format="pdf")
            graph.write(f"test_pdfs/{self.config_name}_{kernelName}.dot")

    def buildCFG(self, kernel):
        out = "temp/0.cpp"
        apply_directives(kernel, f"kernels/{kernel}.cpp", "", out)
        full_invocation = self.invocation + f" --top {kernel} --src {out}"
        graphOutput = subprocess.run(full_invocation, shell=True, capture_output=True, text=True)

        graph = pgv.AGraph(string=graphOutput.stdout)
        node_list = graph.nodes()
        edge_list = graph.edges()

        bb_edge_list = defaultdict(set)
        node_bb_dict = {}

        bb_list = []
        num_bbs = 0
        for node in node_list:
            bbID = int(node.attr["bbID"])
            node_bb_dict[node] = bbID
            bb_list.append(bbID)
            num_bbs = max(num_bbs, bbID + 1)
        bb_list = torch.tensor(bb_list, dtype=torch.int64)

        for edge in edge_list:
            if not (edge.attr["flowType"] == "control" or edge.attr["flowType"] == "call"):
                continue
                
            sourceIndex = 0
            targetIndex = 1

            if edge.attr["style"] == "dashed":
                sourceIndex = 1
                targetIndex = 0

            source_bb = node_bb_dict[edge[sourceIndex]]
            target_bb = node_bb_dict[edge[targetIndex]]
            if source_bb != target_bb:
                bb_edge_list[source_bb].add(target_bb)


        output_source_list = np.array([[]])
        output_target_list = np.array([[]])

        for sourceBB, targetBBs in bb_edge_list.items():
            for targetBB in targetBBs:
                output_source_list = np.concatenate([output_source_list, formatForEncoder([sourceBB])], axis=1)
                output_target_list = np.concatenate([output_target_list, formatForEncoder([targetBB])], axis=1)                

        bb_batch = torch.zeros(num_bbs, dtype=torch.int64)

        output_edge_list = torch.tensor(np.concatenate([output_source_list, output_target_list]), dtype=torch.int64)
        return output_edge_list, bb_list, num_bbs, bb_batch

    def getNodeArray(self, node_list):
        encoded_node_list = []
        for node in node_list:
            temp_array = np.array([[]])
            for one_hot_encoder in self.one_hot_node_encoders:
                input = formatForEncoder([node.attr[one_hot_encoder["key"]].replace(" ", "")])
                one_hot = one_hot_encoder["encoder"].transform(input)
                temp_array = np.concatenate([temp_array, one_hot], axis=1)
            for normalize_encoder in self.normalized_node_encoders:
                input = float(node.attr[normalize_encoder["key"]])

                temp = ((input / float(normalize_encoder["max"])) * 2) - 1
                temp_array = np.concatenate([temp_array, [[temp]]], axis=1)
            for normalize_encoder in self.log2_normalized_node_encoders:
                input = float(node.attr[normalize_encoder["key"]])
                max = float(normalize_encoder["max"])
                log_input = log2(input)
                temp = ((log_input / max) * 2) - 1
                temp_array = np.concatenate([temp_array, [[temp]]], axis=1)
            for i in range(240):
                temp_array = np.concatenate([temp_array, [[0]]], axis=1)

            encoded_node_list.append(temp_array[0])
        
        node_array = np.asarray(encoded_node_list, dtype=np.float32)
        node_array = torch.from_numpy(node_array)

        return node_array
    

    def getEdgeAttributeArray(self, edges, biEdges):
        encoded_edge_list = []

        for edge in edges:
            temp_array = np.array([[]])
            for one_hot_encoder in self.one_hot_edge_encoders:
                input = formatForEncoder([edge.attr[one_hot_encoder["key"]]])
                one_hot = one_hot_encoder["encoder"].transform(input)
                temp_array = np.concatenate([temp_array, one_hot], axis=1)

            encoded_edge_list.append(temp_array[0])
        
        edge_array = np.asarray(encoded_edge_list, dtype=np.float32)
        if(biEdges):
            edge_array = np.concatenate([edge_array, edge_array])

        edge_array = torch.from_numpy(edge_array)
        return edge_array
    

    # def getIndexEdgeAttributeDict(self, edges, biEdges):
    #     # just support one for now
    #     index_edge_encoder = self.index_edge_encoders[0]

    #     temp_array = np.array([])

    #     for edge in edges:
    #         flowType = edge.attr[index_edge_encoder["key"]]
    #         flowIndex = [index_edge_encoder["tags"].index(flowType)]
    #         temp_array = np.concatenate([temp_array, flowIndex])
        
    #     if biEdges:
    #         temp_array = np.concatenate([temp_array, temp_array])

    #     temp_array = np.expand_dims(temp_array, axis=1)
    #     edge_array = torch.tensor(temp_array, dtype=torch.int64)
    #     return edge_array

    def processRow(self, kernel, i, j, row, biEdges, returnGraph=False):
        pragmadFile = "temp/" + str(j) + ".cpp"
        apply_directives(kernel, f"kernels/{kernel}.cpp", row[0], pragmadFile)
        full_invocation = self.invocation + f" --top {kernel} --src {pragmadFile}"
        graphOutput = subprocess.run(full_invocation, shell=True, capture_output=True, text=True)


        graph = pgv.AGraph(string=graphOutput.stdout)

        nodeArray = self.getNodeArray(graph.nodes())

        edgeConnectionArray = tuplesToCOO(graph.edges(), biEdges)
        edgeConnectionArray = torch.from_numpy(edgeConnectionArray)
        
        edgeAttributeArray = self.getEdgeAttributeArray(graph.edges(), biEdges)

        normalized_luts = ((float(row[1]) / self.max_luts) * 2) - 1
        normalized_ffs = ((float(row[2]) / self.max_ffs) * 2) - 1
        if(self.max_brams != 0):
            normalized_brams = ((float(row[3]) / self.max_brams) * 2) - 1
        else:
            normalized_brams = -1.0
        normalized_dsps = ((float(row[4]) / self.max_dsps) * 2) - 1
        normalized_clock = ((float(row[5]) / self.max_clock) * 2) - 1

        latency = row[6]
        # -log2(500) scales better on the lower numbers
        latency_log = log2(latency + 600) - log2(600)

        normalized_latency = ( (latency_log / self.max_latency_log) * 2 ) - 1

        y = np.array([[normalized_luts, normalized_ffs, normalized_brams, normalized_dsps, normalized_clock, normalized_latency]], dtype=np.float32)
        y = torch.from_numpy(y)

        max_y = {}
        max_y["LUTs"] = self.max_luts
        max_y["FFs"] = self.max_ffs
        max_y["BRAMs"] = self.max_brams
        max_y["DSPs"] = self.max_dsps
        max_y["clock"] = self.max_clock
        max_y["latency"] = self.max_latency

        bb_list = []
        for node in graph.nodes():
            bbID = int(node.attr["bbID"])
            bb_list.append(bbID)

        bb_list = torch.tensor(bb_list, dtype=torch.int64)

        data = CustomData(x=nodeArray, 
                    edge_index=edgeConnectionArray, 
                    edge_attr=edgeAttributeArray, 
                    y=y, 
                    max_y=max_y,
                    kernel=kernel,
                    pragmas=row[0],
                    cfg_edge_index= self.cfg_edge_index[kernel],
                    cfg_select = bb_list,
                    num_bbs = self.num_bbs[kernel],
                    bb_batch = self.bb_batch[kernel]
                    )
        
        # data.__inc__ = CustomInc
        

        torch.save(data, osp.join(self.saveDir, f'data_{self.index + i + j}.pt')) 

        if returnGraph:
            return graph
        
    def processColumn(self, kernel, result, j, num_processes, biEdges, returnGraph=False):
        for i in range(j, len(result), num_processes):
            row = result[i]
            pragmadFile = "temp/" + str(j) + ".cpp"
            apply_directives(kernel, f"kernels/{kernel}.cpp", row[0], pragmadFile)
            full_invocation = self.invocation + f" --top {kernel} --src {pragmadFile}"
            graphOutput = subprocess.run(full_invocation, shell=True, capture_output=True, text=True)

            graph = pgv.AGraph(string=graphOutput.stdout)
            nodeArray = self.getNodeArray(graph.nodes())


            edgeConnectionArray = tuplesToCOO(graph.edges(), biEdges)
            edgeConnectionArray = torch.from_numpy(edgeConnectionArray)
            
            edgeAttributeArray = self.getEdgeAttributeArray(graph.edges(), biEdges)


            normalized_luts = ((float(row[1]) / self.max_luts) * 2) - 1
            normalized_ffs = ((float(row[2]) / self.max_ffs) * 2) - 1
            if(self.max_brams != 0):
                normalized_brams = ((float(row[3]) / self.max_brams) * 2) - 1
            else:
                normalized_brams = -1.0
            normalized_dsps = ((float(row[4]) / self.max_dsps) * 2) - 1
            normalized_clock = ((float(row[5]) / self.max_clock) * 2) - 1



            latency = row[6]
            # -log2(500) scales better on the lower numbers
            latency_log = log2(latency + 600) - log2(600)

            normalized_latency = ( (latency_log / self.max_latency_log) * 2 ) - 1

            y = np.array([[normalized_luts, normalized_ffs, normalized_brams, normalized_dsps, normalized_clock, normalized_latency]], dtype=np.float32)
            y = torch.from_numpy(y)

            max_y = {}
            max_y["LUTs"] = self.max_luts
            max_y["FFs"] = self.max_ffs
            max_y["BRAMs"] = self.max_brams
            max_y["DSPs"] = self.max_dsps
            max_y["clock"] = self.max_clock
            max_y["latency"] = self.max_latency

            bb_list = []
            for node in graph.nodes():
                bbID = int(node.attr["bbID"])
                bb_list.append(bbID)

            bb_list = torch.tensor(bb_list, dtype=torch.int64)

            data = CustomData(x=nodeArray, 
                        edge_index=edgeConnectionArray, 
                        edge_attr=edgeAttributeArray, 
                        y=y, 
                        max_y=max_y,
                        kernel=kernel,
                        pragmas=row[0],
                        cfg_edge_index= self.cfg_edge_index[kernel],
                        cfg_select = bb_list,
                        num_bbs = self.num_bbs[kernel],
                        bb_batch = self.bb_batch[kernel]
                        )
            
            # data.__inc__ = CustomInc
            

            torch.save(data, osp.join(self.saveDir, f'data_{self.index + i}.pt')) 

        if returnGraph:
            return graph
        
    def custom_error_callback(error):
        print(f'Got error: {error}')

    def generateDataset(self, biEdges):
        startTime = time.time()
        self._processed_file_names = []

        self.getMaxValues()

        cnx = mysql.connector.connect(user='user', password='password', host='localhost', auth_plugin='mysql_native_password')
        cursor = cnx.cursor()

        num_kernels = 0

        

        for k, (kernelName, id) in enumerate(self.kernels.items()):

            # kernelList = ["matrix_vector_product_with_bias_input_layer", "merge", "hist", "backprop", "bfs", "last_step_scan", "stencil3d", "add_bias_to_activations", "ss_sort","aes256_encrypt_ecb", "aes_addRoundKey_cpy", 'soft_max', "gemm", "viterbi", "md_kernel", "get_oracle_activations1", "update", "update_weights", "local_scan", "ellpack", "fft", "bbgemm", "twiddles8", "sum_scan", "get_oracle_activations2", "take_difference"]

            # if kernelName not in kernelList:
            #     continue

            query = f"""
            SELECT 
            db4hls.configuration.config_script, 
            db4hls.resource_results.hls_lut,
            db4hls.resource_results.hls_ff,
            db4hls.resource_results.hls_bram,
            db4hls.resource_results.hls_dsp,
            db4hls.performance_results.estimated_clock,
            db4hls.performance_results.average_latency
            FROM db4hls.configuration
            JOIN db4hls.configuration_space ON db4hls.configuration.id_configuration_space = db4hls.configuration_space.id_configuration_space
            JOIN db4hls.implementation ON db4hls.configuration.hash_configuration = db4hls.implementation.hash_configuration
            LEFT JOIN db4hls.resource_results ON db4hls.implementation.id_resource_results = db4hls.resource_results.id_resource_result
            LEFT JOIN db4hls.performance_results ON db4hls.implementation.id_performance_results = db4hls.performance_results.id_performance_result
            WHERE db4hls.configuration_space.id_configuration_space = {id} AND db4hls.performance_results.average_latency > 0
            """

            cursor.execute(query)
        
            result = cursor.fetchall()

            if k > 24:
                num_kernels = num_kernels + len(result)
                self.index = self.index + len(result)
                continue
   
            cfg_edges, cfg_select, num_bbs, bb_batch = self.buildCFG(kernelName)
            self.cfg_edge_index[kernelName] = cfg_edges
            self.cfg_select[kernelName] = cfg_select
            self.num_bbs[kernelName] = num_bbs
            self.bb_batch[kernelName] = bb_batch

            print(f"Processing kernel: {kernelName}")
            num_processes = 12

            # for i in tqdm(range(0, len(result), num_processes), desc="Generating Graphs", unit=f"{num_processes} configurations"):
                # num_data = min(num_processes, len(result) - i)
                # with multiprocessing.Pool(processes=num_processes) as pool:
                #     # input = [(kernelName, i, j, result[i+j], biEdges, valid_results) for j in range(num_processes)]
                #     input = [(kernelName, i, j, result, biEdges, False) for j in range(num_data)]
                #     out = pool.starmap_async(self.processRows, input, error_callback=CustomDataset.custom_error_callback)
                    
                #     # # Close the pool to prevent any more tasks from being submitted
                #     pool.close()
                #     out.wait()
                #     # Join the worker processes to clean up resources
                #     pool.join()
            

            a = time.time()
            with multiprocessing.Pool(processes=num_processes) as pool:
                # input = [(kernelName, i, j, result[i+j], biEdges, valid_results) for j in range(num_processes)]
                input = [(kernelName, result, j, num_processes, biEdges, False) for j in range(num_processes)]
                out = pool.starmap_async(self.processColumn, input, error_callback=CustomDataset.custom_error_callback)
                
                # # Close the pool to prevent any more tasks from being submitted
                pool.close()
                out.wait()
                # Join the worker processes to clean up resources
                pool.join()
            b = time.time()
            # print(b-a, " seconds")
            # print(kernelName, len(result))
            num_kernels = num_kernels + len(result)
            self.index = self.index + len(result)
                
        
        endTime = time.time()

        print(f"Time to save {num_kernels} graphs: " + str((endTime - startTime)))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.saveDir, f'data_{idx}.pt'))
        return data