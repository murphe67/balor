
class Encoder:
    pass

one_hot_encoder_method = "one_hot"
normalize_float_method = "normalized"
log2_normalized_method = "log_normalized"
index_method = "index"

baseline_nodes = ["[external]", "alloca", "store", "load", "icmp", "br", "getelementptr", "sub", "mul", "add", "fsub", "fmul", "fadd", "sext", "ret", "bitcast", "truncate", "call", "div", "sitofp", ";undefinedfunction", "fcmp", "zext", "trunc", "ashr", "and", "or", "xor", "shl", "fdiv", "phi", "fneg"]

pragma_nodes = ["resourceAllocation_bram2p", "resourceAllocation_bram1p", "cyclicArrayPartition1", "cyclicArrayPartition2", "blockArrayPartition1", "blockArrayPartition2", "completeArrayPartition1", "completeArrayPartition2", "unroll"]

type_nodes = ["f32", "i64", "i32", "i1", "i8", "::prob_t[140][64]", "::prob_t[140]", "::int32_t[2048]", "::uint8_t[32]", "const::uint8_t[256]", "double[64]", "int[8]","double[3]", "double[192]", "double[4096]", "double[832]", "void", "float[64][64]", "float[64]"]

baseline_edges = ["dataflow", "call", "control"]

converted_nodes = ["[external]", "ret", "externalArray", "externalScalar", "localScalar", "localArray", "globalArray", "arrayParameter", "store", "load", "cmp", "br", "sub", "mul", "getelementptr", "add", "specifyAddress", "div", "call", "sitofp", "cos", "sin", "ashr", "and", "or", "xor", "shl", "phi", "fneg", "exp", "sqrt"]
converted_edges = ["dataflow", "control", "call", "address"]

conversion_args = " --allocas_to_mem_elems --remove_sexts --remove_single_target_branches --drop_func_call_proc"



def getKeyTextEncoder(tags):
    keyTextEncoder = Encoder()
    keyTextEncoder.type = "node"
    keyTextEncoder.label = "keyText"
    keyTextEncoder.method = one_hot_encoder_method
    keyTextEncoder.tags = tags

    return keyTextEncoder

def getNumericEncoder():
    numericEncoder = Encoder()
    numericEncoder.type = "node"
    numericEncoder.label = "numeric"
    numericEncoder.method = normalize_float_method
    numericEncoder.tags = ["256"]
    return numericEncoder

def getUnrollFactorEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "fullUnrollFactor"
    encoder.method = normalize_float_method
    encoder.tags = ["256"]
    return encoder


def getHierarchicalUnrollFactorEncoder(index):
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = f"unrollFactor{index+1}"
    encoder.method = normalize_float_method
    encoder.tags = ["256"]
    return encoder

def getPartitionFactorEncoder(dim):
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = f"partitionFactor{dim}"
    encoder.method = normalize_float_method
    encoder.tags = ["256"]
    return encoder


def getFlowEncoder(tags, indexFlowType):
    flowEncoder = Encoder()
    flowEncoder.type = "edge"
    flowEncoder.label = "flowType"
    if indexFlowType:
        flowEncoder.method = index_method
    else:
        flowEncoder.method = one_hot_encoder_method
    flowEncoder.tags = tags
    return flowEncoder

def getBBEncoder():
    bbIDEncoder = Encoder()
    bbIDEncoder.type = "node"
    bbIDEncoder.label = "bbID"
    bbIDEncoder.method = one_hot_encoder_method
    bbIDEncoder.tags = []
    for i in range(0, 200):
        bbIDEncoder.tags.append(str(i))

    return bbIDEncoder

def getNodeTypeEncoder():
    nodeTypeEncoder = Encoder()
    nodeTypeEncoder.type = "node"
    nodeTypeEncoder.label = "nodeType"
    nodeTypeEncoder.method = one_hot_encoder_method
    nodeTypeEncoder.tags = ["instruction", "pragma", "variable", "constant"]

    return nodeTypeEncoder

def getFuncIDEncoder():
    funcIDEncoder = Encoder()
    funcIDEncoder.type = "node"
    funcIDEncoder.label = "funcID"
    funcIDEncoder.method = one_hot_encoder_method
    funcIDEncoder.tags = []
    for i in range(0, 20):
        funcIDEncoder.tags.append(str(i))

    return funcIDEncoder

def getPositionEncoder():
    positionEncoder = Encoder()
    positionEncoder.type = "edge"
    positionEncoder.label = "edgeOrder"
    positionEncoder.method = one_hot_encoder_method
    positionEncoder.tags = []
    for i in range(0, 20):
        positionEncoder.tags.append(str(i))

    return positionEncoder

def getTypeEncoder(proxyPrograml, reduceIteratorBitwidth):
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "datatype"
    encoder.method = one_hot_encoder_method
    encoder.tags = list(type_nodes)
    encoder.tags.append("NA")
    if proxyPrograml:
        encoder.tags.append("double")
    else:
        encoder.tags.append("f64")

    if reduceIteratorBitwidth:
        bitwidths = ["i" + str(j) for j in range(13)]
        encoder.tags += bitwidths


    return encoder

def getPartitionEncoder(dim):
    partitionEncoder = Encoder()
    partitionEncoder.type = "node"
    partitionEncoder.label = f"partition{dim}"
    partitionEncoder.method = one_hot_encoder_method
    partitionEncoder.tags = ["none", "cyclic", "block", "complete"]

    return partitionEncoder

def getInlinedEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "inlined"
    encoder.method = one_hot_encoder_method
    encoder.tags = ["not_inlined", "inlined"]

    return encoder

def getInlinedEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "inlined"
    encoder.method = one_hot_encoder_method
    encoder.tags = ["not_inlined", "inlined"]

    return encoder

def getNumCallsEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "numCalls"
    encoder.method = normalize_float_method
    encoder.tags = ["256"]

    return encoder

def getNumCallSitesEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "numCallSites"
    encoder.method = normalize_float_method
    encoder.tags = ["8"]

    return encoder

def getDataTypeEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "datatype"
    encoder.method = one_hot_encoder_method
    encoder.tags = ["int", "float", "NA", "void"]

    return encoder

def getBitwidthEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "bitwidth"
    encoder.method = normalize_float_method
    encoder.tags = ["64"]

    return encoder

def getTripcountEncoder():
    encoder = Encoder()
    encoder.type = "node"
    encoder.label = "tripcount"
    encoder.method = log2_normalized_method
    encoder.tags = ["18"]

    return encoder

class Config():
    def __init__(self):
        self.proxyPrograml = True

        self.encodeBBID = True
        self.encodeFuncID = True
        self.encodeEdgeOrder = True
        self.encodeNodeType = True

        self.absorbTypes = False
        self.absorbPragmas = False

        self.convertAllocas = False

        self.ignoreControlFlow = False
        self.memoryOnlyControlFlow = False

        self.addNumCalls = False
        self.reduceIteratorBitwidths = False

        self.oneHotEncodeTypes = True
        self.indexFlowType = False

        self.encodeTripcount = False

        self.hierarchicalUnroll = False

        self.invocation = "~/Documents/AIR/bin/AIR --hide_values"

    def save(self):
        self.encoders = []

        if self.proxyPrograml:
            self.invocation += " --proxy_programl"

        if self.encodeNodeType:
            self.invocation += " --add_node_type"
            self.encoders.append(getNodeTypeEncoder())


        #need for cfg
        self.invocation += " --add_bb_id"
        if self.encodeBBID:
            self.encoders.append(getBBEncoder())


        if self.encodeFuncID:
            self.encoders.append(getFuncIDEncoder())
            self.invocation += " --add_func_id"

        if self.encodeEdgeOrder:
            self.invocation += " --add_edge_order"
            self.encoders.append(getPositionEncoder())

        if self.convertAllocas:
            self.invocation += conversion_args
            self.keyTextTags = list(converted_nodes)
            self.flowTags = list(converted_edges)
        else:
            self.keyTextTags = list(baseline_nodes)
            self.flowTags =  list(baseline_edges)

        if self.oneHotEncodeTypes:
            if self.absorbTypes:
                self.encoders.append(getTypeEncoder(self.proxyPrograml, self.reduceIteratorBitwidths))
                self.invocation += " --absorb_types --one_hot_types"
            else:
                self.keyTextTags += list(type_nodes)
                if self.proxyPrograml:
                    self.keyTextTags += ["double"]
                else:
                    self.keyTextTags += ["f64"]
                self.invocation += " --one_hot_types"
        else:
            assert(self.absorbTypes)
            self.invocation += " --absorb_types"
            
            self.encoders.append(getDataTypeEncoder())
            self.encoders.append(getBitwidthEncoder())

        if self.absorbPragmas:
            self.invocation += " --absorb_pragmas"
            self.encoders.append(getPartitionEncoder(1))
            self.encoders.append(getPartitionEncoder(2))
            self.encoders.append(getInlinedEncoder())
            self.encoders.append(getUnrollFactorEncoder())
            self.encoders.append(getPartitionFactorEncoder(1))
            self.encoders.append(getPartitionFactorEncoder(2))
        else:
            self.encoders.append(getNumericEncoder())
            self.keyTextTags +=  list(pragma_nodes)
            self.flowTags += list(["pragma"])

        if self.ignoreControlFlow:
            self.invocation += " --ignore_control_flow --ignore_call_edges"
        else:
            self.encoders.append(getFlowEncoder(self.flowTags, self.indexFlowType))

        if self.memoryOnlyControlFlow:
            self.invocation += " --only_memory_control_flow"

        if self.addNumCalls:
            self.invocation += " --add_num_calls"
            self.encoders.append(getNumCallsEncoder())
            self.encoders.append(getNumCallSitesEncoder())

        if self.reduceIteratorBitwidths:
            self.invocation += " --reduce_iterator_bitwidth"

        if self.encodeTripcount:
            self.encoders.append(getTripcountEncoder())

        if self.hierarchicalUnroll:
            for i in range(3):
                self.encoders.append(getHierarchicalUnrollFactorEncoder(i))


        self.encoders.append(getKeyTextEncoder(self.keyTextTags))


        with open("configs/" + self.filename, 'w') as file:
            file.write(f"{self.invocation}\n")
            
            for encoder in self.encoders:
                output = f"{encoder.type} {encoder.label} {encoder.method}"

                for tag in encoder.tags:
                    output = output + f" {tag}"
                output = output + "\n"
                file.write(output)

    def output(self):
        for encoder in self.encoders:
            print(encoder.label)
        quit()

class Baseline_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "baseline.txt"
        
class A1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "A1.txt"
        self.absorbPragmas = True

class A2_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "A2.txt"
        self.encodeTripcount = True

class B1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "B1.txt"
        self.absorbPragmas = True
        self.encodeTripcount = True

class C1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "C1.txt"
        self.absorbPragmas = True
        self.encodeTripcount = True
        self.absorbTypes = True


class D1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "D1.txt"
        self.absorbPragmas = True
        self.encodeTripcount = True
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False

class E1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "E1.txt"
        self.absorbPragmas = True
        self.encodeTripcount = True
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.encodeTripcount = True
        self.hierarchicalUnroll = True

class F1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "F1.txt"
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False

class G1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "G1.txt"
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.absorbPragmas = True

class H1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "H1.txt"
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.absorbPragmas = True
        self.hierarchicalUnroll = True

class J1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "J1.txt"
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.absorbPragmas = True
        self.hierarchicalUnroll = True
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False
        self.oneHotEncodeTypes = False
        
class K1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "K1.txt"
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.absorbPragmas = True
        self.hierarchicalUnroll = True
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False

class L1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "L1.txt"
        self.absorbTypes = True
        self.convertAllocas = True
        self.proxyPrograml = False
        self.encodeNodeType = False
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False

class M1_Config(Config):
    def __init__(self):
        super().__init__()
        self.filename = "M1.txt"
        self.encodeNodeType = False
        self.encodeBBID = False
        self.encodeFuncID = False
        self.encodeEdgeOrder = False

if __name__ == "__main__":
    config1 = Baseline_Config()
    config1.save()

    config2 = A1_Config()
    config2.save()

    config3 = A2_Config()
    config3.save()

    config4 = B1_Config()
    config4.save()

    config5 = C1_Config()
    config5.save()

    config6 = D1_Config()
    config6.save()

    config7 = E1_Config()
    config7.save()

    config8 = F1_Config()
    config8.save()

    config9 = G1_Config()
    config9.save()

    config10 = H1_Config()
    config10.save()

    config11 = J1_Config()
    config11.save()

    config12 = K1_Config()
    config12.save()

    config13 = L1_Config()
    config13.save()

    config14 = M1_Config()
    config14.save()