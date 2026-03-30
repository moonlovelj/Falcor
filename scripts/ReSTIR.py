from falcor import *

def render_graph_ReSTIR():
    g = RenderGraph("ReSTIR")
    ReSTIR = createPass("ReSTIR")
    g.addPass(ReSTIR, "ReSTIR")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")

#     AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
#     g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    # GBufferRT = createPass("GBufferRT")
    # g.addPass(GBufferRT, "GBufferRT")

    # AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    # g.addPass(AccumulatePass, "AccumulatePass")

    # g.addEdge("GBufferRT.posW", "ReSTIR.posW")
    # g.addEdge("GBufferRT.normW", "ReSTIR.normalW")
    # g.addEdge("GBufferRT.tangentW", "ReSTIR.tangentW")
    # g.addEdge("GBufferRT.faceNormalW", "ReSTIR.faceNormalW")
    # g.addEdge("GBufferRT.texC", "ReSTIR.texC")
    # g.addEdge("GBufferRT.texGrads", "ReSTIR.texGrads")
    # g.addEdge("GBufferRT.mtlData", "ReSTIR.mtlData")
    # g.addEdge("GBufferRT.vbuffer", "ReSTIR.vbuffer")

    g.addEdge("VBufferRT.vbuffer", "ReSTIR.vbuffer")
    g.addEdge("VBufferRT.viewW", "ReSTIR.viewW")
    g.addEdge("VBufferRT.depth", "ReSTIR.depth")
    g.addEdge("VBufferRT.mvec", "ReSTIR.mvec")

    # g.markOutput("ReSTIR.color")

    # g.addEdge("ReSTIR.color", "AccumulatePass.input")
    # g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.addEdge("ReSTIR.color", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")

    return g

ReSTIR = render_graph_ReSTIR()
try: m.addGraph(ReSTIR)
except NameError: None
