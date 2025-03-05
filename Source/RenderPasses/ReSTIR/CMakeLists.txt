add_plugin(ReSTIR)

target_sources(ReSTIR PRIVATE
    ReSTIR.cpp
    ReSTIR.h
    ReSTIR.rt.slang
    GenerateSamples.cs.slang
    SamplesTypes.slangh
)

target_copy_shaders(ReSTIR RenderPasses/ReSTIR)

target_source_group(ReSTIR "RenderPasses")
