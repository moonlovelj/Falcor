add_plugin(ReSTIR)

target_sources(ReSTIR PRIVATE
    ReSTIR.cpp
    ReSTIR.h
    PathTracing.rt.slang
    GenerateSamples.rt.slang
    Shading.rt.slang
    ReSTIRCommon.slang
    ReSTIRStructs.slangh
)

target_copy_shaders(ReSTIR RenderPasses/ReSTIR)

target_source_group(ReSTIR "RenderPasses")
