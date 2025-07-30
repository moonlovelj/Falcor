add_plugin(ReSTIR)

target_sources(ReSTIR PRIVATE
    ReSTIR.cpp
    ReSTIR.h
    PathTracing.rt.slang
    ReSTIRGenerateSamples.rt.slang
    ReSTIRTemporalReuse.rt.slang
    ReSTIRSpatialReuse.rt.slang
    ReSTIRShading.rt.slang
    ReSTIRCommon.slang
    ReSTIRStructs.slangh
    ReSTIRLightHelpers.slang
)

target_copy_shaders(ReSTIR RenderPasses/ReSTIR)

target_source_group(ReSTIR "RenderPasses")
