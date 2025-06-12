/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ReSTIR.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Core/API/RenderContext.h"
#include "ReSTIRStructs.slangh"
#include "Scene/Scene.h"

namespace
{
const char kShaderFile[] = "RenderPasses/ReSTIR/PathTracing.rt.slang";
const std::string kGenerateSamplesFilename = "RenderPasses/ReSTIR/GenerateSamples.rt.slang";
const std::string kShadingFilename = "RenderPasses/ReSTIR/Shading.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const ChannelList kOutputChannels = {
    // clang-format off
        { "color",          "gOutputColor",                "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float }
    // clang-format on
};

const ChannelList kInputChannels = {
    // clang-format off
        { "vbuffer",        "gVBuffer",                    "Visibility buffer in packed format" },
        { "viewW",          "gViewW",                      "World-space view direction (xyz float format)", true /* optional */ }
    // clang-format on
};

const ChannelList kSamplesOutputChannels = {
    // clang-format off
        { "WY",           "gWY",                                    "WY", false, ResourceFormat::RGBA32Float },
        { "wsum",         "gwsum",                                  "wsum", false, ResourceFormat::RGBA32Float },
        { "phat",         "gphat",                                  "phat", false, ResourceFormat::RGBA32Float },
        // { "ReSTIRColor",  "gOutputColor",                           "ReSTIRColor", false, ResourceFormat::RGBA32Float }
    // clang-format on
};

const ChannelList kSamplesInputChannels = {
    // clang-format off
        { "vbuffer",        "gVBuffer",                    "Visibility buffer in packed format" },
        { "viewW",          "gViewW",                      "World-space view direction (xyz float format)", true /* optional */ }
    // clang-format on
};

const ChannelList kShadingOutputChannels = {
    // clang-format off
    { "ShadingColor",  "gOutputColor",                           "ReSTIRColor", false, ResourceFormat::RGBA32Float }
    // clang-format on
};

const ChannelList kShadingInputChannels = {
    // clang-format off

    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";

} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIR>();
}

ReSTIR::ReSTIR(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

Properties ReSTIR::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    return props;
}

RenderPassReflection ReSTIR::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    addRenderPassOutputs(reflector, kSamplesOutputChannels);
    addRenderPassOutputs(reflector, kShadingOutputChannels);

    return reflector;
}

void ReSTIR::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    // Check for scene changes that require shader recompilation.
    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }


    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (is_set(mUpdateFlags, IScene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            mRecompile = true;
        }
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getILightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            mpEmissiveSampler = nullptr;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        mpEmissiveSampler->update(pRenderContext, mpScene->getLightCollection(pRenderContext));
        auto defines = mpEmissiveSampler->getDefines();
        if (mSamplesTracer.pProgram->addDefines(defines))
            mRecompile = true;
        if (mTracer.pProgram->addDefines(defines))
            mRecompile = true;
    }


    // generateSamples(pRenderContext, renderData);
    {

        const auto& pOutputColor = renderData.getTexture("color");
        FALCOR_ASSERT(pOutputColor);

        uint2 ScreenDim = uint2(pOutputColor->getWidth(), pOutputColor->getHeight());
        uint elemCount = ScreenDim.x * ScreenDim.y;
        mpSamplesBuffer = mpDevice->createStructuredBuffer(sizeof(Reservoir), elemCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);

        setStaticParams(mSamplesTracer.pProgram.get());

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        // TODO: This should be moved to a more general mechanism using Slang.
        mSamplesTracer.pProgram->addDefines(getValidResourceDefines(kSamplesInputChannels, renderData));
        mSamplesTracer.pProgram->addDefines(getValidResourceDefines(kSamplesOutputChannels, renderData));

        // if (!mTracer.pVars)
        prepareSamplesVars();
        FALCOR_ASSERT(mSamplesTracer.pVars);

        // Get dimensions of ray dispatch.
        const uint2 targetDim = renderData.getDefaultTextureDims();
        FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

        auto var = mSamplesTracer.pVars->getRootVar();
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
        // Set up screen space pixel angle for texture LOD using ray cones
        // var["CB"]["gScreenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(targetDim.y);
        // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
        auto bind = [&](const ChannelDesc& desc)
        {
            if (!desc.texname.empty())
            {
                var[desc.texname] = renderData.getTexture(desc.name);
            }
        };
        for (auto channel : kInputChannels)
            bind(channel);
        for (auto channel : kSamplesOutputChannels)
            bind(channel);

        // Spawn the rays.
        mpScene->raytrace(pRenderContext, mSamplesTracer.pProgram.get(), mSamplesTracer.pVars, uint3(targetDim, 1));
    }

    {
        setStaticParams(mShadingTracer.pProgram.get());

        mShadingTracer.pProgram->addDefines(getValidResourceDefines(kSamplesInputChannels, renderData));
        mShadingTracer.pProgram->addDefines(getValidResourceDefines(kSamplesOutputChannels, renderData));

        prepareShadingVars();
        FALCOR_ASSERT(mShadingTracer.pVars);

        // Get dimensions of ray dispatch.
        const uint2 targetDim = renderData.getDefaultTextureDims();
        FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

        auto var = mShadingTracer.pVars->getRootVar();
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
        // Set up screen space pixel angle for texture LOD using ray cones
        // var["CB"]["gScreenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(targetDim.y);
        // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
        auto bind = [&](const ChannelDesc& desc)
        {
            if (!desc.texname.empty())
            {
                var[desc.texname] = renderData.getTexture(desc.name);
            }
        };
        for (auto channel : kInputChannels)
            bind(channel);
        for (auto channel : kShadingOutputChannels)
            bind(channel);

        // Spawn the rays.
        mpScene->raytrace(pRenderContext, mShadingTracer.pProgram.get(), mShadingTracer.pVars, uint3(targetDim, 1));
    }

    {
        setStaticParams(mTracer.pProgram.get());

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        // TODO: This should be moved to a more general mechanism using Slang.
        mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
        mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

        // if (!mTracer.pVars)
        prepareVars();
        FALCOR_ASSERT(mTracer.pVars);

        // Get dimensions of ray dispatch.
        const uint2 targetDim = renderData.getDefaultTextureDims();
        FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

        auto var = mTracer.pVars->getRootVar();
        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
        // Set up screen space pixel angle for texture LOD using ray cones
        // var["CB"]["gScreenSpacePixelSpreadAngle"] = mpScene->getCamera()->computeScreenSpacePixelSpreadAngle(targetDim.y);
        // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
        auto bind = [&](const ChannelDesc& desc)
        {
            if (!desc.texname.empty())
            {
                var[desc.texname] = renderData.getTexture(desc.name);
            }
        };
        for (auto channel : kInputChannels)
            bind(channel);
        for (auto channel : kOutputChannels)
            bind(channel);

        // Spawn the rays.
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    }

    mFrameCount++;
}

void ReSTIR::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.var("Candidate Num", mCandidateNum, 1u, 256u);
    widget.tooltip("Candidate Num of ReSTIR.", true);

    dirty |= widget.var("C Cap", mCCap, 1u, 40u);
    widget.tooltip("C Cap.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use Nee", mUseNee);
    widget.tooltip("Use Nee.\nIf disabled only bsdf is sampled (when max bounces > 0).", true);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void ReSTIR::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mUpdateFlagsConnection = {};
    mUpdateFlags = IScene::UpdateFlags::None;

    mSamplesTracer.pProgram = nullptr;
    mSamplesTracer.pBindingTable = nullptr;
    mSamplesTracer.pVars = nullptr;

    mShadingTracer.pProgram = nullptr;
    mShadingTracer.pBindingTable = nullptr;
    mShadingTracer.pVars = nullptr;

    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (mpScene->hasProceduralGeometry())
        {
            logWarning("ReSTIR: This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        mUpdateFlagsConnection = mpScene->getUpdateFlagsSignal().connect([&](IScene::UpdateFlags flags) { mUpdateFlags |= flags; });

        // if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        // {
        //     logError("ReSTIR: This render pass only supports triangles. Other types of geometry will be ignored.");
        // }

           // Generate samples pass
        ProgramDesc descCompute;
        descCompute.addShaderModules(mpScene->getShaderModules());
        descCompute.addTypeConformances(mpScene->getTypeConformances());
        descCompute.addShaderLibrary(kGenerateSamplesFilename).csEntry("main");
        mpGenerateSamples = ComputePass::create(mpDevice, descCompute, mpScene->getSceneDefines(), false);

        {
            // Create ray tracing program.
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kGenerateSamplesFilename);
            desc.addTypeConformances(mpScene->getTypeConformances());
            desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
            desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
            desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

            mSamplesTracer.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
            auto& sbt = mSamplesTracer.pBindingTable;
            sbt->setRayGen(desc.addRayGen("rayGen"));
            sbt->setMiss(0, desc.addMiss("shadowMiss"));
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));
            mSamplesTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        }

        {
            // ray tracing program for shading
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kShadingFilename);
            desc.addTypeConformances(mpScene->getTypeConformances());
            desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
            desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
            desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

            mShadingTracer.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
            auto& sbt = mShadingTracer.pBindingTable;
            sbt->setRayGen(desc.addRayGen("rayGen"));
            sbt->setMiss(0, desc.addMiss("shadowMiss"));
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));
            mShadingTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        }

        {
            // Create ray tracing program.
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kShaderFile);
            desc.addTypeConformances(mpScene->getTypeConformances());
            desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
            desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
            desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

            mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
            auto& sbt = mTracer.pBindingTable;
            sbt->setRayGen(desc.addRayGen("rayGen"));
            sbt->setMiss(0, desc.addMiss("scatterMiss"));
            sbt->setMiss(1, desc.addMiss("shadowMiss"));

            // if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
            {
                sbt->setHitGroup(
                    0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("scatterClosestHit", "scatterAnyHit")
                );
                sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));
            }

            mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        }
    }
}

void ReSTIR::prepareSamplesVars()
{
     FALCOR_ASSERT(mSamplesTracer.pProgram);
     {
        // Configure program.
        mSamplesTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
        mSamplesTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

        // Create program variables for the current program.
        // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
        mSamplesTracer.pVars = RtProgramVars::create(mpDevice, mSamplesTracer.pProgram, mSamplesTracer.pBindingTable);

        auto reflector = mSamplesTracer.pProgram->getReflector()->getParameterBlock("gReSTIRData");
        mpReSTIRDataBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(mpReSTIRDataBlock);

        // Bind utility classes into shared data.
        auto var = mSamplesTracer.pVars->getRootVar();
        var["gReservoir"] = mpSamplesBuffer;
        mpSampleGenerator->bindShaderData(var);

        if (mpEnvMapSampler && mpReSTIRDataBlock)
            mpEnvMapSampler->bindShaderData(mpReSTIRDataBlock->getRootVar()["envMapSampler"]);

        if (mpEmissiveSampler && mpReSTIRDataBlock)
            mpEmissiveSampler->bindShaderData(mpReSTIRDataBlock->getRootVar()["emissiveSampler"]);
    }

}

void ReSTIR::prepareShadingVars()
{
     FALCOR_ASSERT(mShadingTracer.pProgram);
     {
        // Configure program.
        mShadingTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
        mShadingTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

        // Create program variables for the current program.
        // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
        mShadingTracer.pVars = RtProgramVars::create(mpDevice, mShadingTracer.pProgram, mShadingTracer.pBindingTable);

        auto reflector = mShadingTracer.pProgram->getReflector()->getParameterBlock("gReSTIRData");
        ref<ParameterBlock>  ReSTIRDataBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(ReSTIRDataBlock);

        // Bind utility classes into shared data.
        auto var = mShadingTracer.pVars->getRootVar();
        var["gReservoir"] = mpSamplesBuffer;
        mpSampleGenerator->bindShaderData(var);

        if (mpEnvMapSampler && ReSTIRDataBlock)
            mpEnvMapSampler->bindShaderData(ReSTIRDataBlock->getRootVar()["envMapSampler"]);

        if (mpEmissiveSampler && ReSTIRDataBlock)
            mpEmissiveSampler->bindShaderData(ReSTIRDataBlock->getRootVar()["emissiveSampler"]);
    }

}

void ReSTIR::prepareVars()
{

    FALCOR_ASSERT(mTracer.pProgram);
    {
        // Configure program.
        mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
        mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

        // Create program variables for the current program.
        // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
        mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

        auto reflector = mTracer.pProgram->getReflector()->getParameterBlock("gReSTIRData");
        mpReSTIRDataBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(mpReSTIRDataBlock);

        // Bind utility classes into shared data.
        auto var = mTracer.pVars->getRootVar();
        // var["gSamples"] = mpSamplesBuffer;
        mpSampleGenerator->bindShaderData(var);

        if (mpEnvMapSampler && mpReSTIRDataBlock)
            mpEnvMapSampler->bindShaderData(mpReSTIRDataBlock->getRootVar()["envMapSampler"]);

        if (mpEmissiveSampler && mpReSTIRDataBlock)
            mpEmissiveSampler->bindShaderData(mpReSTIRDataBlock->getRootVar()["emissiveSampler"]);
    }
}

void ReSTIR::setStaticParams(Program* pProgram) const
{
    DefineList defines;
    defines.add("MAX_BOUNCES", std::to_string(mMaxBounces));
    defines.add("CANDIDATE_NUM", std::to_string(mCandidateNum));
    defines.add("C_CAP", std::to_string(mCCap));
    defines.add("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    defines.add("USE_NEE", mUseNee ? "1" : "0");
    // defines.add("TEX_LOD_MODE", std::to_string(static_cast<uint32_t>(mTexLODMode)));
    // defines.add("RAY_CONE_MODE", std::to_string(static_cast<uint32_t>(mRayConeMode)));
    // defines.add("VISUALIZE_SURFACE_SPREAD", mVisualizeSurfaceSpread ? "1" : "0");
    // defines.add("RAY_CONE_FILTER_MODE", std::to_string(static_cast<uint32_t>(mRayConeFilterMode)));
    // defines.add("RAY_DIFF_FILTER_MODE", std::to_string(static_cast<uint32_t>(mRayDiffFilterMode)));
    defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    defines.add("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    // defines.add("USE_ROUGHNESS_TO_VARIANCE", mUseRoughnessToVariance ? "1" : "0");
    // defines.add("USE_FRESNEL_AS_BRDF", mUseFresnelAsBRDF ? "1" : "0");
    pProgram->addDefines(defines);
}

void ReSTIR::generateSamples(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "generateSamples");

    const auto& pOutputColor = renderData.getTexture("color");
    FALCOR_ASSERT(pOutputColor);

    uint2 ScreenDim = uint2(pOutputColor->getWidth(), pOutputColor->getHeight());
    bool isScreenDimChanged = any(mScreenDim != ScreenDim);
    mScreenDim = ScreenDim;

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = 16 * 16;
    FALCOR_ASSERT(mpGenerateSamples->getThreadGroupSize().x == tileSize);
    FALCOR_ASSERT(mpGenerateSamples->getThreadGroupSize().y == 1 && mpGenerateSamples->getThreadGroupSize().z == 1);
    uint2 screenTiles = div_round_up(ScreenDim, {16,16});

    // Additional specialization. This shouldn't change resource declarations.
    // mpGenerateSamples->addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
    // mpGenerateSamples->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");
    // mpGenerateSamples->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");
    // mpGenerateSamples->addDefine("OUTPUT_NRD_ADDITIONAL_DATA", mOutputNRDAdditionalData ? "1" : "0");

    uint32_t elemCount = screenTiles.x * tileSize * screenTiles.y;

    if (isScreenDimChanged)
    {
        mpSamplesBuffer = mpDevice->createStructuredBuffer(sizeof(Reservoir), elemCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
        // mpSamplesTexture = mpDevice->createTexture2D(screenTiles.x * 16, screenTiles.y * 16, ResourceFormat::RGBA32Uint, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    }

    mpGenerateSamples->setVars(nullptr);
    mpGenerateSamples->getRootVar()["gReservoir"] = mpSamplesBuffer;
    // mpGenerateSamples->getRootVar()["gOutputTexture"] = mpSamplesTexture;
    // mpGenerateSamples->setVars(nullptr);
    // Bind resources.
    // auto var = mpGeneratePaths->getRootVar()["CB"]["gPathGenerator"];
    // bindShaderData(var, renderData, false);

    // mpScene->bindShaderData(mpGeneratePaths->getRootVar()["gScene"]);


    // Launch one thread per pixel.
    // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.


    mpGenerateSamples->execute(pRenderContext, { screenTiles.x * tileSize, screenTiles.y, 1u });
}
