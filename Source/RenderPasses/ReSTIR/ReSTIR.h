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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Materials/TexLODTypes.slang" // Using the enum with Mip0, RayCones, etc
#include "Core/API/Buffer.h"
#include "Core/Pass/ComputePass.h"


using namespace Falcor;

class ReSTIR : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIR, "ReSTIR", "Insert pass description here.");

    static ref<ReSTIR> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<ReSTIR>(pDevice, props);
    }

    ReSTIR(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void prepareSamplesVars();
    void prepareVars();
    void setStaticParams(Program* pProgram) const;
    void generateSamples(RenderContext* pRenderContext, const RenderData& renderData);

    ref<Scene>                      mpScene;
     /// GPU sample generator.
    ref<SampleGenerator>            mpSampleGenerator;
    bool                            mRecompile = false;

    std::unique_ptr<EnvMapSampler>  mpEnvMapSampler = nullptr;            ///< Environment map sampler or nullptr if not used.
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler  = nullptr;    ///< Emissive light sampler or nullptr if not used.

    ref<ParameterBlock>             mpReSTIRDataBlock;

    sigs::Connection                mUpdateFlagsConnection; ///< Connection to the UpdateFlags signal.
    IScene::UpdateFlags             mUpdateFlags = IScene::UpdateFlags::None;

    ref<ComputePass>                mpGenerateSamples;
    ref<Buffer>                     mpSamplesBuffer;
    // ref<Texture>                    mpSamplesTexture;

        // Ray tracing program.
    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mTracer;

    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mSamplesTracer;

    uint2 mScreenDim = uint2(0, 0);

    // Frame count since scene was loaded.
    uint mFrameCount = 0;

    bool mOptionsChanged = false;

    /// Max number of indirect bounces (0 = none).
    uint mMaxBounces = 3;
    /// Compute direct illumination (otherwise indirect only).
    bool mComputeDirect = true;

    bool mUseNee = true;

    uint mCandidateNum = 1;
};
