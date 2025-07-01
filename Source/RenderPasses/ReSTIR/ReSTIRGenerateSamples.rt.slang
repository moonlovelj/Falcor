#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"
#include "Scene/SceneDefines.slangh"
#include "ReSTIRCommon.slang"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Math.PackedFormats;
import Scene.HitInfo;
import Utils.Math.Ray;
import Rendering.Materials.InteriorList;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;

RWStructuredBuffer<Reservoir> gReservoir;
RWTexture2D<float4> gWY;
RWTexture2D<float4> gwsum;
RWTexture2D<float4> gphat;

ReSTIRSample GenerateSample(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    ReSTIRSample Sample;

    float3 outColor = 0.f;
    LightSample ls = {};
    uint lobeTypes = mi.getLobeTypes(sd);
    bool sampleUpperHemisphere = (lobeTypes & (uint)LobeType::NonDeltaReflection) != 0;
    bool sampleLowerHemisphere = (lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0;
    // Sample a light.
    bool validSample = generateLightSample(sd, sampleUpperHemisphere, sampleLowerHemisphere, sg, ls);
    if (validSample)
    {
        Sample.isValid = true;
        Sample.lightSample = ls;
    }
    else
    {
        Sample.isValid = false;
    }

    return Sample;
}


[shader("miss")] void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")] void shadowAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}



[shader("raygeneration")] void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);
    const HitInfo hit = HitInfo(gVBuffer[pixel]);

    float3 outColor = float3(0.f);

    Reservoir r;
    InitReservoir(r);
    float3 W_Y = 0;
    float3 w_sum = 0;
    float3 Y_p_hat = 0;

    if (hit.isValid())
    {
        // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);

        // Create material instance at shading point.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);


        // 目标函数 p _hat  = f * L * G * V
        // m_i = 1 / M
        // p_Xi = 光源采样pdf
        // W_Xi = 1 / p_Xi
        // w_i = m_i * p_hat * W_Xi
        // 因此w_i = (1/M)*p_hat/p_Xi
        // 假设最终选出的样本是Y，Y的无偏贡献权重W_Y= r.weightSum / p_hat_Y;
        if (kComputeDirect)
        {
            const float m_i = 1.f / kCandidateNum;
            for (uint i = 0; i < kCandidateNum; i++)
            {
                ReSTIRSample sample = GenerateSample(sd, mi, sg);
                if (sample.isValid)
                {
                    float p_hat = CalculatePhat(sd, mi, sample.lightSample, sg);
                    float w_i = m_i * p_hat / sample.lightSample.pdf;
                    UpdateReservoir(r, sample, w_i, sg, 1);
                }
            }

            if (r.outputSample.isValid)
            {
                float p_hat = CalculatePhat(sd, mi, r.outputSample.lightSample, sg);
                if (p_hat > 0)
                {
                    r.outputSample.W = r.weightSum / p_hat;
                }
                else
                {
                    r.outputSample.W = 0.f;
                }

                // debug
                W_Y = r.outputSample.W;
                Y_p_hat = p_hat;
                w_sum = r.weightSum;
                //
            }
            else
            {
                W_Y = 0;
                Y_p_hat = 0;
                w_sum = 0;
            }

            gReservoir[pixel.x + pixel.y * frameDim.x] = r;
        }
    }
    else
    {
        gReservoir[pixel.x + pixel.y * frameDim.x] = r;
    }

    gWY[pixel] = float4(float3(W_Y), 1);
    gphat[pixel] = float4(float3(Y_p_hat), 1);
    gwsum[pixel] = float4(float3(w_sum), 1);
}
