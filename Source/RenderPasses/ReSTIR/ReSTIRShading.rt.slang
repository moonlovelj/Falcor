
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

StructuredBuffer<Reservoir> gReservoir;

// Outputs
RWTexture2D<float4> gOutputColor;

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

        Reservoir r = gReservoir[pixel.x + pixel.y * frameDim.x];
        if (r.outputSample.isValid)
        {
            // const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, r.outputSample.lightSample.dir) >= 0.f ? sd.faceN : -sd.faceN);

            float3 origin;
            float3 dir;
            float distance;
            GetShadowRayData(sd, r.outputSample.lightSample, origin, dir, distance);
            // float3 toLight = normalize(r.outputSample.lightSample.posW - sd.posW);
            float3 toLight = CalcToLight(sd, r.outputSample.lightSample);

            float V = (float)traceShadowRay(origin, dir, distance);
            float G = evalLightSampleG(sd, r.outputSample.lightSample);
            outColor = (mi.eval(sd, toLight, sg)) * (r.outputSample.lightSample.Li) * G * V * r.outputSample.W;
        }
        // else
        // {
        //     if (kUseEnvLight)
        //     {
        //         float3 rayOrigin = sd.computeRayOrigin();
        //         BSDFSample bsdfSample;
        //         if (mi.sample(sd, sg, bsdfSample, kUseImportanceSampling))
        //         {
        //             if (bsdfSample.isLobe(LobeType::Transmission))
        //             {
        //                 rayOrigin = sd.computeRayOrigin(false);
        //             }
        //             bool V = traceShadowRay(rayOrigin, bsdfSample.wo, FLT_MAX);
        //             if (V)
        //             {
        //                 outColor += bsdfSample.weight * gScene.envMap.eval(bsdfSample.wo);
        //             }
        //         }
        //     }
        // }

        outColor += mi.getProperties(sd).emission;
    }
    else
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
    }

    gOutputColor[pixel] = float4(outColor, 1.f);
}

