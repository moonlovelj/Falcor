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

Texture2D<float2> gMotionVectors;

StructuredBuffer<Reservoir> gPrevFrameReservoir;
StructuredBuffer<Reservoir> gCurrentFrameReservoirRead;
RWStructuredBuffer<Reservoir> gCurrentFrameReservoirWrite;
RWTexture2D<float4> gTemporalReuseWY;
RWTexture2D<float4> gTemporalReusewsum;
RWTexture2D<float4> gTemporalReusephat;

Texture2D<float4> gDepth;
Texture2D<PackedHitInfo> gPrevVbuffer;

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

    float W_Y = 0;
    float Y_p_hat = 0;
    float w_sum = 0;

    // Reservoir CurrentReservoir = gReservoirRead[pixel.x + pixel.y * frameDim.x];
    const uint bufferIndex = pixel.x + pixel.y * frameDim.x;
    Reservoir currentReservoir = gCurrentFrameReservoirRead[bufferIndex];

    Reservoir r;
    InitReservoir(r);

    if (hit.isValid())
    {
        // // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayDir);

        // Create material instance at shading point.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        Reservoir neighborReservoirs[MAX_NEIGHBOR_COUNT + 1];
        ShadingData sds[MAX_NEIGHBOR_COUNT + 1];
        IMaterialInstance mis[MAX_NEIGHBOR_COUNT + 1];
        uint NumValidNeighbors = 0;


        if (currentReservoir.outputSample.isValid)
        {
            // float pHat = CalculatePhat(sd, mi, currentReservoir.outputSample.lightSample, sg);
            // float wi = pHat * currentReservoir.M;
            // UpdateReservoir(r, currentReservoir.outputSample, wi, sg, currentReservoir.M);
            sds[NumValidNeighbors] = sd;
            mis[NumValidNeighbors] = mi;
            neighborReservoirs[NumValidNeighbors++] = currentReservoir;
        }


        float2 motionVector = gMotionVectors[pixel];
        int2 prevPixel = pixel + motionVector * frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        if (isValidScreenRegion(prevPixel, frameDim))
        {
            Reservoir prevReservoir = gPrevFrameReservoir[prevPixel.x + prevPixel.y * frameDim.x];
            const HitInfo prevHit = HitInfo(gPrevVbuffer[prevPixel]);
            if (prevHit.isValid() && prevReservoir.outputSample.isValid)
            {
                const float3 prevRayDir = gScene.camera.computeNonNormalizedRayDirPinholePrevFrame(prevPixel, frameDim);
                ShadingData prevSd = loadShadingData(prevHit, prevRayDir);
                let prevMi = gScene.materials.getMaterialInstance(prevSd, lod);

                sds[NumValidNeighbors] = prevSd;
                mis[NumValidNeighbors] = prevMi;
                neighborReservoirs[NumValidNeighbors++] = prevReservoir;
            }
        }

        // for (uint i = 0; i < NumValidNeighbors; i++)
        // {
        //     float m = CalculateBalanceHeuristicMIS(i, neighborReservoirs, sds, mis, NumValidNeighbors, sg);
        //     float w_i = m * CalculatePhat(sd, mi, neighborReservoirs[i].outputSample.lightSample, sg) * neighborReservoirs[i].outputSample.W;
        //     UpdateReservoir(r, neighborReservoirs[i].outputSample, w_i, sg, neighborReservoirs[i].M);
        // }

         for (uint i = 0; i < NumValidNeighbors; i++)
        {
            // float m = CalculateBalanceHeuristicMIS(i, neighborReservoirs, sds, mis, NumValidNeighbors, sg);
            // float w_i = m * CalculatePhat(sd, mi, neighborReservoirs[i].outputSample.lightSample, sg) * neighborReservoirs[i].outputSample.W;

            float w_i = CalculatePhat(sd, mi, neighborReservoirs[i].outputSample.lightSample, sg) * neighborReservoirs[i].outputSample.W * neighborReservoirs[i].M;
            UpdateReservoir(r, neighborReservoirs[i].outputSample, w_i, sg, neighborReservoirs[i].M);
        }

        // r.M = min(r.M, kCCap);

        if (r.outputSample.isValid)
        {
            float Z = 0.f;
            for (uint i = 0; i < NumValidNeighbors; i++)
            {
                // float m = CalculateBalanceHeuristicMIS(i, neighborReservoirs, sds, mis, NumValidNeighbors, sg);
                // float w_i = m * CalculatePhat(sd, mi, neighborReservoirs[i].outputSample.lightSample, sg) * neighborReservoirs[i].outputSample.W;

                if (CalculatePhat(sds[i], mis[i], r.outputSample.lightSample, sg) > 0)
                {
                    Z += neighborReservoirs[i].M;
                }

            }

            float p_hat = CalculatePhat(sd, mi, r.outputSample.lightSample, sg);
            if (p_hat > kSmallFloat && Z > kSmallFloat)
            {
                r.outputSample.W = r.weightSum / p_hat / Z;
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
    }
    else
    {
        r = currentReservoir;
    }

    gCurrentFrameReservoirWrite[bufferIndex] = r;

    gTemporalReuseWY[pixel] = float4(float3(W_Y), 1.f);
    gTemporalReusewsum[pixel] = float4(float3(w_sum), 1.f);
    gTemporalReusephat[pixel] = float4(float3(Y_p_hat), 1.f);
}

