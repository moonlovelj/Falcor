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

static const int kSpatialReuseRadius = SPATIAL_REUSE_RADIUS;

RWStructuredBuffer<Reservoir> gReservoirRead;
RWStructuredBuffer<Reservoir> gReservoirWrite;
RWTexture2D<float4> gSpatialReuseWY;
RWTexture2D<float4> gSpatialReusewsum;
RWTexture2D<float4> gSpatialReusephat;

Texture2D<float4> gDepth;

// 32个样本的蓝噪声偏移表
static const float2 kNeighborOffsets[MAX_NEIGHBOR_COUNT] =
{
    float2(0.48579, -0.58985), float2(-0.53123, 0.52834),
    float2(0.12323, 0.78771), float2(-0.72332, -0.34252),
    float2(0.85098, 0.21857), float2(-0.16938, -0.89233),
    float2(0.41354, 0.21422), float2(-0.21995, 0.38298),
    float2(0.74861, -0.34131), float2(-0.84175, 0.04615),
    float2(0.24879, -0.20739), float2(-0.43578, -0.66952),
    float2(0.01655, -0.53938), float2(0.93839, 0.34443),
    float2(-0.95462, -0.29792), float2(0.25244, 0.96761),
    float2(0.67578, 0.6436), float2(-0.64177, -0.09908),
    float2(-0.0763, 0.03434), float2(0.53664, -0.84381),
    float2(-0.38942, -0.27863), float2(0.14364, -0.98962),
    float2(-0.47775, 0.82522), float2(0.89437, -0.1115),
    float2(-0.73717, 0.46583), float2(0.20367, -0.6628),
    float2(-0.15174, -0.01017), float2(0.53238, 0.2863),
    float2(-0.17424, 0.7188), float2(0.64042, -0.00949),
    float2(-0.6151, -0.59869), float2(0.01211, 0.39537)
};

static const uint kNeighborOffsetCount = MAX_NEIGHBOR_COUNT;
static const uint kNeighborOffsetMask = 31; // 32 - 1

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

bool NeighborRejection(in ShadingData sd, in IMaterialInstance mi, in ShadingData sdNeighbor, in IMaterialInstance miNeighbor
    , float current_depth, float neighbor_depth, float3 cam_pos)
{
    if (dot(sd.getOrientedFaceNormal(), sdNeighbor.getOrientedFaceNormal()) < 0.5f)
    {
        return true;
    }

    float centralDist = length(cam_pos-sd.posW);
    float neighborDist = length(cam_pos-sdNeighbor.posW);

    if (abs(centralDist - neighborDist) >= 0.1f * centralDist)
    {
        return true;
    }

    return false;
}

bool GetNeighborPixelInfo(const uint2 pixel, const uint2 frameDim, out ShadingData sd, out IMaterialInstance mi)
{
    const HitInfo hitInfo = HitInfo(gVBuffer[pixel]);
    if (hitInfo.isValid())
    {
        const float3 primaryRayOrigin = gScene.camera.getPosition();
        const float3 rayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);
        sd = loadShadingData(hitInfo, rayDir);
        let lod = ExplicitLodTextureSampler(0.f);
        mi = gScene.materials.getMaterialInstance(sd, lod);
        return true;
    }

    return false;
}

// float GenPairwiseMIS_canonical(Reservoir r, in ShadingData sd, in IMaterialInstance mi, in uint2 neighborPixels[MAX_NEIGHBOR_COUNT],
//     in Reservoir neighborReservoirs[MAX_NEIGHBOR_COUNT], uint NumValidNeighbors, float c_total, float c_current, uint2 frameDim, inout SampleGenerator sg)
// {
//     float m_c = c_current / c_total;
//     float m_numerator = c_current * CalculatePhat(sd, mi, r.outputSample.lightSample, sg);
//     for (uint j = 0; j < NumValidNeighbors; j++)
//     {
//         uint2 neighborPixel = neighborPixels[j];
//         ShadingData neighborSd;
//         IMaterialInstance neighborMi;
//         if (GetNeighborPixelInfo(neighborPixel, frameDim, neighborSd, neighborMi))
//         {
//             float m_denominator = m_numerator + (c_total - c_current) * CalculatePhat(neighborSd, neighborMi, neighborReservoirs[j].outputSample.lightSample, sg);
//             if (m_denominator > 0.f)
//             {
//                 m_c += neighborReservoirs[j].M / c_total * (m_numerator / max(FLT_MIN, m_denominator));
//             }
//         }
//     }

//     return m_c;
// }

// float GenPairwiseMIS_noncanonical( in ShadingData sd, in IMaterialInstance mi, Reservoir neighbor_r,
//     float c_total, float c_current, uint2 neighbor_pixel, uint2 frameDim, inout SampleGenerator sg)
// {
//     ShadingData neighborSd;
//     IMaterialInstance neighborMi;
//     if (GetNeighborPixelInfo(neighbor_pixel, frameDim, neighborSd, neighborMi))
//     {
//         float m_numerator = (c_total-c_current) * CalculatePhat(neighborSd, neighborMi, neighbor_r.outputSample.lightSample, sg);
//         float m_denominator = m_numerator + c_current * CalculatePhat(sd, mi, neighbor_r.outputSample.lightSample, sg);
//         if (m_denominator > 0.f)
//         {
//             return neighbor_r.M / c_total * (m_numerator / max(FLT_MIN, m_denominator));
//         }
//     }

//     return 0.f;
// }

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

    Reservoir CurrentReservoir = gReservoirRead[pixel.x + pixel.y * frameDim.x];

    if (hit.isValid() && CurrentReservoir.outputSample.isValid)
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
        uint2 neighborPixels[MAX_NEIGHBOR_COUNT + 1];
        ShadingData sds[MAX_NEIGHBOR_COUNT + 1];
        IMaterialInstance mis[MAX_NEIGHBOR_COUNT + 1];
        uint NumValidNeighbors = 0;

        float c_total = 0.f;
        float c_current = 0.f;

        uint startIndex = uint(sampleNext1D(sg) * kNeighborOffsetCount);
        for (uint i = 0; i < kSpatialReuseSampleCount && i < MAX_NEIGHBOR_COUNT; i++)
        {
            uint offsetIndex = (startIndex + i) & kNeighborOffsetMask;
            float2 offset = kNeighborOffsets[offsetIndex];
            int2 neighborPixel = int2(pixel + offset * kSpatialReuseRadius);

            if (all(neighborPixel == pixel)) continue;
            if (!isValidScreenRegion(neighborPixel, frameDim)) continue;

            Reservoir neighborReservoir = gReservoirRead[neighborPixel.x + neighborPixel.y * frameDim.x];
            if (neighborReservoir.outputSample.isValid)
            {
                ShadingData neighborSd;
                IMaterialInstance neighborMi;
                if (GetNeighborPixelInfo(neighborPixel, frameDim, neighborSd, neighborMi) &&
                    !NeighborRejection(sd, mi, neighborSd, neighborMi, gDepth[pixel].r, gDepth[neighborPixel].r, primaryRayOrigin))
                {
                    neighborPixels[NumValidNeighbors] = neighborPixel;
                    sds[NumValidNeighbors] = neighborSd;
                    mis[NumValidNeighbors] = neighborMi;
                    neighborReservoirs[NumValidNeighbors++] = neighborReservoir;
                    c_total += neighborReservoir.M;
                }
            }
        }

        c_current = CurrentReservoir.M;
        c_total += c_current;

        neighborPixels[NumValidNeighbors] = pixel;
        sds[NumValidNeighbors] = sd;
        mis[NumValidNeighbors] = mi;
        neighborReservoirs[NumValidNeighbors++] = CurrentReservoir;

        Reservoir r;
        InitReservoir(r);

        for (uint i = 0; i < NumValidNeighbors; i++)
        {
            // float m = CalculateBalanceHeuristicMIS(i, neighborReservoirs, sds, mis, NumValidNeighbors, sg);
            // float w_i = m * CalculatePhat(sd, mi, neighborReservoirs[i].outputSample.lightSample, sg) * neighborReservoirs[i].outputSample.W;

            float w_i = CalculatePhat(sd, mi, neighborReservoirs[i].outputSample.lightSample, sg) * neighborReservoirs[i].outputSample.W * neighborReservoirs[i].M;
            UpdateReservoir(r, neighborReservoirs[i].outputSample, w_i, sg, neighborReservoirs[i].M);
        }

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
            Y_p_hat = p_hat * 0.01;
            w_sum = r.weightSum;
            //
        }

        gReservoirWrite[pixel.x + pixel.y * frameDim.x] = r;
    }
    else
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
        gReservoirWrite[pixel.x + pixel.y * frameDim.x] = CurrentReservoir;
    }

    gSpatialReuseWY[pixel] = float4(float3(W_Y), 1.f);
    gSpatialReusewsum[pixel] = float4(float3(w_sum), 1.f);
    gSpatialReusephat[pixel] = float4(float3(Y_p_hat), 1.f);
}

