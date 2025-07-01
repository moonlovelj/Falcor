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

#define MAX_NEIGHBOR_COUNT 32

RWStructuredBuffer<Reservoir> gReservoirRead;
RWStructuredBuffer<Reservoir> gReservoirWrite;
RWTexture2D<float4> gSpatialReuseWY;
RWTexture2D<float4> gSpatialReusewsum;
RWTexture2D<float4> gSpatialReusephat;

Texture2D<float4> gDepth;

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

int2 RandomPixelNeighborPixel(int2 currentPixel, uint2 frameDim, inout SampleGenerator sg)
{
    int Radius = kSpatialReuseRadius;

    int2 offset = 0;
    // 使用循环来避免选择自己，这在统计上是纯粹的
    // 由于选中自己的概率极低，这个循环几乎永远不会执行超过一次
    do {
        // 2. 生成 [-R, R] 范围内的随机整数偏移
        float rand_x = sampleNext1D(sg); // [0, 1)
        float rand_y = sampleNext1D(sg); // [0, 1)

        offset.x = int(floor(rand_x * (2 * Radius + 1))) - Radius;
        offset.y = int(floor(rand_y * (2 * Radius + 1))) - Radius;

    } while (offset.x == 0 && offset.y == 0);

    // 3. 计算邻居坐标
    int2 neighbor_coord = currentPixel + offset;

    // 4. 关键：处理屏幕边界
    //    使用 clamp 将坐标限制在 [0, screen_dim-1] 的有效范围内
    neighbor_coord.x = clamp(neighbor_coord.x, 0, (int)frameDim.x - 1);
    neighbor_coord.y = clamp(neighbor_coord.y, 0, (int)frameDim.y - 1);

    return neighbor_coord;
}

bool NeighborRejection(in ShadingData sd, in IMaterialInstance mi, in ShadingData sdNeighbor, in IMaterialInstance miNeighbor
    , float current_depth, float neighbor_depth)
{
    if (dot(sd.getOrientedFaceNormal(), sdNeighbor.getOrientedFaceNormal()) < 0.9f)
    {
        return true;
    }

    if (length(sd.posW - sdNeighbor.posW) > 200.f)
    {
        return true;
    }

    if (abs(current_depth - neighbor_depth) > 0.01f)
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
        sd = loadShadingData(hitInfo, primaryRayOrigin, rayDir);
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

/**
 * 使用平衡启发式，计算指定候选样本的 MIS 权重。
 * @param candidate_idx 要计算权重的候选样本在 all_reservoirs 数组中的索引。
 * @param all_reservoirs 包含所有候选者（canonical + non-canonical）的 Reservoir 数组。
 * @param all_contexts 包含所有候选者对应上下文（ShadingData, etc.）的数组。
 * @param num_total_candidates 候选者总数。
 */
float CalculateBalanceHeuristicMIS(
    uint candidate_idx,
    const uint2 frameDim,
    const Reservoir all_reservoirs[MAX_NEIGHBOR_COUNT + 1],
    const uint2 all_pixels[MAX_NEIGHBOR_COUNT + 1],
    const uint num_total_candidates,
    inout SampleGenerator in_sg
)
{
    uint2 origin_pixel = all_pixels[candidate_idx];
    Reservoir r_origin = all_reservoirs[candidate_idx];
    float c_origin = r_origin.M;
    ShadingData sd_origin;
    IMaterialInstance mi_origin;
    if (!GetNeighborPixelInfo(origin_pixel, frameDim, sd_origin, mi_origin))
    {
        return 0.f;
    }

    // --- 分子 ---
    float numerator = c_origin * CalculatePhat(sd_origin, mi_origin, r_origin.outputSample.lightSample, in_sg);

    // --- 分母 ---
    float denominator = 0.f;
    for (uint k = 0; k < num_total_candidates; k++)
    {
        Reservoir r_other = all_reservoirs[k];
        float c_other = r_other.M;

        ShadingData sd_other;
        IMaterialInstance mi_other;
        if (GetNeighborPixelInfo(all_pixels[k], frameDim, sd_other, mi_other))
        {
            // 计算其他候选者的贡献
            if (k == candidate_idx)
            {
                // 如果是原始候选者，使用 numerator
                denominator += numerator;
            }
            else
            {
                // 否则，使用其他候选者的上下文
                denominator += c_other * CalculatePhat(sd_other, mi_other, r_other.outputSample.lightSample, in_sg);
            }
        }
    }

    if (denominator > 0.f)
    {
        return numerator / max(FLT_MIN, denominator);
    }

    return 0.f;
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

    Reservoir CurrentReservoir = gReservoirRead[pixel.x + pixel.y * frameDim.x];

    if (hit.isValid() && CurrentReservoir.outputSample.isValid)
    {
        // // Load shading data.
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

        Reservoir neighborReservoirs[MAX_NEIGHBOR_COUNT + 1];
        uint2 neighborPixels[MAX_NEIGHBOR_COUNT + 1];
        uint NumValidNeighbors = 0;

        float c_total = 0.f;
        float c_current = 0.f;

        for (uint i = 0; i < kSpatialReuseSampleCount && i < MAX_NEIGHBOR_COUNT; i++)
        {
            uint2 neighborPixel = RandomPixelNeighborPixel(pixel, frameDim, sg);
            Reservoir neighborReservoir = gReservoirRead[neighborPixel.x + neighborPixel.y * frameDim.x];
            if (neighborReservoir.outputSample.isValid)
            {
                ShadingData neighborSd;
                IMaterialInstance neighborMi;
                if (GetNeighborPixelInfo(neighborPixel, frameDim, neighborSd, neighborMi) &&
                    !NeighborRejection(sd, mi, neighborSd, neighborMi, gDepth[pixel].r, gDepth[neighborPixel].r))
                {
                    neighborPixels[NumValidNeighbors] = neighborPixel;
                    neighborReservoirs[NumValidNeighbors++] = neighborReservoir;
                    c_total += neighborReservoir.M;
                }
            }
        }

        c_current = CurrentReservoir.M;
        c_total += c_current;

        neighborPixels[NumValidNeighbors] = pixel;
        neighborReservoirs[NumValidNeighbors++] = CurrentReservoir;

        Reservoir r;
        InitReservoir(r);

        for (uint i = 0; i < NumValidNeighbors; i++)
        {
            uint2 neighborPixel = neighborPixels[i];
            ShadingData neighborSd;
            IMaterialInstance neighborMi;
            if (GetNeighborPixelInfo(neighborPixel, frameDim, neighborSd, neighborMi))
            {
                float m = CalculateBalanceHeuristicMIS(i, frameDim, neighborReservoirs, neighborPixels, NumValidNeighbors, sg);
                float w_i = m * CalculatePhat(sd, mi, neighborReservoirs[i].outputSample.lightSample, sg) * neighborReservoirs[i].outputSample.W;
                UpdateReservoir(r, neighborReservoirs[i].outputSample, w_i, sg, neighborReservoirs[i].M);
            }
        }

        if (r.outputSample.isValid)
        {
            float p_hat = CalculatePhat(sd, mi, r.outputSample.lightSample, sg);
            if (p_hat > 0.f)
            {
                r.outputSample.W = r.weightSum / max(FLT_MIN, p_hat);
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

