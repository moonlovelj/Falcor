#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"
#include "Scene/SceneDefines.slangh"
#include "SamplesTypes.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Utils.Math.PackedFormats;
import Scene.HitInfo;
import Utils.Math.Ray;
import Rendering.Materials.InteriorList;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;

#define is_valid(name) (is_valid_##name != 0)
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const float kRayTMax = FLT_MAX;
static const uint kCandidateNum = CANDIDATE_NUM;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;

static const float3 kDefaultBackgroundColor = float3(0.f, 0.f, 0.f);
static const float kSmallNumber = 1e-6f;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    uint2 gOutputDim;
}

Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

RWStructuredBuffer<ReSTIRSample> gOutputSamples;
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gWY;
RWTexture2D<float4> gwsum;
RWTexture2D<float4> gphat;


struct ReSTIRData
{
    // Samplers
    EnvMapSampler envMapSampler;          ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.
};

ParameterBlock<ReSTIRData> gReSTIRData;

/** Types of samplable lights.
 */
enum class LightType
{
    EnvMap,
    Emissive,
    Analytic
};

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};


/**
 * Helper to load G-buffer data and prepare shading data.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};

    const TriangleHit triangleHit = hit.getTriangleHit();
    v = gScene.getVertexData(triangleHit);
    materialID = gScene.getMaterialID(triangleHit.instanceID);

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    return sd;
}

/** Generates a light sample on the environment map.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEnvMapSample(const ShadingData sd, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    if (!kUseEnvLight)
        return false;

    // Sample environment map.
    EnvMapSample lightSample;
    if (!gReSTIRData.envMapSampler.sample(sampleNext2D(sg), lightSample))
        return false;

    // Setup returned sample.
    ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    ls.pdf = lightSample.pdf;
    ls.origin = computeRayOrigin(sd.posW, dot(sd.faceN, lightSample.dir) >= 0 ? sd.faceN : -sd.faceN);
    ls.distance = kRayTMax;
    ls.dir = lightSample.dir;
    ls.G = 1;

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEmissiveSample(const ShadingData sd, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights)
        return false;

    TriangleLightSample tls;
    if (!gReSTIRData.emissiveSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(), upperHemisphere, sg, tls))
        return false;

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = computeRayOrigin(sd.posW, dot(sd.faceN, lightPos - sd.posW) >= 0 ? sd.faceN : -sd.faceN);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);
    ls.G = max(FLT_MIN, dot(tls.normalW, -tls.dir) / max(FLT_MIN, tls.distance * tls.distance)); // Cosine term and distance squared term.

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the analytic lights.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateAnalyticLightSample(const ShadingData sd, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    uint lightCount = gScene.getLightCount();
    if (!kUseAnalyticLights || lightCount == 0)
        return false;

    // Sample analytic light source selected uniformly from the light list.
    // TODO: Sample based on estimated contributions as pdf.
    uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

    // Sample local light source.
    AnalyticLightSample lightSample;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, lightSample))
        return false;

    // Setup returned sample.
    ls.pdf = lightSample.pdf / lightCount;
    ls.Li = lightSample.Li * lightCount;
    // Offset shading position to avoid self-intersection.
    ls.origin = computeRayOrigin(sd.posW, dot(sd.faceN, lightSample.dir) >= 0 ? sd.faceN : -sd.faceN);
    // Analytic lights do not currently have a geometric representation in the scene.
    // Do not worry about adjusting the ray length to avoid self-intersections at the light.
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;
    ls.G = max(FLT_MIN, dot(lightSample.normalW, -lightSample.dir) / max(FLT_MIN, lightSample.distance * lightSample.distance)); // Cosine term and distance squared term.

    return any(ls.Li > 0.f);
}

/** Return the probabilities for selecting different light types.
    \param[out] p Probabilities.
*/
void getLightTypeSelectionProbabilities(out float p[3])
{
    // Set relative probabilities of the different sampling techniques.
    // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
    p[0] = kUseEnvLight ? 1.f : 0.f;
    p[1] = kUseEmissiveLights ? 1.f : 0.f;
    p[2] = kUseAnalyticLights ? 1.f : 0.f;

    // Normalize probabilities. Early out if zero.
    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f)
        return;

    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;
}

float getEnvMapSelectionProbability()
{
    float p[3];
    getLightTypeSelectionProbabilities(p);
    return p[0];
}
float getEmissiveSelectionProbability()
{
    float p[3];
    getLightTypeSelectionProbabilities(p);
    return p[1];
}
float getAnalyicSelectionProbability()
{
    float p[3];
    getLightTypeSelectionProbabilities(p);
    return p[2];
}

/** Select a light type for sampling.
    \param[out] lightType Selected light type.
    \param[out] pdf Probability for selected type.
    \param[in,out] sg Sample generator.
    \return Return true if selection is valid.
*/
bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
{
    float p[3];
    getLightTypeSelectionProbabilities(p);

    float u = sampleNext1D(sg);

    [unroll] for (lightType = 0; lightType < 3; ++lightType)
    {
        if (u < p[lightType])
        {
            pdf = p[lightType];
            return true;
        }
        u -= p[lightType];
    }

    lightType = {};
    pdf = {};

    return false;
}

/** Samples a light source in the scene.
     This function first stochastically selects a type of light source to sample,
     and then calls that the sampling function for the chosen light type.
     \param[in] vertex Path vertex.
     \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
     \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
     \param[in,out] sg Sample generator.
     \param[out] ls Struct describing valid samples.
     \return True if the sample is valid and has nonzero contribution, false otherwise.
 */
bool generateLightSample(
    const ShadingData sd,
    const bool sampleUpperHemisphere,
    const bool sampleLowerHemisphere,
    inout SampleGenerator sg,
    out LightSample ls
)
{
    ls = {};

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg))
        return false;

    bool valid = false;
    if (kUseEnvLight && lightType == (uint)LightType::EnvMap)
        valid = generateEnvMapSample(sd, sg, ls);
    if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
    {
        // Emissive light samplers have an option to exclusively sample the upper hemisphere.
        bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
        valid = generateEmissiveSample(sd, upperHemisphere, sg, ls);
    }
    if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
    {
        valid = generateAnalyticLightSample(sd, sg, ls);
    }
    if (!valid)
        return false;

    // Reject samples in non-requested hemispheres.
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
        return false;

    // Account for light type selection.
    ls.lightType = lightType;
    ls.pdf *= selectionPdf;
    ls.Li /= selectionPdf;

    return true;
}

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera)
{
    if (is_valid(gViewW))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gViewW[launchIndex].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

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

struct Reservoir
{
    /* data */
    ReSTIRSample outputSample;
    float weightSum = 0;
    uint M = 0;
};

void InitReservoir(inout Reservoir r)
{
    r.weightSum = 0;
    r.M = 0;

    r.outputSample.isValid = false;
    r.outputSample.W = 0;

    r.outputSample.lightSample.Li = 0;
    r.outputSample.lightSample.pdf = 0;
    r.outputSample.lightSample.origin = 0;
    r.outputSample.lightSample.distance = 0;
    r.outputSample.lightSample.dir = 0;
    r.outputSample.lightSample.lightType = 0;
    r.outputSample.lightSample.G = 0;
}

void UpdateReservoir(inout Reservoir r, in ReSTIRSample inSample, float weight, inout SampleGenerator sg)
{
    r.weightSum += weight;
    r.M += 1;
    if (r.weightSum > 0.f && sampleNext1D(sg) < weight / r.weightSum)
    {
        r.outputSample = inSample;
    }
}

float luminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
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

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before.
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        0 /* hitIdx */,
        rayTypeCount,
        0 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
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
    float W_Y = 0;
    float w_sum = 0;
    float Y_p_hat = 0;

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


        gOutputSamples[pixel.x + pixel.y * frameDim.x].isValid = false;
        // 目标函数 p _hat  = f * L * G * V
        // m_i = 1 / M
        // p_Xi = 光源采样pdf
        // W_Xi = 1 / p_Xi
        // w_i = m_i * p_hat * W_Xi
        // 因此w_i = (1/M)*p_hat/p_Xi
        if (kComputeDirect)
        {
            for (uint i = 0; i < kCandidateNum; i++)
            {
                ReSTIRSample sample = GenerateSample(sd, mi, sg);
                const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, sample.lightSample.dir) >= 0.f ? sd.faceN : -sd.faceN);
                float V = (float)traceShadowRay(origin, sample.lightSample.dir, sample.lightSample.distance);
                float p_hat = luminance(mi.eval(sd, sample.lightSample.dir, sg)) * luminance(sample.lightSample.Li) * V; // G和pdf包含在Li中了
                float w_i = 1.0 / kCandidateNum * p_hat;
                UpdateReservoir(r, sample, w_i, sg);
            }

            if (r.outputSample.isValid)
            {
                const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, r.outputSample.lightSample.dir) >= 0.f ? sd.faceN : -sd.faceN);
                float V = (float)traceShadowRay(origin, r.outputSample.lightSample.dir, r.outputSample.lightSample.distance);
                float p_hat = luminance(mi.eval(sd, r.outputSample.lightSample.dir, sg)) * luminance(r.outputSample.lightSample.Li * r.outputSample.lightSample.pdf) * r.outputSample.lightSample.G * V;
                r.outputSample.W = r.weightSum / max(FLT_MIN, p_hat);

                // debug
                W_Y = r.outputSample.W;
                Y_p_hat = p_hat;
                w_sum = r.weightSum;
                //

                outColor = (mi.eval(sd, r.outputSample.lightSample.dir, sg)) * (r.outputSample.lightSample.Li * r.outputSample.lightSample.pdf) * r.outputSample.lightSample.G * V * r.outputSample.W;
            }

            outColor += mi.getProperties(sd).emission;

            gOutputSamples[pixel.x + pixel.y * frameDim.x] = r.outputSample;
        }
    }
    else
    {
        gOutputSamples[pixel.x + pixel.y * frameDim.x] = r.outputSample;

        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
    }

    gWY[pixel] = float4(float3(W_Y), 1);
    gphat[pixel] = float4(float3(Y_p_hat), 1);
    gwsum[pixel] = float4(float3(w_sum), 1);
    gOutputColor[pixel] = float4(outColor, 1.0f);
}
