
singleton Material(tig_road_rubber_sticky)
{
    mapTo = "tig_road_rubber_sticky";
    diffuseMap[0] = "levels/tig/art/road/road_asphalt_2lane_d.dds";
    doubleSided = "0";
    translucentBlendOp = "LerpAlpha";
    normalMap[0] = "levels/tig/art/road/road_asphalt_2lane_n.dds";
    specularPower[0] = "1";
    useAnisotropic[0] = "1";
    materialTag0 = "RoadAndPath";
    materialTag1 = "beamng";
    specularMap[0] = "levels/tig/art/road/road_asphalt_2lane_s.dds";
    reflectivityMap[0] = "levels/tig/art/road/road_rubber_sticky_d.dds";
    cubemap = "global_cubemap_metalblurred";
    translucent = "1";
    translucentZWrite = "1";
    alphaTest = "0";
    alphaRef = "255";
    castShadows = "0";
    specularStrength[0] = "0";
};




singleton Material(tig_line_white)
{
    mapTo = "tig_line_white";
    doubleSided = "0";
    translucentBlendOp = "LerpAlpha";
    normalMap[0] = "levels/tig/art/road/line_white_n.dds";
    specularPower[0] = "1";
    useAnisotropic[0] = "1";
    materialTag0 = "RoadAndPath";
    materialTag1 = "beamng";
    //cubemap = "cubemap_road_sky_reflection";
    //specularMap[0] = "levels/tig/art/road/line_white_s.dds";
    translucent = "1";
    translucentZWrite = "1";
    alphaTest = "0";
    alphaRef = "255";
    castShadows = "0";
    specularStrength[0] = "0";
   colorMap[0] = "levels/tig/art/road/line_white_d.dds";
   annotation = "SOLID_LINE";
   specularStrength0 = "0";
   specularColor0 = "1 1 1 1";
   materialTag2 = "driver_training";
};

singleton Material(tig_line_yellow)
{
    mapTo = "tig_line_yellow";
    doubleSided = "0";
    translucentBlendOp = "LerpAlpha";
    normalMap[0] = "levels/tig/art/road/line_white_n.dds";
    specularPower[0] = "1";
    useAnisotropic[0] = "1";
    materialTag0 = "RoadAndPath";
    materialTag1 = "beamng";
    //cubemap = "cubemap_road_sky_reflection";
    //specularMap[0] = "levels/tig/art/road/line_yellowblack_s.dds";
    translucent = "1";
    translucentZWrite = "1";
    alphaTest = "0";
    alphaRef = "255";
    castShadows = "0";
    specularStrength[0] = "0";
    annotation = "SOLID_LINE";
   colorMap[0] = "levels/tig/art/road/line_yellow_d.dds";
   specularStrength0 = "0";
};
