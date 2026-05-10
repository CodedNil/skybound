stackcheck:
    spirv-dis assets/shaders/raymarch.spv | grep "OpVariable" | grep "Function" | wc -l

shadercheck:
    spirv-dis assets/shaders/raymarch.spv -o assets/shaders/raymarch.spvasm
