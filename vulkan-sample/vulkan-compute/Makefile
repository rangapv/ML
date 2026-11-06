UNAME := $(shell uname)
CXXFLAGS = -std=c++17 -O2

# Shader compiler: glslc (Vulkan SDK) if present, otherwise glslangValidator
GLSLC := $(shell command -v glslc 2>/dev/null)
ifeq ($(GLSLC),)
SHADER_CMD = glslangValidator -V shader.comp -o shader.spv
else
SHADER_CMD = glslc shader.comp -o shader.spv
endif

ifeq ($(UNAME),Darwin)
# macOS: set VULKAN_SDK (e.g. source ~/VulkanSDK/<version>/setup-env.sh) or edit this line
VULKAN_SDK ?= $(HOME)/VulkanSDK/1.3.xxx.x/macOS
CXXFLAGS += -I$(VULKAN_SDK)/include
LDFLAGS = -L$(VULKAN_SDK)/lib -lvulkan -Wl,-rpath,$(VULKAN_SDK)/lib
RUNENV = VK_ICD_FILENAMES=$(VULKAN_SDK)/share/vulkan/icd.d/MoltenVK_icd.json
else
LDFLAGS = -lvulkan
RUNENV =
endif

compute: compute.cpp shader.spv
	$(CXX) $(CXXFLAGS) -o compute compute.cpp $(LDFLAGS)

shader.spv: shader.comp
	$(SHADER_CMD)

.PHONY: run clean
run: compute
	$(RUNENV) ./compute

clean:
	rm -f compute shader.spv
