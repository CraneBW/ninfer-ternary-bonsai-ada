// Source-level Companion model-config oracle for v3 header-only artifacts.
// Compile this against the Companion checkout's actual model config parsers;
// it never loads model weights or starts a GPU runtime.
#include "models/gemma4/config.h"
#include "models/qwen3_5/config.h"
#include "models/qwen4_exp/config.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

using Json = nlohmann::json;

Json read_header_directory(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::array<unsigned char, 32> header{};
    if (!in.read(reinterpret_cast<char*>(header.data()), header.size()) ||
        std::string_view(reinterpret_cast<const char*>(header.data()), 8) !=
            std::string_view("NINFER\0\3", 8)) {
        throw std::runtime_error("not a complete v3 header: " + path);
    }
    std::uint64_t bytes = 0;
    for (int i = 0; i < 8; ++i) {
        bytes |= static_cast<std::uint64_t>(header[8 + i]) << (8 * i);
    }
    if (bytes == 0 || bytes > 64ULL * 1024 * 1024) {
        throw std::runtime_error("unsupported v3 directory size");
    }
    std::string text(bytes, '\0');
    if (!in.read(text.data(), text.size())) {
        throw std::runtime_error("incomplete v3 JSON directory");
    }
    return Json::parse(text);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: validate_cpp_v3_configs <header.part>\n";
        return 2;
    }
    try {
        const Json directory = read_header_directory(argv[1]);
        const auto& components = directory.at("components");
        const auto& config = components.at("text").at("config");
        const auto& architecture = config.at("architectures").at(0).get_ref<const std::string&>();
        if (architecture == "Gemma4ForCausalLM") {
            const auto text = ninfer::models::gemma4::parse_text_config(config);
            const auto& assistant = components.at("mtp").at("config");
            (void)ninfer::models::gemma4::parse_assistant_config(assistant, text);
            const auto& vision = components.at("vision").at("config");
            (void)ninfer::models::gemma4::parse_vision_config(vision, text);
            std::cout << "Gemma4 text+MTP+Vision native configs OK: "
                      << text.num_hidden_layers << " layers\n";
        } else if (architecture == "Qwen4ExpForCausalLM") {
            const auto text = ninfer::models::qwen4_exp::parse_text_config(config);
            const auto& mtp = components.at("mtp").at("config");
            (void)ninfer::models::qwen4_exp::parse_mtp_config(mtp, text);
            ninfer::artifact::Directory vision_directory;
            ninfer::artifact::Component vision_component;
            vision_component.config = components.at("vision").at("config");
            if (components.at("vision").contains("target")) {
                vision_component.target =
                    components.at("vision").at("target").get<std::string>();
            }
            vision_directory.components.emplace("vision", std::move(vision_component));
            (void)ninfer::models::qwen3_5::parse_vision_config(vision_directory);
            std::cout << "Qwen4Exp text+MTP+Vision native configs OK: "
                      << text.num_hidden_layers << " layers, "
                      << text.num_experts << " experts\n";
        } else if (architecture == "Qwen3_5ForCausalLM" ||
                   architecture == "Qwen3_5MoeForCausalLM") {
            ninfer::artifact::Directory input;
            input.metadata = directory.value("metadata", Json::object());
            for (const auto& [name, component] : components.items()) {
                ninfer::artifact::Component parsed;
                parsed.config = component.at("config");
                if (component.contains("target") && component.at("target").is_string()) {
                    parsed.target = component.at("target").get<std::string>();
                }
                input.components.emplace(name, std::move(parsed));
            }
            ninfer::models::LoadOptions options;
            options.vision = components.contains("vision");
            auto parsed = ninfer::models::qwen3_5::parse_config(input, options);
            if (components.contains("mtp")) {
                options.speculative = ninfer::SpeculativeBackend::Mtp;
                parsed = ninfer::models::qwen3_5::parse_config(input, options);
            }
            if (components.contains("dflash2")) {
                options.speculative = ninfer::SpeculativeBackend::DFlash2;
                parsed = ninfer::models::qwen3_5::parse_config(input, options);
            }
            std::cout << "Qwen3.5/Bonsai native configs OK: "
                      << parsed.text.num_hidden_layers << " layers, Hadamard "
                      << parsed.text.hadamard_block_size << "\n";
        } else {
            throw std::runtime_error("not a supported Companion config");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
