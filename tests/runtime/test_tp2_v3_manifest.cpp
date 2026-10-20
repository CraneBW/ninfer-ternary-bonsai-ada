#include "runtime/tp2/v3_rank_manifest.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ninfer_tp2_v3_manifest_test <rank-directory.json>\n";
        return 2;
    }
    try {
        std::ifstream file(argv[1]);
        if (!file) { throw std::runtime_error("cannot open rank-directory.json"); }
        nlohmann::json directory;
        file >> directory;
        const auto manifest = ninfer::runtime::tp2::inspect_v3_rank_manifest(directory);
        // 48 GDN layers with 8 projections, 16 attention with 5,
        // and 64 MLP blocks with three projections.
        const std::size_t expected = 48 * 8 + 16 * 5 + 64 * 3;
        if (manifest.projections.size() != expected) {
            throw std::runtime_error("unexpected number of rank-local projection bindings");
        }
        std::cout << "rank=" << manifest.rank << " projections="
                  << manifest.projections.size() << "\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
