// Recompute the `id` of drift-corpus records from their redacted text, so a
// capture made under an older shape_key() folds with today's. The redacted
// text is never changed -- only the dedup key derived from it.
//
//   ./build/drift_rekey < capture.jsonl > rekeyed.jsonl
//
// Lines that do not parse or lack `redacted` pass through untouched.
#include "drift_capture.h"
#include <iostream>
#include <string>

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        try {
            nlohmann::json rec = nlohmann::json::parse(line);
            if (rec.is_object() && rec.contains("redacted") && rec["redacted"].is_string())
                rec["id"] = q27::drift_hex(q27::shape_hash(rec["redacted"].get<std::string>()));
            std::cout << rec.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
        } catch (...) {
            std::cout << line << '\n';
        }
    }
    return 0;
}
